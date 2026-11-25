"""Estimate a conditional student's code length or transmit finite ORC indices.

Each observed forecast horizon is one message. The leading estimate is the
signed sum of pre-update EMA log likelihood ratios, in bits per trained model.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

import torch

from ..generators.base import Generator
from ..model.moirai import Mixture, MoiraiModule
from ..splits import source_description, validate_splits
from ..training import TrainConfig, param_groups, weights_digest, window_nll
from ..windows import WindowBatch, WindowSpec, WindowStream, window_batch
from . import _requential_codec as codec
from .base import write_estimator_result
from .registry import register

_LN2 = math.log(2)
_KAPPA = 5.21


def _public(batch: WindowBatch) -> WindowBatch:
    return replace(batch, target=batch.target.masked_fill(batch.prediction_mask.unsqueeze(-1), 0))


def _with_draw(public: WindowBatch, draw: torch.Tensor) -> WindowBatch:
    mask = public.prediction_mask.unsqueeze(-1) & public.observed_mask
    values = torch.where(mask, draw[..., : public.target.shape[-1]], public.target)
    return replace(public, target=values)


def _check_finite(value: torch.Tensor, label: str, step: int) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {label} at step {step}")


def _ratio_bits(
    student: Mixture, teacher: Mixture, batch: WindowBatch, patch: int, step: int
) -> torch.Tensor:
    bits = (
        window_nll(student, batch, patch).double() - window_nll(teacher, batch, patch).double()
    ) / _LN2
    _check_finite(bits, "log likelihood ratio", step)
    return bits


class _Learner:
    """A raw model, its EMA, and its independent AdamW and warmup clocks."""

    def __init__(
        self,
        model: MoiraiModule,
        spec: WindowSpec,
        config: TrainConfig,
        alpha: float,
        teacher: bool = False,
    ) -> None:
        self.raw = copy.deepcopy(model).to(config.device).train()
        self.ema = copy.deepcopy(self.raw).eval().requires_grad_(False)
        self.forward = torch.compile(self.raw) if config.compile else self.raw
        self.ema_forward = torch.compile(self.ema) if config.compile else self.ema
        self.optimizer = torch.optim.AdamW(
            param_groups(self.raw, config.weight_decay),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            fused=config.fused_adam and config.device.startswith("cuda"),
        )
        self.spec = spec
        self.config = config
        self.alpha = alpha
        self.teacher = teacher
        self.clock = 0

    def _distribution(self, batch: WindowBatch, ema: bool) -> Mixture:
        precision = self.config.eval_precision if ema else self.config.precision
        with torch.autocast(batch.target.device.type, torch.bfloat16, enabled=precision == "bf16"):
            model = self.ema_forward if ema else self.forward
            return model(**batch.model_inputs(), patch_size=self.spec.patch_size)

    @torch.no_grad()
    def predict(self, batch: WindowBatch) -> Mixture:
        return self._distribution(batch, ema=True)

    def update(self, batch: WindowBatch, elapsed: int) -> None:
        cfg = self.config
        self.clock += 1
        for group in self.optimizer.param_groups:
            group["lr"] = cfg.lr * min(1.0, self.clock / max(1, cfg.warmup_steps))
        nll = window_nll(self._distribution(batch, ema=False), batch, self.spec.patch_size)
        _check_finite(
            nll, "teacher training NLL" if self.teacher else "student training NLL", elapsed
        )
        self.optimizer.zero_grad(set_to_none=True)
        (nll.mean() / self.spec.horizon).backward()
        norm = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), cfg.grad_clip)
        _check_finite(norm, "gradient norm", elapsed)
        self.optimizer.step()
        timescale = max(50 if self.teacher else 0, self.alpha * elapsed)
        with torch.no_grad():
            torch._foreach_lerp_(
                list(self.ema.parameters()),
                list(self.raw.parameters()),
                -math.expm1(-1 / timescale),
            )

    def project(self, student: _Learner) -> None:
        self.raw.load_state_dict(student.raw.state_dict())
        self.ema.load_state_dict(student.ema.state_dict())
        self.optimizer.load_state_dict(copy.deepcopy(student.optimizer.state_dict()))
        self.clock = student.clock

    @torch.no_grad()
    def evaluate(self, source: Generator, split: dict[str, int], step: int) -> float:
        size = self.config.eval_batch_size
        first, n = split["first_seed"], split["n_windows"]
        total = 0.0
        for start in range(first, first + n, size):
            batch = window_batch(source, range(start, min(start + size, first + n)), self.spec).to(
                self.config.device
            )
            losses = window_nll(self.predict(batch), batch, self.spec.patch_size)
            _check_finite(losses, "validation or held-out NLL", step)
            total += losses.double().sum().item()
        return total / n


def _code_window(
    student: Mixture,
    teacher: Mixture,
    public: WindowBatch,
    seed: int,
    step: int,
    window: int,
    candidates: int,
) -> tuple[int, torch.Tensor]:
    ratios, exponentials = [], []
    for index in range(1, candidates + 1):
        draw = codec.proposal(student, seed, step, window, index)
        synthetic = _with_draw(public, draw)
        bits = _ratio_bits(student, teacher, synthetic, public.target.shape[-1], step + 1)
        ratios.append(bits.item() * _LN2)
        exponentials.append(codec.arrival_increment(seed, step, window, index))
    index = codec.ordered_index(ratios, exponentials)
    return index, codec.proposal(student, seed, step, window, index)


def _projection_schedule(explicit: Sequence[int] | None, budget: int) -> set[int]:
    if explicit is not None:
        return set(explicit)
    steps = set()
    t = 100
    while t < budget:
        steps.add(t)
        t = round(1.5 * t)
    return steps


@register("requential")
class RequentialEstimator:
    """Code one conditional horizon per message and train once per batch.

    Args:
        teacher_ema: EMA timescale as a fraction of elapsed real updates, with a 50-update minimum.
        student_ema: EMA timescale as a fraction of elapsed student updates.
        projection_steps: Completed student updates before a projection. Omit for geometric spacing.
        kl_scale: Must equal one to preserve units of bits.
        public_first_seed: Start of the public range, defaulting to the end of teacher training.
        validation_first_seed: Start of projection validation, defaulting to the end of public contexts.
        validation_windows: Projection validation size, defaulting to the final evaluation size.
        mc_repeats: Independent direct teacher draws per message for the nonlinear bound estimate.
        sampling_seed: Shared seed for synthetic draws and the ORC race.
        mode: Direct teacher sampling or finite ordered random coding ('orc').
        candidates: Number of proposals per ORC message.
    """

    name = "requential"

    def __init__(
        self,
        teacher_ema: float = 0.1,
        student_ema: float = 0.01,
        projection_steps: Sequence[int] | None = None,
        kl_scale: float = 1.0,
        public_first_seed: int | None = None,
        validation_first_seed: int | None = None,
        validation_windows: int | None = None,
        mc_repeats: int = 8,
        sampling_seed: int = 0,
        mode: str = "direct",
        candidates: int = 16,
    ) -> None:
        if kl_scale != 1:
            raise ValueError("kl_scale must equal one")
        if not all(math.isfinite(x) and x > 0 for x in (teacher_ema, student_ema)):
            raise ValueError("EMA timescales must be finite and positive")
        if projection_steps is not None and any(
            type(t) is not int or t < 1 for t in projection_steps
        ):
            raise ValueError("projection_steps must contain positive integers")
        if mode not in {"direct", "orc"}:
            raise ValueError("mode must be 'direct' or 'orc'")
        if (
            type(mc_repeats) is not int
            or mc_repeats < 2
            or type(candidates) is not int
            or candidates < 1
        ):
            raise ValueError(
                "mc_repeats must be at least two and candidates must be positive integers"
            )
        if type(sampling_seed) is not int:
            raise ValueError("sampling_seed must be an integer")
        for value in (public_first_seed, validation_first_seed, validation_windows):
            if value is not None and type(value) is not int:
                raise ValueError("split starts and sizes must be integers")
        self.kl_scale = kl_scale
        self.teacher_ema = teacher_ema
        self.student_ema = student_ema
        self.projection_steps = None if projection_steps is None else list(projection_steps)
        self.public_first_seed = public_first_seed
        self.validation_first_seed = validation_first_seed
        self.validation_windows = validation_windows
        self.mc_repeats = mc_repeats
        self.sampling_seed = sampling_seed
        self.mode = mode
        self.candidates = candidates

    def _public_start(self, config: TrainConfig) -> int:
        return (
            config.first_seed + config.n_windows
            if self.public_first_seed is None
            else self.public_first_seed
        )

    def _public_batch(
        self, source: Generator, spec: WindowSpec, config: TrainConfig, step: int
    ) -> WindowBatch:
        first = self._public_start(config) + step * config.batch_size
        batch = window_batch(source, range(first, first + config.batch_size), spec)
        return _public(batch.to(config.device))

    def _splits(
        self, model: MoiraiModule, source: Generator, spec: WindowSpec, cfg: TrainConfig
    ) -> dict:
        if (
            cfg.batch_size <= 0
            or cfg.n_windows <= 0
            or cfg.n_windows % cfg.batch_size
            or cfg.eval_batch_size <= 0
        ):
            raise ValueError(
                "positive batch sizes and a positive, batch-divisible training budget required"
            )
        if (
            source.length != spec.series_length
            or spec.patch_size not in model.config.patch_sizes
            or spec.n_tokens > model.config.max_seq_len
        ):
            raise ValueError("incompatible source, model, or window shape")
        components = set(model.config.components)
        if not components or components - {"normal_fixed", "student_t", "log_normal"}:
            raise ValueError("requential coding requires continuous density components")
        if "normal_fixed" in components and not (0 < model.config.normal_fixed_scale < math.inf):
            raise ValueError("normal_fixed_scale must be finite and positive")
        public = self._public_start(cfg)
        validation = (
            self.validation_first_seed
            if self.validation_first_seed is not None
            else public + cfg.n_windows
        )
        splits = {
            "teacher_train": {"first_seed": cfg.first_seed, "n_windows": cfg.n_windows},
            "public_context": {"first_seed": public, "n_windows": cfg.n_windows},
            "projection_validation": {
                "first_seed": validation,
                "n_windows": cfg.heldout_windows
                if self.validation_windows is None
                else self.validation_windows,
            },
            "final_eval": {"first_seed": cfg.heldout_first_seed, "n_windows": cfg.heldout_windows},
        }
        validate_splits(source, splits)
        return splits

    @torch.no_grad()
    def _message_batch(
        self, student: _Learner, teacher: _Learner, public: WindowBatch, step: int
    ) -> tuple[WindowBatch, dict, list[int]]:
        p, q = student.predict(public), teacher.predict(public)
        # Pure log-normal laws have a context-dependent lower endpoint.
        if set(q.names) == {"log_normal"} and not torch.equal(q.loc, p.loc):
            raise ValueError(f"teacher and student supports differ at step {step + 1}")
        patch = student.spec.patch_size
        seed = self.sampling_seed
        draw = codec.seeded_draw(q, seed, "direct", step)[0]
        synthetic = _with_draw(public, draw)
        bits = _ratio_bits(p, q, synthetic, patch, step + 1)
        repeats = codec.seeded_draw(q, seed, "overhead", step, count=self.mc_repeats)
        samples = torch.stack(
            [_ratio_bits(p, q, _with_draw(public, d), patch, step + 1) for d in repeats]
        )
        d = samples.mean(dim=0).clamp_min(0)
        overhead = 2 * torch.log1p(d) / _LN2 + _KAPPA
        # Delete one independent draw per message to estimate overhead uncertainty.
        leave_one_out = ((samples.sum(dim=0) - samples) / (self.mc_repeats - 1)).clamp_min(0)
        overhead_loo = 2 * torch.log1p(leave_one_out) / _LN2
        jackknife_var = (self.mc_repeats - 1) * overhead_loo.var(dim=0, unbiased=False).sum()
        record = {
            "student_step": step + 1,
            "estimated_kl_bits": bits.sum().item(),
            "estimated_overhead_bits": overhead.sum().item(),
            "overhead_jackknife_variance_bits2": jackknife_var.item(),
            "plugin_expected_length_bound_bits": (d + overhead).sum().item(),
            "conditional_mc_variance_bits2": samples.var(dim=0).sum().item(),
        }
        indices = []
        if self.mode == "orc":
            draws = []
            for i in range(len(public)):
                index, draw = _code_window(
                    codec.window_law(p, i),
                    codec.window_law(q, i),
                    public.slice(i, i + 1),
                    seed,
                    step,
                    i,
                    self.candidates,
                )
                indices.append(index)
                draws.append(draw)
            synthetic = _with_draw(public, torch.cat(draws))
        return synthetic, record, indices

    def run(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        out_dir: str | Path,
    ) -> dict:
        """Train within the real-window budget and write the checkpoint and measurement."""
        splits = self._splits(model, generator, spec, config)
        source = source_description(generator)
        student = _Learner(model, spec, config, self.student_ema)
        teacher = _Learner(model, spec, config, self.teacher_ema, teacher=True)
        schedule = _projection_schedule(self.projection_steps, config.n_steps)
        steps, events, indices = [], [], []
        real = iter(
            WindowStream(
                generator,
                spec,
                config.batch_size,
                config.first_seed,
                config.n_steps,
                workers=config.stream_workers,
            )
        )
        real_steps = 0

        def train_teacher() -> None:
            nonlocal real_steps
            real_steps += 1
            teacher.update(next(real).to(config.device), real_steps)

        while real_steps < config.n_steps:
            step = student.clock
            if step in schedule:
                validation = splits["projection_validation"]
                before = teacher.evaluate(generator, validation, step)
                teacher.project(student)
                loss = teacher.evaluate(generator, validation, step)
                start = real_steps
                while loss > before and real_steps < config.n_steps:
                    train_teacher()
                    loss = teacher.evaluate(generator, validation, step)
                events.append(
                    {
                        "student_step": step,
                        "message_count": step * config.batch_size,
                        "real_windows_before": start * config.batch_size,
                        "recovery_windows": (real_steps - start) * config.batch_size,
                        "previous_nll": before,
                        "recovered_nll": loss,
                        "complete": loss <= before,
                    }
                )
                if real_steps == config.n_steps:
                    break
            public = self._public_batch(generator, spec, config, step)
            synthetic, record, messages = self._message_batch(student, teacher, public, step)
            student.update(synthetic, step + 1)
            train_teacher()
            steps.append(record)
            indices.extend(messages)

        heldout = {
            key: learner.evaluate(generator, splits["final_eval"], student.clock)
            for key, learner in (("teacher", teacher), ("student", student))
        }
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        # Each published report retains its own checkpoint and optional stream on retries.
        bundle = Path(mkdtemp(prefix="requential-", dir=out))
        checkpoint = bundle / "student.pt"
        result_path = bundle / "estimator_result.json"
        artifacts = {"student_checkpoint": str(checkpoint), "result": str(result_path)}
        result = {
            "mode": self.mode,
            "estimated_kl_bits": math.fsum(s["estimated_kl_bits"] for s in steps),
            "estimated_overhead_bits": math.fsum(s["estimated_overhead_bits"] for s in steps),
            "plugin_expected_length_bound_bits": math.fsum(
                s["plugin_expected_length_bound_bits"] for s in steps
            ),
            "estimated_kl_mc_se_bits": math.sqrt(
                math.fsum(s["conditional_mc_variance_bits2"] for s in steps)
            ),
            "estimated_overhead_mc_se_bits": math.sqrt(
                math.fsum(s["overhead_jackknife_variance_bits2"] for s in steps)
            ),
            "mc_repeats": self.mc_repeats,
            "kappa_bits": _KAPPA,
            "overhead_convention": "Per horizon: floor the independent repeat mean at zero for the plug-in bound only.",
            "setup": "Conditional on exact shared initialization, update algorithm, public contexts, corpus if used, and seeds.",
            "framing": "Message count and meaningful bit length are shared metadata outside the payload.",
            "student_steps": student.clock,
            "real_windows": real_steps * config.batch_size,
            "synthetic_windows": student.clock * config.batch_size,
            "message_count": student.clock * config.batch_size,
            "heldout_nats_per_window": heldout,
            "steps": steps,
            "projections": events,
            "splits": splits,
            "runtime": {
                "torch_version": str(torch.__version__),
                "cpu_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            "source": source,
            "spec": spec.model_dump(mode="json"),
            "model": model.config.model_dump(mode="json"),
            "train": config.model_dump(mode="json"),
            "estimator_config": vars(self).copy(),
            "initial_weights_sha256": weights_digest(model),
            "artifacts": artifacts,
        }
        if self.mode == "orc":
            payload, length = codec.pack_indices(indices)
            stream = bundle / "indices.bin"
            stream.write_bytes(payload)
            artifacts["stream"] = str(stream)
            result.update(realized_message_bits=length, padding_bits=8 * len(payload) - length)
        torch.save(
            {
                "model": student.raw.state_dict(),
                "ema": [p.detach().cpu() for p in student.ema.parameters()],
                "model_config": result["model"],
                "spec": result["spec"],
                "step": student.clock,
            },
            checkpoint,
        )
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        write_estimator_result(out, result)
        return result

    def decode(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        payload: bytes,
        bit_length: int,
        message_count: int,
    ) -> tuple[MoiraiModule, MoiraiModule]:
        """Replay a framed ORC stream from the exact shared initialization and public contexts."""
        if self.mode != "orc":
            raise ValueError("decoding requires mode='orc'")
        indices = codec.unpack_indices(payload, bit_length, self.candidates)
        if (
            config.batch_size < 1
            or len(indices) != message_count
            or message_count > config.n_windows
            or message_count % config.batch_size
        ):
            raise ValueError("invalid ORC message count or incomplete student batch")
        validate_splits(
            generator,
            {
                "public_context": {
                    "first_seed": self._public_start(config),
                    "n_windows": config.n_windows,
                }
            },
        )
        student = _Learner(model, spec, config, self.student_ema)
        for offset in range(0, message_count, config.batch_size):
            step = student.clock
            public = self._public_batch(generator, spec, config, step)
            p = student.predict(public)
            draws = [
                codec.proposal(codec.window_law(p, i), self.sampling_seed, step, i, index)
                for i, index in enumerate(indices[offset : offset + config.batch_size])
            ]
            student.update(_with_draw(public, torch.cat(draws)), step + 1)
        return student.raw, student.ema
