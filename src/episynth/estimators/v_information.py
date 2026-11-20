r"""Budget-limited predictive V-information for context → forecast windows.

Following Xu et al. (ICLR 2020), fit context-free and conditional predictors,
select on validation, and measure paired test NLL reduction in bits/window.
The predictive family includes the option to ignore context. Test uncertainty
conditions on the fitted models; optimization error is not a PAC guarantee.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import torch

from ..generators import Generator
from ..model.moirai import MoiraiConfig, MoiraiModule
from ..splits import source_description, validate_splits
from ..training import TrainConfig, Trainer, window_nll
from ..windows import WindowSpec, WindowStream
from .registry import register

_LN2 = math.log(2)
_MANIFEST = "v_information.json"


class VInformationResult(TypedDict):
    """Paired test measurement; entropy and error fields are bits per window."""

    method: Literal["v_information"]
    units: Literal["bits_per_window"]
    unconditional_entropy_bits_per_window: float
    conditional_entropy_bits_per_window: float
    v_information_bits_per_window: float
    standard_error_bits_per_window: float
    n_test_windows: int
    metadata: dict[str, Any]


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def _read_losses(path: Path, n: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size != n * np.dtype("float32").itemsize:
        raise ValueError(f"missing or incomplete loss artifact: {path}")
    values = np.fromfile(path, dtype=np.float32).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite losses: {path}")
    return values


def _weights_digest(model: MoiraiModule) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _stream(generator: Generator, spec: WindowSpec, config: TrainConfig,
            first: int, n: int) -> WindowStream:
    return WindowStream(generator, spec, config.eval_batch_size, first_seed=first,
                        n_batches=math.ceil(n / config.eval_batch_size),
                        workers=config.stream_workers, seed_limit=first + n)


def _normalization(generator: Generator, spec: WindowSpec, config: TrainConfig) -> tuple[float, float]:
    """Parallel-variance update over train horizon points only, in float64."""
    count, mean, m2 = 0, 0.0, 0.0
    for batch in _stream(generator, spec, config, config.first_seed, config.n_windows):
        mask = batch.prediction_mask.unsqueeze(-1) & batch.observed_mask
        values = batch.target[mask].double().numpy()
        if not np.isfinite(values).all() or not values.size:
            raise ValueError("nonfinite or empty training horizon")
        size, part_mean = values.size, float(values.mean())
        delta = part_mean - mean
        m2 += float(((values - part_mean) ** 2).sum()) + delta**2 * count * size / (count + size)
        mean += delta * size / (count + size)
        count += size
    scale = math.sqrt(m2 / count + 1e-5)
    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError("nonfinite training normalization")
    return mean, scale


def _load_model(path: Path, variant: str) -> MoiraiModule:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = MoiraiModule(MoiraiConfig.model_validate(state["model_config"]))
    model.load_state_dict(state["model"])
    if variant == "ema":
        with torch.no_grad():
            for parameter, average in zip(model.parameters(), state["ema"], strict=True):
                parameter.copy_(average)
    return model


@torch.no_grad()
def _score(model: MoiraiModule, generator: Generator, spec: WindowSpec, config: TrainConfig,
           first: int, n: int, path: Path) -> None:
    """Atomic, streamed, ordered scoring; no full test batch on the accelerator."""
    model = model.to(config.device).eval()
    temporary = path.with_suffix(path.suffix + ".tmp")
    device_type = torch.device(config.device).type
    with temporary.open("wb") as output:
        for batch in _stream(generator, spec, config, first, n):
            batch = batch.to(config.device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=config.eval_precision == "bf16"):
                mixture = model(**batch.model_inputs(), patch_size=spec.patch_size)
                losses = window_nll(mixture, batch, spec.patch_size).float().cpu().numpy()
            if not np.isfinite(losses).all():
                raise ValueError("nonfinite evaluation losses")
            losses.tofile(output)
    _read_losses(temporary, n)
    temporary.replace(path)


def _choose(candidates: list[dict]) -> dict:
    return min(candidates, key=lambda c: (c["validation_nats_per_window"],
                                         c["branch"] != "unconditional", c["step"]))


@register("v_information")
class VInformationEstimator:
    """Fit two predictors and report paired held-out predictive information.

    Args:
        variant: Select checkpoints using their EMA (default) or raw weights.
        test_first_seed: First seed of the untouched test split.
        test_windows: Test windows; at least two for paired standard error.
    """

    name = "v_information"

    def __init__(self, variant: str = "ema", test_first_seed: int = 2_000_000_000,
                 test_windows: int = 100_000) -> None:
        if variant not in ("ema", "raw"):
            raise ValueError("variant must be 'ema' or 'raw'")
        if not isinstance(test_windows, int) or isinstance(test_windows, bool) or test_windows < 2:
            raise ValueError("test_windows must be an integer >= 2")
        if not isinstance(test_first_seed, int) or isinstance(test_first_seed, bool) or test_first_seed < 0:
            raise ValueError("test_first_seed must be a nonnegative integer")
        self.variant = variant
        self.test_first_seed = test_first_seed
        self.test_windows = test_windows

    def run(self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path) -> VInformationResult:
        """Train/resume, select using validation only, then evaluate on test.

        The input model is left unchanged. Each fit receives the full n_windows
        budget; the normalization pass is additional data preparation.
        """
        splits = {
            "train": {"first_seed": config.first_seed, "n_windows": config.n_windows},
            "validation": {"first_seed": config.heldout_first_seed, "n_windows": config.heldout_windows},
            "test": {"first_seed": self.test_first_seed, "n_windows": self.test_windows},
        }
        validate_splits(generator, splits)
        if (config.batch_size <= 0 or config.eval_batch_size <= 0
                or config.n_windows % config.batch_size):
            raise ValueError("positive batch sizes and divisible training budget required")
        if (not config.checkpoint_fractions or 1.0 not in config.checkpoint_fractions
                or any(not 0 < f <= 1 for f in config.checkpoint_fractions)):
            raise ValueError("checkpoint_fractions must lie in (0, 1] and include 1.0")
        if (generator.length != spec.series_length or spec.patch_size not in model.config.patch_sizes
                or spec.n_tokens > model.config.max_seq_len):
            raise ValueError("incompatible generator, window or model shape")
        if "negative_binomial" in model.config.components:
            raise ValueError("V-information requires continuous density components")
        out = Path(out_dir)
        request = {"source": source_description(generator), "spec": spec.model_dump(mode="json"),
                   "train": config.model_dump(mode="json"), "model": model.config.model_dump(mode="json"),
                   "variant": self.variant, "splits": splits, "initial_weights_sha256": _weights_digest(model)}
        manifest_path = out / _MANIFEST
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema_version") != 1 or manifest.get("request") != request:
                raise ValueError("incompatible V-information run configuration or initialization")
        else:
            if any((out / branch).exists() for branch in ("unconditional", "conditional")):
                raise ValueError("branch artifacts exist without a V-information manifest")
            out.mkdir(parents=True, exist_ok=True)
            loc, scale = _normalization(generator, spec, config)
            manifest = {"schema_version": 1, "request": request,
                        "normalization": {"loc": loc, "scale": scale}, "complete": False}
            _write_json(manifest_path, manifest)
        normalization = manifest["normalization"]
        effective = {}
        for branch in ("unconditional", "conditional"):
            cfg = model.config.model_dump()
            cfg.update(scaling=True, fixed_scaling=(normalization["loc"], normalization["scale"]),
                       ignore_context=branch == "unconditional")
            effective[branch] = MoiraiConfig.model_validate(cfg).model_dump(mode="json")
        if "models" in manifest and manifest["models"] != effective:
            raise ValueError("incompatible saved normalization or model configuration")
        if manifest.get("complete"):
            result = self.from_run_dir(out)
            _write_json(out / "estimator_result.json", dict(result))
            return result
        manifest["models"] = effective
        _write_json(manifest_path, manifest)
        candidates = []
        for branch in ("unconditional", "conditional"):
            branch_model = copy.deepcopy(model)
            branch_model.config = MoiraiConfig.model_validate(effective[branch])
            trainer = Trainer(branch_model, generator, spec, config, out / branch)
            trainer.run()
            suffix = self.variant
            if config.eval_precision != config.precision:
                suffix += "_" + config.eval_precision
            for step, fraction in sorted(trainer.checkpoint_steps().items()):
                checkpoint = out / branch / f"ckpt_{fraction:g}.pt"
                if not checkpoint.is_file():
                    raise ValueError(f"missing validation checkpoint: {checkpoint}")
                losses_path = out / branch / f"heldout_{suffix}_{fraction:g}.f32"
                # A process may stop before validation or leave a damaged artifact.
                try:
                    losses = _read_losses(losses_path, config.heldout_windows)
                except ValueError:
                    candidate_model = _load_model(checkpoint, self.variant)
                    _score(candidate_model, generator, spec, config, config.heldout_first_seed,
                           config.heldout_windows, losses_path)
                    del candidate_model
                    losses = _read_losses(losses_path, config.heldout_windows)
                candidates.append({"branch": branch, "checkpoint": str(checkpoint.relative_to(out)),
                                   "step": step, "variant": self.variant,
                                   "validation_nats_per_window": float(losses.mean())})
            # Release parameters, optimizer and cached validation windows before the next fit.
            del trainer, branch_model
        baseline = _choose([c for c in candidates if c["branch"] == "unconditional"])
        conditional = _choose([baseline] + [c for c in candidates if c["branch"] == "conditional"])
        selection = {"unconditional": baseline, "conditional": conditional}
        if "selection" in manifest and manifest["selection"] != selection:
            raise ValueError("saved validation selection no longer matches artifacts")
        manifest["selection"] = selection
        _write_json(manifest_path, manifest)
        for role in ("unconditional", "conditional"):
            test_path = out / f"test_{role}.f32"
            if role == "conditional" and conditional == baseline:
                temporary = test_path.with_suffix(".f32.tmp")
                shutil.copyfile(out / "test_unconditional.f32", temporary)
                temporary.replace(test_path)
            else:
                selected_model = _load_model(out / selection[role]["checkpoint"], self.variant)
                _score(selected_model, generator, spec, config, self.test_first_seed, self.test_windows, test_path)
                del selected_model
        u = _read_losses(out / "test_unconditional.f32", self.test_windows)
        c = _read_losses(out / "test_conditional.f32", self.test_windows)
        temporary = out / "pvi_bits.f64.tmp"
        ((u - c) / _LN2).tofile(temporary)
        temporary.replace(out / "pvi_bits.f64")
        manifest["artifacts"] = {"unconditional": "test_unconditional.f32",
                                 "conditional": "test_conditional.f32", "pvi": "pvi_bits.f64"}
        manifest["complete"] = True
        _write_json(manifest_path, manifest)
        result = self.from_run_dir(out)
        _write_json(out / "estimator_result.json", dict(result))
        return result

    def from_run_dir(self, out_dir: str | Path) -> VInformationResult:
        """Read a completed measurement; reject missing, partial or nonfinite arrays.

        Negative empirical information is preserved. This method performs no
        fitting, selection, test generation or filesystem writes.
        """
        out = Path(out_dir)
        manifest = json.loads((out / _MANIFEST).read_text())
        if manifest.get("schema_version") != 1 or not manifest.get("complete"):
            raise ValueError("incomplete or unsupported V-information measurement")
        n = manifest["request"]["splits"]["test"]["n_windows"]
        if not isinstance(n, int) or n < 2:
            raise ValueError("invalid test size in manifest")
        u = _read_losses(out / "test_unconditional.f32", n)
        c = _read_losses(out / "test_conditional.f32", n)
        difference = (u - c) / _LN2
        pvi_path = out / "pvi_bits.f64"
        if not pvi_path.is_file() or pvi_path.stat().st_size != n * 8:
            raise ValueError("missing or incomplete PVI artifact")
        saved = np.fromfile(pvi_path, dtype=np.float64)
        if not np.isfinite(saved).all() or not np.array_equal(saved, difference):
            raise ValueError("PVI artifact does not match paired test losses")
        return {
            "method": "v_information", "units": "bits_per_window",
            "unconditional_entropy_bits_per_window": float(u.mean() / _LN2),
            "conditional_entropy_bits_per_window": float(c.mean() / _LN2),
            "v_information_bits_per_window": float(difference.mean()),
            "standard_error_bits_per_window": float(difference.std(ddof=1) / math.sqrt(n)),
            "n_test_windows": n,
            "metadata": {"family": "Moirai + optional ignorance; pointwise product over horizon",
                         "request": manifest["request"], "normalization": manifest["normalization"],
                         "models": manifest["models"], "selection": manifest["selection"],
                         "artifacts": manifest["artifacts"],
                         "uncertainty": "paired test sampling error conditional on fitted models"},
        }
