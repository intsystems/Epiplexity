r"""Обучение модели на потоке окон с логом лосса каждого окна, EMA весов и отложенным набором.

Протокол: каждое окно модель видит один раз, лосс окна записывается до шага по нему, поэтому
сумма лоссов есть prequential-длина кода, а площадь над финальным лоссом — оценка эпиплексии
(Finzi и др., 2026, arXiv:2601.03220). Шаг обучения постоянный после линейного разогрева,
веса усредняются экспоненциально. Чекпойнты (текущие веса, EMA, оптимизатор) сохраняются на
заданных долях прогона, на каждом считается лосс обоих вариантов весов на отложенном наборе в
точности обучения (это финальный лосс для площади) и, если задана другая, в ней тоже.

Файлы в каталоге прогона: `config.json`; `window_loss.f32` и `window_log_scale.f32` — лосс
окна в натах (сумма по горизонту, в исходных единицах ряда) и логарифм масштаба нормировки
окна, по номеру окна, float32 без заголовка; `steps.csv` — шаги; `heldout.jsonl` — оценки на
отложенном наборе; `heldout_<вариант>_<доля>.f32` — лоссы окон отложенного набора на каждом
чекпойнте по вариантам `raw`, `ema` (точность обучения) и `raw_<точность>`, `ema_<точность>`
(другая точность оценки), float32 по номеру окна набора, для бутстрепа финального лосса;
`ckpt_<доля>.pt` и `last.pt`. Лосс в нормированных единицах равен лоссу окна минус горизонт,
умноженный на логарифм масштаба: :math:`\ell_{\mathrm{norm}} = \ell_{\mathrm{original}} - h\log s`.
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn

from .generators.base import Generator
from .model.moirai import (
    Mixture,
    MoiraiConfig,
    MoiraiModule,
    MultiInSizeLinear,
    MultiOutSizeLinear,
    RMSNorm,
)
from .splits import source_description, validate_splits
from .windows import WindowBatch, WindowSpec, WindowStream, collect_windows

RESUME_RUNTIME_FIELDS = frozenset({"device", "compile", "eval_batch_size", "stream_workers", "log_every", "fused_adam"})


class TrainConfig(BaseModel):
    r"""Параметры прогона обучения.

    Attributes:
        n_windows: Бюджет: число обучающих окон; кратно `batch_size`.
        batch_size: Окон на шаг.
        lr: Шаг обучения после разогрева.
        warmup_steps: Длина линейного разогрева в шагах.
        weight_decay: Коэффициент регуляризации весов линейных слоёв.
        betas: Моменты AdamW.
        eps: Слагаемое AdamW.
        grad_clip: Порог нормы градиента.
        ema_fraction: Постоянная времени EMA весов в долях прогона.
        checkpoint_fractions: Доли промежуточных чекпойнтов; финальный сохраняется всегда.
        heldout_windows: Размер отложенного набора.
        heldout_first_seed: Сид первого окна отложенного набора; диапазон не пересекается с обучающим.
        first_seed: Сид первого обучающего окна.
        eval_batch_size: Окон на батч при оценке.
        precision: Точность прямого прохода при обучении; лосс всегда в float32.
        eval_precision: Дополнительная точность оценки на отложенном наборе (fp32 или bf16).
        compile: Компилировать ли модель `torch.compile` для шагов обучения.
        fused_adam: Использовать ли fused AdamW (только CUDA).
        init_seed: Сид инициализации весов.
        device: Устройство.
        log_every: Через сколько шагов писать строку в `steps.csv`.
        stream_workers: Число процессов генерации окон; 0 — один фоновый поток.
    """

    n_windows: int
    batch_size: int
    lr: float = 1e-3
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    grad_clip: float = 1.0
    ema_fraction: float = 0.02
    checkpoint_fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    heldout_windows: int = 100_000
    heldout_first_seed: int = 1_000_000_000
    first_seed: int = 0
    eval_batch_size: int = 1024
    precision: Literal["fp32", "bf16"] = "bf16"
    eval_precision: Literal["fp32", "bf16"] = "fp32"
    compile: bool = False
    fused_adam: bool = True
    init_seed: int = 0
    device: str = "cuda"
    log_every: int = 50
    stream_workers: int = 0

    @property
    def n_steps(self) -> int:
        """Число шагов прогона."""
        return self.n_windows // self.batch_size


def window_nll(mixture: Mixture, batch: WindowBatch, patch_size: int) -> torch.Tensor:
    """Лосс каждого окна `(B,)` в натах: сумма минус лог-плотности по наблюдаемым точкам горизонта."""
    nll = -mixture.log_prob(batch.target)[..., :patch_size]
    mask = batch.prediction_mask.unsqueeze(-1) & batch.observed_mask[..., :patch_size]
    return nll.masked_fill(~mask, 0).sum(dim=(1, 2))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Группы параметров как в uni2ts: регуляризуются веса линейных слоёв, остальное нет."""
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if name.endswith("bias") or isinstance(module, (RMSNorm, nn.Embedding)):
                no_decay.append(p)
            elif isinstance(module, (nn.Linear, MultiInSizeLinear, MultiOutSizeLinear)):
                decay.append(p)
            else:
                no_decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    """Прогон обучения одной модели на одном генераторе с записью всех величин в каталог.

    Общий движок обучения: окна, чекпойнты, EMA весов, возобновление и heldout-оценка общие для
    всех протоколов оценки эпиплексии. Протокол-специфичный шаг вынесен в переопределяемый метод
    :meth:`compute_loss`, который возвращает лосс каждого окна батча и (необязательно) логарифм
    масштаба нормировки для записи. По умолчанию реализует prequential-протокол (площадь над
    кривой лосса): подклассы, например requential-учитель-студент, переопределяют `compute_loss`.

    Args:
        model: Модель; инициализирована извне, веса заменяются при возобновлении.
        generator: Генератор рядов.
        spec: Форма задачи.
        config: Параметры прогона.
        out_dir: Каталог прогона; существующий `last.pt` в нём возобновляет прогон.

    Attributes:
        step: Число сделанных шагов.
        ema: Экспоненциально усреднённые веса в порядке `model.parameters()`.
    """

    def __init__(self, model: MoiraiModule, generator: Generator, spec: WindowSpec, config: TrainConfig, out_dir: str | Path) -> None:
        assert config.n_windows % config.batch_size == 0
        validate_splits(generator, {
            "train": {"first_seed": config.first_seed, "n_windows": config.n_windows},
            "heldout": {"first_seed": config.heldout_first_seed, "n_windows": config.heldout_windows},
        })
        self.model = model.to(config.device)
        self.generator = generator
        self.spec = spec
        self.config = config
        self.out_dir = Path(out_dir)
        self.step = 0
        self.params = [p for p in self.model.parameters()]
        self.ema = [p.detach().clone().float() for p in self.params]
        self.ema_decay = math.exp(-1.0 / max(config.ema_fraction * config.n_steps, 1.0))
        use_fused = config.fused_adam and str(config.device).startswith("cuda")
        self.optimizer = torch.optim.AdamW(
            param_groups(self.model, config.weight_decay), lr=config.lr, betas=config.betas, eps=config.eps, fused=use_fused
        )
        self._forward = torch.compile(self.model) if config.compile else self.model
        self._heldout: WindowBatch | None = None

    def lr_at(self, step: int) -> float:
        """Шаг обучения на данном шаге: линейный разогрев, затем постоянный."""
        if self.config.warmup_steps == 0:
            return self.config.lr
        return self.config.lr * min(1.0, (step + 1) / self.config.warmup_steps)

    def compute_loss(self, batch: WindowBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Лосс каждого окна батча и (необязательно) лог масштаба нормировки.

        Протокол-специфичный шаг. Prequential-реализация возвращает `window_nll` по окну и лог
        масштаба нормировки модели; requential-реализация переопределяет и возвращает KL-оценку.

        Args:
            batch: Окна на устройстве модели.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: лосс `(B,)` в натах на окно и либо лог
                масштаба нормировки `(B,)`, либо `None`, если протокол его не пишет.
        """
        mix = self._forward(**batch.model_inputs(), patch_size=self.spec.patch_size)
        per_window = window_nll(mix, batch, self.spec.patch_size)
        scale_log = torch.log(mix.scale[:, -1, 0]).detach().float().cpu().numpy()
        return per_window, scale_log

    def _autocast(self, precision: str):
        return torch.autocast(device_type=str(self.config.device).split(":")[0], dtype=torch.bfloat16, enabled=precision == "bf16")

    def checkpoint_steps(self) -> dict[int, float]:
        """Шаги чекпойнтов с их долями прогона; финальный шаг включён всегда."""
        steps = {max(1, round(f * self.config.n_steps)): f for f in self.config.checkpoint_fractions}
        steps[self.config.n_steps] = 1.0
        return steps

    def _validate_saved_config(self, saved: dict, *, checkpoint: bool = False) -> None:
        """Reject a different run before replacing weights or modifying its logs."""
        if "source" not in saved:
            raise ValueError("cannot resume legacy run without source metadata; use a new output directory")
        train_key, model_key = ("config", "model_config") if checkpoint else ("train", "model")
        expected = {
            "train": self.config.model_dump(mode="json", exclude=RESUME_RUNTIME_FIELDS),
            "spec": self.spec.model_dump(mode="json"),
            "model": self.model.config.model_dump(mode="json"),
            "generator": self.generator.name,
            "source": source_description(self.generator),
        }
        actual = {
            "train": {k: v for k, v in saved.get(train_key, {}).items() if k not in RESUME_RUNTIME_FIELDS},
            "spec": saved.get("spec"),
            "model": saved.get(model_key),
            "generator": saved.get("generator"),
            "source": saved["source"],
        }
        # Checkpoints retain Python tuples; config.json contains JSON lists.
        actual = json.loads(json.dumps(actual))
        for key, value in expected.items():
            if actual[key] != value:
                raise ValueError(f"incompatible saved {key} configuration; use a new output directory")

    def save(self, path: Path) -> None:
        """Атомарно сохраняет веса, EMA, оптимизатор и номер шага."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": [t.cpu() for t in self.ema],
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.model_dump(),
                "spec": self.spec.model_dump(),
                "model_config": self.model.config.model_dump(),
                "generator": self.generator.name,
                "source": source_description(self.generator),
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path: Path) -> None:
        """Восстанавливает состояние из чекпойнта этого же прогона."""
        state = torch.load(path, map_location=self.config.device, weights_only=False)
        self._validate_saved_config(state, checkpoint=True)
        if not isinstance(state["step"], int) or not 0 <= state["step"] <= self.config.n_steps:
            raise ValueError("checkpoint step is outside the configured training budget")
        self.model.load_state_dict(state["model"])
        self.ema = [t.to(self.config.device) for t in state["ema"]]
        self.optimizer.load_state_dict(state["optimizer"])
        # Loading an optimizer also restores its saved runtime options.
        for group in self.optimizer.param_groups:
            group["fused"] = self.config.fused_adam and str(self.config.device).startswith("cuda")
        self.step = state["step"]

    def heldout(self) -> WindowBatch:
        """Отложенный набор окон, генерируется один раз и держится на устройстве."""
        if self._heldout is None:
            seeds = np.arange(self.config.heldout_first_seed, self.config.heldout_first_seed + self.config.heldout_windows)
            self._heldout = collect_windows(self.generator, self.spec, seeds, workers=self.config.stream_workers).to(self.config.device)
        return self._heldout

    @torch.no_grad()
    def evaluate(self, batch: WindowBatch, use_ema: bool, precision: str | None = None, save_to: Path | None = None) -> dict[str, float]:
        """Средний лосс на окно и на точку по батчу окон для текущих или EMA весов.

        Args:
            batch: Окна на устройстве модели.
            use_ema: Подставить ли EMA веса на время оценки.
            precision: Точность прямого прохода; `None` — `eval_precision` из конфига.
            save_to: Куда записать лоссы окон float32 по порядку окон батча; `None` — не писать.

        Returns:
            dict[str, float]: `nats_per_window`, `nats_per_point`, `n_windows`.
        """
        precision = precision or self.config.eval_precision
        was_training = self.model.training
        backup = [p.detach().clone() for p in self.params] if use_ema else None
        try:
            if use_ema:
                for p, e in zip(self.params, self.ema, strict=True):
                    p.copy_(e)
            self.model.eval()
            losses = []
            for i in range(0, len(batch), self.config.eval_batch_size):
                part = batch.slice(i, i + self.config.eval_batch_size)
                with self._autocast(precision):
                    per_window, _ = self.compute_loss(part)
                losses.append(per_window.double().cpu().numpy())
        finally:
            if backup is not None:
                for p, b in zip(self.params, backup, strict=True):
                    p.copy_(b)
            self.model.train(was_training)
        per_window = np.concatenate(losses)
        saved_losses = per_window.astype(np.float32)
        if not np.isfinite(saved_losses).all():
            raise ValueError("heldout evaluation produced nonfinite losses")
        if save_to is not None:
            temporary = save_to.with_suffix(save_to.suffix + ".tmp")
            saved_losses.tofile(temporary)
            temporary.replace(save_to)
        n = len(per_window)
        return {"nats_per_window": float(per_window.sum() / n), "nats_per_point": float(per_window.sum() / n / self.spec.horizon), "n_windows": n}

    def run(self) -> None:
        """Выполняет или возобновляет прогон до конца бюджета."""
        cfg = self.config
        self.out_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.out_dir / "config.json"
        last = self.out_dir / "last.pt"
        if not config_path.exists() and not last.exists() and any(
            (self.out_dir / name).exists() for name in ("window_loss.f32", "window_log_scale.f32", "steps.csv", "heldout.jsonl")
        ):
            raise ValueError("training artifacts exist without config.json or last.pt; use a new output directory")
        if config_path.exists():
            self._validate_saved_config(json.loads(config_path.read_text()))
        if last.exists():
            self.load(last)
        if not config_path.exists():
            temporary = config_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"train": cfg.model_dump(), "spec": self.spec.model_dump(), "model": self.model.config.model_dump(),
                     "generator": self.generator.name, "source": source_description(self.generator)},
                    indent=1,
                    ensure_ascii=False,
                )
            )
            temporary.replace(config_path)
        if self.step:
            for name in ("window_loss.f32", "window_log_scale.f32"):
                path = self.out_dir / name
                if not path.is_file() or path.stat().st_size != cfg.n_windows * np.dtype(np.float32).itemsize:
                    raise ValueError(f"missing or incomplete training log: {path}")
        self._truncate_steps()
        if self.step == cfg.n_steps:
            self._finalize()
            return
        mode = "r+" if self.step else "w+"
        loss_log = np.memmap(self.out_dir / "window_loss.f32", dtype=np.float32, mode=mode, shape=(cfg.n_windows,))
        scale_log = np.memmap(self.out_dir / "window_log_scale.f32", dtype=np.float32, mode=mode, shape=(cfg.n_windows,))
        steps_path = self.out_dir / "steps.csv"
        new_csv = not steps_path.exists()
        with open(steps_path, "a", newline="") as steps_file:
            writer = csv.writer(steps_file)
            if new_csv:
                writer.writerow(["step", "windows_seen", "nats_per_point", "lr", "grad_norm", "seconds"])
            self._train_loop(loss_log, scale_log, writer, steps_file)
        loss_log.flush()
        scale_log.flush()
        self._finalize()

    def _truncate_steps(self) -> None:
        """Drop rows written after the last durable checkpoint before resuming."""
        path = self.out_dir / "steps.csv"
        if not path.exists():
            return
        with path.open(newline="") as file:
            rows = list(csv.reader(file))
        retained = rows[:1]
        for row in rows[1:]:
            if row and row[0].isdigit() and int(row[0]) <= self.step:
                retained.append(row)
        if rows != retained:
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as file:
                csv.writer(file).writerows(retained)
            temporary.replace(path)

    def _evaluation_variants(self) -> list[tuple[str, bool, str]]:
        variants = [("raw", False, self.config.precision), ("ema", True, self.config.precision)]
        if self.config.eval_precision != self.config.precision:
            variants.extend((f"{name}_{self.config.eval_precision}", use_ema, self.config.eval_precision)
                            for name, use_ema in (("raw", False), ("ema", True)))
        return variants

    def _heldout_records(self) -> list[dict]:
        path = self.out_dir / "heldout.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        records = []
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Old append-based logs can end in an interrupted record.
                if i != len(lines) - 1:
                    raise ValueError(f"malformed heldout log: {path}") from None
        return records

    def _finalize(self) -> None:
        """Repair final evaluation after a stop between saving weights and scores."""
        checkpoint = self.out_dir / "ckpt_1.pt"
        if not checkpoint.exists():
            self.save(checkpoint)
        variants = self._evaluation_variants()
        path = self.out_dir / "heldout.jsonl"
        record = {}
        if path.exists():
            try:
                record = json.loads(path.read_text().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                record = {}
        if isinstance(record, dict) and record.get("step") == self.config.n_steps and record.get("fraction") == 1.0:
            complete = True
            for name, _, _ in variants:
                scores = record.get(name, {})
                path = self.out_dir / f"heldout_{name}_1.f32"
                if (not isinstance(scores, dict) or scores.get("n_windows") != self.config.heldout_windows
                        or any(not isinstance(scores.get(key), (int, float)) or not math.isfinite(scores[key])
                               for key in ("nats_per_window", "nats_per_point"))
                        or not path.is_file()
                        or path.stat().st_size != self.config.heldout_windows * np.dtype(np.float32).itemsize
                        or not np.isfinite(np.fromfile(path, dtype=np.float32)).all()):
                    complete = False
                    break
            if complete:
                return
        self._evaluate_checkpoint(1.0)

    def _evaluate_checkpoint(self, fraction: float) -> None:
        """Write all scores first, then atomically publish their completion record."""
        held = self.heldout()
        record = {"step": self.step, "fraction": fraction}
        for name, use_ema, precision in self._evaluation_variants():
            record[name] = self.evaluate(held, use_ema=use_ema, precision=precision,
                                         save_to=self.out_dir / f"heldout_{name}_{fraction:g}.f32")
        records = [r for r in self._heldout_records() if r.get("step") != self.step and r.get("fraction") != fraction]
        records.append(record)
        path = self.out_dir / "heldout.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(r) + "\n" for r in records))
        temporary.replace(path)

    def _train_loop(self, loss_log, scale_log, writer, steps_file) -> None:
        cfg = self.config
        ckpts = self.checkpoint_steps()
        last = self.out_dir / "last.pt"
        stream = WindowStream(self.generator, self.spec, cfg.batch_size, cfg.first_seed, cfg.n_steps, first_batch=self.step, workers=cfg.stream_workers)
        self.model.train()
        t0 = time.perf_counter()
        acc_loss, acc_n = 0.0, 0
        for batch in stream:
            batch = batch.to(cfg.device, non_blocking=True)
            lr = self.lr_at(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            with self._autocast(cfg.precision):
                per_window, scale_log_vals = self.compute_loss(batch)
            loss = per_window.mean() / self.spec.horizon
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
            self.optimizer.step()
            with torch.no_grad():
                torch._foreach_mul_(self.ema, self.ema_decay)
                torch._foreach_add_(self.ema, [p.detach().float() for p in self.params], alpha=1 - self.ema_decay)
            start = self.step * cfg.batch_size
            loss_log[start : start + cfg.batch_size] = per_window.detach().float().cpu().numpy()
            if scale_log_vals is not None:
                scale_log[start : start + cfg.batch_size] = scale_log_vals
            self.step += 1
            acc_loss += loss.item() * cfg.batch_size
            acc_n += cfg.batch_size
            if self.step % cfg.log_every == 0 or self.step == cfg.n_steps:
                writer.writerow([self.step, self.step * cfg.batch_size, acc_loss / acc_n, lr, float(grad_norm), time.perf_counter() - t0])
                steps_file.flush()
                acc_loss, acc_n = 0.0, 0
            if self.step in ckpts:
                loss_log.flush()
                scale_log.flush()
                self.save(self.out_dir / f"ckpt_{ckpts[self.step]:g}.pt")
                self.save(last)
                self._evaluate_checkpoint(ckpts[self.step])


def build_model(config: MoiraiConfig, init_seed: int) -> MoiraiModule:
    """Модель с детерминированной инициализацией."""
    torch.manual_seed(init_seed)
    return MoiraiModule(config)
