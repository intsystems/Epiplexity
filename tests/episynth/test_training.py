"""Обучение: короткий прогон на CPU, файлы прогона, возобновление."""

import json

import numpy as np

from episynth.generators import GENERATORS
from episynth.model.moirai import MoiraiConfig
from episynth.training import TrainConfig, Trainer, build_model
from episynth.windows import WindowSpec


def _setup(tmp_path, n_windows: int):
    spec = WindowSpec(series_length=256, context_max=128, horizon=32, patch_size=16, context_lengths=(64, 128))
    cfg = TrainConfig(
        n_windows=n_windows,
        batch_size=8,
        warmup_steps=2,
        heldout_windows=16,
        eval_batch_size=8,
        precision="fp32",
        device="cpu",
        log_every=1,
        checkpoint_fractions=(0.5, 1.0),
    )
    model = build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(16,), components=("student_t", "normal_fixed")), 0)
    return Trainer(model, GENERATORS["waveform"](length=256), spec, cfg, tmp_path / "run"), cfg


def test_smoke_run_writes_logs_and_resumes(tmp_path):
    """Прогон пишет лосс каждого окна, шаги и отложенные оценки; повторный запуск возобновляет и не меняет файлы."""
    trainer, cfg = _setup(tmp_path, 48)
    trainer.run()
    out = tmp_path / "run"
    losses = np.fromfile(out / "window_loss.f32", dtype=np.float32)
    assert losses.shape == (48,) and np.isfinite(losses).all() and (losses != 0).all()
    scales = np.fromfile(out / "window_log_scale.f32", dtype=np.float32)
    assert np.isfinite(scales).all()
    held = [json.loads(line) for line in (out / "heldout.jsonl").read_text().splitlines()]
    assert [h["fraction"] for h in held] == [0.5, 1.0]
    assert all(np.isfinite(h["ema"]["nats_per_point"]) for h in held)
    assert (out / "ckpt_0.5.pt").exists() and (out / "ckpt_1.pt").exists()
    steps = (out / "steps.csv").read_text().splitlines()
    assert len(steps) == 1 + cfg.n_steps
    again, _ = _setup(tmp_path, 48)
    again.run()
    assert again.step == cfg.n_steps
    assert np.array_equal(losses, np.fromfile(out / "window_loss.f32", dtype=np.float32))
