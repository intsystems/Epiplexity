"""Training: short run on CPU, run files, resumption."""

import csv
import json

import numpy as np
import pytest
import torch

from episynth.generators import GENERATORS
from episynth.model.moirai import MoiraiConfig
from episynth.training import TrainConfig, Trainer, build_model, window_nll
from episynth.windows import WindowSpec, make_windows


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
    """Run writes the loss of each window, steps, and held-out evaluations; restart resumes and does not change files."""
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
    per_window = np.fromfile(out / "heldout_ema_1.f32", dtype=np.float32)
    assert per_window.shape == (16,) and abs(per_window.mean() - held[-1]["ema"]["nats_per_window"]) < 1e-3
    assert (out / "heldout_raw_0.5.f32").exists()
    steps = (out / "steps.csv").read_text().splitlines()
    assert len(steps) == 1 + cfg.n_steps
    again, _ = _setup(tmp_path, 48)
    again.run()
    assert again.step == cfg.n_steps
    assert np.array_equal(losses, np.fromfile(out / "window_loss.f32", dtype=np.float32))


def _tiny(tmp_path, *, trainer_class=Trainer, config=None, generator=None, spec=None, model_config=None):
    spec = spec or WindowSpec(series_length=32, context_max=8, horizon=4, patch_size=4,
                              context_lengths=(8,), dequantize_width=0)
    config = config or TrainConfig(n_windows=8, batch_size=4, heldout_windows=4,
                                   eval_batch_size=4, checkpoint_fractions=(0.5,),
                                   device="cpu", precision="fp32", eval_precision="fp32",
                                   fused_adam=False, warmup_steps=0, log_every=1)
    model_config = model_config or MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(4,),
                                                components=("student_t", "normal_fixed"))
    model = build_model(model_config, config.init_seed)
    return trainer_class(model, generator or GENERATORS["waveform"](length=32), spec, config, tmp_path / "run")


def test_trainer_rejects_overlapping_splits_before_creating_output(tmp_path):
    trainer = _tiny(tmp_path)
    config = trainer.config.model_copy(update={"heldout_first_seed": 4})
    with pytest.raises(ValueError, match="splits overlap"):
        _tiny(tmp_path, config=config)
    assert not trainer.out_dir.exists()


@pytest.mark.parametrize("change", ["train", "spec", "model", "source"])
def test_resume_rejects_changed_identity_without_modifying_artifacts(tmp_path, change):
    trainer = _tiny(tmp_path)
    trainer.run()
    original = {p.name: p.read_bytes() for p in trainer.out_dir.iterdir()}
    kwargs = {}
    if change == "train":
        kwargs["config"] = trainer.config.model_copy(update={"first_seed": 50})
    elif change == "spec":
        kwargs["spec"] = trainer.spec.model_copy(update={"dequantize_width": 0.1})
    elif change == "model":
        kwargs["model_config"] = trainer.model.config.model_copy(update={"scaling": False})
    else:
        kwargs["generator"] = GENERATORS["ou"](length=32)
    with pytest.raises(ValueError, match="incompatible saved"):
        _tiny(tmp_path, **kwargs).run()
    assert {p.name: p.read_bytes() for p in trainer.out_dir.iterdir()} == original


def test_checkpoint_identity_is_checked_before_weights_are_loaded(tmp_path):
    trainer = _tiny(tmp_path)
    trainer.run()
    checkpoint = trainer.out_dir / "last.pt"
    saved = torch.load(checkpoint, weights_only=False)
    del saved["source"]
    torch.save(saved, checkpoint)
    resumed = _tiny(tmp_path)
    before = [p.detach().clone() for p in resumed.params]
    with pytest.raises(ValueError, match="without source metadata"):
        resumed.run()
    assert all(torch.equal(p, b) for p, b in zip(resumed.params, before, strict=True))
    assert resumed.step == 0


def test_resume_allows_runtime_overrides_and_always_has_final_evaluation(tmp_path):
    trainer = _tiny(tmp_path)
    trainer.run()
    assert (trainer.out_dir / "ckpt_1.pt").is_file()
    assert (trainer.out_dir / "heldout_ema_1.f32").is_file()
    config = trainer.config.model_copy(update={"eval_batch_size": 2, "log_every": 2, "fused_adam": True})
    resumed = _tiny(tmp_path, config=config)
    resumed.run()
    assert resumed.step == config.n_steps
    assert all(group["fused"] is False for group in resumed.optimizer.param_groups)


def test_resume_repairs_evaluation_interrupted_after_final_checkpoint(tmp_path):
    class InterruptedEvaluation(Trainer):
        def evaluate(self, *args, **kwargs):
            if self.step == self.config.n_steps:
                raise RuntimeError("interrupted evaluation")
            return super().evaluate(*args, **kwargs)

    trainer = _tiny(tmp_path, trainer_class=InterruptedEvaluation)
    with pytest.raises(RuntimeError, match="interrupted evaluation"):
        trainer.run()
    assert torch.load(trainer.out_dir / "last.pt", weights_only=False)["step"] == trainer.config.n_steps
    losses = (trainer.out_dir / "window_loss.f32").read_bytes()
    resumed = _tiny(tmp_path)
    resumed.run()
    records = [json.loads(line) for line in (resumed.out_dir / "heldout.jsonl").read_text().splitlines()]
    assert [(r["step"], r["fraction"]) for r in records] == [(1, 0.5), (2, 1.0)]
    assert (resumed.out_dir / "window_loss.f32").read_bytes() == losses
    assert np.fromfile(resumed.out_dir / "heldout_ema_1.f32", dtype=np.float32).shape == (4,)


@pytest.mark.parametrize("damage", ["missing_log", "missing_record", "truncated_record", "partial_tail", "nonfinite_record",
                                     "missing_losses", "partial_losses", "nonfinite_losses"])
def test_completed_resume_repairs_final_artifacts_once(tmp_path, damage):
    trainer = _tiny(tmp_path)
    trainer.run()
    path = trainer.out_dir / "heldout.jsonl"
    records = path.read_text().splitlines()
    if damage == "missing_log":
        path.unlink()
    elif damage == "missing_record":
        path.write_text(records[0] + "\n")
    elif damage == "truncated_record":
        path.write_text(records[0] + '\n{"step": 2, "fraction":')
    elif damage == "partial_tail":
        path.write_text(path.read_text() + '{"step":')
    elif damage == "nonfinite_record":
        final = json.loads(records[-1])
        final["ema"]["nats_per_window"] = float("nan")
        path.write_text(records[0] + "\n" + json.dumps(final) + "\n")
    elif damage == "missing_losses":
        (trainer.out_dir / "heldout_ema_1.f32").unlink()
    elif damage == "partial_losses":
        np.array([0.0], dtype=np.float32).tofile(trainer.out_dir / "heldout_ema_1.f32")
    else:
        np.full(4, np.nan, dtype=np.float32).tofile(trainer.out_dir / "heldout_ema_1.f32")
    resumed = _tiny(tmp_path)
    resumed.run()
    repaired = path.read_bytes()
    records = [json.loads(line) for line in repaired.splitlines()]
    assert [r["fraction"] for r in records] == ([1.0] if damage == "missing_log" else [0.5, 1.0])
    assert records[-1]["ema"]["n_windows"] == 4
    losses = np.fromfile(trainer.out_dir / "heldout_ema_1.f32", dtype=np.float32)
    assert losses.shape == (4,) and np.isfinite(losses).all()
    # A second resume must not evaluate again or append another completion row.
    resumed.evaluate = lambda *args, **kwargs: pytest.fail("completed run evaluated twice")
    resumed.run()
    assert path.read_bytes() == repaired


def test_unidentified_training_artifacts_are_not_overwritten(tmp_path):
    trainer = _tiny(tmp_path)
    trainer.out_dir.mkdir()
    path = trainer.out_dir / "window_loss.f32"
    path.write_bytes(b"unidentified partial run")
    with pytest.raises(ValueError, match="artifacts exist without"):
        trainer.run()
    assert path.read_bytes() == b"unidentified partial run"
    assert not (trainer.out_dir / "config.json").exists()


def test_nonfinite_evaluation_does_not_replace_saved_losses(tmp_path, monkeypatch):
    trainer = _tiny(tmp_path)
    trainer.model.eval()
    path = tmp_path / "existing.f32"
    np.ones(4, dtype=np.float32).tofile(path)
    monkeypatch.setattr(trainer, "compute_loss", lambda batch: (torch.full((len(batch),), float("nan")), None))
    with pytest.raises(ValueError, match="nonfinite losses"):
        trainer.evaluate(trainer.heldout(), use_ema=False, save_to=path)
    assert not trainer.model.training
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.float32), np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("training", [False, True])
def test_failed_ema_evaluation_restores_weights_and_original_mode(tmp_path, monkeypatch, training):
    trainer = _tiny(tmp_path)
    trainer.model.train(training)
    before = [p.detach().clone() for p in trainer.params]
    for parameter in trainer.ema:
        parameter.add_(1)

    def fail(batch):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(trainer, "compute_loss", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        trainer.evaluate(trainer.heldout(), use_ema=True)
    assert trainer.model.training is training
    assert all(torch.equal(p, b) for p, b in zip(trainer.params, before, strict=True))


def test_masked_unsupported_values_do_not_contaminate_nll():
    spec = WindowSpec(series_length=16, context_max=8, horizon=4, patch_size=4,
                      context_lengths=(4,), dequantize_width=0)
    batch = make_windows(np.ones((1, 16), dtype=np.float32), [0], spec)
    model = build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(4,),
                                    components=("log_normal",), scaling=False), 0)
    mixture = model(**batch.model_inputs(), patch_size=4)
    mask = batch.prediction_mask.unsqueeze(-1) & batch.observed_mask
    assert torch.isneginf(mixture.log_prob(batch.target)[~mask]).any()
    loss = window_nll(mixture, batch, 4)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("extra_width", [0, 4])
def test_unobserved_nan_values_preserve_scored_losses_and_finite_gradients(extra_width):
    spec = WindowSpec(series_length=32, context_max=12, horizon=4, patch_size=4,
                      context_lengths=(5,), dequantize_width=0)
    batch = make_windows(np.arange(1, 33, dtype=np.float32)[None], [0], spec)
    batch.target = torch.nn.functional.pad(batch.target, (0, extra_width))
    batch.observed_mask = torch.nn.functional.pad(batch.observed_mask, (0, extra_width), value=False)
    original = batch.target.clone()
    model = build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(4, 8),
                                    components=("student_t", "normal_fixed")), 0)
    mixture = model(**batch.model_inputs(), patch_size=4)
    scored = batch.prediction_mask.unsqueeze(-1) & batch.observed_mask[..., :4]
    expected = (-mixture.log_prob(batch.target)[..., :4]).masked_fill(~scored, 0).sum((1, 2))
    expected.sum().backward()
    expected_gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    batch.target[~batch.observed_mask] = float("nan")
    mixture = model(**batch.model_inputs(), patch_size=4)
    actual = window_nll(mixture, batch, 4)
    torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
    actual.sum().backward()
    for parameter, gradient in zip(model.parameters(), expected_gradients, strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    torch.testing.assert_close(batch.target[batch.observed_mask], original[batch.observed_mask], rtol=0, atol=0)
    assert torch.isnan(batch.target[~batch.observed_mask]).all()


def test_resume_discards_step_rows_after_checkpoint(tmp_path):
    class InterruptedTraining(Trainer):
        def compute_loss(self, batch):
            if self.step == 3 and self.model.training:
                raise RuntimeError("interrupted training")
            return super().compute_loss(batch)

    config = _tiny(tmp_path).config.model_copy(update={"n_windows": 16})
    trainer = _tiny(tmp_path, config=config, trainer_class=InterruptedTraining)
    with pytest.raises(RuntimeError, match="interrupted training"):
        trainer.run()
    assert torch.load(trainer.out_dir / "last.pt", weights_only=False)["step"] == 2
    with (trainer.out_dir / "steps.csv").open(newline="") as file:
        assert [int(r["step"]) for r in csv.DictReader(file)] == [1, 2, 3]
    _tiny(tmp_path, config=config).run()
    with (trainer.out_dir / "steps.csv").open(newline="") as file:
        assert [int(r["step"]) for r in csv.DictReader(file)] == [1, 2, 3, 4]


@pytest.mark.parametrize("headerless_rows", [[], [["1", "4", "0", "0", "0", "1"]]])
def test_resume_restores_step_log_header(tmp_path, headerless_rows):
    trainer = _tiny(tmp_path)
    trainer.out_dir.mkdir(parents=True)
    path = trainer.out_dir / "steps.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(headerless_rows)
    trainer.step = 1
    trainer._truncate_steps()
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["step", "windows_seen", "nats_per_point", "lr", "grad_norm", "seconds"], *headerless_rows]
