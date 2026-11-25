"""Conditional code-length, training, and replay checks."""

import json
import math
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from episynth.estimators import build
from episynth.model.moirai import Mixture, MoiraiConfig
from episynth.train import DataConfig, build_generator
from episynth.training import TrainConfig, build_model
from episynth.windows import WindowSpec, window_batch


def experiment():
    spec = WindowSpec(
        series_length=32,
        context_max=16,
        context_lengths=(16,),
        horizon=4,
        patch_size=4,
    )
    model = build_model(
        MoiraiConfig(
            d_model=64,
            num_layers=1,
            patch_sizes=(4, 8),
            components=("student_t",),
        ),
        0,
    )
    train = TrainConfig(
        n_windows=32,
        batch_size=4,
        warmup_steps=1,
        device="cpu",
        precision="fp32",
        heldout_first_seed=72,
        heldout_windows=8,
        eval_batch_size=4,
        checkpoint_fractions=(1.0,),
    )
    source = build_generator(
        DataConfig(generator="waveform"), spec.series_length
    )
    return model, source, spec, train


def test_delta_stream():
    from episynth.estimators._requential_codec import (
        pack_indices,
        unpack_indices,
    )

    # Codes for 1, 2, 3, 4, 8: 1 | 0100 | 0101 | 01100 | 00100000
    payload, bits = pack_indices([1, 2, 3, 4, 8])
    assert bits == 22
    assert payload == int("101000101011000010000000", 2).to_bytes(3, "big")
    assert unpack_indices(payload, bits, 8) == [1, 2, 3, 4, 8]
    for data, length in [
        (payload[:-1], bits),
        (payload, bits - 1),
        (payload + b"x", bits),
        (b"\x81", 1),
        (b"\x00", 8),
    ]:
        with pytest.raises(ValueError):
            unpack_indices(data, length, 8)
    with pytest.raises(ValueError):
        unpack_indices(payload, bits, 7)


def test_ordered_race():
    from episynth.estimators._requential_codec import ordered_index

    rng = np.random.default_rng(321)
    for n, expected in [(1, 0.25), (64, 0.75)]:
        selected = []
        for _ in range(1800):
            proposals = rng.random(n) < 0.25
            ratio = np.where(proposals, 3.0, 1 / 3)
            increments = rng.exponential(size=n)
            index = ordered_index(np.log(ratio), increments)
            arrivals = np.cumsum(increments / np.arange(n, 0, -1))
            assert index == np.argmin(arrivals / ratio) + 1
            selected.append(proposals[index - 1])
        assert abs(np.mean(selected) - expected) < 0.04


def test_gaussian_bits_masks_and_public_targets():
    from episynth.estimators.requential import _public, _ratio_bits, _with_draw

    _, source, spec, _ = experiment()
    batch = window_batch(source, [32] * 1024, spec)
    batch.target[batch.prediction_mask] = float("nan")
    public = _public(batch)
    assert torch.isfinite(public.target).all()
    assert torch.equal(
        public.target[~batch.prediction_mask],
        batch.target[~batch.prediction_mask],
    )
    assert torch.count_nonzero(public.target[batch.prediction_mask]) == 0
    b, s, _ = batch.target.shape
    logits = torch.zeros(b, s, 8, 1)
    loc = torch.zeros(b, s, 1)

    def gaussian(mu):
        return Mixture(
            logits,
            [{"loc": torch.full((b, s, 8), mu)}],
            ["normal_fixed"],
            1.0,
            loc,
            loc + 1,
        )

    p, q = gaussian(0.0), gaussian(1.0)
    draw = q.sample(1)[0]
    synthetic = _with_draw(public, draw)
    assert (_ratio_bits(q, q, synthetic, 4, 0) == 0).all()
    bits = _ratio_bits(p, q, synthetic, 4, 0)
    # Four observed N(1,1) coordinates; tolerance > five MC standard errors.
    assert bits.mean().item() == pytest.approx(2 / math.log(2), abs=0.48)
    assert torch.equal(bits, _ratio_bits(p, q, _with_draw(public, draw), 4, 0))
    public.observed_mask[:, -1, 2:] = False
    half = _with_draw(public, torch.ones_like(draw))
    assert (half.target[:, -1, 2:] == 0).all()
    assert _ratio_bits(p, q, half, 4, 0).sum().item() == pytest.approx(
        b / math.log(2), rel=1e-6
    )
    with pytest.raises(FloatingPointError, match="step 17"):
        _ratio_bits(
            p,
            q,
            replace(half, target=torch.full_like(half.target, float("nan"))),
            4,
            17,
        )


def test_direct_training_and_matched_windows(tmp_path, monkeypatch):
    model, source, spec, train = experiment()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    real = []
    generate = source.generate

    def record(seeds):
        if min(seeds) < 32:
            real.extend(seeds)
        return generate(seeds)

    monkeypatch.setattr(source, "generate", record)
    est = build("requential", projection_steps=[], mc_repeats=4)
    result = est.run(model, source, spec, train, tmp_path / "req")
    assert real == list(range(32))
    assert all(
        torch.equal(v, initial[k]) for k, v in model.state_dict().items()
    )
    real.clear()
    build("prequential").run(model, source, spec, train, tmp_path / "pre")
    assert real == list(range(32))
    assert result["message_count"] == result["real_windows"] == 32
    assert result["student_steps"] == 8
    assert "realized_message_bits" not in result
    assert result["estimated_overhead_mc_se_bits"] > 0
    assert result["steps"][0]["estimated_kl_bits"] == 0
    assert result["steps"][0]["estimated_overhead_bits"] == pytest.approx(
        4 * 5.21
    )
    assert result["estimated_kl_bits"] == pytest.approx(
        sum(x["estimated_kl_bits"] for x in result["steps"])
    )
    assert result["splits"] == {
        k: {"first_seed": a, "n_windows": n}
        for k, a, n in [
            ("teacher_train", 0, 32),
            ("public_context", 32, 32),
            ("projection_validation", 64, 8),
            ("final_eval", 72, 8),
        ]
    }
    assert (
        json.loads((tmp_path / "req/estimator_result.json").read_text())
        == result
    )
    json.dumps(result, allow_nan=False)


def test_repeated_run_preserves_previous_student(tmp_path):
    model, source, spec, train = experiment()
    train = train.model_copy(update={"n_windows": 4})
    first = build("requential", mc_repeats=2).run(
        model, source, spec, train, tmp_path
    )
    checkpoint = Path(first["artifacts"]["student_checkpoint"])
    saved = checkpoint.read_bytes()
    second = build("requential", sampling_seed=8, mc_repeats=2).run(
        model, source, spec, train, tmp_path
    )
    assert checkpoint.read_bytes() == saved
    assert (
        json.loads((tmp_path / "estimator_result.json").read_text()) == second
    )


def test_invalid_setup(tmp_path):
    model, source, spec, train = experiment()
    for kwargs, match in [
        ({"kl_scale": 2}, "kl_scale"),
        ({"public_first_seed": 0}, "overlap"),
    ]:
        with pytest.raises(ValueError, match=match):
            build("requential", **kwargs).run(
                model, source, spec, train, tmp_path
            )
    model.config.components = ("negative_binomial", "student_t")
    with pytest.raises(ValueError, match="continuous"):
        build("requential").run(model, source, spec, train, tmp_path)


@pytest.mark.parametrize(
    ("lower_endpoint", "valid"), [(-8.0, True), (10.0, False)]
)
def test_log_normal_support(tmp_path, lower_endpoint, valid):
    model, source, spec, train = experiment()
    config = model.config.model_copy(
        update={
            "components": ("log_normal",),
            "fixed_scaling": (lower_endpoint, 1.0),
        }
    )
    model = build_model(config, 3)
    train = train.model_copy(update={"n_windows": 8})
    est = build("requential", mc_repeats=2)
    if valid:
        result = est.run(model, source, spec, train, tmp_path)
        assert math.isfinite(result["estimated_kl_bits"])
    else:
        with pytest.raises(
            FloatingPointError, match="teacher training NLL at step 1"
        ):
            est.run(model, source, spec, train, tmp_path)


def test_invalid_predictive_parameters_identify_step(tmp_path):
    model, source, spec, train = experiment()
    with torch.no_grad():
        model.param_proj.proj["components"][0]["loc"].bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="step 1"):
        build("requential").run(model, source, spec, train, tmp_path)


def test_projection_copies_optimizer_and_clocks():
    from episynth.estimators.requential import _Learner

    model, source, spec, train = experiment()
    student = _Learner(model, spec, train, 0.01)
    teacher = _Learner(model, spec, train, 0.1, teacher=True)
    batch = window_batch(source, range(4), spec)
    student.update(batch, 1)
    teacher.project(student)
    assert teacher.clock == student.clock == 1
    for a, b in zip(
        teacher.optimizer.state.values(),
        student.optimizer.state.values(),
        strict=True,
    ):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert (
                torch.equal(a[key], b[key])
                and a[key].data_ptr() != b[key].data_ptr()
            )
    teacher.update(batch, 2)
    assert teacher.clock == 2 and student.clock == 1
    assert any(
        not torch.equal(t, s)
        for t, s in zip(
            teacher.raw.parameters(), student.raw.parameters(), strict=True
        )
    )


def _capture_updates(monkeypatch, horizons):
    from episynth.estimators.requential import _Learner

    update = _Learner.update

    def capture(self, batch, elapsed):
        if not self.teacher:
            horizons.append(batch.target[batch.prediction_mask].clone())
        return update(self, batch, elapsed)

    monkeypatch.setattr(_Learner, "update", capture)


@pytest.mark.parametrize(
    ("sampling_seed", "recovered"), [(1, True), (2, False)]
)
def test_orc_fresh_process_replay(
    tmp_path, monkeypatch, sampling_seed, recovered
):
    model, source, spec, train = experiment()
    settings = dict(
        mode="orc",
        candidates=5,
        mc_repeats=3,
        projection_steps=[2],
        sampling_seed=sampling_seed,
    )
    horizons = []
    _capture_updates(monkeypatch, horizons)
    rng = torch.get_rng_state()
    result = build("requential", **settings).run(
        model, source, spec, train, tmp_path
    )
    assert torch.equal(rng, torch.get_rng_state())
    (event,) = result["projections"]
    assert event["student_step"] == 2 and event["message_count"] == 8
    assert event["recovery_windows"] > 0
    assert event["complete"] == recovered
    if not recovered:
        assert result["student_steps"] == 2 and result["message_count"] == 8
    else:
        assert result["student_steps"] > 2
    assert result["student_steps"] + event["recovery_windows"] // 4 == 8
    assert len(horizons) * 4 == result["message_count"]
    assert (
        result["realized_message_bits"]
        <= 8 * Path(result["artifacts"]["stream"]).stat().st_size
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        check=True,
    )
    decoded = torch.load(tmp_path / "decoded.pt", weights_only=True)
    assert torch.equal(torch.stack(horizons), decoded["horizons"])
    from episynth.evaluation.run import _load_model

    for variant in ("raw", "ema"):
        restored = _load_model(
            Namespace(
                checkpoint=result["artifacts"]["student_checkpoint"],
                config_json=None,
                weights=variant,
            )
        )
        weights = restored.state_dict()
        assert any(
            not torch.equal(model.state_dict()[k], v)
            for k, v in weights.items()
        )
        assert all(
            torch.equal(v, decoded[variant][k]) for k, v in weights.items()
        )


def test_cli_matches_direct_api(tmp_path, monkeypatch):
    from episynth.estimators.run import main as estimator_main
    from episynth.train import main as train_main

    model, source, spec, train = experiment()
    settings = {"projection_steps": [], "mc_repeats": 3}
    expected = build("requential", **settings).run(
        model, source, spec, train, tmp_path / "api"
    )
    path = tmp_path / "cpu.json"
    path.write_text(
        json.dumps(
            {
                "data": {"generator": "waveform"},
                "model": expected["model"],
                "spec": expected["spec"],
                "train": expected["train"],
                "estimator": "requential",
                "estimator_config": settings,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(path), "--out", str(tmp_path / "train")],
    )
    train_main()
    estimator_main(
        [
            "--estimator",
            "requential",
            "--generator",
            "waveform",
            "--config-json",
            str(path),
            "--estimator-config",
            json.dumps(settings),
            "--out",
            str(tmp_path / "cli"),
        ]
    )
    for name in ("train", "cli"):
        actual = json.loads(
            (tmp_path / name / "estimator_result.json").read_text()
        )
        assert {k: v for k, v in actual.items() if k != "artifacts"} == {
            k: v for k, v in expected.items() if k != "artifacts"
        }


def test_projection_schedule():
    from episynth.estimators.requential import _projection_schedule

    assert _projection_schedule(None, 400) == {100, 150, 225, 338}
    assert _projection_schedule([], 400) == set()
    assert _projection_schedule([2, 7], 400) == {2, 7}


if __name__ == "__main__":
    from unittest.mock import patch

    from episynth.estimators.requential import _Learner

    root = Path(sys.argv[1])
    result = json.loads((root / "estimator_result.json").read_text())
    model, source, spec, train = experiment()
    horizons = []
    update = _Learner.update

    def capture(self, batch, elapsed):
        horizons.append(batch.target[batch.prediction_mask].clone())
        return update(self, batch, elapsed)

    with patch.object(_Learner, "update", capture):
        raw, ema = build("requential", **result["estimator_config"]).decode(
            model,
            source,
            spec,
            train,
            Path(result["artifacts"]["stream"]).read_bytes(),
            result["realized_message_bits"],
            result["message_count"],
        )
    torch.save(
        dict(
            horizons=torch.stack(horizons),
            raw=raw.state_dict(),
            ema=ema.state_dict(),
        ),
        root / "decoded.pt",
    )
