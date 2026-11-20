"""Оценка: метрики по окнам, сведение, наив, прогнозчик Moirai, загрузка GIFT-Eval при наличии снимка."""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pytest
import torch

from episynth.evaluation import run as evaluation_run
from episynth.evaluation.forecasters import MoiraiForecaster, SeasonalNaive
from episynth.evaluation.gift_eval import GiftDataset, config_name, seasonality
from episynth.evaluation.metrics import aggregate, crps_from_samples, seasonal_error, window_stats
from episynth.model.moirai import MoiraiConfig, MoiraiModule
from episynth.training import build_model

DATA = os.environ.get("GIFT_EVAL_DIR")


def test_seasonality_and_config_names():
    """Сезонность по правилу gluonts и имена конфигураций как в лидерборде."""
    assert seasonality("15T") == 96 and seasonality("H") == 24 and seasonality("D") == 1 and seasonality("7T") == 1
    assert config_name("m4_yearly", "A-DEC", "short") == "m4_yearly/A/short"
    assert config_name("saugeenday/D", "D", "short") == "saugeen/D/short"
    assert config_name("bizitobs_application", "10S", "short") == "bizitobs_application/10S/short"


def test_crps_and_seasonal_error_closed_forms():
    """CRPS по одному сэмплу равен модулю ошибки, по двум — средней ошибке минус четверть их разности."""
    y = np.array([1.0, -2.0])
    assert np.allclose(crps_from_samples(np.array([[0.5, 0.0]]), y), [0.5, 2.0])
    two = np.array([[0.0, 0.0], [2.0, 1.0]])
    expected = (np.abs(two - y).mean(0)) - np.abs(two[0] - two[1]) / 4
    assert np.allclose(crps_from_samples(two, y), expected)
    se, se_sq = seasonal_error(np.array([1.0, 2.0, 4.0, 7.0]), 2)
    assert se == 4.0 and se_sq == 17.0
    assert seasonal_error(np.array([1.0, 2.0]), 5)[0] == 1.0


def test_window_stats_and_aggregate_handle_nan_and_zero_seasonal_error():
    """NaN в правде не считаются, окна с нулевой сезонной ошибкой выпадают из масштабированных метрик."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((50, 4)) + 1.0
    label = np.array([1.0, np.nan, 2.0, 0.0])
    r1 = window_stats(label, samples, np.array([0.0, 1.0, 3.0, 2.0]), 1, nll=np.ones(4), context_length=4)
    assert r1["n_valid"] == 3 and r1["sum_nll"] == 3 and r1["seasonal_error"] > 0
    r2 = window_stats(label, samples, np.full(5, 3.0), 1)
    agg = aggregate([r1, r2])
    assert agg["n_excluded"] == 1 and agg["n_windows"] == 2 and agg["n_points"] == 6
    median = np.quantile(samples, 0.5, axis=0)
    manual = np.abs(label - median)[[0, 2, 3]].sum() / r1["seasonal_error"] / 3
    assert np.isclose(agg["MASE[0.5]"], manual)
    assert 0 <= agg["coverage_0.9"] <= 1


def test_seasonal_naive_forecast():
    """Повтор последнего сезона, короткая история даёт последнее значение, NaN заполняются вперёд."""
    fc = SeasonalNaive(2).predict([np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0]), np.array([1.0, np.nan, 3.0, np.nan])], 3)
    assert fc.samples.shape == (1, 3, 3)
    assert fc.samples[0, 0].tolist() == [3.0, 4.0, 3.0] and fc.samples[0, 1].tolist() == [5.0] * 3
    assert fc.samples[0, 2].tolist() == [3.0, 3.0, 3.0]


def test_moirai_forecaster_both_protocols():
    """Прогнозчик даёт сэмплы и лог-плотность нужной формы при NaN и коротких историях в обоих протоколах."""
    model = build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(16,), components=("student_t", "normal_fixed")), 0)
    hist = [np.sin(np.arange(300) / 5.0), np.array([1.0, 2.0, np.nan, 4.0, 5.0]), np.full(40, np.nan)]
    labels = [np.ones(10), np.array([1.0, np.nan] * 5), np.ones(10)]
    for protocol, ctx in (("ours", 128), ("notebook", 100)):
        fc = MoiraiForecaster(model, 16, protocol, context_max=ctx, horizon_max=32, n_samples=7, batch_size=2).predict(hist, 10, labels)
        assert fc.samples.shape == (7, 3, 10) and np.isfinite(fc.samples).all()
        assert fc.nll.shape == (3, 10) and np.isfinite(fc.nll[0]).all()
        assert fc.context_length.tolist()[:2] == [ctx if protocol == "ours" else 100, 5]


@pytest.fixture
def native_checkpoint(tmp_path, monkeypatch):
    """Маленькая native-модель с нестандартными настройками и сохраняемым буфером."""
    class BufferedMoirai(MoiraiModule):
        def __init__(self, config):
            super().__init__(config)
            self.register_buffer("saved_buffer", torch.tensor(-1.0))

    monkeypatch.setattr(evaluation_run, "MoiraiModule", BufferedMoirai)
    model = BufferedMoirai(MoiraiConfig(
        d_model=64, num_layers=1, patch_sizes=(16,), components=("student_t", "normal_fixed"),
        normal_fixed_scale=0.25, scaling=False, ignore_context=True,
    ))
    model.saved_buffer.fill_(7.0)
    ema = [p.detach().clone() + 0.01 for p in model.parameters()]
    checkpoint = tmp_path / "native.pt"
    torch.save({"model_config": model.config.model_dump(), "model": model.state_dict(), "ema": ema}, checkpoint)
    return checkpoint, model, ema


def _model_args(checkpoint, **changes):
    return argparse.Namespace(**{
        "checkpoint": str(checkpoint), "config_json": None, "protocol": "ours", "patch_size": 16,
        "components": ["student_t", "normal_fixed", "log_normal"], "weights": "ema", **changes,
    })


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_native_checkpoint_restores_config_parameters_and_buffers(native_checkpoint, weights):
    checkpoint, original, ema = native_checkpoint
    restored = evaluation_run._load_model(_model_args(checkpoint, weights=weights))
    assert restored.config == original.config
    assert restored.saved_buffer.item() == 7.0
    expected = list(original.parameters()) if weights == "raw" else ema
    assert all(torch.equal(p, e) for p, e in zip(restored.parameters(), expected, strict=True))


def test_native_checkpoint_accepts_matching_explicit_config(native_checkpoint, tmp_path):
    checkpoint, original, _ = native_checkpoint
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": original.config.model_dump(mode="json")}))
    restored = evaluation_run._load_model(_model_args(checkpoint, config_json=str(config)))
    assert restored.config == original.config


def test_safetensors_retains_explicit_configuration_loading(native_checkpoint, tmp_path):
    from safetensors.torch import save_file

    _, original, _ = native_checkpoint
    checkpoint = tmp_path / "model.safetensors"
    save_file(original.state_dict(), str(checkpoint))
    config = tmp_path / "model.json"
    config.write_text(original.config.model_dump_json())
    restored = evaluation_run._load_model(_model_args(checkpoint, config_json=str(config)))
    assert restored.config == original.config
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in original.state_dict().items())


@pytest.fixture
def evaluation_case(tmp_path, monkeypatch, native_checkpoint):
    checkpoint, _, _ = native_checkpoint
    data, out = tmp_path / "data", tmp_path / "evaluation"
    for name, freq in (("m4_yearly", "A-DEC"), ("m4_quarterly", "Q-DEC")):
        directory = data / name
        directory.mkdir(parents=True)
        table = pa.table({"item_id": ["series"], "freq": [freq], "target": [np.sin(np.arange(64) / 3).tolist()]})
        with pa.OSFile(str(directory / "data-00000.arrow"), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    def invoke(*extra):
        monkeypatch.setattr(sys, "argv", [
            "evaluate", "--data", str(data), "--out", str(out), "--model", "moirai",
            "--checkpoint", str(checkpoint), "--patch-size", "16", "--context-max", "32",
            "--horizon-max", "16", "--n-samples", "3", "--batch-size", "2", "--configs", "m4_yearly", *extra,
        ])
        evaluation_run.main()

    return invoke, out, data, checkpoint


def _output_snapshot(out):
    return {str(p.relative_to(out)): p.read_bytes() if p.is_file() else None for p in out.rglob("*")}


def test_evaluation_manifest_allows_resume_and_additional_configs(evaluation_case, native_checkpoint, monkeypatch):
    invoke, out, data, checkpoint = evaluation_case
    original_evaluate = evaluation_run.evaluate_config
    evaluated = []

    def record_evaluation(dataset, forecaster):
        evaluated.append(dataset.config)
        return original_evaluate(dataset, forecaster)

    monkeypatch.setattr(evaluation_run, "evaluate_config", record_evaluation)
    invoke()
    manifest = json.loads((out / "evaluation.json").read_text())
    assert manifest["data"] == str(data.resolve())
    assert manifest["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["model_config"] == native_checkpoint[1].config.model_dump(mode="json")
    first_windows = (out / "windows" / "m4_yearly__A__short.csv").read_bytes()
    invoke("--configs", "m4_yearly", "m4_quarterly")
    assert evaluated == ["m4_yearly/A/short", "m4_quarterly/Q/short"]
    assert (out / "windows" / "m4_yearly__A__short.csv").read_bytes() == first_windows
    with open(out / "summary.csv") as f:
        assert [r["config"] for r in csv.DictReader(f)] == evaluated


@pytest.mark.parametrize("change", ["weights", "protocol", "n_samples", "data", "checkpoint"])
def test_evaluation_rejects_stale_results_before_mutating_output(evaluation_case, change):
    invoke, out, data, checkpoint = evaluation_case
    invoke()
    before = _output_snapshot(out)
    extra = {
        "weights": ("--weights", "raw"), "protocol": ("--protocol", "notebook"),
        "n_samples": ("--n-samples", "5"), "data": ("--data", str(data / "other_snapshot")), "checkpoint": (),
    }[change]
    if change == "checkpoint":
        state = torch.load(checkpoint, weights_only=False)
        state["model"]["saved_buffer"].add_(1)
        torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="new --out directory"):
        invoke(*extra)
    assert _output_snapshot(out) == before


def test_native_config_mismatch_preserves_existing_evaluation(evaluation_case, native_checkpoint, tmp_path):
    invoke, out, _, _ = evaluation_case
    invoke()
    before = _output_snapshot(out)
    config = tmp_path / "conflicting.json"
    config.write_text(native_checkpoint[1].config.model_copy(update={"normal_fixed_scale": 0.5}).model_dump_json())
    with pytest.raises(ValueError, match="does not match the checkpoint model_config"):
        invoke("--config-json", str(config))
    assert _output_snapshot(out) == before


@pytest.mark.parametrize("manifest", [None, "{}", "{broken"])
def test_existing_summary_requires_matching_manifest(tmp_path, monkeypatch, manifest):
    out = tmp_path / "evaluation"
    out.mkdir()
    (out / "summary.csv").write_text("config,model\nm4_yearly/A/short,seasonal_naive\n")
    if manifest is not None:
        (out / "evaluation.json").write_text(manifest)
    before = _output_snapshot(out)
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--data", str(tmp_path / "data"), "--out", str(out),
        "--model", "seasonal_naive", "--configs", "m4_yearly",
    ])
    with pytest.raises(ValueError, match="new --out directory"):
        evaluation_run.main()
    assert _output_snapshot(out) == before


@pytest.mark.skipif(DATA is None, reason="GIFT_EVAL_DIR не задан")
def test_gift_dataset_windows():
    """Снимок читается, число окон и горизонт как у gift_eval.data.Dataset."""
    ds = GiftDataset(DATA, "m4_hourly")
    assert ds.windows == 1 and ds.prediction_length == 48 and ds.season == 24 and len(ds.series) == 414
    inst = next(ds.test_instances())
    assert len(inst.label) == 48 and len(inst.history) + 48 == len(ds.series[0][1])
    ds = GiftDataset(DATA, "bizitobs_l2c/H")
    assert ds.target_dim > 1 and ds.series[0][0].endswith("_dim0") and ds.config == "bizitobs_l2c/H/short"
