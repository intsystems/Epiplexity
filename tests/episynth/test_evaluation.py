"""Evaluation: window metrics, mixing, naive, Moirai forecaster, download GIFT-Eval if you have a picture."""

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
from episynth.evaluation.metrics import (
    AGGREGATE_FIELDS,
    aggregate,
    crps_from_samples,
    seasonal_error,
    window_stats,
)
from episynth.model.moirai import MoiraiConfig, MoiraiModule
from episynth.training import build_model

DATA = os.environ.get("GIFT_EVAL_DIR")


def test_seasonality_and_config_names():
    """Seasonality according to the rule of gluonts and configuration names as in the leaderboard."""
    assert seasonality("15T") == 96 and seasonality("H") == 24 and seasonality("D") == 1 and seasonality("7T") == 1
    assert config_name("m4_yearly", "A-DEC", "short") == "m4_yearly/A/short"
    assert config_name("saugeenday/D", "D", "short") == "saugeen/D/short"
    assert config_name("bizitobs_application", "10S", "short") == "bizitobs_application/10S/short"


def test_crps_and_seasonal_error_closed_forms():
    """CRPS one sample is equal to the error module, two - the average error minus a quarter of their difference."""
    y = np.array([1.0, -2.0])
    assert np.allclose(crps_from_samples(np.array([[0.5, 0.0]]), y), [0.5, 2.0])
    two = np.array([[0.0, 0.0], [2.0, 1.0]])
    expected = (np.abs(two - y).mean(0)) - np.abs(two[0] - two[1]) / 4
    assert np.allclose(crps_from_samples(two, y), expected)
    se, se_sq = seasonal_error(np.array([1.0, 2.0, 4.0, 7.0]), 2)
    assert se == 4.0 and se_sq == 17.0
    assert seasonal_error(np.array([1.0, 2.0]), 5)[0] == 1.0


def test_window_stats_and_aggregate_handle_nan_and_zero_seasonal_error():
    """NaN doesn't really count, zero-season error windows fall out of scaled metrics."""
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


def _write_arrow_series(root, name, target, freq="A-DEC"):
    directory = root / name
    directory.mkdir(parents=True)
    table = pa.table({"item_id": ["series"], "freq": [freq], "target": [target.tolist()]})
    with pa.OSFile(str(directory / "data-00000.arrow"), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)


@pytest.mark.parametrize("case", ["zeros", "constant", "missing_labels", "zero_labels"])
def test_degenerate_arrow_metrics_keep_schema_and_undefined_values(tmp_path, case):
    target = np.arange(64, dtype=np.float32)
    if case == "zeros":
        target.fill(0)
    elif case == "constant":
        target.fill(3)
    else:
        target[-6:] = np.nan if case == "missing_labels" else 0
    _write_arrow_series(tmp_path, "m4_yearly", target)
    dataset = GiftDataset(tmp_path, "m4_yearly")
    with np.errstate(divide="raise", invalid="raise"):
        rows, result = evaluation_run.evaluate_config(dataset, SeasonalNaive(dataset.season))
    assert tuple(result) == AGGREGATE_FIELDS
    assert len(rows) == result["n_windows"] == 1
    assert result["n_points"] == (0 if case == "missing_labels" else 6)
    assert result["n_excluded"] == (1 if case in {"zeros", "constant"} else 0)
    assert np.isnan(result["nll_per_point"])
    if case == "missing_labels":
        assert all(np.isnan(result[field]) for field in AGGREGATE_FIELDS[3:])
    elif case == "zero_labels":
        assert result["MASE[0.5]"] == result["MAE[0.5]"] == 57
        assert result["sMAPE[0.5]"] == 2
        for field in ("ND[0.5]", "NRMSE[mean]", "mean_weighted_sum_quantile_loss"):
            assert np.isnan(result[field])
    else:
        assert result["MAE[0.5]"] == result["MSE[0.5]"] == result["RMSE[mean]"] == 0
        for field in ("MASE[0.5]", "MSIS", "scaled_crps", "rmsse", "coverage_0.9", "scaled_width_0.9"):
            assert np.isnan(result[field])
        if case == "zeros":
            assert np.isnan(result["sMAPE[0.5]"])
            assert np.isnan(result["mean_weighted_sum_quantile_loss"])
        else:
            assert result["sMAPE[0.5]"] == result["mean_weighted_sum_quantile_loss"] == 0


def test_aggregate_without_windows_returns_complete_schema():
    result = aggregate([])
    assert tuple(result) == AGGREGATE_FIELDS
    assert all(result[field] == 0 for field in AGGREGATE_FIELDS[:3])
    assert all(np.isnan(result[field]) for field in AGGREGATE_FIELDS[3:])


def test_cli_constant_then_ordinary_arrow_data_keep_csv_columns(tmp_path, monkeypatch, capsys):
    data, out = tmp_path / "data", tmp_path / "evaluation"
    _write_arrow_series(data, "m4_yearly", np.full(64, 3, dtype=np.float32))
    _write_arrow_series(data, "m4_quarterly", np.arange(64, dtype=np.float32), freq="Q-DEC")
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--data", str(data), "--out", str(out), "--model", "seasonal_naive",
        "--configs", "m4_yearly", "m4_quarterly",
    ])
    evaluation_run.main()
    assert "MASE nan" in capsys.readouterr().out
    with open(out / "summary.csv") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 2
    for row in table:
        assert tuple(row) == ("config", "model", "seconds", *AGGREGATE_FIELDS)
        assert all(value is not None for value in row.values())
    assert np.isnan(float(table[0]["MASE[0.5]"]))
    dataset = GiftDataset(data, "m4_quarterly")
    _, expected = evaluation_run.evaluate_config(dataset, SeasonalNaive(dataset.season))
    for field in AGGREGATE_FIELDS:
        np.testing.assert_equal(float(table[1][field]), expected[field])


def test_seasonal_naive_forecast():
    """A repeat of the last season, a short story gives the last value, the NaN is filled forward."""
    fc = SeasonalNaive(2).predict([np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0]), np.array([1.0, np.nan, 3.0, np.nan])], 3)
    assert fc.samples.shape == (1, 3, 3)
    assert fc.samples[0, 0].tolist() == [3.0, 4.0, 3.0] and fc.samples[0, 1].tolist() == [5.0] * 3
    assert fc.samples[0, 2].tolist() == [3.0, 3.0, 3.0]


def test_moirai_forecaster_both_protocols():
    """The prognosticator provides samples and log density of the desired shape for NaN and short histories in both protocols."""
    model = build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(16,), components=("student_t", "normal_fixed")), 0)
    hist = [np.sin(np.arange(300) / 5.0), np.array([1.0, 2.0, np.nan, 4.0, 5.0]), np.full(40, np.nan)]
    labels = [np.ones(10), np.array([1.0, np.nan] * 5), np.ones(10)]
    for protocol, ctx in (("ours", 128), ("notebook", 100)):
        fc = MoiraiForecaster(model, 16, protocol, context_max=ctx, horizon_max=32, n_samples=7, batch_size=2).predict(hist, 10, labels)
        assert fc.samples.shape == (7, 3, 10) and np.isfinite(fc.samples).all()
        assert fc.nll.shape == (3, 10) and np.isfinite(fc.nll[0]).all()
        assert fc.context_length.tolist()[:2] == [ctx if protocol == "ours" else 100, 5]


@pytest.fixture
def small_forecast_model():
    return build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(4,), max_seq_len=8,
                                   components=("student_t", "normal_fixed")), 0)


@pytest.mark.parametrize("change", [
    {"protocol": "unknown"}, {"patch_size": 0}, {"patch_size": 8},
    {"context_max": 0}, {"context_max": 15}, {"context_max": 28},
    {"horizon_max": 0}, {"horizon_max": 7}, {"n_samples": 0}, {"batch_size": 0},
])
def test_moirai_forecaster_rejects_invalid_dimensions_before_device_work(small_forecast_model, monkeypatch, change):
    monkeypatch.setattr(small_forecast_model, "to", lambda *a, **k: pytest.fail("invalid configuration must fail before device work"))
    config = {"patch_size": 4, "context_max": 16, "horizon_max": 8, "n_samples": 2, "batch_size": 2, **change}
    with pytest.raises(ValueError):
        MoiraiForecaster(small_forecast_model, **config)


@pytest.mark.parametrize("protocol,horizon,message", [
    ("ours", 9, "horizon_max"), ("ours", 0, "positive integer"),
    ("notebook", 17, "max_seq_len"),
])
def test_moirai_forecaster_rejects_invalid_horizon_before_prediction(small_forecast_model, monkeypatch, protocol, horizon, message):
    forecaster = MoiraiForecaster(small_forecast_model, 4, protocol, context_max=16, horizon_max=8, n_samples=2)
    monkeypatch.setattr(forecaster, "_predict_batch", lambda *a, **k: pytest.fail("invalid horizon must fail before prediction"))
    with pytest.raises(ValueError, match=message):
        forecaster.predict([np.arange(20, dtype=np.float32)], horizon)


def test_notebook_horizon_remains_dynamic(small_forecast_model):
    forecaster = MoiraiForecaster(small_forecast_model, 4, "notebook", context_max=15, horizon_max=8, n_samples=2)
    forecast = forecaster.predict([np.arange(20, dtype=np.float32)], 10, [np.zeros(10, dtype=np.float32)])
    assert forecast.samples.shape == (2, 1, 10)
    assert forecast.nll.shape == (1, 10)


@pytest.fixture
def native_checkpoint(tmp_path, monkeypatch):
    """A small native model with non-standard settings and a saved buffer."""
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


def test_evaluation_rejects_incompatible_summary_columns(evaluation_case):
    invoke, out, _, _ = evaluation_case
    invoke()
    path = out / "summary.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    fields = [field for field in rows[0] if field != "MASE[0.5]"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    before = _output_snapshot(out)
    with pytest.raises(ValueError, match="incompatible fields"):
        invoke("--configs", "m4_yearly", "m4_quarterly")
    assert _output_snapshot(out) == before


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


@pytest.mark.skipif(DATA is None, reason="GIFT_EVAL_DIR is unset")
def test_gift_dataset_windows():
    """The picture reads, the number of windows and horizon is like the gift_eval.data.Dataset."""
    ds = GiftDataset(DATA, "m4_hourly")
    assert ds.windows == 1 and ds.prediction_length == 48 and ds.season == 24 and len(ds.series) == 414
    inst = next(ds.test_instances())
    assert len(inst.label) == 48 and len(inst.history) + 48 == len(ds.series[0][1])
    ds = GiftDataset(DATA, "bizitobs_l2c/H")
    assert ds.target_dim > 1 and ds.series[0][0].endswith("_dim0") and ds.config == "bizitobs_l2c/H/short"
