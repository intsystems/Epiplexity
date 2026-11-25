"""Layer of analysis: area with error, area curve, summaries of quality,
correlations.
"""

import csv
import json

import numpy as np
import pytest

from episynth.analysis import (
    area_curve,
    correlation_table,
    quality_table,
    run_area,
    spearman_ci,
)


def _write_run(tmp_path, losses, held):
    np.asarray(losses, dtype=np.float32).tofile(tmp_path / "window_loss.f32")
    np.asarray(held, dtype=np.float32).tofile(tmp_path / "heldout_ema_1.f32")


def test_run_area_and_curve(tmp_path):
    """The area is equal to the sum of the loss minus N per final loss; the
    error is equal to N per standard error; the curve ends with the area.
    """
    losses = [5.0, 4.0, 3.0, 2.0]
    held = [2.0, 2.0, 1.0, 3.0]
    _write_run(tmp_path, losses, held)
    a = run_area(tmp_path)
    assert (
        a.n_windows == 4
        and a.l_fin == 2.0
        and abs(a.area - (14.0 - 8.0)) < 1e-6
    )
    assert (
        abs(a.area_se - 4 * np.std(held, ddof=1) / 2) < 1e-6
        and a.n_heldout == 4
    )
    ns, curve = area_curve(tmp_path, points=4)
    assert (
        ns.tolist() == [1, 2, 3, 4]
        and abs(curve[-1] - a.area) < 1e-6
        and abs(curve[0] - 3.0) < 1e-6
    )


@pytest.mark.parametrize("reader", [run_area, area_curve])
@pytest.mark.parametrize("artifact", ["window_loss.f32", "heldout_ema_1.f32"])
@pytest.mark.parametrize(
    "damage", ["missing", "empty", "partial_float", "nonfinite"]
)
def test_area_readers_reject_invalid_vectors(
    tmp_path, reader, artifact, damage
):
    _write_run(tmp_path, [5, 4, 3, 2], [2, 2, 1, 3])
    path = tmp_path / artifact
    if damage == "missing":
        path.unlink()
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "partial_float":
        path.write_bytes(path.read_bytes() + b"\x00")
    else:
        np.asarray([2, 2, np.nan, np.inf], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError):
        reader(tmp_path)


@pytest.mark.parametrize("reader", [run_area, area_curve])
def test_area_readers_require_two_heldout_windows(tmp_path, reader):
    _write_run(tmp_path, [5, 4], [2])
    with pytest.raises(ValueError, match="at least 2"):
        reader(tmp_path)


def _write_trainer_metadata(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "train": {"n_windows": 4, "heldout_windows": 4, "batch_size": 2},
    }))
    (tmp_path / "heldout.jsonl").write_text(
        json.dumps({"step": 2, "fraction": 1.0}) + "\n"
    )


def test_area_readers_accept_completed_trainer_run(tmp_path):
    _write_run(tmp_path, [5, 4, 3, 2], [2, 2, 1, 3])
    _write_trainer_metadata(tmp_path)
    assert run_area(tmp_path).area == 6
    assert area_curve(tmp_path)[1][-1] == 6


@pytest.mark.parametrize("reader", [run_area, area_curve])
@pytest.mark.parametrize("artifact", ["window_loss.f32", "heldout_ema_1.f32"])
@pytest.mark.parametrize("n_losses", [3, 5])
def test_area_readers_require_configured_loss_counts(
    tmp_path, reader, artifact, n_losses
):
    _write_run(tmp_path, [5, 4, 3, 2], [2, 2, 1, 3])
    _write_trainer_metadata(tmp_path)
    np.ones(n_losses, dtype=np.float32).tofile(tmp_path / artifact)
    with pytest.raises(ValueError, match="expected 4 losses"):
        reader(tmp_path)


@pytest.mark.parametrize("reader", [run_area, area_curve])
@pytest.mark.parametrize(
    "record",
    [
        None,
        "",
        '{"step":',
        '{"step": 1, "fraction": 1}',
        '{"step": 2, "fraction": 0.5}',
    ],
)
def test_preallocated_logs_require_final_completion_record(
    tmp_path, reader, record
):
    _write_run(tmp_path, [5, 4, 0, 0], [2, 2, 1, 3])
    _write_trainer_metadata(tmp_path)
    path = tmp_path / "heldout.jsonl"
    if record is None:
        path.unlink()
    else:
        path.write_text(record)
    with pytest.raises(ValueError, match="final heldout record"):
        reader(tmp_path)


@pytest.mark.parametrize("reader", [run_area, area_curve])
@pytest.mark.parametrize("marker", ["last.pt", "heldout.jsonl"])
def test_trainer_artifacts_require_configuration(tmp_path, reader, marker):
    _write_run(tmp_path, [5, 4, 3, 2], [2, 2, 1, 3])
    (tmp_path / marker).touch()
    with pytest.raises(ValueError, match="missing Trainer configuration"):
        reader(tmp_path)


def _write_summary(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "model", "scaled_crps"])
        w.writeheader()
        w.writerows(rows)


def test_quality_table_and_correlations(tmp_path):
    """Ranks, attitudes to naive and Spearman's correlation with intervals on
    toy summaries.
    """
    configs = ["a", "b", "c"]
    _write_summary(
        tmp_path / "naive.csv",
        [{"config": c, "model": "naive", "scaled_crps": 1.0} for c in configs],
    )
    _write_summary(
        tmp_path / "m1.csv",
        [
            {"config": c, "model": "m", "scaled_crps": v}
            for c, v in zip(configs, [0.5, 0.5, 2.0], strict=True)
        ],
    )
    _write_summary(
        tmp_path / "m2.csv",
        [
            {"config": c, "model": "m", "scaled_crps": v}
            for c, v in zip(configs, [0.8, 0.9, 0.9], strict=True)
        ],
    )
    q = quality_table(
        {"m1": tmp_path / "m1.csv", "m2": tmp_path / "m2.csv"},
        tmp_path / "naive.csv",
    )
    assert q["m1"]["mean_rank"] == 4 / 3 and q["m2"]["mean_rank"] == 5 / 3
    assert (
        abs(q["m1"]["geo_mean_ratio"] - (0.5 * 0.5 * 2.0) ** (1 / 3)) < 1e-9
        and q["m1"]["share_better"] == 2 / 3
    )
    assert q["m2"]["median_ratio"] == 0.9 and q["m2"]["n_configs"] == 3
    x = np.arange(11, dtype=float)
    rho, lo, hi = spearman_ci(x, x**2, n_boot=200)
    assert (
        abs(rho - 1.0) < 1e-12
        and abs(lo - 1.0) < 1e-9
        and abs(hi - 1.0) < 1e-9
    )
    rho, lo, hi = spearman_ci(x, -x, n_boot=200)
    assert abs(rho + 1.0) < 1e-12
    table = correlation_table(
        {"p": dict(zip("abcde", [1, 2, 3, 4, 5], strict=True))},
        dict(zip("abcde", [2, 1, 4, 3, 5], strict=True)),
        n_boot=100,
    )
    assert (
        table["p"]["n"] == 5
        and 0 < table["p"]["rho"] < 1
        and table["p"]["lo"] <= table["p"]["rho"] <= table["p"]["hi"]
    )


@pytest.mark.parametrize(
    "x,y",
    [
        ([], []),
        ([1], [2]),
        ([1, 1, 1], [1, 2, 3]),
        ([1, 2, 3], [2, 2, 2]),
        ([1, np.nan], [2, 3]),
        ([1, 2], [2, np.inf]),
    ],
)
def test_undefined_spearman_returns_nan_without_crashing(x, y):
    assert np.isnan(spearman_ci(np.asarray(x), np.asarray(y), n_boot=10)).all()


def test_empty_bootstrap_keeps_defined_point_estimate():
    rho, lo, hi = spearman_ci(
        np.array([0, 1]), np.array([0, 1]), n_boot=1, seed=0
    )
    assert rho == pytest.approx(1)
    assert np.isnan(lo) and np.isnan(hi)


def test_constant_predictor_does_not_abort_correlation_table():
    table = correlation_table(
        {"constant": {"a": 1, "b": 1}, "varying": {"a": 1, "b": 2}},
        {"a": 3, "b": 4},
        n_boot=20,
    )
    assert np.isnan(table["constant"]["rho"])
    assert table["varying"]["rho"] == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs", [{"n_boot": 0}, {"n_boot": 1.5}, {"level": 0}, {"level": 1}]
)
def test_spearman_rejects_invalid_bootstrap_settings(kwargs):
    with pytest.raises(ValueError):
        spearman_ci(np.arange(3), np.arange(3), **kwargs)


def test_spearman_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="equal length"):
        spearman_ci(np.arange(3), np.arange(4))
