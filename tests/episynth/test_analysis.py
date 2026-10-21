"""Слой анализа: площадь с ошибкой, кривая площади, сводки качества, корреляции."""

import csv

import numpy as np

from episynth.analysis import area_curve, correlation_table, quality_table, run_area, spearman_ci


def _write_run(tmp_path, losses, held):
    np.asarray(losses, dtype=np.float32).tofile(tmp_path / "window_loss.f32")
    np.asarray(held, dtype=np.float32).tofile(tmp_path / "heldout_ema_1.f32")


def test_run_area_and_curve(tmp_path):
    """Площадь равна сумме лоссов минус N на финальный лосс; ошибка равна N на стандартную ошибку; кривая кончается площадью."""
    losses = [5.0, 4.0, 3.0, 2.0]
    held = [2.0, 2.0, 1.0, 3.0]
    _write_run(tmp_path, losses, held)
    a = run_area(tmp_path)
    assert a.n_windows == 4 and a.l_fin == 2.0 and abs(a.area - (14.0 - 8.0)) < 1e-6
    assert abs(a.area_se - 4 * np.std(held, ddof=1) / 2) < 1e-6 and a.n_heldout == 4
    ns, curve = area_curve(tmp_path, points=4)
    assert ns.tolist() == [1, 2, 3, 4] and abs(curve[-1] - a.area) < 1e-6 and abs(curve[0] - 3.0) < 1e-6


def _write_summary(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "model", "scaled_crps"])
        w.writeheader()
        w.writerows(rows)


def test_quality_table_and_correlations(tmp_path):
    """Ранги, отношения к наиву и корреляция Спирмена с интервалом на игрушечных сводках."""
    configs = ["a", "b", "c"]
    _write_summary(tmp_path / "naive.csv", [{"config": c, "model": "naive", "scaled_crps": 1.0} for c in configs])
    _write_summary(tmp_path / "m1.csv", [{"config": c, "model": "m", "scaled_crps": v} for c, v in zip(configs, [0.5, 0.5, 2.0], strict=True)])
    _write_summary(tmp_path / "m2.csv", [{"config": c, "model": "m", "scaled_crps": v} for c, v in zip(configs, [0.8, 0.9, 0.9], strict=True)])
    q = quality_table({"m1": tmp_path / "m1.csv", "m2": tmp_path / "m2.csv"}, tmp_path / "naive.csv")
    assert q["m1"]["mean_rank"] == 4 / 3 and q["m2"]["mean_rank"] == 5 / 3
    assert abs(q["m1"]["geo_mean_ratio"] - (0.5 * 0.5 * 2.0) ** (1 / 3)) < 1e-9 and q["m1"]["share_better"] == 2 / 3
    assert q["m2"]["median_ratio"] == 0.9 and q["m2"]["n_configs"] == 3
    x = np.arange(11, dtype=float)
    rho, lo, hi = spearman_ci(x, x**2, n_boot=200)
    assert abs(rho - 1.0) < 1e-12 and abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9
    rho, lo, hi = spearman_ci(x, -x, n_boot=200)
    assert abs(rho + 1.0) < 1e-12
    table = correlation_table({"p": dict(zip("abcde", [1, 2, 3, 4, 5], strict=True))}, dict(zip("abcde", [2, 1, 4, 3, 5], strict=True)), n_boot=100)
    assert table["p"]["n"] == 5 and 0 < table["p"]["rho"] < 1 and table["p"]["lo"] <= table["p"]["rho"] <= table["p"]["hi"]
