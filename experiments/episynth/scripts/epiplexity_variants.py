"""Варианты оценки эпиплексии по сохранённым лоссам и их корреляции Спирмена с качеством.

Запуск: python epiplexity_variants.py <каталог результатов> <куда писать таблицу.md>
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from episynth.analysis import quality_table, spearman_ci

PREFIX = os.environ.get("SERIES_PREFIX", "series_")
root, out_md = Path(sys.argv[1]), Path(sys.argv[2])
GENS = ["kernelsynth", "ets", "waveform", "sde", "stepfunction", "tsi", "arima", "fbm", "garch", "chaotic", "timesynth"]
GENS = [g for g in GENS if (Path(sys.argv[1]) / "runs" / f"{PREFIX}{g}" / "heldout_ema_1.f32").exists() and (Path(sys.argv[1]) / "eval" / f"{PREFIX}{g}" / "summary.csv").exists()]
H = 64


def load(g):
    """Лоссы окон, отложенные лоссы по чекпойнтам и финальные значения прогона."""
    d = root / "runs" / f"{PREFIX}{g}"
    ell = np.fromfile(d / "window_loss.f32", dtype=np.float32).astype(np.float64)
    recs = [json.loads(line) for line in (d / "heldout.jsonl").read_text().splitlines()]
    fr = np.array([r["fraction"] for r in recs])
    ema = np.array([r["ema"]["nats_per_window"] for r in recs])
    raw = np.array([r["raw"]["nats_per_window"] for r in recs])
    return ell, fr, ema, raw


def curve_area(fr, vals, lfin, n):
    """Площадь над lfin под кусочно-линейной кривой отложенного лосса по долям прогона; до первого чекпойнта — значение первого."""
    x = np.concatenate([[0.0], fr]) * n
    y = np.concatenate([[vals[0]], vals]) - lfin
    return float(np.trapezoid(y, x))


est = {name: {} for name in (
    "A_ema", "A_raw", "A_train_tail", "A_min_ema", "A_pos_ema", "A_first_half_ema", "A_second_half_ema",
    "A_heldout_curve_ema", "A_heldout_curve_raw", "A_rel_ema", "sum_loss", "L_fin_ema", "L_fin_raw", "L_start", "drop_ema",
)}
for g in GENS:
    ell, fr, ema, raw = load(g)
    n = len(ell)
    s = ell.sum()
    l_ema, l_raw = ema[-1], raw[-1]
    tail = ell[-int(0.02 * n):].mean()
    half = n // 2
    est["A_ema"][g] = s - n * l_ema
    est["A_raw"][g] = s - n * l_raw
    est["A_train_tail"][g] = s - n * tail
    est["A_min_ema"][g] = s - n * ema.min()
    est["A_pos_ema"][g] = np.maximum(ell - l_ema, 0).sum()
    est["A_first_half_ema"][g] = ell[:half].sum() - half * l_ema
    est["A_second_half_ema"][g] = ell[half:].sum() - (n - half) * l_ema
    est["A_heldout_curve_ema"][g] = curve_area(fr, ema, l_ema, n)
    est["A_heldout_curve_raw"][g] = curve_area(fr, raw, l_raw, n)
    l0 = ell[: int(0.01 * n)].mean()
    est["A_rel_ema"][g] = (s - n * l_ema) / (n * max(l0 - l_ema, 1e-9))
    est["sum_loss"][g] = s
    est["L_fin_ema"][g] = l_ema
    est["L_fin_raw"][g] = l_raw
    est["L_start"][g] = l0
    est["drop_ema"][g] = l0 - l_ema

DESCR = {
    "A_ema": "Σℓ − N·L_fin, L_fin EMA на отложенном (основная)",
    "A_raw": "Σℓ − N·L_fin, L_fin текущих весов на отложенном",
    "A_train_tail": "Σℓ − N·(средний обучающий лосс последних 2 % окон), без отложенного набора",
    "A_min_ema": "Σℓ − N·min по чекпойнтам L_ema",
    "A_pos_ema": "Σ max(ℓ − L_fin_ema, 0): только положительные превышения",
    "A_first_half_ema": "площадь над L_fin_ema по первой половине окон",
    "A_second_half_ema": "площадь над L_fin_ema по второй половине окон",
    "A_heldout_curve_ema": "площадь под кусочно-линейной кривой отложенного лосса EMA по 7 чекпойнтам над L_fin",
    "A_heldout_curve_raw": "то же по кривой текущих весов",
    "A_rel_ema": "A_ema / (N·(L_start − L_fin)): доля площади от прямоугольника полного падения",
    "sum_loss": "Σℓ без вычета (длина кода)",
    "L_fin_ema": "финальный лосс EMA",
    "L_fin_raw": "финальный лосс текущих весов",
    "L_start": "лосс первого 1 % окон",
    "drop_ema": "L_start − L_fin_ema",
}

evals = {g: root / "eval" / f"{PREFIX}{g}" / "summary.csv" for g in GENS}
naive = root / "eval" / "seasonal_naive" / "summary.csv"
qc = quality_table(evals, naive, metric="scaled_crps")
qm = quality_table(evals, naive, metric="MASE[0.5]")
qn = quality_table(evals, naive, metric="nll_per_point")
targets = {
    "CRPS/наив геом": {g: qc[g]["geo_mean_ratio"] for g in GENS},
    "CRPS/наив медиана": {g: qc[g]["median_ratio"] for g in GENS},
    "ранг CRPS": {g: qc[g]["mean_rank"] for g in GENS},
    "MASE/наив геом": {g: qm[g]["geo_mean_ratio"] for g in GENS},
    "NLL GIFT, ранг": {g: qn[g]["mean_rank"] for g in GENS},
}
samples = {"все 11": GENS, "без Chaotic, StepFunction": [g for g in GENS if g not in ("chaotic", "stepfunction")]}

lines = ["# Варианты оценки эпиплексии: корреляции Спирмена с качеством\n",
         "Значение в ячейке — ρ Спирмена между оценкой и целью по генераторам; меньше цель — лучше модель, поэтому отрицательная ρ значит «больше эпиплексия — лучше». В скобках 95 % бутстреп-интервал для основных столбцов.\n"]
for sname, keep in samples.items():
    lines.append(f"\n## {sname} (n = {len(keep)})\n")
    header = "| оценка | " + " | ".join(targets) + " |"
    lines += [header, "|" + "---|" * (len(targets) + 1)]
    for name, vals in est.items():
        row = [f"`{name}`"]
        for tname, t in targets.items():
            x = np.array([vals[g] for g in keep]); y = np.array([t[g] for g in keep])
            if tname in ("CRPS/наив геом", "NLL GIFT, ранг"):
                rho, lo, hi = spearman_ci(x, y, n_boot=1000)
                row.append(f"{rho:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            else:
                row.append(f"{spearmanr(x, y).statistic:+.2f}")
        lines.append("| " + " | ".join(row) + " |")
lines.append("\n## Обозначения\n")
lines += [f"- `{k}` — {v}" for k, v in DESCR.items()]
lines.append("\n## Значения оценок, млн нат (кроме A_rel и L в натах на окно)\n")
lines.append("| генератор | " + " | ".join(est) + " |")
lines.append("|" + "---|" * (len(est) + 1))
for g in GENS:
    cells = []
    for k, v in est.items():
        val = v[g]
        cells.append(f"{val:.3f}" if k == "A_rel_ema" else (f"{val:.2f}" if k.startswith("L_") or k == "drop_ema" else f"{val / 1e6:.1f}"))
    lines.append(f"| {g} | " + " | ".join(cells) + " |")
out_md.write_text("\n".join(lines) + "\n")
print("\n".join(lines[:60]))
