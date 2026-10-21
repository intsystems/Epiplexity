"""Корреляция оценок эпиплексии с итоговым качеством как функция числа окон по чекпойнтам.

Для доли прогона f: A(f) = Σ_{i≤fN} ℓ_i − fN·L_ema(f), падение(f) = L_start − L_ema(f), L_ema(f) — отложенный лосс EMA
на чекпойнте f. Цель — CRPS к наиву финальной модели (геометрическое среднее по задачам).
Запуск: python corr_vs_windows.py <результаты> <каталог рисунков>
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from episynth.analysis import quality_table, spearman_ci  # noqa: E402

PREFIX = os.environ.get("SERIES_PREFIX", "series_")
root, out = Path(sys.argv[1]), Path(sys.argv[2])
GENS = ["kernelsynth", "ets", "waveform", "sde", "stepfunction", "tsi", "arima", "fbm", "garch", "chaotic", "timesynth"]
GENS = [g for g in GENS if (root / "runs" / f"{PREFIX}{g}" / "heldout_ema_1.f32").exists() and (root / "eval" / f"{PREFIX}{g}" / "summary.csv").exists()]
evals = {g: root / "eval" / f"{PREFIX}{g}" / "summary.csv" for g in GENS}
qc = quality_table(evals, root / "eval" / "seasonal_naive" / "summary.csv", metric="scaled_crps")
qm = quality_table(evals, root / "eval" / "seasonal_naive" / "summary.csv", metric="MASE[0.5]")

data = {}
for g in GENS:
    d = root / "runs" / f"{PREFIX}{g}"
    ell = np.fromfile(d / "window_loss.f32", dtype=np.float32).astype(np.float64)
    recs = [json.loads(line) for line in (d / "heldout.jsonl").read_text().splitlines()]
    cum = np.cumsum(ell)
    l0 = ell[: int(0.01 * len(ell))].mean()
    rows = {}
    for r in recs:
        n = r["step"] * 1024
        lf = r["ema"]["nats_per_window"]
        rows[r["fraction"]] = {"A": cum[n - 1] - n * lf, "drop": l0 - lf, "L": lf, "n": n}
    data[g] = rows
fractions = sorted(data[GENS[0]])

samples = {"все 11": GENS, "без Chaotic и StepFunction": [g for g in GENS if g not in ("chaotic", "stepfunction")]}
targets = {"CRPS/наив": qc, "MASE/наив": qm}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
for col, (sname, keep) in enumerate(samples.items()):
    for row, (tname, q) in enumerate(targets.items()):
        ax = axes[row, col]
        y = np.array([q[g]["geo_mean_ratio"] for g in keep])
        for est, color, label in (("A", "C0", "площадь $A$"), ("drop", "C1", "падение $L_{start} - L(n)$"), ("L", "C2", "сам лосс $L(n)$")):
            rhos, los, his, ns = [], [], [], []
            for f in fractions:
                x = np.array([data[g][f][est] for g in keep])
                rho, lo, hi = spearman_ci(x, y, n_boot=1000)
                rhos.append(rho); los.append(lo); his.append(hi); ns.append(data[keep[0]][f]["n"] / 1e6)
            ax.plot(ns, rhos, marker="o", color=color, label=label)
            ax.fill_between(ns, los, his, color=color, alpha=0.12, lw=0)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(-0.6, color="gray", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"{sname}, цель {tname} финальной модели", fontsize=10)
        ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xlabel("окон обучения, млн")
        if col == 0:
            ax.set_ylabel("ρ Спирмена с качеством")
axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
axes[0, 0].set_ylim(-1.05, 1.05)
fig.tight_layout()
fig.savefig(out / "fig_corr_vs_windows.pdf")
fig.savefig(out / "fig_corr_vs_windows.png", dpi=130)
for sname, keep in samples.items():
    y = np.array([qc[g]["geo_mean_ratio"] for g in keep])
    print(sname)
    for f in fractions:
        n = data[keep[0]][f]["n"]
        vals = []
        for est in ("A", "drop", "L"):
            x = np.array([data[g][f][est] for g in keep]); rho, lo, hi = spearman_ci(x, y, n_boot=1000)
            vals.append(f"{est} {rho:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"  n={n / 1e6:.2f} млн: " + "; ".join(vals))
