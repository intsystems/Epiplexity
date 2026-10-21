"""Метрика к наиву против эпиплексии с прямой регрессии и горизонталью TimesFM-3 (опубликованные числа лидерборда GIFT-Eval на тех же 55 задачах).

Запуск: SERIES_PREFIX=s2_ python series_figures4.py <каталог результатов> <каталог рисунков> <каталог published>
Пишет fig_{crps,mase,mwsql}_vs_epiplexity_lines.{pdf,png}.
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from episynth.analysis import quality_table, run_area, spearman_ci  # noqa: E402

PREFIX = os.environ.get("SERIES_PREFIX", "series_")
root, out, pub = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
ORDER = ["kernelsynth", "ets", "waveform", "sde", "stepfunction", "tsi", "arima", "fbm", "garch", "chaotic", "timesynth"]
NAMES = {"kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE", "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm", "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth"}
gens = [g for g in ORDER if (root / "runs" / f"{PREFIX}{g}" / "heldout_ema_1.f32").exists()]
evals = {g: root / "eval" / f"{PREFIX}{g}" / "summary.csv" for g in gens}
naive = root / "eval" / "seasonal_naive" / "summary.csv"
colors = {g: plt.cm.tab20(i / 11) for i, g in enumerate(ORDER)}
SUBSET_EXCL = ("chaotic", "waveform")

area = {g: run_area(root / "runs" / f"{PREFIX}{g}", "ema").area / 1e6 for g in gens}
tables = {
    "crps": ("CRPS", quality_table(evals, naive, metric="scaled_crps")),
    "mase": ("MASE", quality_table(evals, naive, metric="MASE[0.5]")),
    "mwsql": ("CRPS лидерборда (MWSQL)", quality_table(evals, naive, metric="mean_weighted_sum_quantile_loss")),
}

# TimesFM-3: опубликованные числа лидерборда на тех же 55 задачах; знаменатель — наш точечный наив
# (его MASE совпадает с опубликованным до 4e-8, его MWSQL равен ND, который совпадает с опубликованным ND).
ours_naive = pd.read_csv(naive).set_index("config")
pub_tfm = pd.read_csv(pub / "gift_eval_timesfm-3.csv").set_index("dataset")
pub_naive = pd.read_csv(pub / "gift_eval_seasonal_naive.csv").set_index("dataset")
cfgs = sorted(ours_naive.index)
gm = lambda r: float(np.exp(np.log(r).mean()))  # noqa: E731
tfm = {
    "mase": gm(pub_tfm.loc[cfgs, "eval_metrics/MASE[0.5]"].to_numpy() / ours_naive.loc[cfgs, "MASE[0.5]"].to_numpy()),
    "mwsql": gm(pub_tfm.loc[cfgs, "eval_metrics/mean_weighted_sum_quantile_loss"].to_numpy() / ours_naive.loc[cfgs, "mean_weighted_sum_quantile_loss"].to_numpy()),
}
tfm["crps"] = tfm["mwsql"]
tfm_lb = gm(pub_tfm.loc[cfgs, "eval_metrics/mean_weighted_sum_quantile_loss"].to_numpy() / pub_naive.loc[cfgs, "eval_metrics/mean_weighted_sum_quantile_loss"].to_numpy())
print(f"TimesFM-3 к нашему точечному наиву: MASE {tfm['mase']:.3f}, MWSQL {tfm['mwsql']:.3f}; к наиву лидерборда (вероятностному) MWSQL {tfm_lb:.3f}")

gen_handles = [Line2D([], [], marker="o", color=colors[g], ls="", label=NAMES[g]) for g in gens]
for key, (label, table) in tables.items():
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.array([area[g] for g in gens]); y = np.array([table[g]["geo_mean_ratio"] for g in gens])
    for g in gens:
        ax.scatter(area[g], table[g]["geo_mean_ratio"], s=80, color=colors[g], zorder=3)
    xs = np.linspace(0, x.max() * 1.03, 50)
    line_handles = []
    for keep_name, keep, style in (("все 11", gens, dict(color="k", lw=1.2)), ("без Chaotic, Waveform", [g for g in gens if g not in SUBSET_EXCL], dict(color="k", lw=1.0, ls=":"))):
        xi = np.array([area[g] for g in keep]); yi = np.array([table[g]["geo_mean_ratio"] for g in keep])
        b, a = np.polyfit(xi, yi, 1)
        rho, lo, hi = spearman_ci(xi, yi, n_boot=2000)
        h, = ax.plot(xs, a + b * xs, zorder=2, label=f"МНК, {keep_name}: ρ = {rho:+.2f} [{lo:+.2f}; {hi:+.2f}]", **style)
        line_handles.append(h)
        print(f"{key} {keep_name}: наклон {b:.4f}/млн нат, Спирмен {rho:+.2f} [{lo:+.2f}, {hi:+.2f}], Пирсон {np.corrcoef(xi, yi)[0,1]:+.2f}")
    h = ax.axhline(tfm[key], color="tab:red", lw=1.2, ls="--", label=f"TimesFM-3 (лидерборд): {tfm[key]:.3f}")
    line_handles.append(h)
    h = ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="сезонный наив")
    line_handles.append(h)
    ax.set_xlabel("эпиплексия $A$, млн нат")
    ax.set_ylabel(f"{label} / сезонный наив")
    ax.set_ylim(min(y.min(), tfm[key]) - 0.06, y.max() + 0.22)
    leg1 = ax.legend(handles=line_handles, fontsize=8, frameon=False, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=gen_handles, fontsize=8, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out / f"fig_{key}_vs_epiplexity_lines.pdf")
    fig.savefig(out / f"fig_{key}_vs_epiplexity_lines.png", dpi=160)
print("written:", sorted(p.name for p in out.iterdir() if p.name.endswith("_lines.png")))
