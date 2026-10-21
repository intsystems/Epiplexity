"""Три определения по оси x: длина кода без вычета, площадь с вычетом L_fin EMA, площадь с вычетом L_fin текущих весов."""
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from episynth.analysis import quality_table, run_area

PREFIX = os.environ.get("SERIES_PREFIX", "series_")
root, out = Path(sys.argv[1]), Path(sys.argv[2])
ORDER = ["kernelsynth", "ets", "waveform", "sde", "stepfunction", "tsi", "arima", "fbm", "garch", "chaotic", "timesynth"]
NAMES = {"kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE", "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm", "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth"}
colors = {g: plt.cm.tab20(i / 11) for i, g in enumerate(ORDER)}
gens = [g for g in ORDER if (root / "runs" / f"{PREFIX}{g}" / "heldout_ema_1.f32").exists()]
evals = {g: root / "eval" / f"{PREFIX}{g}" / "summary.csv" for g in gens}
naive = root / "eval" / "seasonal_naive" / "summary.csv"
ema = {g: run_area(root / "runs" / f"{PREFIX}{g}", "ema") for g in gens}
raw = {g: run_area(root / "runs" / f"{PREFIX}{g}", "raw") for g in gens}
xs = {
    "длина кода $\\sum_i \\ell_i$ без вычета, млн нат": {g: ema[g].sum_loss / 1e6 for g in gens},
    "$A = \\sum_i \\ell_i - N L_{fin}$, $L_{fin}$ EMA-весов, млн нат": {g: ema[g].area / 1e6 for g in gens},
    "$A$ с $L_{fin}$ текущих весов, млн нат": {g: raw[g].area / 1e6 for g in gens},
}
handles = [Line2D([], [], marker="o", color=colors[g], ls="", label=NAMES[g]) for g in gens]
for key, metric, fname in (("CRPS", "scaled_crps", "fig_crps_vs_three_areas"), ("MASE", "MASE[0.5]", "fig_mase_vs_three_areas")):
    q = quality_table(evals, naive, metric=metric)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (label, xv) in zip(axes, xs.items()):
        for g in gens:
            ax.scatter(xv[g], q[g]["geo_mean_ratio"], s=80, color=colors[g], zorder=3)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"{key} / сезонный наив")
    axes[-1].legend(handles=handles, fontsize=8, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf"); fig.savefig(out / f"{fname}.png", dpi=140)
print({g: (round(ema[g].sum_loss/1e6,1), round(ema[g].area/1e6,1), round(raw[g].area/1e6,1)) for g in gens})
