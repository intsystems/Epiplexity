"""Рисунки серии: главный — метрика к наиву против эпиплексии; вспомогательные — кривые обучения с площадью.

Запуск: python series_figures2.py <каталог результатов> <каталог рисунков>
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from episynth.analysis import quality_table, run_area  # noqa: E402

PREFIX = os.environ.get("SERIES_PREFIX", "series_")
root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ORDER = ["kernelsynth", "ets", "waveform", "sde", "stepfunction", "tsi", "arima", "fbm", "garch", "chaotic", "timesynth"]
NAMES = {"kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE", "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm", "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth"}
gens = [g for g in ORDER if (root / "runs" / f"{PREFIX}{g}" / "heldout_ema_1.f32").exists()]
evals = {g: root / "eval" / f"{PREFIX}{g}" / "summary.csv" for g in gens}
naive = root / "eval" / "seasonal_naive" / "summary.csv"
colors = {g: plt.cm.tab20(i / 11) for i, g in enumerate(ORDER)}
H = 64

area = {g: run_area(root / "runs" / f"{PREFIX}{g}", "ema") for g in gens}
crps = quality_table(evals, naive, metric="scaled_crps")
mase = quality_table(evals, naive, metric="MASE[0.5]")

# --- главный график: средняя метрика к наиву против эпиплексии
from matplotlib.lines import Line2D  # noqa: E402

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for g in gens:
    x = area[g].area / 1e6
    ax.scatter(x, crps[g]["geo_mean_ratio"], s=70, color=colors[g], marker="o", zorder=3)
    ax.scatter(x, mase[g]["geo_mean_ratio"], s=70, color=colors[g], marker="s", facecolors="none", linewidths=1.8, zorder=3)
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.7, 0.8, 1.0, 1.2, 1.5])
ax.set_yticklabels(["0.7", "0.8", "1.0", "1.2", "1.5"])
ax.set_xlabel("эпиплексия $A$, млн нат")
ax.set_ylabel("метрика / сезонный наив")
metric_handles = [Line2D([], [], marker="o", color="k", ls="", label="CRPS"), Line2D([], [], marker="s", markerfacecolor="none", color="k", ls="", label="MASE")]
gen_handles = [Line2D([], [], marker="o", color=colors[g], ls="", label=NAMES[g]) for g in gens]
leg1 = ax.legend(handles=metric_handles, loc="upper left", fontsize=9, frameon=False)
ax.add_artist(leg1)
ax.legend(handles=gen_handles, loc="upper right", fontsize=7.5, ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(out / "fig_main_metric_vs_epiplexity.pdf")
fig.savefig(out / "fig_main_metric_vs_epiplexity.png", dpi=160)

# --- вспомогательный: кривые обучения с площадью над финальным лоссом
fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharex=True)
for ax, g in zip(axes.flat, gens, strict=False):
    losses = np.fromfile(root / "runs" / f"{PREFIX}{g}" / "window_loss.f32", dtype=np.float32).astype(np.float64) / H
    bins = 20480
    n = len(losses) // bins * bins
    curve = losses[:n].reshape(-1, bins).mean(1)
    xs = (np.arange(len(curve)) + 0.5) * bins / 1e6
    lfin = area[g].l_fin / H
    lraw = run_area(root / "runs" / f"{PREFIX}{g}", "raw").l_fin / H
    ax.plot(xs, curve, color=colors[g], lw=1.4)
    ax.axhline(lraw, color="k", lw=0.9, ls=":")
    ax.axhline(lfin, color="k", lw=0.9, ls="--")
    ax.fill_between(xs, curve, lraw, where=curve > lraw, color=colors[g], alpha=0.35, lw=0)
    ax.fill_between(xs, np.minimum(curve, lraw), lfin, where=np.full(len(xs), lraw > lfin), color=colors[g], alpha=0.12, lw=0)
    a_raw = run_area(root / "runs" / f"{PREFIX}{g}", "raw").area / 1e6
    ax.set_title(f"{NAMES[g]}: $A$ = {a_raw:.0f} (raw) / {area[g].area / 1e6:.0f} (EMA) млн нат", fontsize=9)
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("окон, млн")
for ax in axes[:, 0]:
    ax.set_ylabel("лосс, нат/точку")
axes.flat[-1].axis("off")
fig.suptitle("Лосс окна до шага (среднее по 20 тыс. окон); точки — $L_{fin}$ текущих весов, пунктир — $L_{fin}$ EMA; тёмная закраска — $A$ над текущими весами, светлая — добавка от EMA", fontsize=10)
fig.tight_layout()
fig.savefig(out / "fig_train_curves_area.pdf")
fig.savefig(out / "fig_train_curves_area.png", dpi=130)

# --- один генератор крупно
g = "waveform"
fig, ax = plt.subplots(figsize=(6.4, 4.2))
losses = np.fromfile(root / "runs" / f"{PREFIX}{g}" / "window_loss.f32", dtype=np.float32).astype(np.float64) / H
bins = 10240
n = len(losses) // bins * bins
curve = losses[:n].reshape(-1, bins).mean(1)
xs = (np.arange(len(curve)) + 0.5) * bins / 1e6
lfin = area[g].l_fin / H
ax.plot(xs, curve, color=colors[g], lw=1.5, label="лосс окна до шага")
ax.axhline(lfin, color="k", lw=1, ls="--", label=f"финальный лосс EMA = {lfin:.2f}")
ax.fill_between(xs, curve, lfin, where=curve > lfin, color=colors[g], alpha=0.25, lw=0, label=f"эпиплексия $A$ = {area[g].area / 1e6:.1f} млн нат")
ax.set_xlabel("окон, млн")
ax.set_ylabel("нат/точку")
ax.set_title(NAMES[g])
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out / "fig_train_curve_waveform.pdf")
print("figures:", sorted(p.name for p in out.iterdir() if p.name.startswith("fig_")))

# --- по одной метрике на график
for key, table, fname in (("CRPS", crps, "fig_crps_vs_epiplexity"), ("MASE", mase, "fig_mase_vs_epiplexity")):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for g in gens:
        ax.scatter(area[g].area / 1e6, table[g]["geo_mean_ratio"], s=80, color=colors[g], zorder=3)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("эпиплексия $A$, млн нат")
    ax.set_ylabel(f"{key} / сезонный наив")
    ax.legend(handles=gen_handles, fontsize=8, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png", dpi=160)
print("single-metric figures written")
