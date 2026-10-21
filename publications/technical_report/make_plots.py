"""Build publication figures only from saved numerical arrays; no fitted curves."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
GENERATORS = ["kernelsynth", "ets", "waveform", "sde", "stepfunction", "tsi", "arima", "fbm", "garch", "chaotic", "timesynth"]
NAMES = {"kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE", "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm", "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth"}
COLORS = dict(zip(GENERATORS, plt.get_cmap("tab20").colors[::2] + (plt.get_cmap("tab20").colors[1],)))
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#303c46", "text.color": "#172d3d",
    "xtick.color": "#52616b", "ytick.color": "#52616b",
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=200)
    plt.close(fig)


def gallery():
    arrays = np.load(HERE / "data" / "generator_examples.npz")
    labels = {
        "kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE",
        "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm",
        "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth",
    }
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 3.65), sharex=True, layout="constrained")
    for ax, (key, label) in zip(axes.flat, labels.items()):
        for row, color, alpha in zip(arrays[key], ("#1c6a93", "#d08038"), (1.0, 0.8)):
            z = (row.astype(float) - row.mean()) / max(float(row.std()), 1e-12)
            ax.plot(z, lw=0.85, color=color, alpha=alpha)
        ax.set_title(label, fontsize=9, loc="left", pad=2, weight="bold")
        ax.set_xlim(0, 255)
        ax.set_xticks([0, 128, 255])
        ax.tick_params(labelsize=7, pad=1)
        ax.grid(axis="y", alpha=0.15)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.03, 0.72, "Generated for this report\n\nBlue: seed 3\nOrange: seed 11\n256 points per series\nStandardized for display",
        transform=axes.flat[-1].transAxes, fontsize=8, va="top", linespacing=1.4,
    )
    fig.supxlabel("Time index", fontsize=9)
    fig.supylabel("Standardized value", fontsize=9)
    save(fig, "generator_gallery")


def windows():
    arrays = np.load(HERE / "data" / "window_examples.npz")
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 2.85), sharex=True, layout="constrained")
    for ax, row, observed, c, seed in zip(
        axes, arrays["target"], arrays["observed"], arrays["context_length"], arrays["seeds"], strict=True,
    ):
        start, boundary, stop = 512 - int(c), 512, 576
        context = row[start:boundary]
        z = (row - context.mean()) / max(float(context.std()), 1e-12)
        ax.axvspan(0, start, color="#edf0f3", lw=0)
        ax.axvspan(boundary, stop, color="#fff0dc", lw=0)
        ax.plot(np.arange(start, boundary), z[start:boundary], color="#1c6a93", lw=0.85)
        ax.plot(np.arange(boundary - 1, stop), z[boundary - 1:stop], color="#c2762f", lw=0.85)
        ax.axvline(boundary, color="#6d737b", ls="--", lw=0.65)
        ax.set_ylabel(f"c = {int(c)}", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(0, 576)
        ax.text(0.01, 0.83, f"seed {int(seed)}", transform=ax.transAxes, fontsize=7, va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    axes[0].text(200, 0.25, "masked padding", color="#697682", fontsize=8, ha="center")
    axes[0].set_title("Observed context (blue) and forecast target (orange); h = 64, patch size = 32", fontsize=9, loc="left")
    axes[-1].set_xlabel("Position in the padded window")
    axes[-1].set_xticks([0, 128, 256, 384, 512, 576])
    save(fig, "forecast_windows")


def read_rows(name):
    with (HERE / "data" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def training_results():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    base = HERE.parents[1] / "experiments" / "episynth"
    for series, ax in enumerate(axes, 1):
        for gen in GENERATORS:
            prefix = "series_" if series == 1 else "s2_"
            with (base / f"series{series}_2026-10-19/runs/{prefix}{gen}/train_curve.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            x = np.array([int(r["windows_seen"]) for r in rows]) / 1e6
            y = np.array([float(r["nats_per_point"]) for r in rows])
            edges = np.unique(np.geomspace(1, len(x) + 1, 100).astype(int)) - 1
            edges[-1] = len(x)
            xb, yb = [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                if hi > lo:
                    xb.append(x[lo:hi].mean())
                    yb.append(y[lo:hi].mean())
            ax.plot(xb, yb, color=COLORS[gen], lw=1.1, label=NAMES[gen])
        ax.set_xscale("log")
        ax.set_xlabel("Training windows (millions)")
        ax.set_title(f"Series {series}: {'2' if series == 1 else '40'} million windows", fontsize=10, loc="left")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Pre-update NLL (nats / target point)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=7.5)
    fig.subplots_adjust(bottom=0.27, wspace=0.12, left=0.09, right=0.99, top=0.9)
    save(fig, "training_results")


def transfer_results():
    rows = read_rows("run_statistics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True, layout="constrained")
    offsets = [
        {"kernelsynth": (-43, -10), "ets": (-9, -14), "waveform": (5, 8), "sde": (7, 6), "stepfunction": (-23, 8), "tsi": (-23, -15), "arima": (4, -13), "fbm": (4, -13), "garch": (-12, 9), "chaotic": (-24, 8), "timesynth": (5, 3)},
        {"kernelsynth": (-38, 0), "ets": (-9, -14), "waveform": (-30, 9), "sde": (-8, 10), "stepfunction": (6, 2), "tsi": (-8, -13), "arima": (5, -7), "fbm": (6, -13), "garch": (-15, 10), "chaotic": (-26, 8), "timesynth": (-42, -5)},
    ]
    short = {"kernelsynth": "Kernel", "stepfunction": "Step", "timesynth": "Time", "waveform": "Wave", "chaotic": "Chaos"}
    for series, ax in enumerate(axes, 1):
        for row in rows:
            if int(row["series"]) != series or int(row["init_seed"]) != 0:
                continue
            gen = row["generator"]
            x, y = float(row["area_ema_mnats"]), float(row["crps_geo"])
            ax.scatter(x, y, c=[COLORS[gen]], s=28, zorder=3)
            ax.annotate(short.get(gen, NAMES[gen]), (x, y), xytext=offsets[series - 1][gen],
                        textcoords="offset points", fontsize=7.5, color=COLORS[gen])
        ax.axhline(1, color="#6a747c", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0.59, 1.74)
        ax.set_xlabel("Excess area A (million nats; log scale)")
        ax.set_title(f"Series {series} ({'2M' if series == 1 else '40M'}); all 11: rho = {'-0.18' if series == 1 else '0.00'}", fontsize=9, loc="left")
        ax.grid(alpha=0.13)
    repeat = next(r for r in rows if r["run"] == "s2_waveform_seed1")
    origin = next(r for r in rows if r["run"] == "s2_waveform")
    x, y = float(repeat["area_ema_mnats"]), float(repeat["crps_geo"])
    axes[1].scatter(x, y, facecolors="white", edgecolors=[COLORS["waveform"]], s=35, zorder=4)
    axes[1].plot([float(origin["area_ema_mnats"]), x], [float(origin["crps_geo"]), y], color=COLORS["waveform"], lw=0.8, ls=":")
    axes[1].annotate("Wave seed 1", (x, y), xytext=(-51, -10), textcoords="offset points", fontsize=7, color=COLORS["waveform"])
    axes[0].set_ylabel("Geometric mean scaled CRPS / seasonal naive")
    save(fig, "transfer_results")


def prefix_results():
    rows = read_rows("prefix_correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharey=True, layout="constrained")
    for series, ax in enumerate(axes, 1):
        for pred, color, label in [
            ("area_mnats", "#1c6a93", "Area at checkpoint"),
            ("final_loss_nats_per_point", "#c2762f", "Checkpoint loss"),
            ("loss_drop", "#56834c", "Loss drop"),
        ]:
            keep = [r for r in rows if int(r["series"]) == series and r["target"] == "crps_geo" and r["predictor"] == pred]
            ax.plot([float(r["n_windows"]) / 1e6 for r in keep], [float(r["rho"]) for r in keep],
                    marker="o", ms=3, lw=1.2, color=color, label=label)
        ax.set_xscale("log")
        ax.axhline(0, color="#777777", lw=0.65)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Series {series}: all 11 generators", loc="left", fontsize=9)
        ax.set_xlabel("Windows observed (millions)")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Spearman rho with final CRPS ratio")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, "prefix_results")


if __name__ == "__main__":
    gallery()
    windows()
    training_results()
    transfer_results()
    prefix_results()
    print("Wrote three experiment figures and two supplementary descriptive figures (PDF and PNG)")
