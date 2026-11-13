"""Plot the full generator comparison with minimal annotations."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BLUE, GRAY, ORANGE = "#1556b0", "#8b96a4", "#bb682c"
NAMES = {"kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE",
         "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm",
         "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#818891", "axes.labelcolor": "#303842",
                     "xtick.color": "#525b66", "ytick.color": "#525b66",
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def read_rows(name):
    with (HERE / "data" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def transfer(target="crps_geo", name="transfer_results"):
    rows = read_rows("run_statistics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True, layout="constrained")
    for series, ax in enumerate(axes, 1):
        primary = [r for r in rows if int(r["series"]) == series and int(r["init_seed"]) == 0]
        ax.scatter([float(r["area_ema_mnats"]) for r in primary],
                   [float(r[target]) for r in primary], s=25, color=BLUE)
        offsets = {"kernelsynth": (-48, 6), "ets": (-8, 6),
                   "chaotic": (-39, -14), "waveform": (-54, 6)}
        if series == 1 and target == "crps_geo":
            offsets["stepfunction"] = (5, 3)
        for row in primary:
            gen = row["generator"]
            if gen in offsets:
                ax.annotate(NAMES[gen], (float(row["area_ema_mnats"]), float(row[target])),
                            xytext=offsets[gen], textcoords="offset points", fontsize=8)
        if series == 2:
            repeat = next(r for r in rows if r["run"] == "s2_waveform_seed1")
            x, y = float(repeat["area_ema_mnats"]), float(repeat[target])
            ax.scatter(x, y, facecolors="white", edgecolors=BLUE, s=30, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim((7, 180) if series == 1 else (24, 2800))
        ax.set_title(f"{2 if series == 1 else 40}M training windows", loc="left", fontsize=10)
        ax.set_xlabel("Excess area (million nats)")
        if target == "crps_geo":
            ax.axhline(1, color=GRAY, lw=0.7, ls="--", zorder=0)
            ax.set_ylim(0.53, 1.72)
    axes[0].set_ylabel("CRPS ratio to seasonal naive" if target == "crps_geo" else "Downstream NLL (nats / point)")
    save(fig, name)


def training():
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 4.5), layout="constrained")
    base = HERE.parents[1] / "experiments" / "episynth"
    for ax, (gen, label) in zip(axes.flat, NAMES.items()):
        for series, color in [(1, GRAY), (2, BLUE)]:
            prefix = "series_" if series == 1 else "s2_"
            path = base / f"series{series}_2026-10-19/runs/{prefix}{gen}/train_curve.csv"
            with path.open() as f:
                rows = list(csv.DictReader(f))
            x = np.array([int(r["windows_seen"]) for r in rows]) / 1e6
            y = np.array([float(r["nats_per_point"]) for r in rows])
            edges = np.unique(np.geomspace(1, len(x) + 1, 70).astype(int)) - 1
            edges[-1] = len(x)
            ax.plot([x[a:b].mean() for a, b in zip(edges[:-1], edges[1:])],
                    [y[a:b].mean() for a, b in zip(edges[:-1], edges[1:])], color=color, lw=1)
        ax.set_xscale("log")
        ax.set_title(label, loc="left", fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flat[-1].axis("off")
    axes.flat[-1].plot([], [], color=GRAY, label="2M budget")
    axes.flat[-1].plot([], [], color=BLUE, label="40M budget")
    axes.flat[-1].legend(frameon=False, loc="center", fontsize=9)
    fig.supxlabel("Training windows (millions)", fontsize=9)
    fig.supylabel("Pre-update NLL (nats / target point)", fontsize=9)
    save(fig, "training_results")


def prefixes():
    rows = read_rows("prefix_correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, layout="constrained")
    for series, ax in enumerate(axes, 1):
        for pred, color, label in [("area_mnats", BLUE, "Area"),
                                   ("final_loss_nats_per_point", GRAY, "Checkpoint loss"),
                                   ("loss_drop", ORANGE, "Loss reduction")]:
            keep = [r for r in rows if int(r["series"]) == series and r["target"] == "crps_geo" and r["predictor"] == pred]
            ax.plot([float(r["n_windows"]) / 1e6 for r in keep], [float(r["rho"]) for r in keep], color=color, label=label, lw=1.4)
        ax.set_xscale("log")
        ax.axhline(0, color="#bac0c6", lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{2 if series == 1 else 40}M budget", loc="left", fontsize=10)
        ax.set_xlabel("Windows observed (millions)")
    axes[0].set_ylabel("Spearman correlation with final CRPS")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "prefix_results")


if __name__ == "__main__":
    transfer()
    transfer("nll_mean_nats_per_point", "transfer_nll")
    training()
    prefixes()
    print("Wrote full-set transfer, NLL, training, and prefix figures.")
