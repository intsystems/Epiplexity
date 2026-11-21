"""Plot generator samples, the archived comparison, and learning curves."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BLUE, GRAY, ORANGE = "#1556b0", "#8b96a4", "#bb682c"
NAMES = {"kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE",
         "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm",
         "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth"}
# Label centers in axes coordinates, tuned to keep the archived points legible
# even where their scores are close. Leader lines identify the plotted values.
TRANSFER_LABELS = {
    "crps_geo": {
        1: {
            "fbm": (0.07, 0.14), "sde": (0.10, 0.45), "timesynth": (0.245, 0.54),
            "garch": (0.24, 0.15), "arima": (0.41, 0.32), "ets": (0.42, 0.045),
            "stepfunction": (0.79, 0.94), "tsi": (0.58, 0.25),
            "kernelsynth": (0.73, 0.04), "waveform": (0.77, 0.34), "chaotic": (0.81, 0.63),
        },
        2: {
            "sde": (0.075, 0.44), "fbm": (0.12, 0.15), "garch": (0.22, 0.50),
            "timesynth": (0.41, 0.56), "arima": (0.48, 0.18), "ets": (0.36, 0.05),
            "stepfunction": (0.55, 0.31), "tsi": (0.68, 0.12), "kernelsynth": (0.87, 0.135),
            "waveform": (0.72, 0.80), "chaotic": (0.88, 0.56), "repeat": (0.75, 0.46),
        },
    },
    "nll_mean_nats_per_point": {
        1: {
            "fbm": (0.07, 0.37), "sde": (0.10, 0.08), "timesynth": (0.23, 0.53),
            "garch": (0.24, 0.18), "arima": (0.43, 0.35), "ets": (0.40, 0.08),
            "stepfunction": (0.58, 0.45), "tsi": (0.54, 0.13),
            "kernelsynth": (0.71, 0.042), "waveform": (0.84, 0.29), "chaotic": (0.83, 0.83),
        },
        2: {
            "fbm": (0.115, 0.42), "sde": (0.08, 0.063), "garch": (0.25, 0.18),
            "timesynth": (0.32, 0.43), "arima": (0.46, 0.35), "ets": (0.40, 0.067),
            "stepfunction": (0.53, 0.22), "tsi": (0.69, 0.32), "kernelsynth": (0.79, 0.041),
            "repeat": (0.79, 0.46), "chaotic": (0.86, 0.81),
        },
    },
}
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


def generator_examples():
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from episynth.generators import GENERATORS

    # The PDF pages follow the rows of the generator table in main.tex.
    seeds = {
        "kernelsynth": 6,
        "ets": 0,
        "waveform": 2,
        "sde": 10,
        "stepfunction": 8,
        "tsi": 11,
        "arima": 4,
        "fbm": 5,
        "garch": 1,
        "chaotic": 7,
        "timesynth": 0,
    }
    with plt.rc_context({"path.simplify": False}), PdfPages(OUT / "generator_examples.pdf") as pdf:
        for name, seed in seeds.items():
            values = GENERATORS[name](length=192).generate([seed])[0]
            fig = plt.figure(figsize=(1.58, 0.18))
            ax = fig.add_axes([0.01, 0.06, 0.98, 0.88])
            ax.plot(values, color="#17688a", linewidth=0.65)
            ax.margins(x=0, y=0.05)
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches=fig.bbox_inches)
            plt.close(fig)


def generator_gallery():
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from episynth.generators import GENERATORS

    # These seeds are distinct from those used in the generator table.
    seeds = (12, 13, 14, 15, 16)
    with plt.rc_context({"path.simplify": False}):
        fig, axes = plt.subplots(len(NAMES), len(seeds), figsize=(6.9, 6.50))
        fig.subplots_adjust(left=0.155, right=0.994, bottom=0.018, top=0.948,
                            wspace=0.17, hspace=0.35)
        for (name, label), row in zip(NAMES.items(), axes):
            samples = GENERATORS[name](length=192).generate(seeds)
            anchor = row[0].get_position()
            fig.text(0.014, (anchor.y0 + anchor.y1) / 2, label, va="center",
                     fontsize=9, weight="semibold", color="#203245")
            for ax, values in zip(row, samples):
                marker = "." if np.ptp(values) == 0 else None
                ax.plot(values, color="#17688a", linewidth=0.72, marker=marker,
                        markevery=[0, len(values) - 1], markersize=3)
                ax.set_axis_off()
                ax.margins(x=0, y=0.12)
            separator = anchor.y0 - 0.008
            fig.add_artist(Line2D([0.014, 0.994], [separator, separator], transform=fig.transFigure,
                                  linewidth=0.55, color="#e2e8ee"))

        fig.text(0.014, 0.976, "Generator", va="center", fontsize=9, weight="semibold", color="#203245")
        for ax, seed in zip(axes[0], seeds):
            position = ax.get_position()
            fig.text((position.x0 + position.x1) / 2, 0.976, f"Seed {seed}",
                     ha="center", va="center", fontsize=9, weight="semibold", color="#203245")
        save(fig, "generator_gallery")


def gift_eval_gallery():
    data = json.loads((HERE / "data" / "gift_eval_examples.json").read_text(encoding="utf-8"))
    with plt.rc_context({"path.simplify": False}):
        fig, axes = plt.subplots(3, 5, figsize=(6.9, 2.13))
        fig.subplots_adjust(left=0.014, right=0.994, bottom=0.018, top=0.869,
                            wspace=0.17, hspace=0.93)
        for ax, example in zip(axes.flat, data["examples"], strict=True):
            ax.text(0.5, 1.55, example["label"], transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, weight="semibold", color="#203245")
            ax.text(0.5, 1.18, example["sampling_interval"], transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#525b66")
            ax.plot(np.asarray(example["values"], dtype=float), color=ORANGE, linewidth=0.72)
            ax.set_axis_off()
            ax.margins(x=0, y=0.12)
        save(fig, "gift_eval_gallery")


def transfer(target="crps_geo", name="transfer_results"):
    rows = read_rows("run_statistics.csv")
    is_crps = target == "crps_geo"
    ink, point, leader = "#203245", "#17688a", "#92a3b2"
    label_box = {"boxstyle": "square,pad=0.08", "facecolor": "white", "linewidth": 0}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.19, top=0.86, wspace=0.13)
    for series, budget, ax in zip((1, 2), (2, 40), axes):
        labels = TRANSFER_LABELS[target][series]
        ax.set_xscale("log")
        ticks = (10, 20, 50, 100) if series == 1 else (30, 100, 300, 1000, 2000)
        ax.set_xticks(ticks, [f"{tick:,}" for tick in ticks])
        ax.set_xlim((7, 180) if series == 1 else (23, 2850))
        ax.set_ylim((0.55, 1.72) if is_crps else (4.60, 6.23))
        ax.set_yticks((0.6, 0.8, 1, 1.2, 1.4, 1.6) if is_crps else (4.8, 5.2, 5.6, 6.0))
        ax.tick_params(which="minor", bottom=False)
        ax.tick_params(which="major", labelsize=8, length=2.5, width=0.6)
        ax.set_title(f"{budget}M training windows", loc="left", fontsize=10, weight="semibold", color=ink)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e9ee", linewidth=0.65)
        if is_crps:
            ax.axhline(1, color="#647589", lw=0.8, ls=(0, (4, 3)), zorder=1)

        for row in rows:
            if int(row["series"]) != series:
                continue
            x, y = float(row["area_ema_mnats"]), float(row[target])
            gen = row["generator"]
            repeat = int(row["init_seed"]) != 0
            if repeat:
                ax.scatter(x, y, s=44, facecolors="none", edgecolors=ORANGE, lw=1.2, zorder=4)
                key = "repeat"
                label = "Waveform (seed 1)" if is_crps else "Waveform (seeds 0, 1)"
            else:
                ax.scatter(x, y, s=20, color=point, edgecolors="white", lw=0.45, zorder=3)
                if not is_crps and series == 2 and gen == "waveform":
                    continue  # The two Waveform NLL observations share a label.
                key, label = gen, NAMES[gen]
            ax.annotate(label, (x, y), xytext=labels[key], textcoords="axes fraction",
                        ha="center", va="center", fontsize=7.8, color=ink, zorder=2.5,
                        bbox=label_box,
                        arrowprops={"arrowstyle": "-", "color": leader, "lw": 0.65,
                                    "shrinkA": 2, "shrinkB": 3.5 if repeat else 2.5})
    axes[0].set_ylabel("CRPS / seasonal naive" if is_crps else "Downstream NLL (nats / point)")
    fig.supxlabel("Excess prequential loss (million nats, log scale)", x=0.536, y=0.025, fontsize=9)
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
    generator_examples()
    generator_gallery()
    gift_eval_gallery()
    transfer()
    transfer("nll_mean_nats_per_point", "transfer_nll")
    training()
    prefixes()
    print("Wrote generator samples, transfer, NLL, training, and prefix figures.")
