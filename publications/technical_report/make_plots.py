"""Render the technical report's window, execution, and learning-curve figures.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WAVEFORM_RUN = (
    ROOT
    / "experiments"
    / "episynth"
    / "series2_2026-10-19"
    / "runs"
    / "s2_waveform"
)
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

WIDTH = 6.30
INK = "#203950"
MUTED = "#506276"
RULE = "#b6c3cf"
GRID = "#e6ebf0"
CONTEXT = "#275b91"
HORIZON = "#a65332"
OUTSIDE = "#8b98a3"
EXCESS = "#256e64"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.tick_params(labelsize=8.5, length=3, width=0.65)


def architecture():
    sys.path.insert(0, str(ROOT / "src"))
    from episynth.generators import GENERATORS
    from episynth.windows import WindowSpec, make_windows, window_uniforms

    sources = (
        ("(a) CPU example", HERE / "examples" / "prequential.json"),
        ("(b) Archived run", WAVEFORM_RUN / "run.json"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2.72), sharey=True)
    fig.subplots_adjust(
        left=0.12, right=0.963, bottom=0.16, top=0.81, hspace=0.69
    )
    for ax, (title, path) in zip(axes, sources, strict=False):
        config = json.loads(path.read_text(encoding="utf-8"))
        spec = WindowSpec(**config["spec"])
        seeds = np.array([config["train"].get("first_seed", 0)])
        data = config["data"]
        generator = GENERATORS[data["generator"]](
            length=spec.series_length, **data.get("generator_args", {})
        )
        series = generator.generate(seeds)
        batch = make_windows(series, seeds, spec)
        context = int(batch.context_length[0])
        window_length = context + spec.horizon
        valid_starts = spec.series_length - window_length + 1
        # Stream 1 selects the crop offset in make_windows.
        start = min(
            int(window_uniforms(seeds, 1, 1)[0, 0] * valid_starts),
            valid_starts - 1,
        )
        split = start + context
        stop = start + window_length
        past = slice(start, split)
        future = slice(split, stop)
        time = np.arange(1, spec.series_length + 1)
        values = series[0]

        style_axes(ax)
        ax.axvspan(start + 0.5, split + 0.5, color="#eaf2f9", linewidth=0)
        ax.axvspan(split + 0.5, stop + 0.5, color="#fbefe8", linewidth=0)
        ax.axvline(
            split + 0.5, color=MUTED, linestyle=(0, (2, 2)), linewidth=0.75
        )
        ax.plot(time, values, color=OUTSIDE, linewidth=0.75)
        ax.plot(time[past], values[past], color=CONTEXT, linewidth=0.95)
        ax.plot(time[future], values[future], color=HORIZON, linewidth=0.95)
        length = spec.series_length
        ax.set(
            xlim=(0.5, length + 0.5),
            xticks=[1, length // 4, length // 2, 3 * length // 4, length],
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_title(
            title, loc="left", fontsize=9.7, fontweight="semibold", pad=6
        )
        ax.set_title(
            f"Series length {length:,}  |  Context {context}  |  Forecast"
            f" horizon {spec.horizon}",
            loc="right",
            fontsize=8.4,
            color=MUTED,
            pad=6,
        )

    fig.supylabel("Series value", x=0.025, y=0.485, fontsize=9)
    fig.supxlabel("Observation index in generated series", y=0.028, fontsize=9)
    keys = (
        ("Observed context", CONTEXT),
        ("Forecast target", HORIZON),
        ("Outside window", OUTSIDE),
    )
    fig.legend(
        handles=[
            Line2D([], [], label=label, color=color, linewidth=2.0)
            for label, color in keys
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 1.00),
        ncols=3,
        frameon=False,
        fontsize=8.7,
    )
    save(fig, "architecture")


def execution():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.81))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.04)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.plot([0.50, 0.50], [0.32, 0.95], color=RULE, linewidth=0.7)

    panels = (
        (0.25, "(a) Each optimizer step", r"Training batch $\mathcal{B}_k$",
         r"Score with $q_{k-1}$", r"Pre-update loss $\ell_i$ for each window",
         r"Then update the model and EMA"),
        (0.75, "(b) Final checkpoint", r"Held-out windows $(C'_j,Y'_j)$",
         r"Score with final EMA weights", r"Mean held-out loss $\bar{\ell}_f$",
         r"Same generator, separate seeds"),
    )
    for x, title, data, operation, result, note in panels:
        ax.text(
            x,
            0.95,
            title,
            ha="center",
            va="top",
            fontsize=10.1,
            fontweight="semibold",
        )
        ax.text(x, 0.73, data, ha="center", va="center", fontsize=10)
        ax.annotate(
            "",
            (x - 0.126, 0.52),
            (x - 0.126, 0.66),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 0.9},
        )
        ax.text(
            x - 0.105, 0.59, operation, fontsize=8.5, color=MUTED, va="center"
        )
        ax.text(x, 0.44, result, ha="center", va="center", fontsize=10)
        ax.text(
            x, 0.33, note, ha="center", va="center", fontsize=8.5, color=MUTED
        )

    ax.plot([0.11, 0.89], [0.225, 0.225], color=RULE, linewidth=0.7)
    ax.text(
        0.5,
        0.107,
        r"Signed excess loss (nats)    $A_N = \sum_{i=1}^N \ell_i -"
        r" N\bar{\ell}_f$",
        ha="center",
        va="center",
        fontsize=11.0,
    )
    save(fig, "execution")


def demo():
    example = json.loads(
        (HERE / "data" / "example_results.json").read_text(encoding="utf-8")
    )
    record = example["prequential_curve"]
    config = json.loads(
        (WAVEFORM_RUN / "config.json").read_text(encoding="utf-8")
    )
    with (WAVEFORM_RUN / "train_curve.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        archive_nll = np.array(
            [float(row["nats_per_point"]) for row in csv.DictReader(handle)]
        )
    with (WAVEFORM_RUN / "heldout.jsonl").open(encoding="utf-8") as handle:
        final = next(
            checkpoint
            for checkpoint in map(json.loads, handle)
            if checkpoint["fraction"] == 1.0
        )

    cpu_nll = np.array(record["loss_nats_per_window"]) / record["horizon"]
    runs = (
        (
            (
                f"(a) CPU example, {len(cpu_nll) * record['batch_size']:,}"
                " windows"
            ),
            cpu_nll,
            record["heldout_nats_per_window"] / record["horizon"],
            record["batch_size"],
            record["horizon"],
        ),
        (
            (
                f"(b) Archived run, {config['train']['n_windows'] / 1e6:.0f}M"
                " windows"
            ),
            archive_nll,
            final["ema"]["nats_per_point"],
            config["train"]["batch_size"],
            config["spec"]["horizon"],
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.15), sharex="col")
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.135,
        top=0.84,
        wspace=0.50,
        hspace=0.24,
    )
    for ax in axes.flat:
        style_axes(ax)

    for column, (title, batch_nll, floor, batch_size, horizon) in enumerate(
        runs
    ):
        loss_ax, excess_ax = axes[:, column]
        x_unit = 1e6 if column else 1
        bit_unit = 1e9 if column else 1
        windows_seen = np.arange(len(batch_nll) + 1) * batch_size / x_unit
        excess_bits = (batch_nll - floor) * batch_size * horizon / np.log(2)
        cumulative = np.r_[0, np.cumsum(excess_bits)] / bit_unit

        groups = np.array_split(batch_nll, min(len(batch_nll), 80))
        heights = np.array([group.mean() for group in groups])
        edges = (
            np.r_[0, np.cumsum([len(group) for group in groups])]
            * batch_size
            / x_unit
        )
        loss_ax.stairs(
            heights,
            edges,
            baseline=floor,
            fill=True,
            color="#dce8f4",
            linewidth=0,
        )
        loss_ax.stairs(
            heights, edges, baseline=None, color=CONTEXT, linewidth=1.05
        )
        loss_ax.axhline(
            floor, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3))
        )
        loss_ax.use_sticky_edges = False
        loss_ax.margins(y=0.12)
        loss_ax.set_title(
            title, loc="left", fontsize=9.2, fontweight="semibold", pad=7
        )

        excess_ax.step(
            windows_seen,
            cumulative,
            where="post",
            color=EXCESS,
            linewidth=1.35,
        )
        excess_ax.scatter(
            windows_seen[-1], cumulative[-1], color=EXCESS, s=12, zorder=3
        )
        endpoint = (
            f"{cumulative[-1]:.3f}" if column else f"{cumulative[-1]:,.1f}"
        )
        excess_ax.annotate(
            endpoint,
            (windows_seen[-1], cumulative[-1]),
            xytext=(-3, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.4,
        )
        ticks = np.arange(0, 41, 10) if column else np.arange(0, 257, 64)
        excess_ax.set(xticks=ticks, xlim=(0, ticks[-1] * 1.02),
                      ylim=(min(0, cumulative.min()), cumulative.max() * 1.25))
        excess_ax.set_xlabel(
            (
                "Training window count (millions)"
                if column
                else "Training window count"
            ),
            fontsize=8.6,
        )
        if not column:
            excess_ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    axes[0, 0].set_ylabel("NLL (nats per point)", fontsize=8.7)
    axes[1, 0].set_ylabel("Cumulative excess (bits)", fontsize=8.7)
    axes[1, 1].set_ylabel(r"Cumulative excess ($10^9$ bits)", fontsize=8.7)
    fig.legend(
        handles=[
            Line2D(
                [], [], color=CONTEXT, linewidth=1.2, label="Pre-update NLL"
            ),
            Line2D(
                [],
                [],
                color=MUTED,
                linewidth=0.95,
                linestyle=(0, (4, 3)),
                label="Final held-out EMA mean",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 1.01),
        ncols=2,
        frameon=False,
        fontsize=8.6,
    )
    save(fig, "demo_loss")


if __name__ == "__main__":
    architecture()
    execution()
    demo()
    print("Rendered the window, execution, and learning-curve figures.")
