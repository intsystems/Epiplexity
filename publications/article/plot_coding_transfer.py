"""Rebuild the three coding comparisons from archived measurements."""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
sys.path.insert(0, str(HERE.parents[1] / "experiments/episynth/scripts"))
from requential_plot import render_figure as render_requential
from v_information_transfer import NAMES, render_figure as render_v_information


def read_table(filename: str) -> dict[tuple[str, int, int], dict]:
    with (DATA / filename).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(r["generator"], int(r["init_seed"]), int(r["n_windows"])): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate run in {filename}")
    return indexed


def arrows(pairs: list[tuple[dict, dict]]) -> None:
    ink, blue, orange = "#203245", "#17688a", "#c36019"
    offsets = {"tsi": (-5, -13), "waveform": (7, 6), "sde": (4, 14),
               "fbm": (3, -11), "arima": (6, -8), "garch": (6, 8)}
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=.09, right=.97, top=.84, bottom=.13)
    ax.set(xscale="log", xlim=(7.7, 255), ylim=(.60, 1.69),
           xlabel="Coding estimate (million bits, log scale)", ylabel="CRPS / seasonal naive")
    ax.set_xticks([8, 10, 15, 20, 30, 50, 100, 200])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", color="#e2e8ee")
    ax.axhline(1, ls=(0, (4, 3)), color="#718096", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    for pre, req in pairs:
        x_pre = float(pre["area_ema_mbits"])
        x_req = req["estimated_kl_bits"] / 1e6
        y = float(req["crps_geo"])
        ax.annotate("", (x_req, y), (x_pre, y), arrowprops=dict(
            arrowstyle="-|>", color="#738493", lw=1.25, mutation_scale=13, shrinkA=3, shrinkB=4))
        ax.scatter(x_pre, y, color=blue, s=36, zorder=4)
        ax.scatter(x_req, y, color=orange, marker="D", s=27, zorder=4)
        generator = req["generator"]
        dx, dy = offsets.get(generator, (6, 7))
        ax.annotate(NAMES[generator], (x_pre, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="center", color=ink, fontsize=10)
    fig.text(.09, .945, "2M training windows", weight="bold", fontsize=18, color=ink)
    legend = [Line2D([], [], marker="o", linestyle="none", color=blue,
                     label="Prequential (excess loss)"),
              Line2D([], [], marker="D", linestyle="none", color=orange,
                     label="Requential (estimated KL per student)")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.08, .928), frameon=False, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"prequential_to_requential_crps_2m.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    pre = read_table("run_statistics.csv")
    req = read_table("requential_vs_crps_2m.csv")
    v_info = read_table("v_information_vs_crps.csv")
    if pre.keys() != v_info.keys() or req.keys() != {k for k in pre if k[2] == 1_999_872}:
        raise ValueError("Coding tables cover different runs from the prequential archive")
    for table, metric in ((req, "estimated_kl_bits"), (v_info, "v_information_bits_per_window")):
        for key, row in table.items():
            if any(row[k] != pre[key][k] for k in ("run", "series", "crps_geo")):
                raise ValueError(f"Transfer result differs for {key}")
            row[metric] = float(row[metric])
    render_v_information(list(v_info.values()), FIGURES / "v_information_vs_crps")
    render_requential(list(req.values()), FIGURES / "requential_vs_crps_2m")
    arrows([(pre[k], row) for k, row in req.items()])


if __name__ == "__main__":
    main()
