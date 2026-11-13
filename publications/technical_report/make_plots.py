"""Render architecture, execution, and the measured CPU learning curve."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BLUE, INK, GRAY = "#1556b0", "#263342", "#84909c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def arrow(ax, start, stop, color=GRAY):
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=10, lw=1, color=color))


def architecture():
    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    ax.set_xlim(-0.1, 9.1)
    ax.set_ylim(-0.2, 3.15)
    ax.axis("off")
    def box(x, y, title, detail):
        ax.add_patch(Rectangle((x, y), 2.4, 0.85, facecolor="white", edgecolor=GRAY, lw=0.8))
        ax.text(x+1.2, y+0.57, title, ha="center", va="center", fontsize=10, color=INK)
        ax.text(x+1.2, y+0.25, detail, ha="center", va="center", fontsize=8, color="#596776")
    box(0, 2.1, "Data source", "Generator / Corpus / Mixture")
    box(3.2, 2.1, "Forecast windows", "WindowSpec / WindowStream")
    box(6.4, 2.1, "Predictive model", "Moirai density and samples")
    box(0, 0.65, "Estimator", 'build("prequential") etc.')
    box(3.2, 0.65, "Trainer", "Updates / EMA / checkpoints")
    box(6.4, 0.65, "Run artifacts", "Losses / weights / metadata")
    arrow(ax, (2.4, 2.52), (3.2, 2.52))
    arrow(ax, (4.4, 2.1), (4.4, 1.5))
    arrow(ax, (6.4, 2.1), (5.5, 1.5))
    arrow(ax, (2.4, 1.07), (3.2, 1.07), BLUE)
    arrow(ax, (5.6, 1.07), (6.4, 1.07))
    arrow(ax, (7.6, 0.65), (7.6, 0.13))
    ax.text(7.6, -0.06, "Analysis and GIFT-Eval", ha="center", va="center", fontsize=9, color=INK)
    save(fig, "architecture")


def execution():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(-0.45, 3.55)
    ax.set_ylim(-0.2, 4.7)
    ax.axis("off")
    for x, label in enumerate(("Window stream", "Trainer", "Model", "Artifacts")):
        ax.text(x, 4.4, label, ha="center", color=INK, fontsize=10)
        ax.plot([x, x], [0, 4.15], color="#d4dae0", lw=0.8)
    for a, b, y, label in [(0, 1, 3.7, "Fresh batch"), (1, 2, 3.0, "Evaluate NLL"),
                            (1, 2, 2.3, "Update weights"), (1, 3, 1.6, "Save scored losses"),
                            (1, 3, 0.6, "Held-out scores and checkpoint")]:
        arrow(ax, (a, y), (b, y), BLUE if label == "Evaluate NLL" else GRAY)
        ax.text((a+b)/2, y+0.12, label, ha="center", va="bottom", fontsize=8)
    save(fig, "execution")


def demo():
    record = json.loads((HERE / "data" / "example_results.json").read_text(encoding="utf-8"))["prequential_curve"]
    curve = np.array(record["loss_nats_per_window"]) / record["horizon"]
    x = np.arange(1, len(curve)+1) * record["batch_size"]
    floor = record["heldout_nats_per_window"] / record["horizon"]
    fig, ax = plt.subplots(figsize=(6, 2.25), layout="constrained")
    ax.plot(x, curve, color=BLUE, lw=1.5)
    ax.axhline(floor, color=GRAY, lw=1, ls="--")
    ax.annotate("Held-out endpoint", (x[-1], floor), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=9, color="#616d7a")
    ax.set_xlabel("Training windows")
    ax.set_ylabel("NLL (nats / target point)")
    save(fig, "demo_loss")


if __name__ == "__main__":
    architecture()
    execution()
    demo()
    print("Wrote component diagram, sequence diagram, and measured demo curve.")
