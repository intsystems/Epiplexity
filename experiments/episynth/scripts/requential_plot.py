"""Plot archived requential measurements against downstream CRPS."""

import math
from pathlib import Path

from v_information_transfer import NAMES


def render_figure(rows: list[dict], stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    panels = len({int(r["series"]) for r in rows})
    fig, grid = plt.subplots(
        1, panels, figsize=(5 * panels, 3.65), sharey=True, dpi=150, squeeze=False
    )
    axes = grid[0]
    fig.subplots_adjust(left=0.15 / panels, right=0.985, top=0.87, bottom=0.21, wspace=0.13)
    ink = "#203245"
    for series, ax in enumerate(axes, 1):
        panel = [r for r in rows if int(r["series"]) == series]
        expected = 11 if series == 1 else 12
        title = f"{2 if series == 1 else 40}M training windows"
        if len(panel) < expected:
            title += f" ({len(panel)}/{expected} measured)"
        ax.set_title(
            title,
            loc="left",
            fontsize=12,
            weight="semibold",
            color=ink,
        )
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#818891")
        ax.tick_params(labelsize=9, colors="#525b66", length=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e9ee", linewidth=0.7)
        ax.axhline(1, color="#62768b", lw=1, ls=(0, (4, 3)))
        ax.set_ylim(0.55, 1.72)
        ax.set_yticks((0.6, 0.8, 1.0, 1.2, 1.4, 1.6))
        ax.margins(x=0.16)
        for r in panel:
            repeat = int(r["init_seed"]) == 1
            ax.scatter(
                r["estimated_kl_bits"] / 1e6,
                float(r["crps_geo"]),
                zorder=3,
                s=50 if repeat else 28,
                facecolor="none" if repeat else "#17688a",
                edgecolor="#bf692a" if repeat else "white",
                linewidth=1.3 if repeat else 0.5,
            )
        fig.canvas.draw()
        bounds = ax.get_window_extent().padded(-3)
        renderer = fig.canvas.get_renderer()
        points = [
            ax.transData.transform((r["estimated_kl_bits"] / 1e6, float(r["crps_geo"])))
            for r in panel
        ]
        obstacles = [Bbox.from_bounds(x - 5, y - 5, 10, 10) for x, y in points]
        # Place labels for the most crowded points first, in display coordinates.
        order = sorted(
            range(len(panel)),
            key=lambda i: min((math.dist(points[i], p) for j, p in enumerate(points) if i != j), default=math.inf),
        )
        for i in order:
            r, (x, y) = panel[i], points[i]
            label = NAMES[r["generator"]] + (" (seed 1)" if int(r["init_seed"]) else "")
            width, height, _ = renderer.get_text_width_height_descent(
                label, FontProperties(size=8.6), False
            )
            w, h = width + 6, max(height + 5, 14)
            candidates = []
            for gap in (5, 13, 24, 38, 54, 76):
                for dx, dy in (
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                    (-1, -1),
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                ):
                    cx = min(max(x + dx * (w / 2 + gap), bounds.xmin + w / 2), bounds.xmax - w / 2)
                    cy = min(max(y + dy * (h / 2 + gap), bounds.ymin + h / 2), bounds.ymax - h / 2)
                    rect = Bbox.from_bounds(cx - w / 2, cy - h / 2, w, h)
                    overlap = sum(
                        a.width * a.height for b in obstacles if (a := Bbox.intersection(rect, b))
                    )
                    candidates.append((overlap * 100 + math.hypot(cx - x, cy - y), cx, cy, rect))
            _, cx, cy, rect = min(candidates, key=lambda item: item[0])
            obstacles.append(rect)
            loc = ax.transAxes.inverted().transform((cx, cy))
            ax.annotate(
                label,
                (r["estimated_kl_bits"] / 1e6, float(r["crps_geo"])),
                xytext=loc,
                textcoords="axes fraction",
                ha="center",
                va="center",
                fontsize=8.6,
                color=ink,
                bbox=dict(boxstyle="square,pad=.06", fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-", lw=0.65, color="#92a3b2", shrinkA=2, shrinkB=3),
            )
    axes[0].set_ylabel("CRPS / seasonal naive", fontsize=10, color=ink)
    fig.supxlabel(
        "Requential model length (estimated KL, million bits per student)",
        x=0.53,
        y=0.025,
        fontsize=11 if panels == 2 else 9,
    )
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext), dpi=200)
    plt.close(fig)
