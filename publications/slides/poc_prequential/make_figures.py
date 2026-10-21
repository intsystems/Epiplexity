"""Build PoC figures from the committed notebook and report; no training is rerun."""

import base64
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)
nb = json.loads((ROOT / "experiments/demos/demo_estimators.ipynb").read_text())
plots = [
    o["data"]["image/png"]
    for c in nb["cells"]
    for o in c.get("outputs", [])
    if "image/png" in o.get("data", {})
]
assert len(plots) == 1
(OUT / "demo_notebook.png").write_bytes(base64.b64decode(plots[0]))
result = None
for c in nb["cells"]:
    for o in c.get("outputs", []):
        if o["output_type"] == "stream":
            t = "".join(o["text"])
            if '"total_bits"' in t:
                result = json.loads(t[t.index("{") : t.rindex("}") + 1])
assert result is not None and result["n_windows"] == 64
# This figure is deliberately schematic; it is not an experimental learning curve.
x = np.linspace(1, 100, 400)
floor = 1.1
y = floor + 2.7 * np.exp(-(x - 1) / 24) - 0.60 * np.exp(-(((x - 76) / 12) ** 2))
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(x, y, color="#124B68", lw=2.6, label="Лосс до обновления весов")
ax.axhline(floor, color="#BD6A23", lw=2, ls="--", label="Финальный held-out лосс")
ax.fill_between(
    x,
    floor,
    y,
    where=y >= floor,
    interpolate=True,
    color="#40AD9D",
    alpha=0.38,
    label="Положительный вклад",
)
ax.fill_between(
    x,
    floor,
    y,
    where=y < floor,
    interpolate=True,
    color="#D76972",
    alpha=0.5,
    label="Отрицательный вклад",
)
ax.set(xlabel="Обучающие окна", ylabel="NLL на окно, наты", xlim=(1, 100), ylim=(0.35, 4.25))
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
ax.set_yticks([])
ax.set_xticks([])
ax.legend(loc="upper right", frameon=False, fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "area_schematic.pdf")
fig.savefig(OUT / "area_schematic.png", dpi=170)
plt.close(fig)
with (ROOT / "publications/technical_report/data/run_statistics.csv").open() as f:
    runs = list(csv.DictReader(f))
with (ROOT / "publications/technical_report/data/correlations.csv").open() as f:
    corrs = [
        r
        for r in csv.DictReader(f)
        if r["subset"] == "all_11"
        and r["predictor"] == "area_ema_mnats"
        and r["target"] == "crps_geo"
    ]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.65))
for ax, series in zip(axs, ["1", "2"], strict=True):
    rows = [r for r in runs if r["series"] == series and r["init_seed"] == "0"]
    assert len(rows) == 11
    for r in rows:
        px, py = float(r["area_ema_mnats"]), float(r["crps_geo"])
        ax.scatter(px, py, s=38, color="#0C8179")
        label = r["generator"].replace("stepfunction", "step").replace("kernelsynth", "kernel")
        ax.annotate(label, (px, py), xytext=(4, 4), textcoords="offset points", fontsize=7.2)
    ax.axhline(1, color="#BD6A23", lw=1.3, ls="--")
    corr = next(r for r in corrs if r["series"] == series)
    ax.set_title(
        ("≈ 2 млн" if series == "1" else "≈ 40 млн")
        + f" окон; Spearman ρ = {float(corr['rho']):.2f}",
        fontsize=12,
    )
    ax.set_xlabel("Знаковая площадь A, млн нат", fontsize=14)
    ax.set_ylabel("CRPS к сезонному наиву (меньше лучше)", fontsize=9)
    ax.margins(x=0.16, y=0.22)
fig.tight_layout()
fig.savefig(OUT / "large_series_context.pdf")
fig.savefig(OUT / "large_series_context.png", dpi=170)
plt.close(fig)
(HERE / "data/provenance.json").write_text(
    json.dumps(
        {
            "notebook": "experiments/demos/demo_estimators.ipynb",
            "notebook_executed_code_cells": 6,
            "demo_result_bits": result,
            "illustrative_figure": "area_schematic is a synthetic illustration, not measured data",
            "report_runs": "publications/technical_report/data/run_statistics.csv",
            "report_correlations": corrs,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print("Exported actual notebook plot, illustrative area diagram and saved-series context.")
