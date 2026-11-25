"""Build concise manuscript tables from the analysis outputs."""

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    with (DATA / "correlations.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        (
            r"Budget & Predictor / sources & $n$ & $\rho$ & Lower 95\% & Upper"
            r" 95\%\\"
        ),
        r"\midrule",
    ]
    for series, label in [("1", "2M"), ("2", "40M")]:
        for predictor, subset, description in [
            ("area_ema_mnats", "all_11", "EMA area, all sources"),
            ("area_raw_mnats", "all_11", "Raw area, all sources"),
            ("final_ema_nats_per_point", "all_11", "Final loss, all sources"),
            (
                "area_ema_mnats",
                "without_chaotic_waveform",
                "EMA area, selected subset",
            ),
        ]:
            row = next(
                r
                for r in rows
                if r["series"] == series
                and r["predictor"] == predictor
                and r["subset"] == subset
                and r["target"] == "crps_geo"
            )
            cells = [
                label,
                description,
                row["n"],
                *[f"{float(row[k]):.2f}" for k in ("rho", "lo", "hi")],
            ]
            lines.append(" & ".join(cells) + r" \\")
        if series == "1":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "correlation_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    p = 0.1
    values = [
        {
            "q": q,
            "crps_divergence": (p - q) ** 2,
            "kl_nats": (
                p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
            ),
        }
        for q in (0.3, 0.001)
    ]
    assert values[0]["crps_divergence"] > values[1]["crps_divergence"]
    assert values[0]["kl_nats"] < values[1]["kl_nats"]
    # Independently evaluate the expected Bernoulli squared score and log
    # score.
    for row in values:
        q = row["q"]
        crps_risk = p * (1 - q) ** 2 + (1 - p) * q ** 2
        log_risk = -p * math.log(q) - (1 - p) * math.log(1 - q)
        entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)
        assert math.isclose(
            crps_risk - p * (1 - p), row["crps_divergence"], abs_tol=1e-12
        )
        assert math.isclose(log_risk - entropy, row["kl_nats"], abs_tol=1e-12)
    lines = [
        r"\begin{tabular}{@{}rrr@{}}",
        r"\toprule",
        r"Predicted success probability & CRPS divergence & KL (nats)\\",
        r"\midrule",
    ]
    for row in values:
        lines.append(
            f"{row['q']:g} & {row['crps_divergence']:.6f} &"
            f" {row['kl_nats']:.6f}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "counterexample.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (DATA / "counterexample.json").write_text(
        json.dumps({"truth": p, "predictions": values}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Generated correlation and counterexample tables.")


if __name__ == "__main__":
    main()
