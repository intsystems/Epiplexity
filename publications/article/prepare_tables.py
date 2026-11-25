"""Build concise manuscript tables from the analysis outputs."""

import csv
import json
import math
from pathlib import Path

from analyze_runs import bootstrap_rho

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def write_coding_correlations():
    with (DATA / "run_statistics.csv").open(newline="", encoding="utf-8") as handle:
        archive = {row["run"]: row for row in csv.DictReader(handle)}
    lines = [r"\begin{tabular}{@{}llrrl@{}}", r"\toprule",
             r"Measure & Budget & $n$ & $\rho$ & 95\% interval\\", r"\midrule"]
    with (DATA / "correlations.csv").open(newline="", encoding="utf-8") as handle:
        prequential = {row["series"]: row for row in csv.DictReader(handle)
                       if row["predictor"] == "area_ema_mnats" and row["subset"] == "all_11"
                       and row["target"] == "crps_geo"}
    for series, budget in (("1", "2M"), ("2", "40M")):
        row = prequential[series]
        rho, lo, hi = (float(row[key]) for key in ("rho", "lo", "hi"))
        lines.append(f"Prequential EMA area & {budget} & {row['n']} & ${rho:.2f}$ & "
                     rf"$[{lo:.2f},\,{hi:.2f}]$ \\")
    for filename, metric, label, budgets in [
        ("requential_vs_crps_2m.csv", "estimated_kl_bits", "Requential KL", {"1": "2M"}),
        ("v_information_vs_crps.csv", "v_information_bits_per_window",
         r"Predictive $\mathcal V$-information", {"1": "2M", "2": "40M"}),
    ]:
        with (DATA / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = {run for run, row in archive.items() if row["series"] in budgets}
        if len(rows) != len(expected) or {row["run"] for row in rows} != expected:
            raise ValueError(f"Unexpected run coverage in {filename}")
        for row in rows:
            match = archive[row["run"]]
            if any(row[key] != match[key] for key in ("series", "generator", "init_seed", "n_windows", "crps_geo")):
                raise ValueError(f"Archive mismatch for {row['run']}")
        for series, budget in budgets.items():
            primary = [row for row in rows if row["series"] == series and row["init_seed"] == "0"]
            stats = bootstrap_rho([float(row[metric]) for row in primary],
                                  [float(row["crps_geo"]) for row in primary])
            lines.append(f"{label} & {budget} & {stats['n']} & ${stats['rho']:.2f}$ & "
                         rf"$[{stats['lo']:.2f},\,{stats['hi']:.2f}]$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "coding_correlation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    write_coding_correlations()
    print("Generated correlation and counterexample tables.")


if __name__ == "__main__":
    main()
