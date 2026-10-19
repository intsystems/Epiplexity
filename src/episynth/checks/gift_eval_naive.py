r"""Сверка нашего конвейера оценки с опубликованными числами seasonal naive GIFT-Eval.

    python -m episynth.checks.gift_eval_naive --data <снимок GiftEval> --published <results/Seasonal_Naive/all_results.csv> \\
        [--out <каталог прогона>] [--configs ...]

Считает сезонный наивный прогноз нашим кодом по краткосрочным конфигурациям и сравнивает
точечные метрики (MASE, sMAPE, ND, MAE, MSE[0.5]) с опубликованными: они зависят только от
данных, окон и формул, а не от интервалов statsforecast. Квантильные метрики опубликованного
наива зависят от его интервалов и здесь не сверяются. Печатает относительное расхождение по
конфигурациям и наибольшее по каждой метрике.
"""

import argparse
import csv
from pathlib import Path

from episynth.evaluation.forecasters import SeasonalNaive
from episynth.evaluation.gift_eval import SHORT_CONFIGS, GiftDataset
from episynth.evaluation.run import evaluate_config

_POINT = {"MASE[0.5]": "eval_metrics/MASE[0.5]", "sMAPE[0.5]": "eval_metrics/sMAPE[0.5]", "ND[0.5]": "eval_metrics/ND[0.5]", "MAE[0.5]": "eval_metrics/MAE[0.5]", "MSE[0.5]": "eval_metrics/MSE[0.5]"}


def main() -> None:
    """Разбор аргументов, прогон наива и сравнение."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--configs", nargs="*", default=list(SHORT_CONFIGS))
    args = parser.parse_args()
    with open(args.published) as f:
        published = {r["dataset"]: r for r in csv.DictReader(f)}
    worst = {k: 0.0 for k in _POINT}
    out_rows = []
    for name in args.configs:
        ds = GiftDataset(args.data, name, "short")
        _, agg = evaluate_config(ds, SeasonalNaive(ds.season))
        pub = published[ds.config]
        rel = {k: abs(agg[k] - float(pub[col])) / max(abs(float(pub[col])), 1e-12) for k, col in _POINT.items()}
        worst = {k: max(v, rel[k]) for k, v in worst.items()}
        out_rows.append({"config": ds.config, "windows": ds.windows, "series": len(ds.series), **{f"rel_{k}": v for k, v in rel.items()}, **{f"ours_{k}": v for k, v in agg.items() if k in rel}})
        print(f"{ds.config:40s} " + " ".join(f"{k} {v:.1e}" for k, v in rel.items()), flush=True)
    print("наибольшее относительное расхождение:", {k: f"{v:.1e}" for k, v in worst.items()})
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "naive_check.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)


if __name__ == "__main__":
    main()
