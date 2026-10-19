"""Сравнение `summary.csv` нашего прогона с опубликованной таблицей модели в GIFT-Eval.

    python -m episynth.checks.gift_eval_published --summary <summary.csv> --published <all_results.csv>

Печатает по конфигурациям опубликованное и наше значение MASE, sMAPE, ND и CRPS лидерборда
(`mean_weighted_sum_quantile_loss`) и относительное расхождение; у сэмплирующих моделей
ожидается расхождение на уровне шума сэмплирования.
"""

import argparse
import csv
from pathlib import Path

_COLS = {
    "MASE[0.5]": "eval_metrics/MASE[0.5]",
    "sMAPE[0.5]": "eval_metrics/sMAPE[0.5]",
    "ND[0.5]": "eval_metrics/ND[0.5]",
    "mean_weighted_sum_quantile_loss": "eval_metrics/mean_weighted_sum_quantile_loss",
}


def main() -> None:
    """Разбор аргументов и печать сравнения."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    args = parser.parse_args()
    with open(args.published) as f:
        published = {r["dataset"]: r for r in csv.DictReader(f)}
    with open(args.summary) as f:
        ours = list(csv.DictReader(f))
    for r in ours:
        pub = published.get(r["config"])
        if pub is None:
            print(f"{r['config']:40s} нет в опубликованных")
            continue
        parts = []
        for k, col in _COLS.items():
            a, b = float(r[k]), float(pub[col])
            parts.append(f"{k.split('[')[0][:5]:5s} {b:.4f} -> {a:.4f} ({(a - b) / max(abs(b), 1e-12):+.1%})")
        print(f"{r['config']:40s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
