"""Сводка серии: площади, финальные лоссы, качество относительно наива и по рангам, корреляции Спирмена.

Запуск: python series_analysis.py <каталог результатов> [gzip_baseline.json]
Ожидает runs/series_<g>/ и eval/series_<g>/summary.csv, eval/seasonal_naive/summary.csv.
"""

import json
import os
import sys
from pathlib import Path

from episynth.analysis import correlation_table, quality_table, run_area

PREFIX = os.environ.get("SERIES_PREFIX", "series_")
root = Path(sys.argv[1])
gz = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
gens = sorted(p.name.removeprefix(PREFIX) for p in (root / "runs").glob(f"{PREFIX}*") if (p / "heldout_ema_1.f32").exists() and "seed" not in p.name)
evals = {g: root / "eval" / f"{PREFIX}{g}" / "summary.csv" for g in gens if (root / "eval" / f"{PREFIX}{g}" / "summary.csv").exists()}
naive = root / "eval" / "seasonal_naive" / "summary.csv"
areas = {g: run_area(root / "runs" / f"{PREFIX}{g}", "ema") for g in gens}
areas_raw = {g: run_area(root / "runs" / f"{PREFIX}{g}", "raw") for g in gens}
quality = {m: quality_table(evals, naive, metric=m) for m in ("scaled_crps", "MASE[0.5]", "mean_weighted_sum_quantile_loss", "nll_per_point")} if evals else {}

print(f"{'генератор':14} {'A_ema, млн':>11} {'±':>6} {'A_raw, млн':>11} {'L_fin ema':>10} {'L_fin raw':>10} {'ранг CRPS':>10} {'CRPS/наив':>10} {'MASE/наив':>10} {'gzip q16':>9}")
for g in gens:
    a, r = areas[g], areas_raw[g]
    q = quality.get("scaled_crps", {}).get(g, {})
    qm = quality.get("MASE[0.5]", {}).get(g, {})
    print(f"{g:14} {a.area / 1e6:11.2f} {a.area_se / 1e6:6.2f} {r.area / 1e6:11.2f} {a.l_fin / 64:10.4f} {r.l_fin / 64:10.4f} {q.get('mean_rank', float('nan')):10.2f} {q.get('geo_mean_ratio', float('nan')):10.3f} {qm.get('geo_mean_ratio', float('nan')):10.3f} {gz.get(g, {}).get('q16_bytes_per_point', float('nan')):9.3f}")

if evals and len(evals) >= 4:
    rank = {g: quality["scaled_crps"][g]["mean_rank"] for g in evals}
    ratio = {g: quality["scaled_crps"][g]["geo_mean_ratio"] for g in evals}
    predictors = {
        "area_ema": {g: areas[g].area for g in evals},
        "area_raw": {g: areas_raw[g].area for g in evals},
        "l_fin_ema": {g: areas[g].l_fin for g in evals},
        "l_fin_raw": {g: areas_raw[g].l_fin for g in evals},
    }
    if gz:
        predictors["gzip_q16"] = {g: gz[g]["q16_bytes_per_point"] for g in evals if g in gz}
    for name, target in (("средний ранг по CRPS (меньше лучше)", rank), ("геом. среднее CRPS/наив", ratio)):
        print(f"\nСпирмен с «{name}», n={len(target)}:")
        for p, v in correlation_table(predictors, target).items():
            print(f"  {p:10} rho {v['rho']:+.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]")
