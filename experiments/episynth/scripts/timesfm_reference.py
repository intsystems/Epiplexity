"""Сопоставимость опубликованных чисел TimesFM-3 (лидерборд GIFT-Eval) с нашим протоколом: совпадение наива по задачам, отношения к наиву по MASE и MWSQL, разрыв между scaled_crps и MWSQL у наших моделей.

Запуск: python timesfm_reference.py <каталог результатов серии> <каталог published>
Печатает сводку и пишет timesfm_reference.json рядом со скриптом.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, pub = Path(sys.argv[1]), Path(sys.argv[2])
prefix = sys.argv[3] if len(sys.argv) > 3 else "s2_"
ours_naive = pd.read_csv(root / "eval" / "seasonal_naive" / "summary.csv").set_index("config")
pub_naive = pd.read_csv(pub / "gift_eval_seasonal_naive.csv").set_index("dataset")
pub_tfm = pd.read_csv(pub / "gift_eval_timesfm-3.csv").set_index("dataset")
configs = sorted(ours_naive.index)
missing = [c for c in configs if c not in pub_naive.index or c not in pub_tfm.index]
print("задач у нас:", len(configs), "нет в опубликованных:", missing)
configs = [c for c in configs if c not in missing]

def gmean(r):
    return float(np.exp(np.log(r).mean()))

# совпадение наива
for col_ours, col_pub in (("MASE[0.5]", "eval_metrics/MASE[0.5]"), ("mean_weighted_sum_quantile_loss", "eval_metrics/mean_weighted_sum_quantile_loss")):
    a = ours_naive.loc[configs, col_ours].to_numpy(); b = pub_naive.loc[configs, col_pub].to_numpy()
    rel = np.abs(a - b) / b
    print(f"наив {col_ours}: макс. отн. расхождение {rel.max():.2e}, медиана {np.median(rel):.2e}, задач с расхождением >1e-3: {(rel > 1e-3).sum()}")
    worst = np.argsort(rel)[-3:]
    for i in worst:
        print(f"   {configs[i]}: наше {a[i]:.6g}, опубл. {b[i]:.6g}")

# TimesFM-3 к наиву (опубликованному) по тем же 55 задачам
tfm = {}
for key, col in (("MASE", "eval_metrics/MASE[0.5]"), ("MWSQL", "eval_metrics/mean_weighted_sum_quantile_loss")):
    r = pub_tfm.loc[configs, col].to_numpy() / pub_naive.loc[configs, col].to_numpy()
    tfm[key] = {"geo_mean_ratio": gmean(r), "median_ratio": float(np.median(r)), "share_better": float((r < 1).mean()), "n": len(configs)}
    print(f"TimesFM-3 {key}/наив: геом. {tfm[key]['geo_mean_ratio']:.3f}, медиана {tfm[key]['median_ratio']:.3f}, лучше наива в {tfm[key]['share_better']:.2f} задач")

# у наших моделей: scaled_crps/наив против MWSQL/наив
rows = []
for d in sorted((root / "eval").glob(f"{prefix}*")):
    if "seed" in d.name:
        continue
    t = pd.read_csv(d / "summary.csv").set_index("config")
    cc = [c for c in configs if c in t.index]
    r_sc = t.loc[cc, "scaled_crps"].to_numpy() / ours_naive.loc[cc, "scaled_crps"].to_numpy()
    r_mw = t.loc[cc, "mean_weighted_sum_quantile_loss"].to_numpy() / ours_naive.loc[cc, "mean_weighted_sum_quantile_loss"].to_numpy()
    r_ma = t.loc[cc, "MASE[0.5]"].to_numpy() / ours_naive.loc[cc, "MASE[0.5]"].to_numpy()
    rows.append((d.name, gmean(r_sc), gmean(r_mw), float(np.median(r_sc)), float(np.median(r_mw)), gmean(r_ma)))
print("\nмодель | scaled_crps/наив геом | MWSQL/наив геом | медианы | MASE/наив геом")
for r in rows:
    print(f"{r[0]:22s} {r[1]:.3f} {r[2]:.3f}   {r[3]:.3f} {r[4]:.3f}   {r[5]:.3f}")
json.dump({"timesfm3": tfm, "ours": {r[0]: {"scaled_crps_geo": r[1], "mwsql_geo": r[2], "scaled_crps_median": r[3], "mwsql_median": r[4], "mase_geo": r[5]} for r in rows}, "configs": configs},
          open(Path(__file__).with_name("timesfm_reference.json"), "w"), indent=1, ensure_ascii=False)
