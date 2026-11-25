"""Compare the published TimesFM-3 GIFT-Eval results with our protocol.\nCheck
seasonal-naive results by task, calculate MASE and MWSQL ratios,\nand compare
scaled CRPS with MWSQL for our models.

Launch: python timesfm_reference.py <series results directory> <catalog
    published> Prints a summary
and writes timesfm_reference.json next to the script.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, pub = Path(sys.argv[1]), Path(sys.argv[2])
prefix = sys.argv[3] if len(sys.argv) > 3 else "s2_"
ours_naive = pd.read_csv(
    root / "eval" / "seasonal_naive" / "summary.csv"
).set_index("config")
pub_naive = pd.read_csv(pub / "gift_eval_seasonal_naive.csv").set_index(
    "dataset"
)
pub_tfm = pd.read_csv(pub / "gift_eval_timesfm-3.csv").set_index("dataset")
configs = sorted(ours_naive.index)
missing = [
    c for c in configs if c not in pub_naive.index or c not in pub_tfm.index
]
print("Local tasks:", len(configs), "Missing from published results:", missing)
configs = [c for c in configs if c not in missing]


def gmean(r):
    return float(np.exp(np.log(r).mean()))


# Compare seasonal-naive metrics.
for col_ours, col_pub in (
    ("MASE[0.5]", "eval_metrics/MASE[0.5]"),
    (
        "mean_weighted_sum_quantile_loss",
        "eval_metrics/mean_weighted_sum_quantile_loss",
    ),
):
    a = ours_naive.loc[configs, col_ours].to_numpy()
    b = pub_naive.loc[configs, col_pub].to_numpy()
    rel = np.abs(a - b) / b
    print(
        f"Seasonal naive {col_ours}: max relative difference {rel.max():.2e},"
        f" median {np.median(rel):.2e}, tasks above 1e-3: {(rel > 1e-3).sum()}"
    )
    worst = np.argsort(rel)[-3:]
    for i in worst:
        print(f"   {configs[i]}: local {a[i]:.6g}, published {b[i]:.6g}")

# TimesFM-3 to naive (published) on the same 55 problems
tfm = {}
for key, col in (
    ("MASE", "eval_metrics/MASE[0.5]"),
    ("MWSQL", "eval_metrics/mean_weighted_sum_quantile_loss"),
):
    r = (
        pub_tfm.loc[configs, col].to_numpy()
        / pub_naive.loc[configs, col].to_numpy()
    )
    tfm[key] = {
        "geo_mean_ratio": gmean(r),
        "median_ratio": float(np.median(r)),
        "share_better": float((r < 1).mean()),
        "n": len(configs),
    }
    print(
        f"TimesFM-3 {key} / seasonal naive: geometric mean"
        f" {tfm[key]['geo_mean_ratio']:.3f}, median"
        f" {tfm[key]['median_ratio']:.3f}, share beating naive"
        f" {tfm[key]['share_better']:.2f}"
    )

# In our models, scaled_crps / naive vs. MWSQL / naive
rows = []
for d in sorted((root / "eval").glob(f"{prefix}*")):
    if "seed" in d.name:
        continue
    t = pd.read_csv(d / "summary.csv").set_index("config")
    cc = [c for c in configs if c in t.index]
    r_sc = (
        t.loc[cc, "scaled_crps"].to_numpy()
        / ours_naive.loc[cc, "scaled_crps"].to_numpy()
    )
    r_mw = (
        t.loc[cc, "mean_weighted_sum_quantile_loss"].to_numpy()
        / ours_naive.loc[cc, "mean_weighted_sum_quantile_loss"].to_numpy()
    )
    r_ma = (
        t.loc[cc, "MASE[0.5]"].to_numpy()
        / ours_naive.loc[cc, "MASE[0.5]"].to_numpy()
    )
    rows.append(
        (
            d.name,
            gmean(r_sc),
            gmean(r_mw),
            float(np.median(r_sc)),
            float(np.median(r_mw)),
            gmean(r_ma),
        )
    )
print(
    "\n model | scaled_crps/naive geom | MWSQL/naive geom | medians |"
    " MASE/naive geom"
)
for r in rows:
    print(
        f"{r[0]:22s} {r[1]:.3f} {r[2]:.3f}   {r[3]:.3f} {r[4]:.3f}  "
        f" {r[5]:.3f}"
    )
with Path(__file__).with_name("timesfm_reference.json").open("w") as out:
    json.dump(
        {
            "timesfm3": tfm,
            "ours": {
                r[0]: {
                    "scaled_crps_geo": r[1],
                    "mwsql_geo": r[2],
                    "scaled_crps_median": r[3],
                    "mwsql_median": r[4],
                    "mase_geo": r[5],
                }
                for r in rows
            },
            "configs": configs,
        },
        out,
        indent=1,
        ensure_ascii=False,
    )
