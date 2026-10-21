"""Собирает лёгкий пакет результатов серии: конфиги, лог шагов, отложенные лоссы по чекпойнтам, кривая обучения по шагам (среднее лосса на точку по 1024 окнам шага), сводки GIFT-Eval.

Запуск: python bundle_results.py <каталог результатов> <префикс прогонов> <каталог пакета>
"""

import shutil
import sys
from pathlib import Path

import numpy as np

root, prefix, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
H, B = 64, 1024
for run in sorted((root / "runs").glob(f"{prefix}*")):
    dst = out / "runs" / run.name
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "run.json", "heldout.jsonl", "steps.csv"):
        if (run / name).exists():
            shutil.copy(run / name, dst / name)
    losses = np.fromfile(run / "window_loss.f32", dtype=np.float32).astype(np.float64) / H
    n = len(losses) // B * B
    per_step = losses[:n].reshape(-1, B).mean(1)
    with open(dst / "train_curve.csv", "w") as f:
        f.write("step,windows_seen,nats_per_point\n")
        for i, v in enumerate(per_step, 1):
            f.write(f"{i},{i * B},{v:.6g}\n")
    print(run.name, len(losses), "окон,", len(per_step), "шагов, лосс первого/последнего шага", f"{per_step[0]:.3f}/{per_step[-1]:.3f}")
for ev in sorted((root / "eval").glob(f"{prefix}*")) + [root / "eval" / "seasonal_naive"]:
    dst = out / "eval" / ev.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(ev / "summary.csv", dst / "summary.csv")
for name in ("gzip_baseline.json", "verify_corpora.log"):
    if (root / name).exists():
        shutil.copy(root / name, out / name)
print("BUNDLE_DONE", out)
