"""Bundle run configs, step logs, held-out checkpoint losses, and GIFT-Eval
summaries. The training curve records mean loss per point over 1,024 windows
per step.

Usage: python bundle_results.py <series directory> <run prefix> <output
    directory>
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
    losses = (
        np.fromfile(run / "window_loss.f32", dtype=np.float32).astype(
            np.float64
        )
        / H
    )
    n = len(losses) // B * B
    per_step = losses[:n].reshape(-1, B).mean(1)
    with open(dst / "train_curve.csv", "w") as f:
        f.write("step,windows_seen,nats_per_point\n")
        for i, v in enumerate(per_step, 1):
            f.write(f"{i},{i * B},{v:.6g}\n")
    print(
        run.name,
        len(losses),
        "window",
        len(per_step),
        "Loss of the first/last step.",
        f"{per_step[0]:.3f}/{per_step[-1]:.3f}",
    )
for ev in sorted((root / "eval").glob(f"{prefix}*")) + [
    root / "eval" / "seasonal_naive"
]:
    dst = out / "eval" / ev.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(ev / "summary.csv", dst / "summary.csv")
for name in ("gzip_baseline.json", "verify_corpora.log"):
    if (root / name).exists():
        shutil.copy(root / name, out / name)
print("BUNDLE_DONE", out)
