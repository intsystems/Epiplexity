"""Run the report's CPU examples and verify their saved measurements."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from episynth.checks.ou_bayes import (
        exact_window_nll,
        phi_nodes,
        pointwise_window_nll,
    )
    from episynth.estimators.prequential import PrequentialEstimator
    from episynth.estimators.v_information import VInformationEstimator
    from episynth.generators import GENERATORS
    from episynth.windows import WindowSpec, window_batch
    torch.set_num_threads(4)
    work = ROOT / "build" / "report_demo"
    work.mkdir(parents=True, exist_ok=True)
    configs = {
        "prequential": HERE / "examples" / "prequential.json",
        "v_information": ROOT / "experiments" / "demos" / "v_information.json",
    }
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONIOENCODING="utf-8",
    )
    outputs = {}
    for name, config in configs.items():
        out = work / name
        subprocess.run(
            [
                sys.executable,
                "-m",
                "episynth.train",
                "--config",
                str(config),
                "--out",
                str(out),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        outputs[name] = json.loads(
            (out / "estimator_result.json").read_text(encoding="utf-8")
        )
    pre = PrequentialEstimator().from_run_dir(work / "prequential")
    for key, value in pre.items():
        assert math.isclose(value, outputs["prequential"][key], abs_tol=1e-8)
    assert math.isclose(
        pre["total_bits"],
        pre["model_part_bits"] + pre["data_part_bits"],
        abs_tol=1e-8,
    )
    vi = VInformationEstimator().from_run_dir(work / "v_information")
    baseline = np.fromfile(
        work / "v_information" / "test_unconditional.f32", dtype=np.float32
    ).astype(float)
    conditional = np.fromfile(
        work / "v_information" / "test_conditional.f32", dtype=np.float32
    ).astype(float)
    gain = (baseline - conditional) / math.log(2)
    assert math.isclose(
        gain.mean(), vi["v_information_bits_per_window"], abs_tol=1e-8
    )
    assert math.isclose(
        gain.std(ddof=1) / math.sqrt(len(gain)),
        vi["standard_error_bits_per_window"],
        abs_tol=1e-8,
    )

    spec = WindowSpec(
        series_length=64,
        context_max=32,
        horizon=16,
        patch_size=8,
        context_lengths=(16, 32),
        dequantize_width=0.0,
    )
    batch = window_batch(GENERATORS["ou"](length=64), np.arange(64), spec)
    oracle = {}
    for name, scorer in [
        ("joint", exact_window_nll),
        ("pointwise", pointwise_window_nll),
    ]:
        coarse = scorer(batch, spec, phi_nodes(512))
        fine = scorer(batch, spec, phi_nodes(1024))
        assert np.isfinite(fine).all()
        difference = float(np.max(np.abs(coarse - fine)))
        assert difference < 1e-3, (name, difference)
        oracle[name] = {
            "nats_per_window": float(fine.mean()),
            "max_quadrature_change": difference,
        }
    cfg = json.loads(configs["prequential"].read_text(encoding="utf-8"))
    losses = np.fromfile(
        work / "prequential" / "window_loss.f32", dtype=np.float32
    ).astype(float)
    floor = (
        np.fromfile(
            work / "prequential" / "heldout_ema_1.f32", dtype=np.float32
        )
        .astype(float)
        .mean()
    )
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    result = {
        "examples": outputs,
        "ou_reference": oracle,
        "prequential_curve": {
            "batch_size": cfg["train"]["batch_size"],
            "horizon": cfg["spec"]["horizon"],
            "loss_nats_per_window": (
                losses.reshape(-1, cfg["train"]["batch_size"]).mean(1).tolist()
            ),
            "heldout_nats_per_window": float(floor),
        },
        "verification": {
            "recomputed_prequential": True,
            "paired_test_gain": True,
            "paired_standard_error": True,
            "ou_quadrature_convergence": True,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pydantic")
        },
        "configuration_sha256": {
            k: hashlib.sha256(v.read_bytes()).hexdigest()
            for k, v in configs.items()
        },
    }
    (data / "example_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Verified prequential recomputation, paired information, and OU"
        " quadrature."
    )


if __name__ == "__main__":
    main()
