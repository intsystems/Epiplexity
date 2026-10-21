"""Recompute saved-run results, redraw figures, and run existing episynth checks.

Analysis uses Chernousovdv's committed series1/series2 experiment outputs.
Supplementary generator examples and test outcomes are new report reproductions.
Run from any directory with Python >=3.10 and the report requirements installed.
All temporary training checkpoints stay in the repository's ignored build/ folder.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from episynth.generators import GENERATORS
from episynth.windows import WindowSpec, window_batch


def regenerate_examples():
    """Save the exact plotted arrays; use the same seed pair for every generator."""
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    seeds = np.array([3, 11], dtype=np.int64)
    arrays = {name: cls(length=256).generate(seeds) for name, cls in GENERATORS.items()}
    if not all(x.shape == (2, 256) and np.isfinite(x).all() for x in arrays.values()):
        raise RuntimeError("A plotted generator produced invalid output")
    np.savez_compressed(data / "generator_examples.npz", seeds=seeds, **arrays)

    spec = WindowSpec(context_lengths=(128, 256, 512), dequantize_width=1e-3)
    batch = window_batch(GENERATORS["waveform"](length=1024), np.arange(24), spec)
    indices = [batch.context_length.tolist().index(c) for c in spec.context_lengths]
    np.savez_compressed(
        data / "window_examples.npz",
        target=batch.target[indices].numpy().reshape(3, -1),
        observed=batch.observed_mask[indices].numpy().reshape(3, -1),
        seeds=batch.seeds[indices].numpy(),
        context_length=batch.context_length[indices].numpy(),
    )
    (data / "figure_protocol.json").write_text(
        json.dumps(
            {
                "provenance": "New descriptive examples generated for this report; not historical experiment results",
                "generator_length": 256,
                "generator_seeds": seeds.tolist(),
                "display_transform": "Each series independently centered and divided by its population standard deviation",
                "window_spec": spec.model_dump(),
                "window_generator": "waveform",
                "selected_window_seeds": batch.seeds[indices].tolist(),
            }, indent=2,
        ), encoding="utf-8",
    )


def run_checks():
    """Execute the author's existing tests, preserving results and skip reasons."""
    work = ROOT / "build" / "technical_report_checks"
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTHONIOENCODING"] = "utf-8"
    # External reference data/checkpoints are absent; avoid unrelated caches.
    env.pop("GIFT_EVAL_DIR", None)
    env.pop("MOIRAI_SMALL_DIR", None)
    command = [
        sys.executable, "-m", "pytest", str(ROOT / "tests" / "episynth"),
        "-q", "-rs", "--junitxml=" + str(work / "junit.xml"),
        "--basetemp=" + str(work / "pytest_tmp"),
    ]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", flush=True)
    (HERE / "data" / "pytest_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if not (work / "junit.xml").exists():
        raise RuntimeError("pytest did not produce a test report")
    tree = ET.parse(work / "junit.xml")
    cases = []
    for node in tree.iter("testcase"):
        status = "passed"
        details = ""
        for result in ("failure", "error", "skipped"):
            child = node.find(result)
            if child is not None:
                status, details = result, child.get("message", "")
                break
        cases.append({"name": node.get("name"), "module": node.get("classname"), "status": status, "details": details})
    result = {"provenance": "Fresh report rerun of existing tests, not an archived author run", "cases": cases}
    result["counts"] = {s: sum(c["status"] == s for c in cases) for s in ("passed", "skipped", "failure", "error")}
    (HERE / "data" / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (HERE / "data" / "check_summary.tex").write_text(
        "\\newcommand{\\ChecksPassed}{" + str(result["counts"]["passed"]) + "}\n"
        + "\\newcommand{\\ChecksSkipped}{" + str(result["counts"]["skipped"]) + "}\n"
        + "\\newcommand{\\ChecksFailed}{" + str(result["counts"]["failure"] + result["counts"]["error"]) + "}\n",
        encoding="utf-8",
    )
    if proc.returncode:
        raise RuntimeError("Existing test suite failed; see data/pytest_output.txt")


def record_provenance():
    git = ["git", "-c", "safe.directory=" + ROOT.as_posix()]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    message = subprocess.check_output(git + ["show", "-s", "--format=fuller", "7d2b001"], cwd=ROOT, text=True)
    files = []
    for path in [Path(r"D:\DOWNLOAD\Telegram Desktop\lit_talk_v2.pdf"), Path(r"D:\DOWNLOAD\Telegram Desktop\proposal_3.4.pdf")]:
        if path.exists():
            files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    packages = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "pydantic", "pyarrow", "safetensors", "matplotlib", "pytest"):
        packages[name] = importlib.metadata.version(name)
    provenance = {
        "report_date": "2026-10-21", "commit": commit, "author_commit_message": message,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "device": "cpu", "torch_threads": torch.get_num_threads(), "context_documents": files,
        "historical_outputs_available": True,
        "historical_outputs": [
            "experiments/episynth/series1_2026-10-19",
            "experiments/episynth/series2_2026-10-19",
        ],
        "analysis_entrypoint": "analyze_runs.py",
        "fresh_reproductions": ["supplementary descriptive examples", "existing test-suite rerun"],
    }
    (HERE / "data" / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checks", action="store_true", help="Recompute analysis and figures without rerunning tests")
    args = parser.parse_args()
    torch.set_num_threads(4)
    regenerate_examples()
    record_provenance()
    if not args.skip_checks:
        run_checks()
    subprocess.run([sys.executable, str(HERE / "analyze_runs.py")], cwd=HERE, check=True)
    subprocess.run([sys.executable, str(HERE / "make_plots.py")], cwd=HERE, check=True)
