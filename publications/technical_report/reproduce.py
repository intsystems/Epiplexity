"""Run the library examples, existing tests, and report figures."""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CHECK_GROUPS = {
    "DataWindows": ("test_generators", "test_corpus", "test_windows"),
    "ModelTraining": ("test_model", "test_training", "test_train_cli"),
    "PrequentialOU": ("test_estimators", "test_ou_bayes"),
    "VInformation": ("test_v_information",),
    "Evaluation": ("test_analysis", "test_evaluation"),
}


def write_check_summary(result):
    lines = ["% Generated from test_results.json by reproduce.py."]
    for group, modules in CHECK_GROUPS.items():
        counts = Counter(case["status"] for case in result["cases"]
                         if case["module"].rsplit(".", 1)[-1] in modules)
        failed = counts["failure"] + counts["error"]
        cells = f"{counts['passed']} & {counts['skipped']} & {failed}"
        lines.append(rf"\newcommand{{\Checks{group}}}{{{cells}}}")
    (HERE / "data" / "check_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_checks():
    """Execute the author's existing tests, preserving results and skip
    reasons.
    """
    work = ROOT / "build" / "publication_split_checks"
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
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", flush=True)
    (HERE / "data" / "pytest_output.txt").write_text(
        proc.stdout + proc.stderr, encoding="utf-8"
    )
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
        cases.append(
            {
                "name": node.get("name"),
                "module": node.get("classname"),
                "status": status,
                "details": details,
            }
        )
    result = {
        "provenance": (
            "Fresh report rerun of existing tests, not an archived author run"
        ),
        "cases": cases,
    }
    result["counts"] = {
        s: sum(c["status"] == s for c in cases)
        for s in ("passed", "skipped", "failure", "error")
    }
    (HERE / "data" / "test_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_check_summary(result)
    if proc.returncode:
        raise RuntimeError(
            "Existing test suite failed; see data/pytest_output.txt"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        [sys.executable, str(HERE / "run_examples.py")], cwd=ROOT, check=True
    )
    if not args.skip_checks:
        run_checks()
    subprocess.run(
        [sys.executable, str(HERE / "make_plots.py")], cwd=HERE, check=True
    )
