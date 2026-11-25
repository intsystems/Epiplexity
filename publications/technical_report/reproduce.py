"""Run the library examples and tests, then generate the report figures."""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CHECK_GROUPS = {
    "DataWindows": ("test_generators", "test_chaotic", "test_corpus", "test_windows"),
    "ModelTraining": ("test_model", "test_training", "test_train_cli"),
    "PrequentialOU": ("test_estimators", "test_ou_bayes"),
    "Requential": ("test_requential", "test_requential_transfer"),
    "VInformation": ("test_v_information", "test_v_information_transfer"),
    "Evaluation": ("test_analysis", "test_evaluation"),
}


def write_check_summary(result):
    cases = [(case["module"].rsplit(".", 1)[-1], case["status"]) for case in result["cases"]]
    grouped_modules = {module for modules in CHECK_GROUPS.values() for module in modules}
    ungrouped = {module for module, _ in cases} - grouped_modules
    if ungrouped:
        raise ValueError(f"No report group for tests in {', '.join(sorted(ungrouped))}")
    lines = ["% Generated from test_results.json by reproduce.py."]
    for group, modules in CHECK_GROUPS.items():
        counts = Counter(status for module, status in cases if module in modules)
        failed = counts["failure"] + counts["error"]
        cells = f"{counts['passed']} & {counts['skipped']} & {failed}"
        lines.append(rf"\newcommand{{\Checks{group}}}{{{cells}}}")
    (HERE / "data" / "check_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_checks():
    """Record the CPU suite results and skip reasons for the current platform."""
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
    with TemporaryDirectory(prefix="check-", dir=work) as scratch:
        junit = Path(scratch) / "junit.xml"
        command = [
            sys.executable, "-m", "pytest", "tests/episynth",
            "-q", "-rs", "--junitxml=" + str(junit),
            "--basetemp=" + str(Path(scratch) / "tmp"),
        ]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        tree = ET.parse(junit) if junit.is_file() else None
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", flush=True)
    (HERE / "data" / "pytest_output.txt").write_text(
        proc.stdout + proc.stderr, encoding="utf-8"
    )
    if tree is None:
        raise RuntimeError("pytest did not produce a test report")
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
        "provenance": "Fresh CPU run of tests/episynth for the technical report",
        "platform": sys.platform,
        "cases": cases,
    }
    result["counts"] = {
        s: sum(c["status"] == s for c in cases)
        for s in ("passed", "skipped", "failure", "error")
    }
    write_check_summary(result)
    (HERE / "data" / "test_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
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
