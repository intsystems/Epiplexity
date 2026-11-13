"""Recompute the archived experiment analysis and article figures."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for script in ("analyze_runs.py", "prepare_tables.py", "make_plots.py"):
        subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, check=True)
