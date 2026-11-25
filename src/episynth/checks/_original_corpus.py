"""The body of the original Mix generator, Don't Pick, counted by a separate
interpreter.

    python _original_corpus.py <clone> <name> <length> <side> <n> <out.npy>

Prints in stdout the generation time in seconds. Runs the interpreter with
versions of the
dependencies of the original (numpy 1.26): under the numpy 2, part of the
original code falls and
silently replaced by spare rows.
"""

import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np

MODULES = {
    "kernelsynth": ("kernelsynth", "KernelSynthesizer"),
    "ets": ("ets", "ETSSynthesizer"),
    "waveform": ("waveform", "WaveformSynthesizer"),
    "sde": ("sde", "SDESynthesizer"),
    "stepfunction": ("stepfunction", "StepFunctionSynthesizer"),
    "tsi": ("tsi", "TSISynthesizer"),
    "arima": ("arima", "ARIMASynthesizer"),
    "fbm": ("fbm", "FBMSynthesizer"),
    "garch": ("garch", "GARCHSynthesizer"),
    "chaotic": ("chaotic", "ChaoticSynthesizer"),
    "timesynth": ("timesynth_gen", "TimeSynthesizer"),
}


def main(
    repo: Path, name: str, length: int, seed: int, n: int, out: Path
) -> None:
    """Counts the body of the original, saves it to `out` and prints the
    generation time.
    """
    pkg = types.ModuleType("mdpgen")
    pkg.__path__ = [str(repo / "generators")]
    sys.modules["mdpgen"] = pkg
    module, cls = MODULES[name]
    gen_cls = getattr(importlib.import_module(f"mdpgen.{module}"), cls)
    t0 = time.perf_counter()
    corpus = gen_cls(length=length, random_seed=seed).generate_series(n)
    elapsed = time.perf_counter() - t0
    np.save(out, corpus)
    print(f"{elapsed:.6f}")


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), a[2], int(a[3]), int(a[4]), int(a[5]), Path(a[6]))
