r"""Generators vs. originals Mix, Don't Pick: playback, distribution, speed.

    python -m episynth.checks.generators --original < clone mix-dont-pick> --out <catalogue>
        [-original-python <interpreter with dependencies of the original>] [-names ets sde ...]

The original is considered a separate interpreter (`_original_corpus.py`) because it needs its own
versions of dependencies: under the numpy 2, the original TimeSynth generator drops in step-by-step
cycles and silently replaces the rows with spares. Each generator is written
"<catalogue>/<name>.json" with three blocks.

- `exact`: hull `reproduce(seed, n)` vs `generate_series(n)` original with the same seat:
  the proportion of rows that matched to the bit and the largest relative error (to the standard
  deviation of the series) among the rest.
- `distribution`: series statistics (standard deviation, scale, middle module)
  The difference, the autocorrelation of the difference, the fraction of the energy of the spectrum
  above a quarter of the Nyquist frequency) in the body of the original and in the body of the main
  mode on other seats, the p-value of the Kolmogorov-Smirnov criterion; the proportion of infinite
  series in both. In generators where the main mode coincides with the reproduction mode, this is a test
  of the criterion itself, not of the code.
- `speed`: rows per second for the original and the main mode.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from episynth.generators import GENERATORS

_HELPER = Path(__file__).with_name("_original_corpus.py")


def original_corpus(python: str, repo: Path, name: str, length: int, seed: int, n: int) -> tuple[np.ndarray, float]:
    """The body of the original `(n, length)` float32 and its generation time in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "corpus.npy"
        proc = subprocess.run(
            [python, "-I", str(_HELPER), str(repo), name, str(length), str(seed), str(n), str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.load(out), float(proc.stdout.strip().splitlines()[-1])


def series_stats(y: np.ndarray) -> dict[str, float]:
    """Statistics of the same series for comparison of distributions."""
    y = y.astype(np.float64)
    d = np.diff(y)
    dc = d - d.mean()
    spec = np.abs(np.fft.rfft(y - y.mean())) ** 2
    cut = len(spec) // 4
    return {
        "std": float(y.std()),
        "range": float(y.max() - y.min()),
        "mean_abs_diff": float(np.abs(d).mean()),
        "acf1_diff": float((dc[1:] * dc[:-1]).sum() / max((dc * dc).sum(), 1e-300)),
        "high_freq_share": float(spec[cut:].sum() / max(spec.sum(), 1e-300)),
    }


def check_one(name: str, repo: Path, python: str, length: int, n_exact: int, n_stat: int, n_speed: int) -> dict:
    """All three checks for one generator."""
    ours = GENERATORS[name](length=length)
    res: dict = {"generator": name, "length": length, "original_python": python}

    seed = 7
    o, t_orig = original_corpus(python, repo, name, length, seed, n_exact)
    e = ours.reproduce(seed, n_exact)
    same = np.array([np.array_equal(a, b) for a, b in zip(o, e, strict=True)])
    o64, e64 = o.astype(np.float64), e.astype(np.float64)
    rel = np.abs(o64 - e64).max(axis=1) / np.maximum(o64.std(axis=1), 1e-12)
    res["exact"] = {
        "n": n_exact,
        "bitwise_share": float(same.mean()),
        "max_rel_diff": float(rel.max()),
        "worst_rows": [int(i) for i in np.argsort(rel)[-3:][::-1] if rel[i] > 0],
    }

    o, _ = original_corpus(python, repo, name, length, 1000, n_stat)
    f = ours.generate(np.arange(10_000_000, 10_000_000 + n_stat))
    fin_o, fin_f = np.isfinite(o).all(axis=1), np.isfinite(f).all(axis=1)
    so = [series_stats(y) for y in o[fin_o]]
    sf = [series_stats(y) for y in f[fin_f]]
    res["distribution"] = {
        "n": n_stat,
        "nonfinite_share": {"original": float(1 - fin_o.mean()), "ours": float(1 - fin_f.mean())},
        "ks_pvalues": {k: float(ks_2samp([a[k] for a in so], [b[k] for b in sf]).pvalue) for k in so[0]},
    }

    n_orig = min(n_speed, n_exact)
    _, t_orig = original_corpus(python, repo, name, length, 3, n_orig)
    t0 = time.perf_counter()
    ours.generate(np.arange(20_000_000, 20_000_000 + n_speed))
    t_ours = time.perf_counter() - t0
    res["speed"] = {"original_series_per_sec": n_orig / t_orig, "ours_series_per_sec": n_speed / t_ours}
    if hasattr(ours, "stats"):
        res["stats"] = dict(ours.stats)
    return res


def main() -> None:
    """Analysis of arguments and launching checks on the list of generators."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-python", default=sys.executable)
    parser.add_argument("--names", nargs="*", default=list(GENERATORS))
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--n-exact", type=int, default=40)
    parser.add_argument("--n-stat", type=int, default=600)
    parser.add_argument("--n-speed", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        res = check_one(name, args.original, args.original_python, args.length, args.n_exact, args.n_stat, args.n_speed)
        (args.out / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        ex, ds, sp = res["exact"], res["distribution"], res["speed"]
        print(
            f"{name:13s} exact {ex['bitwise_share']:.2f} (max rel {ex['max_rel_diff']:.1e}) | "
            f"KS min p {min(ds['ks_pvalues'].values()):.3f} | "
            f"speed {sp['original_series_per_sec']:.0f} -> {sp['ours_series_per_sec']:.0f} rows/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
