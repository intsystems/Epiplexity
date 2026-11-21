"""Verify corpus integrity against the manifest and generator.

Usage: python verify_corpora.py /workspace/corpora [n_rows]
"""

import json
import sys
from pathlib import Path

import numpy as np

from episynth.corpus import Corpus
from episynth.generators import GENERATORS


def main() -> None:
    """Prints by directory: OK or the reason for the discrepancy."""
    root = Path(sys.argv[1])
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    rng = np.random.default_rng(0)
    for gen_dir in sorted(root.iterdir()):
        for part in ("train", "heldout"):
            d = gen_dir / part
            if not (d / "manifest.json").exists():
                print(f"{d}: missing manifest")
                continue
            m = json.loads((d / "manifest.json").read_text())
            expect = 128 + m["n"] * m["length"] * 4
            size = (d / "series.npy").stat().st_size
            if size != expect:
                print(f"{d}: size {size}, expected {expect}")
                continue
            corpus = Corpus(d)
            seeds = m["first_seed"] + np.sort(rng.integers(0, m["n"], n_rows))
            fresh = GENERATORS[m["generator"]](length=m["length"]).generate(seeds)
            same = np.array_equal(corpus.generate(seeds), fresh)
            print(f"{d}: {'OK' if same else 'ROWS DO NOT MATCH'} ({n_rows} sampled, {m['n']} total)", flush=True)


if __name__ == "__main__":
    main()
