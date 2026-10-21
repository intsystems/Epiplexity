"""База gzip: сжимаемость рядов корпуса, байт gzip на точку, по отложенным рядам каждого генератора.

Запуск: python gzip_baseline.py /workspace/corpora [n_series] > gzip_baseline.json
Ряды берутся как есть (float32) и после нормировки по ряду с квантованием до int16 на сетке
из 2^15 уровней размаха, чтобы сжимаемость не зависела от масштаба и битового шума float32.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Печатает JSON {генератор: {raw_bytes_per_point, q16_bytes_per_point}}."""
    root = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    out = {}
    for d in sorted(root.iterdir()):
        path = d / "heldout" / "series.npy"
        if not path.exists():
            continue
        x = np.load(path, mmap_mode="r")[:n].astype(np.float32)
        raw = len(gzip.compress(x.tobytes(), compresslevel=6)) / x.size
        lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
        q = np.round((x - lo) / np.maximum(hi - lo, 1e-12) * 32767).astype(np.int16)
        q16 = len(gzip.compress(q.tobytes(), compresslevel=6)) / x.size
        out[d.name] = {"raw_bytes_per_point": raw, "q16_bytes_per_point": q16, "n_series": int(x.shape[0])}
        print(d.name, round(raw, 3), round(q16, 3), file=sys.stderr, flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
