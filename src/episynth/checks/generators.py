r"""Проверка генераторов против оригиналов Mix, Don't Pick: воспроизведение, распределение, скорость.

    python -m episynth.checks.generators --original <клон mix-dont-pick> --out <каталог> \
        [--original-python <интерпретатор с зависимостями оригинала>] [--names ets sde ...]

Оригинал считается отдельным интерпретатором (`_original_corpus.py`), потому что ему нужны
свои версии зависимостей: под numpy 2 генератор TimeSynth оригинала падает в пошаговых
циклах и молча подменяет ряды запасными. На каждый генератор пишется `<каталог>/<имя>.json`
с тремя блоками.

- `exact`: корпус `reproduce(seed, n)` против `generate_series(n)` оригинала с тем же сидом:
  доля рядов, совпавших до бита, и наибольшая относительная ошибка (к стандартному
  отклонению ряда) среди остальных.
- `distribution`: статистики ряда (стандартное отклонение, размах, средний модуль
  разности, автокорреляция разностей, доля энергии спектра выше четверти частоты
  Найквиста) по корпусу оригинала и по корпусу основного режима на других сидах, p-значения
  критерия Колмогорова–Смирнова; доля неконечных рядов у обоих. У генераторов, где основной
  режим совпадает с режимом воспроизведения, это проверка самого критерия, а не кода.
- `speed`: рядов в секунду у оригинала и у основного режима.
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
    """Корпус оригинала `(n, length)` float32 и время его генерации в секундах."""
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
    """Статистики одного ряда для сравнения распределений."""
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
    """Все три проверки для одного генератора."""
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
    """Разбор аргументов и запуск проверок по списку генераторов."""
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
