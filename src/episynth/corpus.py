"""Корпуса рядов: генерация по сидам в файл процессами и чтение корпуса как генератора.

Корпус есть массив float32 формы `(n, length)` в файле `series.npy` и манифест `manifest.json`
рядом: строка `i` хранит ряд с сидом `first_seed + i`, построенный основным режимом генератора.
Ряд зависит только от сида, поэтому прогон на корпусе даёт те же ряды и окна, что прогон с
генерацией на лету по тем же сидам, а корпус доращивается новым каталогом со следующим диапазоном
сидов. Генерация идёт кусками сидов в отдельных процессах с одним потоком BLAS в каждом
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` выставляются в 1, если не заданы) и
пишет прямо в memmap, так что память не растёт с размером корпуса. Манифест пишется последним и
служит признаком целости: каталог без него недописан. `Corpus` объединяет каталоги одного
генератора с непересекающимися диапазонами сидов, например обучающий и отложенный, и отдаёт ряды
по сидам как генератор; режима воспроизведения у корпуса нет.

Запуск из командной строки:

    python -m episynth.corpus --generator kernelsynth --first-seed 0 --n 2000000 --out corpora/kernelsynth/train
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .generators import GENERATORS
from .generators.base import Generator

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_worker: dict = {}


class CorpusManifest(BaseModel):
    """Описание каталога корпуса.

    Attributes:
        generator: Имя генератора в реестре.
        length: Длина ряда.
        first_seed: Сид первой строки.
        n: Число рядов.
        generator_args: Аргументы конструктора генератора кроме длины.
        nonfinite_rows: Число рядов с неконечными значениями.
        workers: Число процессов генерации.
        elapsed_s: Время генерации в секундах.
        created: Момент записи манифеста в UTC.
    """

    generator: str
    length: int
    first_seed: int
    n: int
    generator_args: dict = {}
    nonfinite_rows: int = 0
    workers: int = 1
    elapsed_s: float = 0.0
    created: str = ""


def available_cpus() -> int:
    """Число ядер, доступных процессу: маска планировщика, ограниченная квотой cgroup, если она есть."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def _init_worker(generator: str, length: int, generator_args: dict, series_path: str) -> None:
    """Строит генератор и открывает memmap корпуса на запись в процессе-работнике."""
    _worker["gen"] = GENERATORS[generator](length=length, **generator_args)
    _worker["series"] = np.load(series_path, mmap_mode="r+")


def _fill(task: tuple[int, int, int]) -> tuple[int, int]:
    """Генерирует строки `[start, stop)` корпуса; возвращает их число и число неконечных рядов."""
    start, stop, first_seed = task
    x = _worker["gen"].generate(np.arange(first_seed + start, first_seed + stop, dtype=np.int64))
    _worker["series"][start:stop] = x
    _worker["series"].flush()
    return stop - start, int((~np.isfinite(x)).any(axis=1).sum())


def generate_corpus(
    generator: str,
    first_seed: int,
    n: int,
    out: str | Path,
    length: int = 1024,
    workers: int | None = None,
    chunk: int = 1024,
    generator_args: dict | None = None,
    verbose: bool = False,
) -> CorpusManifest:
    """Записывает корпус генератора в каталог `out`.

    Args:
        generator: Имя генератора в реестре.
        first_seed: Сид первой строки; строка `i` получает сид `first_seed + i`.
        n: Число рядов.
        out: Каталог корпуса; создаётся, если его нет. Каталог с манифестом не перезаписывается.
        length: Длина ряда.
        workers: Число процессов; `None` — все доступные ядра.
        chunk: Сидов на одно задание процессу.
        generator_args: Аргументы конструктора генератора кроме длины.
        verbose: Печатать ход генерации.

    Returns:
        CorpusManifest: Записанный манифест.

    Raises:
        FileExistsError: В каталоге уже есть манифест.
    """
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"корпус уже записан: {out}")
    if generator not in GENERATORS:
        raise KeyError(f"неизвестный генератор: {generator}")
    generator_args = dict(generator_args or {})
    GENERATORS[generator](length=length, **generator_args)  # ошибка конструктора в процессе-работнике повесила бы пул
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or available_cpus()
    series_path = out / "series.npy"
    np.lib.format.open_memmap(series_path, mode="w+", dtype=np.float32, shape=(n, length)).flush()
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    tasks = [(s, min(s + chunk, n), first_seed) for s in range(0, n, chunk)]
    t0 = time.time()
    done = nonfinite = 0
    next_report = 0.1
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(generator, length, generator_args, str(series_path))) as pool:
        for rows, bad in pool.imap_unordered(_fill, tasks):
            done += rows
            nonfinite += bad
            if verbose and done >= next_report * n:
                elapsed = time.time() - t0
                print(f"{generator}: {done}/{n} рядов, {done / elapsed:.0f} рядов/с, осталось ~{elapsed * (n - done) / done:.0f} с", flush=True)
                next_report += 0.1
    manifest = CorpusManifest(
        generator=generator,
        length=length,
        first_seed=first_seed,
        n=n,
        generator_args=generator_args,
        nonfinite_rows=nonfinite,
        workers=workers,
        elapsed_s=time.time() - t0,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return manifest


class Corpus(Generator):
    """Корпус как генератор: ряды по сидам из одного или нескольких каталогов.

    Args:
        paths: Каталоги корпусов одного генератора с непересекающимися диапазонами сидов.

    Attributes:
        name: Имя генератора корпуса.
        manifests: Манифесты каталогов в порядке `paths`.
    """

    def __init__(self, *paths: str | Path) -> None:
        if not paths:
            raise ValueError("нужен хотя бы один каталог корпуса")
        self.manifests = [CorpusManifest.model_validate_json((Path(p) / "manifest.json").read_text()) for p in paths]
        first = self.manifests[0]
        if any((m.generator, m.length) != (first.generator, first.length) for m in self.manifests):
            raise ValueError("каталоги корпуса от разных генераторов или с разной длиной ряда")
        ranges = sorted((m.first_seed, m.first_seed + m.n) for m in self.manifests)
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:], strict=False)):
            raise ValueError("диапазоны сидов каталогов пересекаются")
        super().__init__(first.length)
        self.name = first.generator
        self._paths = [str(p) for p in paths]
        self._parts: list[tuple[int, int, np.ndarray]] = []
        for p, m in zip(paths, self.manifests, strict=True):
            arr = np.load(Path(p) / "series.npy", mmap_mode="r")
            if arr.shape != (m.n, m.length) or arr.dtype != np.float32:
                raise ValueError(f"файл рядов не соответствует манифесту: {p}")
            self._parts.append((m.first_seed, m.n, arr))

    def __reduce__(self) -> tuple:
        """Сериализация путями каталогов: memmap открывается заново в принимающем процессе."""
        return (Corpus, tuple(self._paths))

    @property
    def seed_ranges(self) -> list[tuple[int, int]]:
        """Полуинтервалы сидов `[first, first + n)` по каталогам."""
        return [(first, first + n) for first, n, _ in self._parts]

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        """Строки корпуса по сидам, float32 без преобразования; сид вне диапазонов — `KeyError`."""
        out = np.empty((len(seeds), self.length), dtype=np.float32)
        covered = np.zeros(len(seeds), dtype=bool)
        for first, n, arr in self._parts:
            mask = (seeds >= first) & (seeds < first + n)
            if mask.any():
                out[mask] = arr[seeds[mask] - first]
                covered |= mask
        if not covered.all():
            raise KeyError(f"сиды вне корпуса: {seeds[~covered][:5].tolist()}")
        return out

    def _exact(self, seed: int) -> np.ndarray:
        """Режима воспроизведения у корпуса нет: всегда `NotImplementedError`."""
        raise NotImplementedError("корпус хранит ряды основного режима; режима воспроизведения у него нет")


def main() -> None:
    """Генерация корпуса из командной строки."""
    parser = argparse.ArgumentParser(description="Записать корпус рядов генератора в каталог.")
    parser.add_argument("--generator", required=True, choices=sorted(GENERATORS))
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--generator-arg", action="append", default=[], metavar="имя=значение", help="аргумент конструктора генератора, значение как JSON, иначе строка; например device=cuda для KernelSynth на GPU при одном процессе")
    args = parser.parse_args()
    generator_args = {}
    for item in args.generator_arg:
        key, _, value = item.partition("=")
        try:
            generator_args[key] = json.loads(value)
        except json.JSONDecodeError:
            generator_args[key] = value
    manifest = generate_corpus(
        args.generator, args.first_seed, args.n, args.out, length=args.length, workers=args.workers, chunk=args.chunk, generator_args=generator_args, verbose=True
    )
    print(manifest.model_dump_json())


if __name__ == "__main__":
    main()
