"""Generate seeded corpora in parallel and use them through the generator
interface.

A corpus contains a float32 array of shape `(n, length)` in `series.npy` and
a `manifest.json`. Row `i` holds the default-mode series at seed
`first_seed + i`. A corpus-backed run therefore uses the same series and
windows as on-the-fly generation with the same seeds. Add a directory for
the next seed range to extend a corpus.

Workers generate chunks of seeds and write directly to a memory-mapped
array, so memory use does not scale with the corpus size. Each worker
defaults to one BLAS thread unless configured through `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, or `MKL_NUM_THREADS`. The manifest is written last
and marks a completed corpus. `Corpus` combines directories for one
generator with disjoint seed ranges, such as training and held-out ranges,
and serves series by seed. Corpora support the default generation mode.

    python -m episynth.corpus --generator kernelsynth --first-seed 0 --n
    2000000 --out corpora/kernelsynth/train
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from inspect import signature
from numbers import Integral
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .generators import GENERATORS
from .generators.base import Generator

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_worker: dict = {}


class CorpusManifest(BaseModel):
    """Metadata for a corpus directory.

    Attributes:
        generator: Name of the generator in the registry.
        length: Series length.
        first_seed: Seed of the first row.
        n: Number of series.
        generator_args: Constructor arguments other than length, with defaults
            expanded.
        nonfinite_rows: Number of series containing nonfinite values.
        workers: Number of generation processes.
        elapsed_s: Generation time in seconds.
        created: Manifest creation time in UTC.
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


def _canonical_generator_args(generator: str, length: int, args: dict) -> dict:
    """Arguments of the constructor with disclosed values by default, without
    row length.
    """
    bound = signature(GENERATORS[generator]).bind(length=length, **args)
    bound.apply_defaults()
    return {
        key: value for key, value in bound.arguments.items() if key != "length"
    }


def available_cpus() -> int:
    """Number of cores available to the process: Planner mask, limited by the
    cgroup quota, if any.
    """
    n = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def _init_worker(
    generator: str, length: int, generator_args: dict, series_path: str
) -> None:
    """Builds a generator and opens the memmap enclosure to record in the
    process-worker.
    """
    _worker["gen"] = GENERATORS[generator](length=length, **generator_args)
    _worker["series"] = np.load(series_path, mmap_mode="r+")


def _fill(task: tuple[int, int, int]) -> tuple[int, int]:
    """Generates the `[start, stop)` strings of the corpus; returns their
    number and number of infinite series.
    """
    start, stop, first_seed = task
    x = _worker["gen"].generate(
        np.arange(first_seed + start, first_seed + stop, dtype=np.int64)
    )
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
    """Records the generator body in the `out` catalog.

    Args:
        generator: Generator's name on the registry.
        first_seed: The first line seat; the `i` string receives the
            `first_seed + i` seat.
        n: Number of rows.
        out: Output directory, created on demand. A completed corpus is never
            overwritten.
        length: The length of the row.
        workers: Number of processes; `None` are all available kernels.
        chunk: Sids on one trial assignment.
        generator_args: Arguments of the generator designer other than length.
        verbose: Print the progress of generation.

    Returns:
        CorpusManifest: A recorded manifesto.

    Raises:
        FileExistsError: There's already a manifesto in the catalog.
    """
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Corpus already exists: {out}")
    if generator not in GENERATORS:
        raise KeyError(f"Unknown generator: {generator}")
    workers = available_cpus() if workers is None else workers
    for name, value in (
        ("n", n),
        ("length", length),
        ("chunk", chunk),
        ("workers", workers),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(first_seed, bool)
        or not isinstance(first_seed, Integral)
        or first_seed < 0
    ):
        raise ValueError("first_seed must be a nonnegative integer")
    n, length, chunk, workers, first_seed = map(
        int, (n, length, chunk, workers, first_seed)
    )
    if first_seed + n > 2**31:
        raise ValueError("corpus seeds must be in [0, 2**31)")
    generator_args = _canonical_generator_args(
        generator, length, dict(generator_args or {})
    )
    GENERATORS[generator](
        length=length, **generator_args
    )  # Designer error in the process-worker would hang the pool
    out.mkdir(parents=True, exist_ok=True)
    series_path = out / "series.npy"
    np.lib.format.open_memmap(
        series_path, mode="w+", dtype=np.float32, shape=(n, length)
    ).flush()
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    tasks = [(s, min(s + chunk, n), first_seed) for s in range(0, n, chunk)]
    t0 = time.time()
    done = nonfinite = 0
    next_report = 0.1
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers,
        initializer=_init_worker,
        initargs=(generator, length, generator_args, str(series_path)),
    ) as pool:
        for rows, bad in pool.imap_unordered(_fill, tasks):
            done += rows
            nonfinite += bad
            if verbose and done >= next_report * n:
                elapsed = time.time() - t0
                print(
                    f"{generator}: {done}/{n} series, {done / elapsed:.0f}"
                    f" series/s, ~{elapsed * (n - done) / done:.0f} s"
                    " remaining",
                    flush=True,
                )
                next_report += 0.1
    if done != n:
        raise RuntimeError(
            f"incomplete corpus generation: wrote {done} of {n} rows"
        )
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
    """Serve a generator's series from one or more corpus directories.

    Args:
        paths: Directories for the same generator with disjoint seed ranges.
    """

    def __init__(self, *paths: str | Path) -> None:
        if not paths:
            raise ValueError("At least one corpus directory is required")
        self.manifests = [
            CorpusManifest.model_validate_json(
                (Path(p) / "manifest.json").read_text()
            )
            for p in paths
        ]
        first = self.manifests[0]
        if any(
            (m.generator, m.length) != (first.generator, first.length)
            for m in self.manifests
        ):
            raise ValueError(
                "Corpus directories have different generators or series"
                " lengths"
            )
        args = _canonical_generator_args(
            first.generator, first.length, first.generator_args
        )
        if any(
            _canonical_generator_args(m.generator, m.length, m.generator_args)
            != args
            for m in self.manifests[1:]
        ):
            raise ValueError(
                "Corpus directories have different generator_args"
            )
        ranges = sorted(
            (m.first_seed, m.first_seed + m.n) for m in self.manifests
        )
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:], strict=False)):
            raise ValueError("Seed ranges of corpus directories overlap")
        super().__init__(first.length)
        self.name = first.generator
        self._paths = [str(p) for p in paths]
        self._parts: list[tuple[int, int, np.ndarray]] = []
        for p, m in zip(paths, self.manifests, strict=True):
            arr = np.load(Path(p) / "series.npy", mmap_mode="r")
            if arr.shape != (m.n, m.length) or arr.dtype != np.float32:
                raise ValueError(
                    f"Series file does not match its manifest: {p}"
                )
            self._parts.append((m.first_seed, m.n, arr))

    def __reduce__(self) -> tuple:
        """Directory path serialization: Memmap is reopened in the receiving
        process.
        """
        return (Corpus, tuple(self._paths))

    @property
    def seed_ranges(self) -> list[tuple[int, int]]:
        """Half-open seed range `[first, first + n)` for each directory."""
        return [(first, first + n) for first, n, _ in self._parts]

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        """Read float32 series by seed without conversion. Raise `KeyError` for
        missing seeds.
        """
        out = np.empty((len(seeds), self.length), dtype=np.float32)
        covered = np.zeros(len(seeds), dtype=bool)
        for first, n, arr in self._parts:
            mask = (seeds >= first) & (seeds < first + n)
            if mask.any():
                out[mask] = arr[seeds[mask] - first]
                covered |= mask
        if not covered.all():
            raise KeyError(
                f"Seeds outside the corpus: {seeds[~covered][:5].tolist()}"
            )
        return out

    def _exact(self, seed: int) -> np.ndarray:
        """Raise `NotImplementedError` because the corpus stores default-mode
        series.
        """
        raise NotImplementedError(
            "The corpus stores series from the default mode and does not"
            " support reproduction mode"
        )


def main() -> None:
    """Generating the body from the command line."""
    parser = argparse.ArgumentParser(
        description="Write a corpus of generated series to a directory."
    )
    parser.add_argument(
        "--generator", required=True, choices=sorted(GENERATORS)
    )
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument(
        "--generator-arg",
        action="append",
        default=[],
        metavar="name=value",
        help=(
            "Generator constructor argument. Values are parsed as JSON or kept"
            " as strings. For example, use device=cuda for KernelSynth on a"
            " GPU with one process."
        ),
    )
    args = parser.parse_args()
    generator_args = {}
    for item in args.generator_arg:
        key, _, value = item.partition("=")
        try:
            generator_args[key] = json.loads(value)
        except json.JSONDecodeError:
            generator_args[key] = value
    manifest = generate_corpus(
        args.generator,
        args.first_seed,
        args.n,
        args.out,
        length=args.length,
        workers=args.workers,
        chunk=args.chunk,
        generator_args=generator_args,
        verbose=True,
    )
    print(manifest.model_dump_json())


if __name__ == "__main__":
    main()
