r"""Build forecasting windows with sampled context lengths, a fixed horizon,
and patches.

Each length-:math:`L` series supplies one window. Draw the context length
:math:`c` from the configured distribution and pick the start uniformly
among positions that fit the context and following horizon :math:`h`.
Add uniform dequantization noise of width `dequantize_width` in units of
the context standard deviation, using the same variance offset as the
model scaler. This gives all generators the same resolution for scoring.

Place points in patches of size :math:`P`, giving
:math:`(c_{\max} + h)/P` tokens. Right-align the context. Tokens before it
are padding (`sample_id = 0`). Keep a partially filled first context
token and mark its observed positions with a mask.

The window draw depends only on the series seed and is independent of the
generator's random stream. A splitmix64 hash of the seed, stream index,
and position supplies the uniform values for context length, start, and
noise, so batching does not change any window.
"""

import collections
import math
import multiprocessing as mp
import os
import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel, model_validator

from .generators.base import Generator

_WINDOW_SALT = np.uint64(0x57494E44)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finalizer is elementally over the uint64 with a
    :math:`2^{64}` overflow.
    """
    z = x + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def window_uniforms(seeds: np.ndarray, n: int, stream: int) -> np.ndarray:
    """Uniform numbers from :math:`[0, 1)` form `(B, n)`, defined by the seat,
    flow number and position.

    Args:
        seeds: Series seeds of shape `(B,)`.
        n: Numbers on the seed.
        stream: Flow Number: Different window sizes take on different streams.

    Returns:
        np.ndarray: float64 `(B, n)`, 53 bit accuracy.
    """
    key = _splitmix64(
        np.asarray(seeds, dtype=np.uint64) ^ _WINDOW_SALT
    ) + np.uint64((stream * int(_GOLDEN)) % 2**64)
    h = _splitmix64(key[:, None] + np.arange(n, dtype=np.uint64)[None, :])
    return (h >> np.uint64(11)).astype(np.float64) * 2.0**-53


class WindowSpec(BaseModel):
    """Shape of a forecasting task.

    Attributes:
        series_length: Length of each generated series.
        context_max: Maximum context length, divisible by the patch size.
        horizon: Forecast horizon, divisible by the patch size.
        patch_size: Number of points per patch.
        context_lengths: Allowed context lengths, each at most `context_max`.
        context_probs: Context-length probabilities. `None` uses equal
            probabilities.
        dequantize_width: Uniform noise width in units of context scale. Zero
            disables noise.
    """

    series_length: int = 1024
    context_max: int = 512
    horizon: int = 64
    patch_size: int = 32
    context_lengths: tuple[int, ...] = (512,)
    context_probs: tuple[float, ...] | None = None
    dequantize_width: float = 1e-3

    @model_validator(mode="after")
    def _check(self) -> "WindowSpec":
        if (
            min(
                self.series_length,
                self.context_max,
                self.horizon,
                self.patch_size,
            )
            <= 0
        ):
            raise ValueError(
                "series_length, context_max, horizon and patch_size must be"
                " positive"
            )
        if (
            self.context_max % self.patch_size != 0
            or self.horizon % self.patch_size != 0
        ):
            raise ValueError(
                "context_max and horizon must be divisible by patch_size"
            )
        if (
            not self.context_lengths
            or min(self.context_lengths) < 1
            or max(self.context_lengths) > self.context_max
        ):
            raise ValueError(
                "context_lengths must be nonempty and lie in [1, context_max]"
            )
        if self.series_length < max(self.context_lengths) + self.horizon:
            raise ValueError(
                "series_length must fit every context and the horizon"
            )
        if self.context_probs is not None:
            if len(self.context_probs) != len(self.context_lengths):
                raise ValueError(
                    "context_probs must have one entry per context length"
                )
            if any(not math.isfinite(p) or p < 0 for p in self.context_probs):
                raise ValueError(
                    "context_probs must be finite and nonnegative"
                )
            total = sum(self.context_probs)
            if not math.isfinite(total) or total <= 0:
                raise ValueError(
                    "context_probs must have a positive finite sum"
                )
        if (
            not math.isfinite(self.dequantize_width)
            or self.dequantize_width < 0
        ):
            raise ValueError("dequantize_width must be finite and nonnegative")
        return self

    @property
    def n_tokens(self) -> int:
        """The number of window tokens, including filling on the left."""
        return (self.context_max + self.horizon) // self.patch_size

    @property
    def n_context_tokens(self) -> int:
        """The number of context tokens at the longest length."""
        return self.context_max // self.patch_size


@dataclass
class WindowBatch:
    """Forecasting windows in the model's input layout.

    Attributes:
        target: Float32 patch values of shape `(B, S, P)`, zero at unobserved
            positions.
        observed_mask: Boolean mask of observed values, shape `(B, S, P)`.
        sample_id: Int64 series IDs of shape `(B, S)`, one for window tokens
            and zero for padding.
        time_id: Int64 token positions of shape `(B, S)`.
        variate_id: Int64 zeros of shape `(B, S)` for univariate series.
        prediction_mask: Boolean mask of horizon tokens, shape `(B, S)`.
        seeds: Int64 series seeds of shape `(B,)`.
        context_length: Int64 context lengths of shape `(B,)`.
    """

    target: torch.Tensor
    observed_mask: torch.Tensor
    sample_id: torch.Tensor
    time_id: torch.Tensor
    variate_id: torch.Tensor
    prediction_mask: torch.Tensor
    seeds: torch.Tensor
    context_length: torch.Tensor

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> "WindowBatch":
        """Same layout on another device."""
        return WindowBatch(
            **{
                k: v.to(device, non_blocking=non_blocking)
                for k, v in self.__dict__.items()
            },
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """`MoiraiModule.forward` arguments other than patch size."""
        return {
            k: getattr(self, k)
            for k in (
                "target",
                "observed_mask",
                "sample_id",
                "time_id",
                "variate_id",
                "prediction_mask",
            )
        }

    def __len__(self) -> int:
        """Number of windows in the batch."""
        return self.target.shape[0]

    def slice(self, start: int, stop: int) -> "WindowBatch":
        """Return a window sub-batch at indices `[start, stop)`."""
        return WindowBatch(
            **{k: v[start:stop] for k, v in self.__dict__.items()}
        )


def make_windows(
    series: np.ndarray, seeds: Sequence[int] | np.ndarray, spec: WindowSpec
) -> WindowBatch:
    """Build one window per series from `(B, L)` values and `(B,)` seeds.

    Args:
        series: `(B, series_length)` rows.
        seeds: Series seeds that determine the window draw.
        spec: The shape of the task.

    Returns:
        WindowBatch: Windows on the CPU.
    """
    seeds = np.array(seeds, dtype=np.int64).reshape(-1)
    B, P, S, h = len(seeds), spec.patch_size, spec.n_tokens, spec.horizon
    total = spec.context_max + h
    series = np.asarray(series, dtype=np.float64)
    choices = np.asarray(spec.context_lengths, dtype=np.int64)
    if spec.context_probs is None:
        idx = np.minimum(
            (window_uniforms(seeds, 1, 0)[:, 0] * len(choices)).astype(
                np.int64
            ),
            len(choices) - 1,
        )
    else:
        cdf = np.cumsum(np.asarray(spec.context_probs, dtype=np.float64))
        idx = np.minimum(
            np.searchsorted(
                cdf / cdf[-1], window_uniforms(seeds, 1, 0)[:, 0], side="right"
            ),
            len(choices) - 1,
        )
    lengths = choices[idx]
    n_starts = spec.series_length - lengths - h + 1
    start = np.minimum(
        (window_uniforms(seeds, 1, 1)[:, 0] * n_starts).astype(np.int64),
        n_starts - 1,
    )
    pos = np.arange(total)[None, :]
    window = np.take_along_axis(
        series,
        np.minimum(start[:, None] + pos, spec.series_length - 1),
        axis=1,
    )
    if spec.dequantize_width > 0:
        in_context = pos < lengths[:, None]
        mean = (window * in_context).sum(1) / lengths
        var = (((window - mean[:, None]) * in_context) ** 2).sum(
            1
        ) / np.maximum(lengths - 1, 1)
        var[lengths < 2] = 0.0
        width = spec.dequantize_width * np.sqrt(var + 1e-5)
        window = (
            window + (window_uniforms(seeds, total, 2) - 0.5) * width[:, None]
        )
    offset = (spec.context_max - lengths)[:, None]
    src = pos - offset
    observed = src >= 0
    values = np.where(
        observed, np.take_along_axis(window, np.maximum(src, 0), axis=1), 0.0
    ).astype(np.float32)
    assert np.array_equal(observed.sum(1), lengths + h)
    target = torch.from_numpy(values.reshape(B, S, P))
    observed_mask = torch.from_numpy(observed.reshape(B, S, P))
    sample_id = observed_mask.any(-1).to(torch.int64)
    time_id = torch.arange(S, dtype=torch.int64).expand(B, S).clone()
    prediction_mask = (
        (torch.arange(S) >= spec.n_context_tokens).expand(B, S).clone()
    )
    return WindowBatch(
        target=target,
        observed_mask=observed_mask,
        sample_id=sample_id,
        time_id=time_id,
        variate_id=torch.zeros(B, S, dtype=torch.int64),
        prediction_mask=prediction_mask,
        seeds=torch.from_numpy(seeds),
        context_length=torch.from_numpy(lengths),
    )


def window_batch(
    generator: Generator, seeds: Sequence[int] | np.ndarray, spec: WindowSpec
) -> WindowBatch:
    """Generate a batch of series by seed and build their windows."""
    return make_windows(generator.generate(seeds), seeds, spec)


_worker_state: dict = {}


def _init_stream_worker(generator: Generator, spec: WindowSpec) -> None:
    """Remembers the generator and the form of the task in the process-worker
    flow.
    """
    _worker_state["generator"] = generator
    _worker_state["spec"] = spec


def _stream_task(seeds: np.ndarray) -> WindowBatch:
    """Generate a batch of windows in a worker process."""
    return window_batch(
        _worker_state["generator"], seeds, _worker_state["spec"]
    )


class WindowStream:
    """The flow of batch windows over increasing seed with generation in the
    background stream or in processes.

    Batch number :math:`k` is built from `first_seed + k * batch_size + [0,
    batch_size)`, so the
    flow is determined, does not depend on the number of employees and resumes
    from any batch
    through `first_batch`. With `workers > 0`, batches are considered a pool of
    processes (start
    `spawn`, one thread BLAS in the employee, the generator is transmitted by
    copy); the number of
    unfinished tasks is limited to `workers + prefetch`, the batch order is
    preserved.

    Args:
        generator: Series generator; `workers > 0` must serialize `pickle`.
        spec: The shape of the task.
        batch_size: The windows are in the batch.
        first_seed: Seed of the first window in the first batch.
        n_batches: How many batches to give out; `None` - without end.
        first_batch: The batch number to start with.
        prefetch: How many batches to prepare for the future.
        workers: Number of processes; 0 is one background stream.
        seed_limit: The upper limit of the seed (not including); the last batch
            may be shorter.
    """

    def __init__(
        self,
        generator: Generator,
        spec: WindowSpec,
        batch_size: int,
        first_seed: int = 0,
        n_batches: int | None = None,
        first_batch: int = 0,
        prefetch: int = 2,
        workers: int = 0,
        seed_limit: int | None = None,
    ) -> None:
        self.generator = generator
        self.spec = spec
        self.batch_size = batch_size
        self.first_seed = first_seed
        self.n_batches = n_batches
        self.first_batch = first_batch
        self.prefetch = prefetch
        self.workers = workers
        self.seed_limit = seed_limit

    def seeds_of(self, batch_index: int) -> np.ndarray:
        """Return the seeds for the given batch index."""
        start = self.first_seed + batch_index * self.batch_size
        stop = (
            start + self.batch_size
            if self.seed_limit is None
            else min(start + self.batch_size, self.seed_limit)
        )
        return np.arange(start, stop, dtype=np.int64)

    def __iter__(self) -> Iterator[WindowBatch]:
        """Yield batches in index order starting at `first_batch`."""
        return self._iter_pool() if self.workers > 0 else self._iter_thread()

    def _iter_pool(self) -> Iterator[WindowBatch]:
        """Batches from a pool of processes with a limited number of missions
        in flight.
        """
        saved = {
            var: os.environ.get(var)
            for var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        }
        for var in saved:
            os.environ[var] = "1"
        try:
            pool = mp.get_context("spawn").Pool(
                self.workers,
                initializer=_init_stream_worker,
                initargs=(self.generator, self.spec),
            )
        finally:
            for var, value in saved.items():
                if value is None:
                    del os.environ[var]
                else:
                    os.environ[var] = value
        pending: collections.deque = collections.deque()
        k = self.first_batch
        try:
            while True:
                while len(pending) < self.workers + self.prefetch and (
                    self.n_batches is None or k < self.n_batches
                ):
                    pending.append(
                        pool.apply_async(_stream_task, (self.seeds_of(k),))
                    )
                    k += 1
                if not pending:
                    return
                yield pending.popleft().get()
        finally:
            pool.terminate()
            pool.join()

    def _iter_thread(self) -> Iterator[WindowBatch]:
        """Batches from a single background stream; closing waits up to a
        second without interrupting current generation.
        """
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def worker() -> None:
            """Puts batchi in the queue in order, at the end of `None`, if the
            error is the exception itself.
            """
            k = self.first_batch
            try:
                while not stop.is_set() and (
                    self.n_batches is None or k < self.n_batches
                ):
                    if not put(
                        window_batch(
                            self.generator, self.seeds_of(k), self.spec
                        )
                    ):
                        return
                    k += 1
            except BaseException as exc:  # noqa: BLE001
                put(exc)
                return
            put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            thread.join(timeout=1.0)


def collect_windows(
    generator: Generator,
    spec: WindowSpec,
    seeds: np.ndarray,
    workers: int = 0,
    batch_size: int = 4096,
) -> WindowBatch:
    """Collect consecutive seeded windows into one batch using `workers`
    processes.

    Args:
        generator: Row generator.
        spec: The shape of the task.
        seeds: Seeds are ordered; the first is the beginning, the number is the
            size.
        workers: Number of processes; 0 is a background stream.
        batch_size: The windows are in the flow batch.

    Returns:
        WindowBatch: The windows on the CPU are in order of seed.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    n = len(seeds)
    assert n > 0 and np.array_equal(seeds, np.arange(seeds[0], seeds[0] + n))
    n_batches = -(-n // batch_size)
    parts = list(
        WindowStream(
            generator,
            spec,
            batch_size,
            first_seed=int(seeds[0]),
            n_batches=n_batches,
            workers=workers,
            seed_limit=int(seeds[0]) + n,
        )
    )
    merged = WindowBatch(
        **{
            k: torch.cat([getattr(p, k) for p in parts])
            for k in parts[0].__dict__
        }
    )
    assert len(merged) == n
    return merged
