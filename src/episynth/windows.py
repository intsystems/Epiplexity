r"""Окна задачи прогноза из рядов генератора: контекст случайной длины, фиксированный горизонт, патчи.

Из ряда длины $L$ берётся одно окно: контекст длины $c$ из заданного распределения и горизонт
$h$ сразу за ним, начало окна равномерно среди допустимых. Окно деквантуется: ко всем его точкам
добавляется равномерный шум ширины `dequantize_width` в единицах стандартного отклонения контекста
(с тем же слагаемым под корнем, что у нормировки модели), чтобы лог-плотность была ограничена
снизу одинаково для всех генераторов. Точки раскладываются по патчам размера $P$ в
$(c_{\max} + h) / P$ токенов: контекст выровнен по правому краю, токены левее начала контекста
помечены как заполнение (`sample_id = 0`), частично заполненный первый токен контекста остаётся
в последовательности с маской наблюдений. Розыгрыш окна зависит только от сида ряда и отделён от
потока генератора: равномерные числа для длины контекста, начала окна и шума деквантования
берутся из хеша splitmix64 от сида, номера потока и позиции, поэтому сборка окон векторизована
по батчу и не зависит от его состава.
"""

import collections
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
    """Финализатор splitmix64 поэлементно над uint64 с переполнением по модулю $2^{64}$."""
    z = x + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def window_uniforms(seeds: np.ndarray, n: int, stream: int) -> np.ndarray:
    """Равномерные числа из $[0, 1)$ формы `(B, n)`, определяемые сидом, номером потока и позицией.

    Args:
        seeds: Сиды рядов `(B,)`.
        n: Чисел на сид.
        stream: Номер потока: разные величины окна берут разные потоки.

    Returns:
        np.ndarray: float64 `(B, n)`, 53 бита точности.
    """
    key = _splitmix64(np.asarray(seeds, dtype=np.uint64) ^ _WINDOW_SALT) + np.uint64((stream * int(_GOLDEN)) % 2**64)
    h = _splitmix64(key[:, None] + np.arange(n, dtype=np.uint64)[None, :])
    return (h >> np.uint64(11)).astype(np.float64) * 2.0**-53


class WindowSpec(BaseModel):
    """Форма задачи прогноза.

    Attributes:
        series_length: Длина ряда генератора.
        context_max: Наибольшая длина контекста; кратна размеру патча.
        horizon: Длина горизонта; кратна размеру патча.
        patch_size: Размер патча.
        context_lengths: Допустимые длины контекста, не больше `context_max`.
        context_probs: Вероятности длин контекста; `None` — равновероятно.
        dequantize_width: Ширина равномерного шума деквантования в единицах масштаба контекста; 0 — без шума.
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
        assert self.context_max % self.patch_size == 0 and self.horizon % self.patch_size == 0
        assert min(self.context_lengths) >= 1 and max(self.context_lengths) <= self.context_max
        assert self.series_length >= max(self.context_lengths) + self.horizon
        assert self.context_probs is None or len(self.context_probs) == len(self.context_lengths)
        return self

    @property
    def n_tokens(self) -> int:
        """Число токенов окна, включая заполнение слева."""
        return (self.context_max + self.horizon) // self.patch_size

    @property
    def n_context_tokens(self) -> int:
        """Число токенов контекста при наибольшей длине."""
        return self.context_max // self.patch_size


@dataclass
class WindowBatch:
    """Батч окон в раскладке модели.

    Attributes:
        target: Значения патчей `(B, S, P)` float32, нули на ненаблюдаемых позициях.
        observed_mask: Булева маска наблюдаемых значений `(B, S, P)`.
        sample_id: Номер ряда `(B, S)` int64: 1 у токенов окна, 0 у заполнения.
        time_id: Индекс токена `(B, S)` int64.
        variate_id: Нули `(B, S)` int64.
        prediction_mask: Булева маска токенов горизонта `(B, S)`.
        seeds: Сиды рядов `(B,)` int64.
        context_length: Длины контекста `(B,)` int64.
    """

    target: torch.Tensor
    observed_mask: torch.Tensor
    sample_id: torch.Tensor
    time_id: torch.Tensor
    variate_id: torch.Tensor
    prediction_mask: torch.Tensor
    seeds: torch.Tensor
    context_length: torch.Tensor

    def to(self, device: str | torch.device, non_blocking: bool = False) -> "WindowBatch":
        """Та же раскладка на другом устройстве."""
        return WindowBatch(
            **{k: v.to(device, non_blocking=non_blocking) for k, v in self.__dict__.items()},
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Аргументы `MoiraiModule.forward` кроме размера патча."""
        return {
            k: getattr(self, k)
            for k in ("target", "observed_mask", "sample_id", "time_id", "variate_id", "prediction_mask")
        }

    def __len__(self) -> int:
        """Число окон в батче."""
        return self.target.shape[0]

    def slice(self, start: int, stop: int) -> "WindowBatch":
        """Подбатч окон с номерами из `[start, stop)`."""
        return WindowBatch(**{k: v[start:stop] for k, v in self.__dict__.items()})


def make_windows(series: np.ndarray, seeds: Sequence[int] | np.ndarray, spec: WindowSpec) -> WindowBatch:
    """Батч окон из рядов `(B, L)` по сидам `(B,)`: по одному окну на ряд.

    Args:
        series: Ряды `(B, series_length)`.
        seeds: Сиды рядов; задают розыгрыш окна.
        spec: Форма задачи.

    Returns:
        WindowBatch: Окна на CPU.
    """
    seeds = np.array(seeds, dtype=np.int64).reshape(-1)
    B, P, S, h = len(seeds), spec.patch_size, spec.n_tokens, spec.horizon
    total = spec.context_max + h
    series = np.asarray(series, dtype=np.float64)
    choices = np.asarray(spec.context_lengths, dtype=np.int64)
    if spec.context_probs is None:
        idx = np.minimum((window_uniforms(seeds, 1, 0)[:, 0] * len(choices)).astype(np.int64), len(choices) - 1)
    else:
        cdf = np.cumsum(np.asarray(spec.context_probs, dtype=np.float64))
        idx = np.minimum(np.searchsorted(cdf / cdf[-1], window_uniforms(seeds, 1, 0)[:, 0], side="right"), len(choices) - 1)
    lengths = choices[idx]
    n_starts = spec.series_length - lengths - h + 1
    start = np.minimum((window_uniforms(seeds, 1, 1)[:, 0] * n_starts).astype(np.int64), n_starts - 1)
    pos = np.arange(total)[None, :]
    window = np.take_along_axis(series, np.minimum(start[:, None] + pos, spec.series_length - 1), axis=1)
    if spec.dequantize_width > 0:
        in_context = pos < lengths[:, None]
        mean = (window * in_context).sum(1) / lengths
        var = (((window - mean[:, None]) * in_context) ** 2).sum(1) / np.maximum(lengths - 1, 1)
        var[lengths < 2] = 0.0
        width = spec.dequantize_width * np.sqrt(var + 1e-5)
        window = window + (window_uniforms(seeds, total, 2) - 0.5) * width[:, None]
    offset = (spec.context_max - lengths)[:, None]
    src = pos - offset
    observed = src >= 0
    values = np.where(observed, np.take_along_axis(window, np.maximum(src, 0), axis=1), 0.0).astype(np.float32)
    assert np.array_equal(observed.sum(1), lengths + h)
    target = torch.from_numpy(values.reshape(B, S, P))
    observed_mask = torch.from_numpy(observed.reshape(B, S, P))
    sample_id = observed_mask.any(-1).to(torch.int64)
    time_id = torch.arange(S, dtype=torch.int64).expand(B, S).clone()
    prediction_mask = (torch.arange(S) >= spec.n_context_tokens).expand(B, S).clone()
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


def window_batch(generator: Generator, seeds: Sequence[int] | np.ndarray, spec: WindowSpec) -> WindowBatch:
    """Батч окон прямо из генератора по сидам."""
    return make_windows(generator.generate(seeds), seeds, spec)


_worker_state: dict = {}


def _init_stream_worker(generator: Generator, spec: WindowSpec) -> None:
    """Запоминает генератор и форму задачи в процессе-работнике потока."""
    _worker_state["generator"] = generator
    _worker_state["spec"] = spec


def _stream_task(seeds: np.ndarray) -> WindowBatch:
    """Батч окон по сидам в процессе-работнике."""
    return window_batch(_worker_state["generator"], seeds, _worker_state["spec"])


class WindowStream:
    """Поток батчей окон по возрастающим сидам с генерацией в фоновом потоке или в процессах.

    Батч с номером $k$ строится из сидов `first_seed + k * batch_size + [0, batch_size)`, поэтому
    поток детерминирован, не зависит от числа работников и возобновляется с любого батча через
    `first_batch`. При `workers > 0` батчи считаются пулом процессов (запуск `spawn`, по одному
    потоку BLAS в работнике, генератор передаётся копированием); число незавершённых заданий
    ограничено `workers + prefetch`, порядок батчей сохраняется.

    Args:
        generator: Генератор рядов; при `workers > 0` должен сериализоваться `pickle`.
        spec: Форма задачи.
        batch_size: Окон в батче.
        first_seed: Сид первого окна первого батча.
        n_batches: Сколько батчей выдать; `None` — без конца.
        first_batch: Номер батча, с которого начать.
        prefetch: Сколько батчей готовить впрок.
        workers: Число процессов; 0 — один фоновый поток.
        seed_limit: Верхняя граница сидов (не включая); последний батч может быть короче.
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
        """Сиды окон батча с данным номером."""
        start = self.first_seed + batch_index * self.batch_size
        stop = start + self.batch_size if self.seed_limit is None else min(start + self.batch_size, self.seed_limit)
        return np.arange(start, stop, dtype=np.int64)

    def __iter__(self) -> Iterator[WindowBatch]:
        """Батчи окон по порядку номеров, начиная с `first_batch`."""
        return self._iter_pool() if self.workers > 0 else self._iter_thread()

    def _iter_pool(self) -> Iterator[WindowBatch]:
        """Батчи из пула процессов с ограниченным числом заданий в полёте."""
        saved = {var: os.environ.get(var) for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        for var in saved:
            os.environ[var] = "1"
        try:
            pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_stream_worker, initargs=(self.generator, self.spec))
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
                while len(pending) < self.workers + self.prefetch and (self.n_batches is None or k < self.n_batches):
                    pending.append(pool.apply_async(_stream_task, (self.seeds_of(k),)))
                    k += 1
                if not pending:
                    return
                yield pending.popleft().get()
        finally:
            pool.terminate()
            pool.join()

    def _iter_thread(self) -> Iterator[WindowBatch]:
        """Батчи из одного фонового потока."""
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def worker() -> None:
            """Кладёт батчи в очередь по порядку, в конце `None`, при ошибке само исключение."""
            k = self.first_batch
            try:
                while not stop.is_set() and (self.n_batches is None or k < self.n_batches):
                    q.put(window_batch(self.generator, self.seeds_of(k), self.spec))
                    k += 1
            except BaseException as exc:  # noqa: BLE001
                q.put(exc)
                return
            q.put(None)

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


def collect_windows(generator: Generator, spec: WindowSpec, seeds: np.ndarray, workers: int = 0, batch_size: int = 4096) -> WindowBatch:
    """Один батч окон по подряд идущим сидам, собранный потоком с `workers` процессами.

    Args:
        generator: Генератор рядов.
        spec: Форма задачи.
        seeds: Подряд идущие сиды; первый задаёт начало, число — размер.
        workers: Число процессов; 0 — фоновый поток.
        batch_size: Окон в батче потока.

    Returns:
        WindowBatch: Окна на CPU в порядке сидов.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    n = len(seeds)
    assert n > 0 and np.array_equal(seeds, np.arange(seeds[0], seeds[0] + n))
    n_batches = -(-n // batch_size)
    parts = list(WindowStream(generator, spec, batch_size, first_seed=int(seeds[0]), n_batches=n_batches, workers=workers, seed_limit=int(seeds[0]) + n))
    merged = WindowBatch(**{k: torch.cat([getattr(p, k) for p in parts]) for k in parts[0].__dict__})
    assert len(merged) == n
    return merged
