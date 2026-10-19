r"""Окна задачи прогноза из рядов генератора: контекст случайной длины, фиксированный горизонт, патчи.

Из ряда длины $L$ берётся одно окно: контекст длины $c$ из заданного распределения и горизонт
$h$ сразу за ним, начало окна равномерно среди допустимых. Окно деквантуется: ко всем его точкам
добавляется равномерный шум ширины `dequantize_width` в единицах стандартного отклонения контекста
(с тем же слагаемым под корнем, что у нормировки модели), чтобы лог-плотность была ограничена
снизу одинаково для всех генераторов. Точки раскладываются по патчам размера $P$ в
$(c_{\max} + h) / P$ токенов: контекст выровнен по правому краю, токены левее начала контекста
помечены как заполнение (`sample_id = 0`), частично заполненный первый токен контекста остаётся
в последовательности с маской наблюдений. Розыгрыш окна зависит только от сида ряда и отделён от
потока генератора.
"""

import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel, model_validator

from .generators.base import Generator

_WINDOW_SALT = 0x57494E44


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
    B, P, S = len(seeds), spec.patch_size, spec.n_tokens
    total = spec.context_max + spec.horizon
    values = np.zeros((B, total), dtype=np.float32)
    observed = np.zeros((B, total), dtype=bool)
    lengths = np.empty(B, dtype=np.int64)
    probs = None if spec.context_probs is None else np.asarray(spec.context_probs, dtype=float)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng([int(seed), _WINDOW_SALT])
        c = int(rng.choice(spec.context_lengths, p=probs))
        start = int(rng.integers(0, spec.series_length - c - spec.horizon + 1))
        window = series[i, start : start + c + spec.horizon].astype(np.float64)
        if spec.dequantize_width > 0:
            var = window[:c].var(ddof=1) if c > 1 else 0.0
            width = spec.dequantize_width * np.sqrt(var + 1e-5)
            window = window + rng.uniform(-width / 2, width / 2, size=len(window))
        offset = spec.context_max - c
        values[i, offset:] = window
        observed[i, offset:] = True
        lengths[i] = c
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


class WindowStream:
    """Поток батчей окон по возрастающим сидам с генерацией в фоновом потоке.

    Батч с номером $k$ строится из сидов `first_seed + k * batch_size + [0, batch_size)`, поэтому
    поток детерминирован и возобновляется с любого батча через `first_batch`.

    Args:
        generator: Генератор рядов.
        spec: Форма задачи.
        batch_size: Окон в батче.
        first_seed: Сид первого окна первого батча.
        n_batches: Сколько батчей выдать; `None` — без конца.
        first_batch: Номер батча, с которого начать.
        prefetch: Сколько батчей готовить впрок.
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
    ) -> None:
        self.generator = generator
        self.spec = spec
        self.batch_size = batch_size
        self.first_seed = first_seed
        self.n_batches = n_batches
        self.first_batch = first_batch
        self.prefetch = prefetch

    def seeds_of(self, batch_index: int) -> np.ndarray:
        """Сиды окон батча с данным номером."""
        start = self.first_seed + batch_index * self.batch_size
        return np.arange(start, start + self.batch_size, dtype=np.int64)

    def __iter__(self) -> Iterator[WindowBatch]:
        """Батчи окна по порядку номеров, начиная с `first_batch`."""
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
