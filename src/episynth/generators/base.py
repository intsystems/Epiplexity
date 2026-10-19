"""Контракт генератора одномерных синтетических рядов.

Каждый генератор пакета повторяет генеративный процесс одноимённого генератора из
Mix, Don't Pick (Nagpal и др., 2026, arXiv:2606.09912; код
github.com/birla-ai-labs/mix-dont-pick, Apache-2.0) и работает в двух режимах.

Основной режим считает ряды пакетом: циклы по времени векторизованы по рядам, сторонние
библиотеки оригинала не нужны, а там, где оригинал делит один сид между выбором параметров
и шумом, потоки случайных чисел разведены. Режим воспроизведения (`exact`) повторяет
оригинал вплоть до порядка обращений к генератору случайных чисел и даёт те же ряды при
тех же сидах; он медленнее и предназначен для проверок, а не для обучения.

Ряд зависит только от своего сида и длины: генерация любого подмножества сидов в любом
порядке и любыми пакетами даёт те же ряды.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np


def original_seeds(random_seed: int, n: int) -> np.ndarray:
    """Сиды рядов, которые оригинал выдаёт корпусу из `n` рядов при сиде корпуса `random_seed`.

    Оригинал разыгрывает их как `default_rng(random_seed).integers(0, 2**31, n)`; у TSI и
    TimeSynth своя схема, см. их `reproduce`.

    Args:
        random_seed: Сид корпуса.
        n: Число рядов.

    Returns:
        np.ndarray: Целочисленные сиды `(n,)`.
    """
    return np.random.default_rng(random_seed).integers(0, 2**31, size=n)


class Generator(ABC):
    """Генератор одномерных синтетических рядов фиксированной длины.

    Args:
        length: Длина ряда.

    Attributes:
        name: Имя генератора в реестре.
    """

    name: ClassVar[str]

    def __init__(self, length: int = 1024) -> None:
        self.length = length

    def generate(self, seeds: Sequence[int] | np.ndarray, exact: bool = False) -> np.ndarray:
        """Строит ряды `(n, length)` float32 по списку сидов, по ряду на сид.

        При `exact=True` ряд совпадает с рядом оригинала для того же сида: до бита там, где
        оригинал считает в numpy тем же порядком операций, иначе до ошибки округления.
        Численно вырожденные ряды (не конечные значения) заменяются так же, как в оригинале.

        Args:
            seeds: Сиды рядов, неотрицательные целые до `2**31`.
            exact: Воспроизводить оригинал вместо основного режима.

        Returns:
            np.ndarray: Ряды `(n, length)` float32.
        """
        seeds_arr = np.asarray(seeds, dtype=np.int64).reshape(-1)
        if exact:
            rows = [self._exact(int(s)) for s in seeds_arr]
            return np.stack(rows).astype(np.float32, copy=False)
        return self._batch(seeds_arr).astype(np.float32, copy=False)

    def reproduce(self, random_seed: int, n: int) -> np.ndarray:
        """Корпус `(n, length)` float32, равный корпусу оригинала при сиде корпуса `random_seed`.

        Повторяет `generate_series(n)` оригинала: схему сидов рядов и режим `exact`.

        Args:
            random_seed: Сид корпуса.
            n: Число рядов.

        Returns:
            np.ndarray: Ряды `(n, length)` float32.
        """
        return self.generate(original_seeds(random_seed, n), exact=True)

    @abstractmethod
    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        """Ряды `(n, length)` float64 основного режима для массива сидов."""

    def _exact(self, seed: int) -> np.ndarray:
        """Ряд `(length,)` float64 режима воспроизведения; по умолчанию основной режим и есть он."""
        return self._batch(np.array([seed], dtype=np.int64))[0]
