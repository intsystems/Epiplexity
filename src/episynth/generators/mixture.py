"""Равная смесь генераторов: сид выбирает компоненту и ряд в ней."""

from collections.abc import Sequence

import numpy as np

from .base import Generator


class Mixture(Generator):
    """Равная смесь генераторов одной длины ряда.

    Сид `s` даёт ряд компоненты `s % k` с сидом `s // k`, где `k` — число компонент: подряд идущие
    сиды обходят компоненты по кругу, и каждый ряд компоненты используется не более одного раза.
    Диапазон сидов смеси `[a, b)` требует у каждой компоненты сидов из `[a // k, b // k]`; для
    корпусов это значит, что отложенный набор смеси с первого сида `h` берёт у корпусов ряды с сида
    `h // k`. Режим воспроизведения передаётся компоненте.

    Args:
        components: Генераторы; длина ряда у всех одна.

    Attributes:
        name: `mixture(` и имена компонент через запятую.
        components: Компоненты в порядке задания.
    """

    def __init__(self, components: Sequence[Generator]) -> None:
        if not components:
            raise ValueError("смесь без компонент")
        if len({g.length for g in components}) != 1:
            raise ValueError("компоненты смеси с разной длиной ряда")
        super().__init__(components[0].length)
        self.components = list(components)
        self.name = "mixture(" + ",".join(g.name for g in self.components) + ")"

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        k = len(self.components)
        out = np.empty((len(seeds), self.length), dtype=np.float32)
        for j, g in enumerate(self.components):
            mask = seeds % k == j
            if mask.any():
                out[mask] = g.generate(seeds[mask] // k)
        return out

    def _exact(self, seed: int) -> np.ndarray:
        k = len(self.components)
        return self.components[seed % k].generate([seed // k], exact=True)[0]
