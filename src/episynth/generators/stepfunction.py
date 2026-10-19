"""Кусочно-постоянные ряды с резкими, линейными и сигмоидными переходами."""

import numpy as np

from .base import Generator

_TRANSITIONS = ["hard", "ramp", "sigmoid"]
_LEVEL_MODES = ["uniform", "random_walk", "clustered"]


class StepFunction(Generator):
    """Ряд из случайного числа сегментов постоянного уровня с переходами между ними.

    Число сегментов геометрическое, длины сегментов из распределения Дирихле, уровни
    равномерные, случайное блуждание или кластеры; переход резкий, линейный или сигмоидный
    (вероятности 0,5, 0,3, 0,2) шириной 1–5 % длины ряда; с вероятностью 0,7 добавляется
    белый шум. Ширина перехода задана долей длины, поэтому при другой длине меняется форма
    перехода в точках. Вырожденный ряд заменяется нулями, как в оригинале. Генерация
    дешёвая, основной режим совпадает с режимом воспроизведения.

    Args:
        length: Длина ряда.
    """

    name = "stepfunction"

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        out = np.empty((len(seeds), self.length))
        for i, seed in enumerate(seeds):
            out[i] = self._one(np.random.default_rng(int(seed)))
        return out

    def _segment_lengths(self, n_seg: int, rng: np.random.Generator) -> np.ndarray:
        proportions = rng.dirichlet(rng.uniform(0.5, 2.0) * np.ones(n_seg))
        lengths = np.maximum(np.floor(proportions * self.length).astype(int), 2)
        deficit = self.length - lengths.sum()
        if deficit > 0:
            for idx in rng.choice(n_seg, size=deficit, replace=True):
                lengths[idx] += 1
        elif deficit < 0:
            for _ in range(-deficit):
                longest = np.argmax(lengths)
                if lengths[longest] > 2:
                    lengths[longest] -= 1
        return lengths

    @staticmethod
    def _levels(n_seg: int, mode: str, rng: np.random.Generator) -> np.ndarray:
        if mode == "uniform":
            return rng.uniform(-5, 5, size=n_seg)
        if mode == "random_walk":
            start = rng.uniform(-3, 3)
            return (
                np.cumsum(rng.normal(0, rng.uniform(0.3, 2.0), size=n_seg))
                - rng.normal(0, rng.uniform(0.3, 2.0))
                + start
            )
        n_clusters = rng.integers(2, min(6, n_seg + 1))
        centers = rng.uniform(-5, 5, size=n_clusters)
        return centers[rng.choice(n_clusters, size=n_seg)] + rng.normal(0, 0.1, size=n_seg)

    def _transitions(self, series, cps, levels, ttype, rng) -> np.ndarray:
        if ttype == "hard":
            return series
        result = series.copy()
        for i, cp in enumerate(cps):
            if i + 1 >= len(levels):
                break
            window = max(3, int(rng.uniform(0.01, 0.05) * self.length))
            half_w = window // 2
            start, end = max(0, cp - half_w), min(self.length, cp + half_w)
            if end - start < 2:
                continue
            t_norm = np.linspace(0, 1, end - start)
            if ttype == "ramp":
                blend = t_norm
            else:
                blend = 1 / (1 + np.exp(-rng.uniform(5, 15) * (t_norm - 0.5)))
            result[start:end] = levels[i] + (levels[i + 1] - levels[i]) * blend
        return result

    def _one(self, rng: np.random.Generator) -> np.ndarray:
        n_seg = min(int(rng.geometric(p=0.15)) + 2, max(3, self.length // 4))
        level_mode = rng.choice(_LEVEL_MODES)
        ttype = rng.choice(_TRANSITIONS, p=[0.5, 0.3, 0.2])
        seg_lengths = self._segment_lengths(n_seg, rng)
        levels = self._levels(n_seg, level_mode, rng)
        series = np.empty(self.length)
        pos, cps = 0, []
        for i, sl in enumerate(seg_lengths):
            end = min(pos + sl, self.length)
            series[pos:end] = levels[i]
            if i > 0:
                cps.append(pos)
            pos = end
        series = self._transitions(series, cps, levels, ttype, rng)
        if rng.random() < 0.7:
            series += rng.normal(0, rng.uniform(0.01, 0.3), size=self.length)
        if not np.all(np.isfinite(series)):
            series = np.zeros(self.length)
        return series
