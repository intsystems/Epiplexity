"""Разложение тренд–сезонность–нерегулярность в восьми схемах сочетания."""

import numpy as np
from scipy.signal import lfilter

from .base import Generator

_TRENDS = ["linear", "nonlinear", "none"]
_SEASONS = ["sinusoidal", "step", "triangular", "impulsive", "none"]
_IRREGS = ["fgn", "fbm", "white"]
_MODELS = list(range(1, 9))


class TSI(Generator):
    """Сумма или произведение тренда, сезонности и нерегулярной компоненты по одной из восьми схем.

    Тренд линейный, нелинейный или нулевой; сезонность синусоидальная, ступенчатая,
    треугольная, импульсная или нулевая с периодом до $L/8$; нерегулярность — AR(1)-шум,
    нормированная накопленная сумма или белый шум. Период разыгрывается от длины ряда.
    Оригинал тянет все ряды из одного генератора случайных чисел подряд и не зависит от сида
    ряда, поэтому его корпус воспроизводится только целиком через `reproduce`; основной режим
    даёт каждому ряду свой генератор от сида.

    Args:
        length: Длина ряда.
    """

    name = "tsi"

    def _trend(self, kind: str, rng: np.random.Generator) -> np.ndarray:
        t = np.linspace(0, 1, self.length)
        if kind == "linear":
            return rng.uniform(-2, 2) * t + rng.uniform(-1, 1)
        if kind == "nonlinear":
            a, b, c, d = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-2, 2), rng.uniform(-1, 1)
            return (a * t + b) * np.sin(2 * np.pi * t) + c * t + d
        return np.zeros(self.length)

    def _season(self, kind: str, rng: np.random.Generator) -> np.ndarray:
        L = self.length
        t = np.arange(L)
        if kind == "sinusoidal":
            period = rng.integers(4, max(8, L // 8))
            return float(rng.uniform(0.5, 2.0)) * np.sin(2 * np.pi / period * t + rng.uniform(0, 2 * np.pi))
        if kind == "step":
            period = int(rng.integers(4, max(8, L // 8)))
            amp = float(rng.uniform(0.5, 2.0))
            return np.where((t % (2 * period)) < period, amp, 0.0)
        if kind == "triangular":
            period = int(rng.integers(4, max(8, L // 8)))
            amp = float(rng.uniform(0.5, 2.0))
            cp = t % (2 * period)
            return np.where(cp < period, amp * (cp / period), amp * (2 - cp / period))
        if kind == "impulsive":
            period = int(rng.integers(4, max(8, L // 8)))
            amp = float(rng.uniform(0.5, 2.0))
            s = np.zeros(L)
            s[::period] = amp
            return np.convolve(s, np.ones(3) / 3, mode="same")
        return np.zeros(L)

    def _irregularity(self, kind: str, rng: np.random.Generator) -> np.ndarray:
        L = self.length
        if kind == "fgn":
            h = rng.uniform(0.3, 0.7)
            noise = rng.normal(0, 0.1, L)
            return lfilter([1.0], [1.0, -h], noise)
        if kind == "fbm":
            cumul = np.cumsum(rng.normal(0, 0.1, L))
            return cumul * (rng.uniform(0.5, 0.9) / np.std(cumul))
        return rng.normal(0, 0.2, L)

    @staticmethod
    def _combine(T, S, N, model: int) -> np.ndarray:
        if model == 1:
            return T + S + N
        if model == 2:
            return (T + S) * (1 + 0.1 * N)
        if model == 3:
            return (T + N) * (1 + 0.1 * S)
        if model == 4:
            return (S + N) * (1 + 0.1 * T)
        if model == 5:
            return T * (1 + 0.1 * S) + N
        if model == 6:
            return T * (1 + 0.1 * N) + S
        if model == 7:
            return S * (1 + 0.1 * N) + T
        return T * (1 + 0.1 * S) * (1 + 0.1 * N)

    def _one(self, rng: np.random.Generator) -> np.ndarray:
        T = self._trend(rng.choice(_TRENDS), rng)
        S = self._season(rng.choice(_SEASONS), rng)
        N = self._irregularity(rng.choice(_IRREGS), rng)
        return self._combine(T, S, N, rng.choice(_MODELS))

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        out = np.empty((len(seeds), self.length))
        for i, seed in enumerate(seeds):
            out[i] = self._one(np.random.default_rng(int(seed)))
        return out

    def reproduce(self, random_seed: int, n: int) -> np.ndarray:
        """Корпус `(n, length)` float32 оригинала: все ряды подряд из одного генератора от `random_seed`.

        Args:
            random_seed: Сид корпуса.
            n: Число рядов.

        Returns:
            np.ndarray: Ряды `(n, length)` float32.
        """
        rng = np.random.default_rng(random_seed)
        return np.stack([self._one(rng) for _ in range(n)]).astype(np.float32)
