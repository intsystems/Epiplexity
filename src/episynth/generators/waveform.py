"""Негладкие периодические волны: пила, меандр, треугольник."""

import numpy as np
from scipy.signal import sawtooth, square

from .base import Generator

_TYPES = ["sawtooth", "square", "triangle"]


class Waveform(Generator):
    """Сумма одной–трёх негладких волн со случайными частотой, амплитудой, фазой и скважностью.

    С вероятностью 0,3 амплитуда модулируется синусом, с вероятностью 0,7 добавляется белый
    шум, с вероятностью 0,3 — линейный тренд. Частоты заданы числом периодов на ряд, поэтому
    распределение формы не зависит от длины. Оригинал уже векторизован по времени, основной
    режим совпадает с режимом воспроизведения.

    Args:
        length: Длина ряда.
    """

    name = "waveform"

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        t = np.linspace(0, 1, self.length, endpoint=False)
        out = np.empty((len(seeds), self.length))
        for i, seed in enumerate(seeds):
            out[i] = self._one(np.random.default_rng(int(seed)), t)
        return out

    def _one(self, rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
        n_waves = int(rng.integers(1, 4))
        combined = np.zeros(self.length)
        for _ in range(n_waves):
            wtype = rng.choice(_TYPES)
            freq = float(rng.uniform(1, 50))
            amplitude = float(rng.uniform(0.3, 3.0))
            phase = float(rng.uniform(0, 2 * np.pi))
            duty = float(rng.uniform(0.2, 0.8))
            angle = 2 * np.pi * freq * t + phase
            if wtype == "sawtooth":
                wave = sawtooth(angle, width=float(rng.choice([0.0, 1.0])))
            elif wtype == "square":
                wave = square(angle, duty=duty)
            else:
                wave = sawtooth(angle, width=0.5)
            combined += amplitude * wave
        if rng.random() < 0.3:
            mod_freq = float(rng.uniform(0.5, 5))
            combined *= 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        if rng.random() < 0.7:
            combined += rng.normal(0, float(rng.uniform(0.01, 0.3)), size=self.length)
        if rng.random() < 0.3:
            combined += rng.uniform(-2, 2) * t
        return combined
