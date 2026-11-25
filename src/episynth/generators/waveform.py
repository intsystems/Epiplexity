"""Smooth periodic waves: saw, meander, triangle."""

import numpy as np
from scipy.signal import sawtooth, square

from .base import Generator

_TYPES = ["sawtooth", "square", "triangle"]


class Waveform(Generator):
    """The sum of one or three non-smooth waves with random frequency,
    amplitude, phase and porosity.

    With the probability of 0,3, the amplitude is modulated by the sine, with
    the probability of
    0,7, white noise is added, with the probability of 0,3 is a linear trend.
    The frequencies are
    given by the number of periods per series, so the distribution of the shape
    is independent of
    the length. The original is already vectored in time, the main mode
    coincides with the playback
    mode.

    Args:
        length: The length of the row.
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
            combined += rng.normal(
                0, float(rng.uniform(0.01, 0.3)), size=self.length
            )
        if rng.random() < 0.3:
            combined += rng.uniform(-2, 2) * t
        return combined
