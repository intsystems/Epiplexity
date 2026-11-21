"""Interface for fixed-length univariate synthetic time-series generators.

Each generator follows the generative process of its counterpart in
Mix, Don't Pick (Nagpal et al., 2026, arXiv:2606.09912,
github.com/birla-ai-labs/mix-dont-pick, Apache-2.0).

The default mode batches across series, vectorizes time recurrences, and
runs without the original external libraries. Where the original shares
one seed for parameter selection and noise, the default mode separates
those random streams. Reproduction mode (`exact`) preserves the original
order of random draws and returns the original series for the same seed.
Use this slower mode for checks.

Each series depends only on its own seed and length, so the subset, order,
and batching of seeds do not change the generated series.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import numpy as np


def original_seeds(random_seed: int, n: int) -> np.ndarray:
    """Recreate the per-series seeds for a corpus of `n` series with seed `random_seed`.

    The original samples these with `default_rng(random_seed).integers(0, 2**31, n)`.
    See the `reproduce` methods of TSI and TimeSynth for their own seed schemes.

    Args:
        random_seed: Corpus seed.
        n: Number of series.

    Returns:
        np.ndarray: Integer seeds of shape `(n,)`.
    """
    return np.random.default_rng(random_seed).integers(0, 2**31, size=n)


class Generator(ABC):
    """Generator of one-dimensional synthetic series of fixed length.

    Args:
        length: The length of the row.

    Attributes:
        name: Generator's name on the registry.
    """

    name: ClassVar[str]

    def __init__(self, length: int = 1024) -> None:
        self.length = length

    def generate(self, seeds: Sequence[int] | np.ndarray, exact: bool = False) -> np.ndarray:
        """Generate one float32 series per seed.

        With `exact=True`, match the reference implementation for the same seed.
        The result is bitwise-identical when NumPy evaluates operations in the
        same order and agrees up to rounding otherwise. Nonfinite series are
        replaced according to the original implementation's fallback.

        Args:
            seeds: Series seeds, nonnegative integers below `2**31`.
            exact: Use the original algorithm's reproduction mode.

        Returns:
            np.ndarray: Float32 series of shape `(n, length)`.
        """
        seeds_arr = np.asarray(seeds, dtype=np.int64).reshape(-1)
        if exact:
            rows = [self._exact(int(s)) for s in seeds_arr]
            return np.stack(rows).astype(np.float32, copy=False)
        return self._batch(seeds_arr).astype(np.float32, copy=False)

    def reproduce(self, random_seed: int, n: int) -> np.ndarray:
        """Reproduce the original float32 corpus for `random_seed`.

        Follow the original `generate_series(n)` by reproducing its per-series
        seed scheme and generating in `exact` mode.

        Args:
            random_seed: Corpus seed.
            n: Number of series.

        Returns:
            np.ndarray: Float32 series of shape `(n, length)`.
        """
        return self.generate(original_seeds(random_seed, n), exact=True)

    @abstractmethod
    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        """`(n, length)` float64 series of the main mode for the array of seats."""

    def _exact(self, seed: int) -> np.ndarray:
        """The `(length,)` float64 series of reproduction modes; by default, the main mode is it."""
        return self._batch(np.array([seed], dtype=np.int64))[0]
