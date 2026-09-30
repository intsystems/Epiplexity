"""Abstract base classes for epiplexity estimators."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEstimator(ABC):
    """Contract every epiplexity estimator must satisfy.

    Implementations return code lengths in *bits* (``units == "bits"``). The
    estimator is split into a stateful ``fit`` (which may train models, compute
    sweeps or draw samples) and a stateless ``estimate`` returning the
    ``(model_part, data_part)`` decomposition in bits.
    """

    name: str = "estimator"
    units: str = "bits"

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> "BaseEstimator":
        """Fit the estimator on data/models; returns self for chaining."""

    @abstractmethod
    def estimate(self, *args: Any, **kwargs: Any) -> tuple[float, float]:
        """Return ``(model_part_bits, data_part_bits)`` for this estimator."""
