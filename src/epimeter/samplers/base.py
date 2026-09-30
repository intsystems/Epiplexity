"""Sampler interface for Bayesian estimators (e.g. SGLD, priors over weights)."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch


class BaseSampler(ABC):
    """Draw parameter samples from a model.

    ``sample`` returns a batch of tensors of ``num_samples`` parameter draws of
    shape ``shape`` (e.g. ``(num_samples,) + model.parameter_shape``).
    """

    @abstractmethod
    def sample(
        self,
        model: Any,
        num_samples: int,
        shape: Sequence[int],
    ) -> torch.Tensor:
        """Draw ``num_samples`` parameter samples shaped as ``shape``."""
