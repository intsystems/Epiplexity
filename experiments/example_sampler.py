"""Example sampler stub wired through the registry.

Placeholder for the parameter samplers used by the Bayesian codes (stage 4+).
It registers a ``dummy`` sampler so the sampler registry/builder path can be
exercised end to end.
"""

import torch

from epimeter.samplers.base import BaseSampler
from epimeter.samplers.registry import register


@register("dummy")
class DummySampler(BaseSampler):
    """Returns zero parameter samples of the requested shape."""

    def sample(self, model, num_samples: int, shape) -> torch.Tensor:
        return torch.zeros((num_samples,) + tuple(shape))
