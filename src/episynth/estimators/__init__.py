r"""Epiplexity estimators: a plug-in interface for selecting and adding algorithms.

The user selects the algorithm by name via :func:`~episynth.estimators.registry.build` or CLI
``python -m episynth.estimators.run --estimator <name>``. The new algorithm is added by one module:
determine the class implementing :class:`~episynth.estimators.base.EpiplexityEstimator` and mark it
with ``@register("name")``; the remaining modules of the library do not need to be changed.
"""

from .base import EpiplexityEstimator
from .registry import available, build, register

__all__ = ["EpiplexityEstimator", "register", "build", "available"]
