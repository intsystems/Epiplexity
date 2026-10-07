"""epimeter: estimating epiplexity with small models on small datasets.

Public interface of the package. Importing the package is side-effect free and
safe even before the estimators are implemented: the concrete classes below are
defined but raise ``NotImplementedError`` until a later stage fills them in.
"""

from ._version import __version__
from .base import BaseEstimator
from .estimators import Bayesian, Prequential, Proxies, Requential
from .models.base import BaseModel
from .models.registry import build, register
from .samplers.base import BaseSampler

__all__ = [
    "BaseEstimator",
    "BaseModel",
    "BaseSampler",
    "Prequential",
    "Requential",
    "Bayesian",
    "Proxies",
    "build",
    "register",
    "__version__",
]
