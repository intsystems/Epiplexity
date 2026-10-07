"""Concrete estimator implementations (skeletons)."""

from ..base import BaseEstimator


class Prequential(BaseEstimator):
    """Online (prequential) code: area above the floor, and its variants.

    Stage 4 implements the online code, the held-out / EDL / training-loss
    floors, and the block-wise variants of Bornschein et al. (2022).
    """

    name = "prequential"

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def estimate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class Requential(BaseEstimator):
    """Two-part / requential MDL code length."""

    name = "requential"

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def estimate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class Bayesian(BaseEstimator):
    """Variational and Laplace Bayesian code lengths."""

    name = "bayesian"

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def estimate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class Proxies(BaseEstimator):
    """Classical criteria and fast proxies: AIC, BIC, HQIC, WAIC, WBIC, EDL,
    and the reservoir score (stage 4)."""

    name = "proxies"

    def fit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def estimate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError
