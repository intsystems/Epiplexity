r"""The Ornstein-Ulenbeck process with a prior on the coefficient: a benchmark with a well-known Bayesian forecast."""

import numpy as np

from .base import Generator


class OU(Generator):
    r"""AR(1) with zero average and unit stationary variance, :math:`\varphi = \sin u`, :math:`u \sim U[u_{\min}, u_{\max}]`.

    :math:`x_0 \sim N(0, 1)`, :math:`x_t = \varphi x_{t-1} + \sqrt{1 -
    \varphi^2}\,\varepsilon_t`Prior on :math:`u` taken from E0`prior_diversity`): :math:`\varphi \in
    [0{,}01;\,0{,}995]`Parameter and noise are taken from the same stream. `default_rng(seed)`:first
    :math:`u`, then :math:`L` The distribution does not depend on the length of the series, serves
    as a test of learning: the exact Bayesian loss of a window is considered to be a priori square.
    `episynth.checks.ou_bayes`.

    Args:
        length: The length of the row.

    Attributes:
        u_min: The lower limit of the prior is :math:`u`.
        u_max: The upper limit of the prior is :math:`u`.
    """

    name = "ou"
    u_min = 0.010000166674167114
    u_max = 1.4707546131833567

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n, L = len(seeds), self.length
        phi = np.empty(n)
        eps = np.empty((n, L))
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            phi[i] = np.sin(rng.uniform(self.u_min, self.u_max))
            eps[i] = rng.standard_normal(L)
        x = np.empty((n, L))
        x[:, 0] = eps[:, 0]
        a = np.sqrt(1 - phi**2)
        for t in range(1, L):
            x[:, t] = phi * x[:, t - 1] + a * eps[:, t]
        return x
