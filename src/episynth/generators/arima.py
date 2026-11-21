"""SARIMA-like processes with original or corrected coefficient selection.

By default, the original Mix generator, Don't Pick, is stored stream, including the orientation
error of the polynomial. Corrected selection is activated explicitly: coefficient_mode="corrected".
"""

from typing import Literal

import numpy as np
from scipy.signal import lfilter

from .base import Generator

_BURN_IN = 500
_MAX_ORDER = 3


_TRIES = 100


def _roots_outside(poly: np.ndarray, *, lag_order: bool = False) -> np.ndarray:
    """All the roots of the `(m, k+1)` polynomial package are larger than 1,05.

    The first coefficient is 1. In the initial mode, the coefficients go down in degrees, as in
    `np.roots`. In `lag_order=True`, they go up in degrees: check the inverse roots inside the
    circle of the radius 1 / 1,05. This avoids dividing into the last coefficient, which can be
    zero. The roots are considered a packet accompanying matrix.
    """
    m, k1 = poly.shape
    k = k1 - 1
    if k == 1:
        moduli = np.abs(-poly[:, 1:])
    else:
        comp = np.zeros((m, k, k))
        comp[:, np.arange(1, k), np.arange(k - 1)] = 1.0
        comp[:, 0, :] = -poly[:, 1:] / poly[:, :1]
        moduli = np.abs(np.linalg.eigvals(comp))
    return np.all(moduli < 1 / 1.05 if lag_order else moduli > 1.05, axis=1)


def _sample_coeffs(k: int, rng: np.random.Generator, sign: float, coefficient_mode: str = "legacy") -> np.ndarray:
    """The coefficients are AR (`sign=-1`) or MA (`sign=+1`) with a given root selection.

    Legacy repeats the error of the original: the polynomial z**k + sign*a[0]*z**(k-1) + ... the
    product of the root modules is equal to |a[-]1]| <= 0,8Therefore, all roots cannot have a
    module. 1,05: 100 Candidates are always rejected and U(-)0,3; 0,3)**k. Corrected checks the lag
    polynomial 1 + sign*sum(a[j]*z**(j+1) After 100 Failures of the same small spare vector are
    guaranteed at k <= 3because
    0,3 * (1,05 + 1,05**2 + 1,05**3) < 1.

    Candidates are played in a block with the return of the generator state in order to maintain the
    order of random numbers of line selection until the first successful candidate.
    """
    state = rng.bit_generator.state
    block = rng.uniform(-0.8, 0.8, size=(_TRIES, k))
    rng.bit_generator.state = state
    ok = _roots_outside(np.concatenate([np.ones((_TRIES, 1)), sign * block], axis=1),
                        lag_order=coefficient_mode == "corrected")
    hits = np.flatnonzero(ok)
    if len(hits) == 0:
        rng.uniform(-0.8, 0.8, size=(_TRIES, k))
        return rng.uniform(-0.3, 0.3, size=k)
    first = int(hits[0])
    if first > 0:
        rng.uniform(-0.8, 0.8, size=(first, k))
    return rng.uniform(-0.8, 0.8, size=k)


def _sample_ar(p: int, rng: np.random.Generator, coefficient_mode: str = "legacy") -> np.ndarray:
    return _sample_coeffs(p, rng, -1.0, coefficient_mode)


def _sample_ma(q: int, rng: np.random.Generator, coefficient_mode: str = "legacy") -> np.ndarray:
    return _sample_coeffs(q, rng, 1.0, coefficient_mode)


class ARIMA(Generator):
    r"""ARIMA(p, d, q) with random orders and, with probability 0,3, seasonal terms.

    Orders :math:`p, q \in \{0..3\}`, :math:`d \in \{0, 1, 2\}`In mode. `legacy` vectors AR/MA,
    including seasonal ones, are actually uniform from :math:`[-0{,}3;\,0{,}3]^k`Inherited error in
    the order of coefficients rejects everything 100 This mode saves the published series down to
    the bit. `corrected` candidates from :math:`[-0{,}8;\,0{,}8]^k` are accepted if the roots of the
    lag AR/MAThe polynomials lie outside the radius circle. 1,05 (stationarity) ARreversibility
    MAafter 100 It's obviously a different distribution of the generator. Seasonal from :math:`\{4,
    7, 12, 24, 52\}` seasonally AR and MA First order and seasonal difference. ARMA counted 500
    Acceleration points are longer than a row, then the tail of the desired length is taken; at
    :math:`d = 2` The distribution does not depend on the length of the series. Recursion is
    considered a packet in rows, elementally the same operations as the original. `exact` regimes
    coincide with one `coefficient_mode`Reproduction of the original generator is provided only for
    `legacy`.

    Args:
        length: The length of the row.
        coefficient_mode: `legacy` saves the original series; `corrected` corrects the selection of coefficients.
    """

    name = "arima"

    def __init__(self, length: int = 1024, coefficient_mode: Literal["legacy", "corrected"] = "legacy") -> None:
        super().__init__(length)
        if coefficient_mode not in ("legacy", "corrected"):
            raise ValueError("coefficient_mode must be 'legacy' or 'corrected'")
        self.coefficient_mode = coefficient_mode

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n, L = len(seeds), self.length
        T = L + _BURN_IN
        ar = np.zeros((n, _MAX_ORDER))
        ma = np.zeros((n, _MAX_ORDER))
        p_arr = np.zeros(n, dtype=np.int64)
        q_arr = np.zeros(n, dtype=np.int64)
        eps = np.empty((n, T))
        meta = []
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            p = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]))
            d = int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
            q = int(rng.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]))
            if p == 0 and q == 0:
                p = 1
            seasonal = rng.random() < 0.3
            m = int(rng.choice([4, 7, 12, 24, 52])) if seasonal else 0
            P = int(rng.choice([0, 1])) if seasonal else 0
            D = int(rng.choice([0, 1])) if seasonal else 0
            Q = int(rng.choice([0, 1])) if seasonal else 0
            sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))
            if p > 0:
                ar[i, :p] = _sample_ar(p, rng, self.coefficient_mode)
            if q > 0:
                ma[i, :q] = _sample_ma(q, rng, self.coefficient_mode)
            eps[i] = rng.normal(0, sigma, size=T)
            p_arr[i], q_arr[i] = p, q
            meta.append((rng, d, seasonal, m, P, D, Q, sigma))

        y = np.zeros((n, T))
        for t in range(T):
            yt = eps[:, t].copy()
            for j in range(_MAX_ORDER):
                if t - j - 1 >= 0:
                    yt = yt + np.where(j < p_arr, ar[:, j] * y[:, t - j - 1], 0.0)
            for j in range(_MAX_ORDER):
                if t - j - 1 >= 0:
                    yt = yt + np.where(j < q_arr, ma[:, j] * eps[:, t - j - 1], 0.0)
            y[:, t] = yt

        out = np.empty((n, L))
        for i, (rng, d, seasonal, m, P, D, Q, sigma) in enumerate(meta):
            yi = y[i]
            if seasonal and m > 0:
                if P > 0:
                    sar = _sample_ar(P, rng, self.coefficient_mode)
                    for r in range(m):
                        yi[r::m] = lfilter([1.0], [1.0, -sar[0]], yi[r::m])
                if Q > 0:
                    sma = _sample_ma(Q, rng, self.coefficient_mode)
                    eps_s = rng.normal(0, sigma * 0.5, size=T)
                    yi[m:] += sma[0] * eps_s[:-m]
                for _ in range(D):
                    yi = yi[m:] - yi[:-m]
            for _ in range(d):
                yi = np.cumsum(yi)
            yi = yi[-L:]
            if not np.all(np.isfinite(yi)):
                yi = rng.normal(0, sigma, size=L)
            out[i] = yi
        return out
