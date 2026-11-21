"""Fractal Brownian motion and fractal Gaussian noise."""

import numpy as np

from .base import Generator


def _autocovariance(k: np.ndarray, hurst: float) -> np.ndarray:
    return 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst) + np.abs(k + 1) ** (2 * hurst))


def _hosking(gn: np.ndarray, hurst: float) -> np.ndarray:
    """Hosking's method, the backup path of the fbm packet at negative eigenvalues."""
    n = len(gn)
    cov = _autocovariance(np.arange(n, dtype=float), hurst)
    fgn = np.zeros(n)
    phi = np.zeros(n)
    psi = np.zeros(n)
    fgn[0] = gn[0]
    v = 1
    phi[0] = 0
    for i in range(1, n):
        phi[i - 1] = cov[i]
        for j in range(i - 1):
            psi[j] = phi[j]
            phi[i - 1] -= psi[j] * cov[i - j - 1]
        phi[i - 1] /= v
        for j in range(i - 1):
            phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
        v *= 1 - phi[i - 1] * phi[i - 1]
        for j in range(i):
            fgn[i] += phi[j] * fgn[i - j - 1]
        fgn[i] += np.sqrt(v) * gn[i]
    return fgn


class FBM(Generator):
    r"""Fractional Brownian motion or fractional Gaussian noise with :math:`H \sim U(0.1, 0.9)`.

    The Davies–Harte method (circulant embedding and FFT) builds increments on
    the unit interval as in `fbm` 0.3.0. A log-uniform factor in :math:`[0.1, 5]`
    scales the increments. With probability 0.5 the generator returns their
    cumulative sum (motion), and otherwise returns the increments (noise).
    The increment scale :math:`(1/L)^H` depends on series length. The original
    uses NumPy's global RNG seeded by the series seed. A separate `RandomState`
    reproduces this stream, so the default and reproduction modes agree, and
    the FFT runs in batches across series.

    Args:
        length: Series length.
    """

    name = "fbm"

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n, L = len(seeds), self.length
        hurst = np.empty(n)
        is_fbm = np.zeros(n, dtype=bool)
        gn = np.empty((n, L))
        states = []
        rngs = []
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            hurst[i] = float(rng.uniform(0.1, 0.9))
            is_fbm[i] = rng.choice(["fbm", "fgn"]) == "fbm"
            legacy = np.random.RandomState(int(seed))
            gn[i] = legacy.normal(0.0, 1.0, L)
            states.append(legacy)
            rngs.append(rng)
        k = np.arange(1, L, dtype=float)
        ac = _autocovariance(k[None, :], hurst[:, None])
        row = np.concatenate(
            [_autocovariance(np.zeros((n, 1)), hurst[:, None]), ac, np.zeros((n, 1)), ac[:, ::-1]], axis=1
        )
        eig = np.fft.fft(row, axis=1).real
        fgn = np.empty((n, L))
        gn2 = np.empty((n, L))
        for i in range(n):
            gn2[i] = states[i].normal(0.0, 1.0, L)
        w = np.zeros((n, 2 * L), dtype=complex)
        w[:, 0] = np.sqrt(eig[:, 0] / (2 * L)) * gn[:, 0]
        w[:, 1:L] = np.sqrt(eig[:, 1:L] / (4 * L)) * (gn[:, 1:L] + 1j * gn2[:, 1:L])
        w[:, L] = np.sqrt(eig[:, L] / (2 * L)) * gn2[:, 0]
        w[:, L + 1 :] = np.sqrt(eig[:, L + 1 :] / (4 * L)) * (gn[:, L - 1 : 0 : -1] - 1j * gn2[:, L - 1 : 0 : -1])
        fgn = np.fft.fft(w, axis=1)[:, :L].real
        out = np.empty((n, L))
        for i in range(n):
            sample = _hosking(gn[i], hurst[i]) if np.any(eig[i] < 0) else fgn[i]
            sample = sample * ((1.0 * 1 / L) ** hurst[i])
            if is_fbm[i]:
                sample = sample.cumsum()
            sample = sample * float(np.exp(rngs[i].uniform(np.log(0.1), np.log(5.0))))
            if not np.all(np.isfinite(sample)):
                sample = rngs[i].standard_normal(L)
            out[i] = sample
        return out
