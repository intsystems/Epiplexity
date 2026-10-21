"""Процесс Орнштейна–Уленбека с переключением режимов по марковской цепи."""

import numpy as np

from .base import Generator

_MAX_REGIMES = 4


class SDE(Generator):
    r"""ОУ-процесс с 2–4 режимами (вероятности 0,5, 0,3, 0,2), сменяемыми марковской цепью.

    У каждого режима свои скорость возврата :math:`\theta`, уровень :math:`\mu` и волатильность
    :math:`\sigma`; матрица переходов разыгрывается по строкам из распределения Дирихле с
    тяжёлой диагональю. Шаг времени единичный, переход между шагами точный (Doob):
    :math:`x_{t+1} = \mu_k + (x_t - \mu_k) e^{-\theta_k} + s_k \varepsilon_t` с
    :math:`s_k^2 = \sigma_k^2 (1 - e^{-2\theta_k}) / (2\theta_k)`. Распределение не зависит от
    длины ряда. Цепь и рекурсия считаются пакетом по рядам поэлементно теми же операциями,
    что у оригинала, поэтому основной режим совпадает с режимом воспроизведения.

    Args:
        length: Длина ряда.
    """

    name = "sde"

    @staticmethod
    def _regime_params(k: int, rng: np.random.Generator):
        thetas = rng.uniform(0.05, 5.0, size=k)
        sigmas = rng.uniform(0.05, 2.0, size=k)
        base_mu = rng.uniform(-3.0, 3.0)
        if k == 1:
            return thetas, np.array([base_mu]), sigmas
        mus = np.empty(k)
        mus[0] = base_mu
        for i in range(1, k):
            mus[i] = mus[i - 1] + rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 4.0)
        return thetas, mus, sigmas

    @staticmethod
    def _transition_cdf(k: int, rng: np.random.Generator) -> np.ndarray:
        diag = rng.uniform(15.0, 50.0)
        cdf = np.ones((_MAX_REGIMES, _MAX_REGIMES))
        for i in range(k):
            alpha = np.full(k, 1.0)
            alpha[i] = diag
            row = rng.dirichlet(alpha).cumsum()
            row /= row[-1]
            cdf[i, :k] = row
        return cdf

    def _ou_scalar(self, regimes, thetas, mus, sigmas, rng) -> np.ndarray:
        """Рекурсия ОУ по одному ряду, как у оригинала; нужна запасному пути."""
        dt = 1.0
        x = np.empty(self.length)
        k0 = regimes[0]
        x[0] = mus[k0] + (sigmas[k0] / np.sqrt(2.0 * thetas[k0])) * rng.standard_normal()
        decay = np.exp(-thetas * dt)
        t_std = np.sqrt(np.maximum((sigmas**2 / (2 * thetas)) * (1 - np.exp(-2 * thetas * dt)), 1e-15))
        noise = rng.standard_normal(self.length - 1)
        for t in range(self.length - 1):
            k = regimes[t + 1]
            x[t + 1] = mus[k] + (x[t] - mus[k]) * decay[k] + t_std[k] * noise[t]
        return x

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n, L = len(seeds), self.length
        thetas = np.ones((n, _MAX_REGIMES))
        mus = np.zeros((n, _MAX_REGIMES))
        sigmas = np.ones((n, _MAX_REGIMES))
        cdf = np.ones((n, _MAX_REGIMES, _MAX_REGIMES))
        regimes = np.empty((n, L), dtype=np.int64)
        u = np.empty((n, L - 1))
        z0 = np.empty(n)
        noise = np.empty((n, L - 1))
        rngs = []
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            rngs.append(rng)
            k = int(rng.choice([2, 3, 4], p=[0.5, 0.3, 0.2]))
            th, mu, sg = self._regime_params(k, rng)
            cdf[i] = self._transition_cdf(k, rng)
            regimes[i, 0] = rng.integers(0, k)
            u[i] = rng.random(L - 1)
            z0[i] = rng.standard_normal()
            noise[i] = rng.standard_normal(L - 1)
            thetas[i, :k], mus[i, :k], sigmas[i, :k] = th, mu, sg
        rows = np.arange(n)
        for t in range(1, L):
            regimes[:, t] = (cdf[rows, regimes[:, t - 1]] <= u[:, t - 1, None]).sum(axis=1)
        dt = 1.0
        decay = np.exp(-thetas * dt)
        t_std = np.sqrt(np.maximum((sigmas**2 / (2 * thetas)) * (1 - np.exp(-2 * thetas * dt)), 1e-15))
        x = np.empty((n, L))
        k0 = regimes[:, 0]
        x[:, 0] = mus[rows, k0] + (sigmas[rows, k0] / np.sqrt(2.0 * thetas[rows, k0])) * z0
        for t in range(L - 1):
            k = regimes[:, t + 1]
            m = mus[rows, k]
            x[:, t + 1] = m + (x[:, t] - m) * decay[rows, k] + t_std[rows, k] * noise[:, t]
        bad = ~np.all(np.isfinite(x), axis=1)
        for i in np.flatnonzero(bad):
            x[i] = self._ou_scalar(
                np.zeros(L, dtype=np.int64), np.array([1.0]), np.array([0.0]), np.array([0.5]), rngs[i]
            )
        return x
