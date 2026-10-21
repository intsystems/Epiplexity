"""Семейство GARCH: GARCH(1,1), GJR-GARCH и EGARCH."""

import numpy as np

from .base import Generator

_BURN_IN = 500
_E_ABS_Z = np.sqrt(2 / np.pi)


class GARCH(Generator):
    r"""Доходности GARCH(1,1), GJR-GARCH или EGARCH, с вероятностью 0,5 накопленные в цены.

    Инновации нормальные или Стьюдента с 3–10 степенями свободы (нормированы к единичной
    дисперсии), среднее нулевое, постоянное или AR(1), безусловная волатильность
    логравномерная из :math:`[0{,}005;\,0{,}5]`; 500 точек разгона отбрасываются. Распределение
    не зависит от длины ряда. Рекурсии считаются пакетом по рядам поэлементно теми же
    операциями, что у оригинала (квадрат берётся как у скалярного `**` Python), поэтому
    основной режим совпадает с режимом воспроизведения.

    Args:
        length: Длина ряда.
    """

    name = "garch"

    @staticmethod
    def _innovations(n: int, kind: str, doff: float, rng: np.random.Generator) -> np.ndarray:
        if kind == "normal":
            return rng.standard_normal(n)
        return rng.standard_t(doff, size=n) / np.sqrt(doff / (doff - 2.0))

    @staticmethod
    def _garch11(omega, alpha, beta, mu, phi, z) -> np.ndarray:
        """Рекурсия GARCH(1,1); параметры — массивы по рядам, `z` формы `(n, T)`."""
        T = z.shape[1]
        sigma2 = omega / (1 - alpha - beta)
        eps = np.sqrt(sigma2) * z[:, 0]
        r = np.empty_like(z)
        r[:, 0] = mu + eps
        for t in range(1, T):
            sigma2 = np.maximum(omega + alpha * np.float_power(eps, 2.0) + beta * sigma2, 1e-8)
            eps = np.sqrt(sigma2) * z[:, t]
            r[:, t] = mu + phi * (r[:, t - 1] - mu) + eps
        return r

    @staticmethod
    def _gjr(omega, alpha, beta, gamma, mu, phi, z) -> np.ndarray:
        T = z.shape[1]
        sigma2 = omega / np.maximum(1 - alpha - gamma / 2 - beta, 1e-6)
        eps = np.sqrt(sigma2) * z[:, 0]
        r = np.empty_like(z)
        r[:, 0] = mu + eps
        for t in range(1, T):
            ind = np.where(eps < 0, 1.0, 0.0)
            sigma2 = np.maximum(omega + (alpha + gamma * ind) * np.float_power(eps, 2.0) + beta * sigma2, 1e-8)
            eps = np.sqrt(sigma2) * z[:, t]
            r[:, t] = mu + phi * (r[:, t - 1] - mu) + eps
        return r

    @staticmethod
    def _egarch(omega, alpha, beta, gamma, mu, phi, z) -> np.ndarray:
        T = z.shape[1]
        log_s2 = omega / (1 - beta)
        eps = np.sqrt(np.exp(log_s2)) * z[:, 0]
        r = np.empty_like(z)
        r[:, 0] = mu + eps
        for t in range(1, T):
            zp = z[:, t - 1]
            log_s2 = np.clip(omega + alpha * (np.abs(zp) - _E_ABS_Z) + gamma * zp + beta * log_s2, -20, 20)
            eps = np.sqrt(np.exp(log_s2)) * z[:, t]
            r[:, t] = mu + phi * (r[:, t - 1] - mu) + eps
        return r

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n, L = len(seeds), self.length
        T = L + _BURN_IN
        model = np.empty(n, dtype=object)
        z = np.empty((n, T))
        mu = np.zeros(n)
        phi = np.zeros(n)
        omega = np.zeros(n)
        a = np.zeros(n)
        b = np.zeros(n)
        g = np.zeros(n)
        rngs = []
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            rngs.append(rng)
            model[i] = rng.choice(["garch11", "gjr_garch", "egarch"])
            innov = rng.choice(["normal", "student_t"])
            mean_type = rng.choice(["zero", "constant", "ar1"])
            z[i] = self._innovations(T, innov, float(rng.uniform(3, 10)), rng)
            mu[i] = float(rng.normal(0, 0.05)) if mean_type == "constant" else 0.0
            phi[i] = float(rng.uniform(-0.3, 0.3)) if mean_type == "ar1" else 0.0
            sigma_unc = float(np.exp(rng.uniform(np.log(0.005), np.log(0.5))))
            if model[i] == "garch11":
                pers = float(rng.uniform(0.70, 0.97))
                a_share = float(rng.uniform(0.02, 0.25))
                a[i], b[i] = pers * a_share, pers - pers * a_share
                omega[i] = sigma_unc**2 * (1 - a[i] - b[i])
            elif model[i] == "gjr_garch":
                g[i] = float(rng.uniform(0.01, 0.15))
                pers = float(rng.uniform(0.70, 0.97))
                a_share = float(rng.uniform(0.02, 0.20))
                a[i] = pers * a_share
                b[i] = max(pers - a[i] - g[i] / 2, 0.01)
                omega[i] = sigma_unc**2 * max(1 - a[i] - g[i] / 2 - b[i], 1e-4)
            else:
                b[i] = float(rng.uniform(0.70, 0.97))
                a[i] = float(rng.uniform(0.05, 0.25))
                g[i] = float(rng.uniform(-0.20, 0.20))
                omega[i] = np.log(sigma_unc**2) * (1 - b[i])
        r = np.empty((n, T))
        with np.errstate(all="ignore"):
            for kind in ("garch11", "gjr_garch", "egarch"):
                idx = np.flatnonzero(model == kind)
                if len(idx) == 0:
                    continue
                if kind == "garch11":
                    r[idx] = self._garch11(omega[idx], a[idx], b[idx], mu[idx], phi[idx], z[idx])
                elif kind == "gjr_garch":
                    r[idx] = self._gjr(omega[idx], a[idx], b[idx], g[idx], mu[idx], phi[idx], z[idx])
                else:
                    r[idx] = self._egarch(omega[idx], a[idx], b[idx], g[idx], mu[idx], phi[idx], z[idx])
        out = np.empty((n, L))
        for i in range(n):
            ri = r[i, _BURN_IN:]
            if not np.all(np.isfinite(ri)):
                z_safe = self._innovations(T, "normal", 5.0, rngs[i])
                ri = self._garch11(
                    np.array([1e-4]), np.array([0.05]), np.array([0.90]), np.array([0.0]), np.array([0.0]), z_safe[None]
                )[0, _BURN_IN:]
            if rngs[i].random() < 0.5:
                ri = np.cumsum(ri)
            out[i] = ri
        return out
