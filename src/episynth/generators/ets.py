"""Модели экспоненциального сглаживания ETS в форме пространства состояний."""

import numpy as np

from .base import Generator

_BURN_IN = 200
_MAX_PERIOD = 52
_LEVEL_FLOOR = 1e-2
_LEVEL_CEIL = 1e6


class ETS(Generator):
    r"""ETS со случайным сочетанием ошибки (A, M), тренда (N, A, затухающий) и сезонности (N, A, M).

    Параметры сглаживания и начальные состояния равномерные, период из
    :math:`\{4, 7, 12, 24, 52\}`; уровень ограничен сверху и снизу, сезонные индексы
    мультипликативной модели зажаты в :math:`[0{,}1;\,10]`. Ряд, вышедший за :math:`10^8` или потерявший
    конечность, заменяется простой аддитивной моделью, как в оригинале. Распределение не
    зависит от длины ряда. Рекурсия состояний считается пакетом по рядам поэлементно теми же
    операциями, что у оригинала, поэтому основной режим совпадает с режимом воспроизведения.

    Args:
        length: Длина ряда.
    """

    name = "ets"

    def _fallback(self, rng: np.random.Generator) -> np.ndarray:
        level = float(rng.uniform(10, 50))
        y = np.empty(self.length)
        for t in range(self.length):
            e = rng.normal(0, 0.1 * level)
            y[t] = level + e
            level = level + 0.05 * e
        return y

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n, L = len(seeds), self.length
        T = L + _BURN_IN
        err_a = np.zeros(n, dtype=bool)
        trend_n = np.zeros(n, dtype=bool)
        season_a = np.zeros(n, dtype=bool)
        season_m = np.zeros(n, dtype=bool)
        period = np.ones(n, dtype=np.int64)
        alpha = np.zeros(n)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        phi = np.ones(n)
        sigma = np.zeros(n)
        level = np.zeros(n)
        b = np.zeros(n)
        s = np.ones((n, _MAX_PERIOD))
        z = np.empty((n, T))
        rngs = []
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(int(seed))
            rngs.append(rng)
            error = rng.choice(["A", "M"])
            trend = rng.choice(["N", "A", "Ad"])
            season = rng.choice(["N", "A", "M"])
            per = int(rng.choice([4, 7, 12, 24, 52])) if season != "N" else 1
            alpha[i] = float(rng.uniform(0.01, 0.3))
            beta[i] = float(rng.uniform(0.001, 0.1)) if trend != "N" else 0.0
            gamma[i] = float(rng.uniform(0.001, 0.15)) if season != "N" else 0.0
            phi[i] = float(rng.uniform(0.8, 0.98)) if trend == "Ad" else 1.0
            if error == "M":
                sigma[i] = float(rng.uniform(0.005, 0.05))
            else:
                sigma[i] = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
            level[i] = float(rng.uniform(10, 50))
            b[i] = float(rng.uniform(-0.1, 0.1)) if trend != "N" else 0.0
            if season == "A":
                si = rng.normal(0, level[i] * 0.05, size=per)
                si -= si.mean()
            elif season == "M":
                si = rng.uniform(0.85, 1.15, size=per)
                si = si / si.mean()
            else:
                si = np.ones(per)
            s[i, :per] = si
            z[i] = rng.standard_normal(T)
            err_a[i] = error == "A"
            trend_n[i] = trend == "N"
            season_a[i] = season == "A"
            season_m[i] = season == "M"
            period[i] = per

        rows = np.arange(n)
        y = np.empty((n, T))
        with np.errstate(all="ignore"):
            for t in range(T):
                idx = t % period
                si = s[rows, idx]
                base = level + phi * b
                yhat = np.where(season_m, base * si, np.where(season_a, base + si, base))
                e = np.where(err_a, (sigma * np.abs(level)) * z[:, t], sigma * z[:, t])
                y[:, t] = np.where(err_a, yhat + e, yhat * (1 + e))
                err_term = np.where(err_a, e, np.where(np.abs(yhat) > 1e-10, e * yhat, e))
                level_old = level
                level = np.where(
                    season_m,
                    level + phi * b + alpha * err_term / np.maximum(np.abs(si), 1e-10),
                    level + phi * b + alpha * err_term,
                )
                level = np.clip(level, -_LEVEL_CEIL, _LEVEL_CEIL)
                small = (np.abs(level) < _LEVEL_FLOOR) & ~err_a
                level = np.where(small, np.where(level != 0, np.sign(level) * _LEVEL_FLOOR, _LEVEL_FLOOR), level)
                b = np.where(trend_n, b, np.clip(phi * b + beta * (level - level_old), -10.0, 10.0))
                denom = np.maximum(np.abs(level_old + phi * b), 1e-10)
                s_new = np.where(
                    season_a,
                    si + gamma * err_term,
                    np.where(season_m, np.clip(si + gamma * err_term / denom, 0.1, 10.0), si),
                )
                s[rows, idx] = s_new
        out = y[:, _BURN_IN:].copy()
        with np.errstate(all="ignore"):
            bad = ~np.all(np.isfinite(out), axis=1) | (np.nanmax(np.abs(out), axis=1) > 1e8)
        for i in np.flatnonzero(bad):
            out[i] = self._fallback(rngs[i])
        return out
