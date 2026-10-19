"""SARIMA-подобные процессы: ARMA, интегрирование, сезонные члены."""

import numpy as np
from scipy.signal import lfilter

from .base import Generator

_BURN_IN = 500
_MAX_ORDER = 3


_TRIES = 100


def _roots_outside(poly: np.ndarray) -> np.ndarray:
    """Для пакета многочленов `(m, k+1)` со старшим коэффициентом 1: все ли корни по модулю больше 1,05.

    Корни считаются как собственные числа сопровождающей матрицы, как в `np.roots`, но для всего
    пакета разом; решения совпадают с `np.roots` поэлементно.
    """
    m, k1 = poly.shape
    k = k1 - 1
    if k == 1:
        return np.abs(-poly[:, 1]) > 1.05
    comp = np.zeros((m, k, k))
    comp[:, np.arange(1, k), np.arange(k - 1)] = 1.0
    comp[:, 0, :] = -poly[:, 1:] / poly[:, :1]
    return np.all(np.abs(np.linalg.eigvals(comp)) > 1.05, axis=1)


def _sample_coeffs(k: int, rng: np.random.Generator, sign: float) -> np.ndarray:
    """Коэффициенты AR (`sign=-1`) или MA (`sign=+1`) с отбраковкой по корням, как у оригинала.

    Оригинал разыгрывает кандидатов по одному до первого удачного (не больше 100), затем берёт
    малые коэффициенты. Здесь кандидаты разыгрываются блоком с возвратом состояния генератора,
    чтобы поток случайных чисел остался тем же, а корни считались одним пакетным вызовом.
    """
    state = rng.bit_generator.state
    block = rng.uniform(-0.8, 0.8, size=(_TRIES, k))
    rng.bit_generator.state = state
    ok = _roots_outside(np.concatenate([np.ones((_TRIES, 1)), sign * block], axis=1))
    hits = np.flatnonzero(ok)
    if len(hits) == 0:
        rng.uniform(-0.8, 0.8, size=(_TRIES, k))
        return rng.uniform(-0.3, 0.3, size=k)
    first = int(hits[0])
    if first > 0:
        rng.uniform(-0.8, 0.8, size=(first, k))
    return rng.uniform(-0.8, 0.8, size=k)


def _sample_ar(p: int, rng: np.random.Generator) -> np.ndarray:
    return _sample_coeffs(p, rng, -1.0)


def _sample_ma(q: int, rng: np.random.Generator) -> np.ndarray:
    return _sample_coeffs(q, rng, 1.0)


class ARIMA(Generator):
    r"""ARIMA(p, d, q) со случайными порядками и, с вероятностью 0,3, сезонными членами.

    Порядки $p, q \in \{0..3\}$, $d \in \{0, 1, 2\}$; коэффициенты равномерные из
    $[-0{,}8;\,0{,}8]$ с отбраковкой по корням (стационарность и обратимость); сезонный
    период из $\{4, 7, 12, 24, 52\}$ с сезонными AR и MA первого порядка и сезонным
    разностью. Рекурсия ARMA считается на 500 точек разгона длиннее ряда, затем берётся
    хвост нужной длины; при $d = 2$ ряды сильно нестационарны. Распределение не зависит от
    длины ряда. Рекурсия считается пакетом по рядам поэлементно теми же операциями, что у
    оригинала, поэтому основной режим совпадает с режимом воспроизведения.

    Args:
        length: Длина ряда.
    """

    name = "arima"

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
                ar[i, :p] = _sample_ar(p, rng)
            if q > 0:
                ma[i, :q] = _sample_ma(q, rng)
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
                    sar = _sample_ar(P, rng)
                    for r in range(m):
                        yi[r::m] = lfilter([1.0], [1.0, -sar[0]], yi[r::m])
                if Q > 0:
                    sma = _sample_ma(Q, rng)
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
