"""SARIMA-подобные процессы с исходным или исправленным отбором коэффициентов.

По умолчанию сохраняется поток исходного генератора Mix, Don't Pick, включая ошибку
ориентации полинома. Исправленный отбор включается явно: coefficient_mode="corrected".
"""

from typing import Literal

import numpy as np
from scipy.signal import lfilter

from .base import Generator

_BURN_IN = 500
_MAX_ORDER = 3


_TRIES = 100


def _roots_outside(poly: np.ndarray, *, lag_order: bool = False) -> np.ndarray:
    """Все ли корни пакета полиномов `(m, k+1)` по модулю больше 1,05.

    Первый коэффициент равен 1. В исходном режиме коэффициенты идут по убыванию степеней,
    как в `np.roots`. При `lag_order=True` они идут по возрастанию степеней: проверяем
    обратные корни внутри круга радиуса 1/1,05. Это избегает деления на последний
    коэффициент, который может быть нулём. Корни считаются пакетной сопровождающей матрицей.
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
    """Коэффициенты AR (`sign=-1`) или MA (`sign=+1`) с заданным отбором по корням.

    Legacy повторяет ошибку оригинала: у полинома z**k + sign*a[0]*z**(k-1) + ...
    произведение модулей корней равно |a[-1]| <= 0,8. Поэтому все корни не могут иметь
    модуль > 1,05: 100 кандидатов всегда отклоняются и берётся U(-0,3; 0,3)**k.
    Corrected проверяет лаговый полином 1 + sign*sum(a[j]*z**(j+1)). После 100 неудач
    тот же малый запасной вектор гарантированно допустим при k <= 3, поскольку
    0,3 * (1,05 + 1,05**2 + 1,05**3) < 1.

    Кандидаты разыгрываются блоком с возвратом состояния генератора, чтобы сохранить
    порядок случайных чисел построчного отбора до первого удачного кандидата.
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
    r"""ARIMA(p, d, q) со случайными порядками и, с вероятностью 0,3, сезонными членами.

    Порядки :math:`p, q \in \{0..3\}`, :math:`d \in \{0, 1, 2\}`. В режиме `legacy` все
    векторы AR/MA, включая сезонные, фактически равномерны из :math:`[-0{,}3;\,0{,}3]^k`:
    унаследованная ошибка порядка коэффициентов отклоняет все 100 исходных кандидатов.
    Этот режим сохраняет опубликованные ряды до бита. В режиме `corrected` кандидаты из
    :math:`[-0{,}8;\,0{,}8]^k` принимаются, если корни лаговых AR/MA-полиномов лежат вне
    круга радиуса 1,05 (стационарность AR, обратимость MA); после 100 неудач используется
    тот же малый запасной вектор. Это явно другое распределение генератора. Сезонный
    период из :math:`\{4, 7, 12, 24, 52\}` с сезонными AR и MA первого порядка и сезонным
    разностью. Рекурсия ARMA считается на 500 точек разгона длиннее ряда, затем берётся
    хвост нужной длины; при :math:`d = 2` ряды сильно нестационарны. Распределение не зависит от
    длины ряда. Рекурсия считается пакетом по рядам поэлементно теми же операциями, что у
    оригинала. Основной и `exact` режимы совпадают при одном `coefficient_mode`;
    воспроизведение исходного генератора обеспечивается только для `legacy`.

    Args:
        length: Длина ряда.
        coefficient_mode: `legacy` сохраняет исходные ряды; `corrected` исправляет отбор коэффициентов.
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
