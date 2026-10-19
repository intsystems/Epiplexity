"""Смеси сигналов в духе библиотеки TimeSynth без самой библиотеки."""

import numpy as np

from .base import Generator

_TYPES = ["sinusoidal", "car", "narma", "pseudoperiodic", "autoregressive"]
_TYPE_P = [0.2, 0.25, 0.2, 0.15, 0.2]
_NOISE_BANK = [("gaussian", s, None) for s in (0.02, 0.05, 0.1, 0.2, 0.3)] + [
    ("red", s, t) for s in (0.05, 0.15, 0.25) for t in (2.0, 5.0, 10.0)
]


class TimeSynth(Generator):
    """Сумма или произведение 1–3 сигналов TimeSynth (синус, CAR, NARMA, псевдопериодический, AR) с шумом.

    Сигналы и их параметры разыгрываются как в генераторе TimeSynth из Mix, Don't Pick,
    сами сигналы и шум (белый или красный) реализованы здесь по коду библиотеки TimeSynth
    (github.com/TimeSynth/TimeSynth, MIT), так что зависимость от неё и от `jitcdde` не
    нужна. Периоды разыгрываются от длины ряда. С вероятностью `irregular_prob` сетка
    времени нерегулярная (прореживание и сдвиги), ряд затем интерполируется на регулярную.
    Оригинал тянет шум из глобального генератора numpy, засеянного сидом ряда, а сид ряда
    есть сид корпуса плюс номер ряда; `irregular_prob` у него разыгрывается от сида корпуса.
    Здесь тот же поток воспроизводится отдельным `RandomState`, и основной режим совпадает с
    режимом воспроизведения при равных `irregular_prob`.

    Args:
        length: Длина ряда.
        irregular_prob: Вероятность нерегулярной сетки времени.
    """

    name = "timesynth"

    def __init__(self, length: int = 1024, irregular_prob: float = 0.2) -> None:
        super().__init__(length)
        self.irregular_prob = irregular_prob

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        out = np.empty((len(seeds), self.length))
        for i, seed in enumerate(seeds):
            out[i] = self._one(int(seed), self.irregular_prob)
        return out

    def reproduce(self, random_seed: int, n: int) -> np.ndarray:
        """Корпус `(n, length)` float32 оригинала: сиды рядов `random_seed + i`, вероятность нерегулярной сетки от `random_seed`.

        Args:
            random_seed: Сид корпуса.
            n: Число рядов.

        Returns:
            np.ndarray: Ряды `(n, length)` float32.
        """
        prob = float(np.random.default_rng(random_seed).uniform(0.1, 0.3))
        return np.stack([self._one(random_seed + i, prob) for i in range(n)]).astype(np.float32)

    def _one(self, seed: int, irregular_prob: float) -> np.ndarray:
        L = self.length
        legacy = np.random.RandomState(seed)
        rng = np.random.default_rng(seed)
        n_signals = rng.integers(1, 4)
        selected = rng.choice(_TYPES, size=n_signals, replace=True, p=_TYPE_P)
        stop_time = L / 1.0
        need_regular = any(s in {"autoregressive", "narma"} for s in selected)
        if need_regular or rng.random() >= irregular_prob:
            times = np.linspace(0, stop_time, L)
        else:
            keep = float(rng.uniform(80, 95))
            times = np.linspace(0, stop_time, L)
            resolution = float(stop_time - 0) / L
            index = np.sort(legacy.choice(L, size=int(keep * L / 100), replace=False))
            times = np.sort(times[index] + legacy.normal(0.0, resolution, size=len(index)))
        combined = None
        for kind in selected:
            comp = np.asarray(self._signal(kind, rng, legacy, times), dtype=np.float32)
            w = float(rng.uniform(0.4, 1.2))
            if combined is None:
                combined = w * comp
            elif rng.random() < 0.8:
                combined = combined + w * comp
            else:
                combined = combined * (1 + 0.05 * w * comp)
        kind, std, tau = _NOISE_BANK[rng.choice(len(_NOISE_BANK))]
        if kind == "gaussian":
            noise = legacy.normal(0.0, std, size=len(times))
        else:
            noise = np.zeros(len(times))
            prev = 0.0
            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                wnoise = legacy.normal(0.0, std, size=1)
                prev = (tau / (tau + dt)) * (dt * wnoise + prev)
                noise[i] = prev[0]
        samples = combined + noise
        if len(samples) != L:
            samples = np.interp(np.linspace(times[0], times[-1], L), times, samples)
        return np.asarray(samples, dtype=np.float32).astype(np.float64)

    def _signal(self, kind: str, rng: np.random.Generator, legacy: np.random.RandomState, times: np.ndarray):
        L = self.length
        if kind == "sinusoidal":
            p = int(rng.integers(4, max(8, L // 8)))
            amplitude = float(rng.uniform(0.5, 3.0))
            frequency = float((L / p) / (L / 1.0))
            ftype = rng.choice([np.sin, np.cos])
            return amplitude * ftype(2 * np.pi * frequency * times)
        if kind == "car":
            ar = float(rng.uniform(0.1, 0.95))
            sigma = float(rng.uniform(0.1, 0.8))
            out = np.zeros(len(times))
            prev = float(rng.uniform(-0.5, 0.5))
            out[0] = prev
            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                noise = legacy.normal(loc=0.0, scale=1.0, size=1)
                prev = (np.power(ar, dt)) * prev + sigma * np.sqrt(1 - np.power(ar, dt)) * noise
                out[i] = prev[0]
            return out
        if kind == "narma":
            order = int(rng.integers(5, 15))
            a = np.array(
                [
                    float(rng.uniform(0.6, 0.9)),
                    float(rng.uniform(0.02, 0.08)),
                    float(rng.uniform(1.0, 2.0)),
                    float(rng.uniform(0.05, 0.15)),
                ]
            )
            inits = np.array(rng.normal(0, 0.1, size=order))
            own = np.random.RandomState(42)
            rand_inits = own.uniform(0, 0.5, size=order)
            rands = np.concatenate((rand_inits, own.uniform(0, 0.5, size=len(times))))
            values = np.concatenate((inits, np.zeros(len(times))))
            n = order
            for t in range(order, len(values)):
                values[t] = a[0] * values[t - 1] + a[1] * values[t - 1] * np.sum(values[t - n : n]) + a[2] * rands[t - n] * rands[t] + a[3]
            return values[order:]
        if kind == "pseudoperiodic":
            p = int(rng.integers(6, max(12, L // 10)))
            amplitude = float(rng.uniform(0.5, 2.5))
            amp_sd = float(rng.uniform(0.05, 0.25))
            freq_sd = float(rng.uniform(0.01, 0.15) * p)
            ftype = rng.choice([np.sin, np.cos])
            freq_arr = legacy.normal(loc=int(p), scale=freq_sd, size=len(times))
            amp_arr = legacy.normal(loc=amplitude, scale=amp_sd, size=len(times))
            return np.multiply(amp_arr, ftype(np.multiply(freq_arr, times)))
        p = int(rng.integers(1, 4))
        phi = rng.uniform(-0.6, 0.6, size=p)
        while np.sum(np.abs(phi)) > 0.9:
            phi = rng.uniform(-0.6, 0.6, size=p)
        sigma = float(rng.uniform(0.1, 0.7))
        prev = [np.array([v], dtype=float) for v in rng.normal(0, 0.2, size=p)]
        ar_param = phi.tolist()
        ar_param.reverse()
        out = np.zeros(len(times))
        for i in range(len(times)):
            ar_value = [prev[j] * ar_param[j] for j in range(len(ar_param))]
            noise = legacy.normal(loc=0.0, scale=sigma, size=1)
            ar_value = np.sum(ar_value) + noise
            prev = prev[1:] + [ar_value]
            out[i] = ar_value[0]
        return out
