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
        """Те же ряды, что `_one`, но рекурсии CAR, AR, NARMA и красного шума считаются пакетом по рядам.

        Первый проход по рядам делает все обращения к генераторам случайных чисел в порядке
        оригинала и запоминает параметры; рекурсии по времени идут векторно по рядам (шумы
        рекурсий берутся заранее одним вектором, поток `RandomState` от этого не меняется);
        второй проход собирает ряд из компонент в исходном порядке операций.
        """
        plans = [self._plan(int(seed), self.irregular_prob) for seed in seeds]
        rec: dict[int, list] = {}
        for i, plan in enumerate(plans):
            for j, comp in enumerate(plan["signals"]):
                if comp["kind"] in ("car", "narma", "autoregressive"):
                    rec.setdefault(i, []).append(j)
        self._solve_car([(plans[i], j) for i in rec for j in rec[i] if plans[i]["signals"][j]["kind"] == "car"])
        self._solve_ar([(plans[i], j) for i in rec for j in rec[i] if plans[i]["signals"][j]["kind"] == "autoregressive"])
        self._solve_narma([(plans[i], j) for i in rec for j in rec[i] if plans[i]["signals"][j]["kind"] == "narma"])
        self._solve_red([plan for plan in plans if plan["noise"]["kind"] == "red"])
        out = np.empty((len(seeds), self.length))
        for i, plan in enumerate(plans):
            out[i] = self._assemble(plan)
        return out

    def _plan(self, seed: int, irregular_prob: float) -> dict:
        """Все розыгрыши одного ряда в порядке `_one`; векторные сигналы считаются сразу, рекурсии откладываются."""
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
        T = len(times)
        signals = []
        for k, kind in enumerate(selected):
            comp: dict = {"kind": kind}
            if kind == "sinusoidal":
                p = int(rng.integers(4, max(8, L // 8)))
                amplitude = float(rng.uniform(0.5, 3.0))
                frequency = float((L / p) / (L / 1.0))
                ftype = rng.choice([np.sin, np.cos])
                comp["values"] = amplitude * ftype(2 * np.pi * frequency * times)
            elif kind == "car":
                comp["ar"] = float(rng.uniform(0.1, 0.95))
                comp["sigma"] = float(rng.uniform(0.1, 0.8))
                comp["prev"] = float(rng.uniform(-0.5, 0.5))
                comp["noise"] = legacy.normal(loc=0.0, scale=1.0, size=T - 1)
            elif kind == "narma":
                order = int(rng.integers(5, 15))
                comp["order"] = order
                comp["a"] = np.array([float(rng.uniform(0.6, 0.9)), float(rng.uniform(0.02, 0.08)), float(rng.uniform(1.0, 2.0)), float(rng.uniform(0.05, 0.15))])
                comp["inits"] = np.array(rng.normal(0, 0.1, size=order))
            elif kind == "pseudoperiodic":
                p = int(rng.integers(6, max(12, L // 10)))
                amplitude = float(rng.uniform(0.5, 2.5))
                amp_sd = float(rng.uniform(0.05, 0.25))
                freq_sd = float(rng.uniform(0.01, 0.15) * p)
                ftype = rng.choice([np.sin, np.cos])
                freq_arr = legacy.normal(loc=int(p), scale=freq_sd, size=T)
                amp_arr = legacy.normal(loc=amplitude, scale=amp_sd, size=T)
                comp["values"] = np.multiply(amp_arr, ftype(np.multiply(freq_arr, times)))
            else:
                p = int(rng.integers(1, 4))
                phi = rng.uniform(-0.6, 0.6, size=p)
                while np.sum(np.abs(phi)) > 0.9:
                    phi = rng.uniform(-0.6, 0.6, size=p)
                comp["sigma"] = float(rng.uniform(0.1, 0.7))
                comp["prev"] = np.array(rng.normal(0, 0.2, size=p), dtype=float)
                comp["phi_rev"] = phi.tolist()[::-1]
                comp["noise"] = legacy.normal(loc=0.0, scale=comp["sigma"], size=T)
            comp["w"] = float(rng.uniform(0.4, 1.2))
            comp["add"] = True if k == 0 else bool(rng.random() < 0.8)
            signals.append(comp)
        kind, std, tau = _NOISE_BANK[rng.choice(len(_NOISE_BANK))]
        noise: dict = {"kind": kind, "std": std, "tau": tau}
        if kind == "gaussian":
            noise["values"] = legacy.normal(0.0, std, size=T)
        else:
            noise["w"] = legacy.normal(0.0, std, size=T - 1)
        return {"times": times, "signals": signals, "noise": noise}

    @staticmethod
    def _dt_matrix(items: list, L: int) -> np.ndarray:
        """Шаги времени `(B, L-1)` рядов с дополнением нулями до общей длины."""
        dt = np.zeros((len(items), L - 1))
        for b, times in enumerate(items):
            d = np.diff(times)
            dt[b, : len(d)] = d
        return dt

    def _solve_car(self, items: list) -> None:
        """CAR: `prev = ar^dt * prev + sigma * sqrt(1 - ar^dt) * noise` векторно по рядам."""
        if not items:
            return
        L = self.length
        B = len(items)
        dt = self._dt_matrix([plan["times"] for plan, _ in items], L)
        ar = np.array([plan["signals"][j]["ar"] for plan, j in items])[:, None]
        sigma = np.array([plan["signals"][j]["sigma"] for plan, j in items])[:, None]
        noise = np.zeros((B, L - 1))
        for b, (plan, j) in enumerate(items):
            nz = plan["signals"][j]["noise"]
            noise[b, : len(nz)] = nz
        powered = np.power(ar, dt)
        coef = sigma * np.sqrt(1 - powered)
        out = np.zeros((B, L))
        prev = np.array([plan["signals"][j]["prev"] for plan, j in items])
        out[:, 0] = prev
        for t in range(1, L):
            prev = powered[:, t - 1] * prev + coef[:, t - 1] * noise[:, t - 1]
            out[:, t] = prev
        for b, (plan, j) in enumerate(items):
            plan["signals"][j]["values"] = out[b, : len(plan["times"])]

    def _solve_red(self, plans: list) -> None:
        """Красный шум: `prev = tau/(tau+dt) * (dt * w + prev)` векторно по рядам."""
        if not plans:
            return
        L = self.length
        B = len(plans)
        dt = self._dt_matrix([plan["times"] for plan in plans], L)
        tau = np.array([plan["noise"]["tau"] for plan in plans])[:, None]
        w = np.zeros((B, L - 1))
        for b, plan in enumerate(plans):
            wn = plan["noise"]["w"]
            w[b, : len(wn)] = wn
        k = tau / (tau + dt)
        out = np.zeros((B, L))
        prev = np.zeros(B)
        for t in range(1, L):
            prev = k[:, t - 1] * (dt[:, t - 1] * w[:, t - 1] + prev)
            out[:, t] = prev
        for b, plan in enumerate(plans):
            plan["noise"]["values"] = out[b, : len(plan["times"])]

    def _solve_ar(self, items: list) -> None:
        """AR(p), p = 1..3: сумма слагаемых в порядке оригинала, затем шум; векторно по рядам одного p."""
        for p in (1, 2, 3):
            group = [(plan, j) for plan, j in items if len(plan["signals"][j]["phi_rev"]) == p]
            if not group:
                continue
            T = self.length
            B = len(group)
            phi = np.array([plan["signals"][j]["phi_rev"] for plan, j in group])
            noise = np.stack([plan["signals"][j]["noise"] for plan, j in group])
            prev = [np.array([plan["signals"][j]["prev"][q] for plan, j in group]) for q in range(p)]
            out = np.zeros((B, T))
            for i in range(T):
                s = prev[0] * phi[:, 0]
                for q in range(1, p):
                    s = s + prev[q] * phi[:, q]
                value = s + noise[:, i]
                prev = prev[1:] + [value]
                out[:, i] = value
            for b, (plan, j) in enumerate(group):
                plan["signals"][j]["values"] = out[b]

    def _solve_narma(self, items: list) -> None:
        """NARMA порядка n: рекурсия оригинала с тем же срезом `values[t-n:n]` (ненулевым только при t < 2n)."""
        T = self.length
        for n in range(5, 15):
            group = [(plan, j) for plan, j in items if plan["signals"][j]["order"] == n]
            if not group:
                continue
            own = np.random.RandomState(42)
            rands = np.concatenate((own.uniform(0, 0.5, size=n), own.uniform(0, 0.5, size=T)))
            B = len(group)
            a = np.array([plan["signals"][j]["a"] for plan, j in group])
            values = np.zeros((B, n + T))
            slice_sum = np.zeros((B, 2 * n))
            for b, (plan, j) in enumerate(group):
                inits = plan["signals"][j]["inits"]
                values[b, :n] = inits
                for t in range(n, 2 * n):
                    slice_sum[b, t] = np.sum(inits[t - n : n])
            for t in range(n, n + T):
                s = slice_sum[:, t] if t < 2 * n else 0.0
                values[:, t] = a[:, 0] * values[:, t - 1] + a[:, 1] * values[:, t - 1] * s + a[:, 2] * rands[t - n] * rands[t] + a[:, 3]
            for b, (plan, j) in enumerate(group):
                plan["signals"][j]["values"] = values[b, n:]

    def _assemble(self, plan: dict) -> np.ndarray:
        """Сборка ряда из компонент и шума в порядке операций `_one`."""
        L = self.length
        times = plan["times"]
        combined = None
        for comp in plan["signals"]:
            values = np.asarray(comp["values"], dtype=np.float32)
            w = comp["w"]
            if combined is None:
                combined = w * values
            elif comp["add"]:
                combined = combined + w * values
            else:
                combined = combined * (1 + 0.05 * w * values)
        samples = combined + plan["noise"]["values"]
        if len(samples) != L:
            samples = np.interp(np.linspace(times[0], times[-1], L), times, samples)
        return np.asarray(samples, dtype=np.float32).astype(np.float64)

    def _exact(self, seed: int) -> np.ndarray:
        return self._one(seed, self.irregular_prob)

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
