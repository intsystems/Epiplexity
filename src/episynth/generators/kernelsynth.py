"""Гауссовские процессы с составными ядрами (KernelSynth)."""

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

from .base import Generator

_PERIODS = [24, 48, 96, 24 * 7, 48 * 7, 96 * 7, 7, 14, 30, 60, 365, 365 * 2, 4, 26, 52, 4, 6, 12, 4, 4 * 10, 10]
_NON_PERIODIC = [
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    Matern(nu=0.5, length_scale=0.1),
    Matern(nu=0.5, length_scale=1.0),
    Matern(nu=0.5, length_scale=10.0),
    Matern(nu=1.5, length_scale=0.1),
    Matern(nu=1.5, length_scale=1.0),
    Matern(nu=1.5, length_scale=10.0),
    Matern(nu=2.5, length_scale=0.1),
    Matern(nu=2.5, length_scale=1.0),
    Matern(nu=2.5, length_scale=10.0),
    WhiteKernel(noise_level=0.1),
    ConstantKernel(),
]
_JITTERS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


class KernelSynth(Generator):
    r"""Гауссовский процесс с ядром из 1–5 базовых (сумма с вероятностью 0,8, иначе произведение) и случайным средним.

    Банк из 41 ядра на сетке $[0, 1]$: 21 периодическое с периодами в точках ряда и 20
    непериодических (линейные, RBF, рациональные квадратичные, Матерна, белый шум,
    константа). Среднее с вероятностью 0,5 нулевое, иначе линейное, квадратичное,
    синусоидальное, логарифмическое или случайное блуждание. Периоды заданы в точках,
    поэтому при другой длине меняется число периодов на ряд.

    Матрицы базовых ядер считаются один раз (41 матриц $L \times L$, 344 МБ при
    $L = 1024$), ковариация ряда собирается из них поэлементно. Гауссовский вектор
    строится разложением Холецкого пакетом по `chunk` рядов с лестницей добавок к
    диагонали от $10^{-12}$ до $10^{-4}$ средней дисперсии (добавка есть белый шум той же
    величины; $10^{-6}$ уже заметно меняет автокорреляцию разностей гладких рядов), при
    неудаче — собственным разложением, как у оригинала. Оригинал берёт один сид и для
    состава ядра, и для гауссовского вектора, так что первые числа вектора связаны с выбором
    ядра; в основном режиме потоки разведены через `SeedSequence`, в режиме воспроизведения
    сохранён оригинал с `multivariate_normal(method="eigh")`. Собственное разложение зависит
    от сборки LAPACK, поэтому режим воспроизведения совпадает с оригиналом до бита только при
    той же сборке numpy; при другой сборке ряды те же по распределению, но другие.

    Args:
        length: Длина ряда.
        max_kernels: Наибольшее число базовых ядер в составе.
        chunk: Сколько рядов раскладывается за один вызов.
        device: Устройство torch для разложения; `None` — numpy на CPU.

    Attributes:
        stats: Сколько рядов построено на каждой ступени добавки и сколько ушло в собственное разложение.
    """

    name = "kernelsynth"

    def __init__(self, length: int = 1024, max_kernels: int = 5, chunk: int = 64, device: str | None = None) -> None:
        super().__init__(length)
        self.max_kernels = max_kernels
        self.chunk = chunk
        self.device = device
        self.x = np.linspace(0, 1, length)
        self._bank = [ExpSineSquared(periodicity=p / length) for p in _PERIODS] + _NON_PERIODIC
        self._mats = np.stack([k(self.x[:, None]) for k in self._bank])
        self._fallback_cov = (ConstantKernel() + WhiteKernel(noise_level=0.1))(self.x[:, None])
        self._mats_dev = None
        if device is not None:
            import torch

            self._mats_dev = torch.as_tensor(self._mats, device=device)
        self.stats = {"series": 0, **{f"jitter_{j:g}": 0 for j in _JITTERS}, "eigh": 0}

    def _mean(self, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > 0.5:
            return np.zeros(self.length)
        t = self.x
        kind = rng.choice(["linear", "quadratic", "sinusoidal", "log_linear", "random_walk"])
        if kind == "linear":
            return rng.uniform(-3, 3) * t + rng.uniform(-2, 2)
        if kind == "quadratic":
            return rng.uniform(-2, 2) * t**2 + rng.uniform(-2, 2) * t + rng.uniform(-1, 1)
        if kind == "sinusoidal":
            return rng.uniform(0.5, 3) * np.sin(2 * np.pi * rng.uniform(1, 8) * t + rng.uniform(0, 2 * np.pi))
        if kind == "log_linear":
            return rng.uniform(-3, 3) * np.log1p(rng.uniform(1, 20) * t) + rng.uniform(-1, 1)
        return np.cumsum(rng.normal(0, rng.uniform(0.01, 0.1), size=self.length))

    def _plan(self, rng: np.random.Generator):
        """Индексы базовых ядер, операции (True — сумма) и среднее, тем же порядком вызовов, что у оригинала."""
        k = rng.integers(1, self.max_kernels + 1)
        idx = rng.choice(len(self._bank), k, replace=True)
        ops = [bool(rng.random() < 0.8) for _ in range(k - 1)]
        return idx, ops, self._mean(rng)

    @staticmethod
    def _compose(mats, idx, ops):
        cov = mats[idx[0]]
        for j, add in zip(idx[1:], ops, strict=True):
            cov = cov + mats[j] if add else cov * mats[j]
        return cov

    def _eigh_sample(self, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        try:
            return rng.multivariate_normal(np.zeros(self.length), cov, method="eigh")
        except np.linalg.LinAlgError:
            return rng.multivariate_normal(np.zeros(self.length), self._fallback_cov, method="eigh")

    def _exact(self, seed: int) -> np.ndarray:
        idx, ops, mean = self._plan(np.random.default_rng(seed))
        return self._eigh_sample(self._compose(self._mats, idx, ops), np.random.default_rng(seed)) + mean

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        out = np.empty((len(seeds), self.length))
        for start in range(0, len(seeds), self.chunk):
            part = seeds[start : start + self.chunk]
            plans, z = [], np.empty((len(part), self.length))
            for i, seed in enumerate(part):
                s_plan, s_noise = np.random.SeedSequence(int(seed)).spawn(2)
                plans.append(self._plan(np.random.default_rng(s_plan)))
                z[i] = np.random.default_rng(s_noise).standard_normal(self.length)
            if self._mats_dev is not None:
                x = self._chol_torch(plans, z)
            else:
                x = self._chol_numpy(plans, z)
            out[start : start + len(part)] = x + np.stack([m for _, _, m in plans])
        self.stats["series"] += len(seeds)
        return out

    def _chol_numpy(self, plans, z: np.ndarray) -> np.ndarray:
        x = np.empty_like(z)
        for i, (idx, ops, _) in enumerate(plans):
            cov = self._compose(self._mats, idx, ops)
            scale = cov.diagonal().mean()
            eye = np.eye(self.length)
            for jit in _JITTERS:
                try:
                    chol = np.linalg.cholesky(cov + jit * scale * eye)
                except np.linalg.LinAlgError:
                    continue
                x[i] = chol @ z[i]
                self.stats[f"jitter_{jit:g}"] += 1
                break
            else:
                x[i] = self._eigh_sample(cov, np.random.default_rng(int(z[i, 0] * 1e6) & 0x7FFFFFFF))
                self.stats["eigh"] += 1
        return x

    def _chol_torch(self, plans, z: np.ndarray) -> np.ndarray:
        import torch

        cov = torch.stack([self._compose(self._mats_dev, idx, ops) for idx, ops, _ in plans])
        zt = torch.as_tensor(z, device=self.device)
        scale = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
        x = torch.empty_like(zt)
        todo = torch.arange(len(plans), device=self.device)
        added = torch.zeros_like(scale)
        for jit in _JITTERS:
            if len(todo) == 0:
                break
            step = jit * scale[todo] - added[todo]
            sub = cov[todo]
            sub.diagonal(dim1=-2, dim2=-1).add_(step[:, None])
            added[todo] += step
            cov[todo] = sub
            chol, info = torch.linalg.cholesky_ex(sub)
            ok = info == 0
            x[todo[ok]] = (chol[ok] @ zt[todo[ok], :, None]).squeeze(-1)
            self.stats[f"jitter_{jit:g}"] += int(ok.sum())
            todo = todo[~ok]
        res = x.cpu().numpy()
        for i in todo.tolist():
            idx, ops, _ = plans[i]
            res[i] = self._eigh_sample(self._compose(self._mats, idx, ops), np.random.default_rng(int(abs(z[i, 0]) * 1e6)))
            self.stats["eigh"] += 1
        return res
