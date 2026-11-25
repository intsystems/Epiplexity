"""Gaussian processes with compound nuclei (KernelSynth)"""

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

_PERIODS = [
    24,
    48,
    96,
    24 * 7,
    48 * 7,
    96 * 7,
    7,
    14,
    30,
    60,
    365,
    365 * 2,
    4,
    26,
    52,
    4,
    6,
    12,
    4,
    4 * 10,
    10,
]
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
    r"""Gaussian process with a composition of base kernels and a random mean.

    Choose one to five kernels, combining them by addition with probability 0.8
    or by multiplication otherwise. The bank has 41 kernels on a grid over
    :math:`[0, 1]`. Of these, 21 are periodic with periods in series points and
    20 are nonperiodic (linear, RBF, rational quadratic, Matérn, white noise,
    or
    constant). The mean is zero with probability 0.5, and otherwise linear,
    quadratic, sinusoidal, logarithmic, or a random walk. The number of cycles
    per series changes with series length because periods are in points.

    The base matrices are computed once (41 matrices of shape :math:`L \times
    L`,
    344 MB at :math:`L=1024`). Each series covariance is composed elementwise.
    Sampling uses batched Cholesky factorization for `chunk` series, with
    diagonal jitter from :math:`10^{-12}` to :math:`10^{-4}` times the mean
    variance. This corresponds to white noise of the same variance.
    Even :math:`10^{-6}` visibly changes the autocorrelation of differences
    of smooth series. If Cholesky fails, an eigendecomposition is used as in
    the original. The original uses one seed for both kernel composition and
    the Gaussian vector, so the first vector draws are linked to the kernel.
    The default mode separates the streams with `SeedSequence`. Reproduction
    uses the original `multivariate_normal(method="eigh")`. Eigendecomposition
    depends on LAPACK, so reproduction matches bit for bit only with the same
    NumPy build. With another build the distribution is the same.

    Args:
        length: Series length.
        max_kernels: Maximum number of base kernels in a composition.
        chunk: Number of series factored in one batch.
        device: PyTorch device for factorization. `None` uses NumPy on CPU.

    Attributes:
        stats: Counts at each jitter level and the eigendecomposition fallback
            count.
    """

    name = "kernelsynth"

    def __init__(
        self,
        length: int = 1024,
        max_kernels: int = 5,
        chunk: int = 64,
        device: str | None = None,
    ) -> None:
        super().__init__(length)
        self.max_kernels = max_kernels
        self.chunk = chunk
        self.device = device
        self.x = np.linspace(0, 1, length)
        self._bank = [
            ExpSineSquared(periodicity=p / length) for p in _PERIODS
        ] + _NON_PERIODIC
        self._mats = np.stack([k(self.x[:, None]) for k in self._bank])
        self._fallback_cov = (ConstantKernel() + WhiteKernel(noise_level=0.1))(
            self.x[:, None]
        )
        self._mats_dev = None
        if device is not None:
            import torch

            self._mats_dev = torch.as_tensor(self._mats, device=device)
        self.stats = {
            "series": 0,
            **{f"jitter_{j:g}": 0 for j in _JITTERS},
            "eigh": 0,
        }

    def __getstate__(self) -> dict:
        """The condition without the kernel bank tensor on the device: it is
        rebuilt after deserialization.
        """
        state = dict(self.__dict__)
        state["_mats_dev"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores the generator and, with a given device, puts the bank of
        cores on it again.
        """
        self.__dict__.update(state)
        if self.device is not None:
            import torch

            self._mats_dev = torch.as_tensor(self._mats, device=self.device)

    def _mean(self, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > 0.5:
            return np.zeros(self.length)
        t = self.x
        kind = rng.choice(
            ["linear", "quadratic", "sinusoidal", "log_linear", "random_walk"]
        )
        if kind == "linear":
            return rng.uniform(-3, 3) * t + rng.uniform(-2, 2)
        if kind == "quadratic":
            return (
                rng.uniform(-2, 2) * t**2
                + rng.uniform(-2, 2) * t
                + rng.uniform(-1, 1)
            )
        if kind == "sinusoidal":
            return rng.uniform(0.5, 3) * np.sin(
                2 * np.pi * rng.uniform(1, 8) * t + rng.uniform(0, 2 * np.pi)
            )
        if kind == "log_linear":
            return rng.uniform(-3, 3) * np.log1p(
                rng.uniform(1, 20) * t
            ) + rng.uniform(-1, 1)
        return np.cumsum(
            rng.normal(0, rng.uniform(0.01, 0.1), size=self.length)
        )

    def _plan(self, rng: np.random.Generator):
        """Indices of base cores, operations (True – sum) and average, the same
        order of calls as the original.
        """
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

    def _eigh_sample(
        self, cov: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        try:
            return rng.multivariate_normal(
                np.zeros(self.length), cov, method="eigh"
            )
        except np.linalg.LinAlgError:
            return rng.multivariate_normal(
                np.zeros(self.length), self._fallback_cov, method="eigh"
            )

    def _exact(self, seed: int) -> np.ndarray:
        idx, ops, mean = self._plan(np.random.default_rng(seed))
        return (
            self._eigh_sample(
                self._compose(self._mats, idx, ops),
                np.random.default_rng(seed),
            )
            + mean
        )

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        out = np.empty((len(seeds), self.length))
        for start in range(0, len(seeds), self.chunk):
            part = seeds[start: start + self.chunk]
            plans, z = [], np.empty((len(part), self.length))
            for i, seed in enumerate(part):
                s_plan, s_noise = np.random.SeedSequence(int(seed)).spawn(2)
                plans.append(self._plan(np.random.default_rng(s_plan)))
                z[i] = np.random.default_rng(s_noise).standard_normal(
                    self.length
                )
            if self._mats_dev is not None:
                x = self._chol_torch(plans, z)
            else:
                x = self._chol_numpy(plans, z)
            out[start: start + len(part)] = x + np.stack(
                [m for _, _, m in plans]
            )
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
                x[i] = self._eigh_sample(
                    cov, np.random.default_rng(int(z[i, 0] * 1e6) & 0x7FFFFFFF)
                )
                self.stats["eigh"] += 1
        return x

    def _chol_torch(self, plans, z: np.ndarray) -> np.ndarray:
        import torch

        cov = torch.stack(
            [self._compose(self._mats_dev, idx, ops) for idx, ops, _ in plans]
        )
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
            res[i] = self._eigh_sample(
                self._compose(self._mats, idx, ops),
                np.random.default_rng(int(abs(z[i, 0]) * 1e6)),
            )
            self.stats["eigh"] += 1
        return res
