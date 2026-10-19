"""Хаотические системы: координата аттрактора Лоренца и уравнение Макки–Гласса."""

import numpy as np

from .base import Generator

_BURN_IN = 2000


class Chaotic(Generator):
    """Координата системы Лоренца (RK4) или решение уравнения Макки–Гласса (Эйлер), поровну.

    Параметры систем, шаг интегрирования и начальные условия случайные; первые 2000 точек
    отбрасываются как переходный процесс. Шума наблюдения нет: ряды детерминированы при
    известных параметрах и начальных условиях. Интегрирование идёт пакетом по рядам
    поэлементно теми же операциями, что у оригинала, поэтому основной режим совпадает с
    режимом воспроизведения до бита.

    Args:
        length: Длина ряда.
    """

    name = "chaotic"

    @staticmethod
    def _draw(rng: np.random.Generator) -> dict:
        system = rng.choice(["lorenz", "mackey_glass"])
        if system == "lorenz":
            return {
                "system": system,
                "sigma": float(rng.uniform(8, 12)),
                "rho": float(rng.uniform(24, 30)),
                "beta": float(rng.uniform(2.0, 3.5)),
                "dt": float(rng.uniform(0.005, 0.02)),
                "x0": rng.uniform(-1, 1, size=3),
            }
        tau = float(rng.uniform(15, 30))
        n_mg = float(rng.choice([8, 9, 10, 11, 12]))
        beta = float(rng.uniform(0.15, 0.25))
        gamma = float(rng.uniform(0.05, 0.15))
        dt = float(rng.uniform(0.5, 2.0))
        hist = int(tau / dt) + 1
        return {
            "system": system,
            "n": n_mg,
            "beta": beta,
            "gamma": gamma,
            "dt": dt,
            "hist": hist,
            "history": 0.9 + rng.uniform(-0.1, 0.1, size=hist),
        }

    @staticmethod
    def _lorenz(sigma, rho, beta, dt, n_steps, x0) -> np.ndarray:
        """RK4 для системы Лоренца; параметры могут быть массивами по рядам."""
        x, y, z = (np.array(v, dtype=float) for v in x0)
        traj = np.empty((n_steps,) + np.shape(x) + (3,))
        for i in range(n_steps):
            traj[i, ..., 0], traj[i, ..., 1], traj[i, ..., 2] = x, y, z
            dx1 = sigma * (y - x)
            dy1 = x * (rho - z) - y
            dz1 = x * y - beta * z
            xm = x + 0.5 * dt * dx1
            ym = y + 0.5 * dt * dy1
            zm = z + 0.5 * dt * dz1
            dx2 = sigma * (ym - xm)
            dy2 = xm * (rho - zm) - ym
            dz2 = xm * ym - beta * zm
            xm2 = x + 0.5 * dt * dx2
            ym2 = y + 0.5 * dt * dy2
            zm2 = z + 0.5 * dt * dz2
            dx3 = sigma * (ym2 - xm2)
            dy3 = xm2 * (rho - zm2) - ym2
            dz3 = xm2 * ym2 - beta * zm2
            xe = x + dt * dx3
            ye = y + dt * dy3
            ze = z + dt * dz3
            dx4 = sigma * (ye - xe)
            dy4 = xe * (rho - ze) - ye
            dz4 = xe * ye - beta * ze
            x = x + dt / 6 * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
            y = y + dt / 6 * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
            z = z + dt / 6 * (dz1 + 2 * dz2 + 2 * dz3 + dz4)
        return traj

    @staticmethod
    def _mackey_glass(params: list[dict], n_steps: int) -> np.ndarray:
        m = len(params)
        hist = np.array([p["hist"] for p in params])
        dt = np.array([p["dt"] for p in params])
        beta = np.array([p["beta"] for p in params])
        gamma = np.array([p["gamma"] for p in params])
        n_exp = np.array([p["n"] for p in params])
        width = hist.max() + n_steps
        x = np.zeros((m, width))
        for i, p in enumerate(params):
            x[i, : p["hist"]] = p["history"]
        rows = np.arange(m)
        with np.errstate(all="ignore"):
            for i in range(hist.min(), width):
                active = (i >= hist) & (i < hist + n_steps)
                x_tau = x[rows, np.maximum(i - hist, 0)]
                prev = x[:, i - 1]
                new = prev + dt * (beta * x_tau / (1 + x_tau**n_exp) - gamma * prev)
                new = np.where(np.isfinite(new), new, prev)
                x[:, i] = np.where(active, new, x[:, i])
        out = np.empty((m, n_steps))
        for i in range(m):
            out[i] = x[i, hist[i] : hist[i] + n_steps]
        return out

    def _finish(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        L = self.length
        if len(series) > L:
            series = series[:L]
        elif len(series) < L:
            series = np.pad(series, (0, L - len(series)), mode="edge")
        if not np.all(np.isfinite(series)):
            series = rng.standard_normal(L)
        return series

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        n_steps = self.length + _BURN_IN
        rngs = [np.random.default_rng(int(s)) for s in seeds]
        params = [self._draw(r) for r in rngs]
        out = np.empty((len(seeds), self.length))
        lor = [i for i, p in enumerate(params) if p["system"] == "lorenz"]
        mg = [i for i, p in enumerate(params) if p["system"] != "lorenz"]
        if lor:
            sigma, rho, beta, dt = (np.array([params[i][key] for i in lor]) for key in ("sigma", "rho", "beta", "dt"))
            x0 = np.stack([params[i]["x0"] for i in lor], axis=1)
            traj = self._lorenz(sigma, rho, beta, dt, n_steps, x0)
            for j, i in enumerate(lor):
                coord = int(rngs[i].integers(0, 3))
                out[i] = self._finish(traj[_BURN_IN:, j, coord], rngs[i])
        if mg:
            raw = self._mackey_glass([params[i] for i in mg], n_steps)
            for j, i in enumerate(mg):
                out[i] = self._finish(raw[j, _BURN_IN:], rngs[i])
        return out
