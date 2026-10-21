r"""Проверка обучения на ОУ: лосс модели на окнах против точного байесовского лосса.

Для генератора `ou` маргиналы при приоре на $\varphi$ считаются квадратурой по $u$ (средние
точки, `n_nodes` узлов). Два эталона в натах на окно, где $c$ — длина контекста, $h$ — горизонт:

- *последовательный*: $-\log M_0(x_{1:c+h}) + \log M_0(x_{1:c})$, каждая точка горизонта при всех
  предыдущих; наименьший достижимый лог-лосс для любой модели;
- *поточечный*: $-\sum_{t=1}^{h} \log \int p(x_{c+t} \mid x_c, \varphi)\, p(\varphi \mid x_{1:c})\, d\varphi$,
  каждая точка горизонта только при контексте, как в лоссе Moirai, где точки горизонта условно
  независимы при контексте; наименьший достижимый лог-лосс в форме модели.

Оба считаются на тех же окнах, что видит модель (после деквантования; шум ширины $10^{-3}$
масштаба меняет их меньше чем на $10^{-5}$ нат на точку). Обученная модель должна подойти
сверху к поточечному эталону; разность эталонов — цена независимости точек горизонта.

    python -m episynth.checks.ou_bayes --run runs/ou [--n-windows 20000] [--device cuda]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..generators.ou import OU
from ..model.moirai import MoiraiConfig, MoiraiModule
from ..training import window_nll
from ..windows import WindowBatch, WindowSpec, window_batch


def phi_nodes(n_nodes: int = 2000) -> np.ndarray:
    r"""Узлы квадратуры средних точек по $u$, переведённые в $\varphi = \sin u$."""
    h = (OU.u_max - OU.u_min) / n_nodes
    return np.sin(OU.u_min + h * (np.arange(n_nodes) + 0.5))


def prefix_loglik(x: np.ndarray, lengths: np.ndarray, phi: np.ndarray) -> np.ndarray:
    r"""$\log p_\varphi(x_{1:\ell})$ для окон `(B, T)` при длинах префиксов `(B,)` и узлах $\varphi$ `(K,)`: матрица `(B, K)`."""
    B = x.shape[0]
    sq = np.cumsum(x**2, axis=1)
    cross = np.cumsum(np.concatenate([np.zeros((B, 1)), x[:, 1:] * x[:, :-1]], axis=1), axis=1)
    idx = np.arange(B)
    s_all = sq[idx, lengths - 1]
    x1 = x[:, 0] ** 2
    a = s_all - x1
    b = cross[idx, lengths - 1]
    c = s_all - x[idx, lengths - 1] ** 2
    v = 1 - phi**2
    q = a[:, None] - 2 * phi[None] * b[:, None] + phi[None] ** 2 * c[:, None]
    return -0.5 * lengths[:, None] * math.log(2 * math.pi) - 0.5 * x1[:, None] - 0.5 * (lengths[:, None] - 1) * np.log(v)[None] - q / (2 * v[None])


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_marginal(x: np.ndarray, lengths: np.ndarray, phi: np.ndarray, chunk: int = 2048) -> np.ndarray:
    r"""$\log M_0(x_{1:\ell})$ для окон `(B, T)` при длинах префиксов `(B,)` и узлах $\varphi$ `(K,)` с равными весами."""
    return np.concatenate([_logsumexp(prefix_loglik(x[i : i + chunk], lengths[i : i + chunk], phi)) - math.log(len(phi)) for i in range(0, len(x), chunk)])


def _left_aligned(batch: WindowBatch) -> tuple[np.ndarray, np.ndarray]:
    """Значения окон, прижатые к левому краю, float64 `(B, c_max + h)`, и длины контекста `(B,)`."""
    vals = batch.target.reshape(len(batch), -1).double().numpy()
    obs = batch.observed_mask.reshape(len(batch), -1).numpy()
    left = np.zeros_like(vals)
    for i in range(len(batch)):
        left[i, : obs[i].sum()] = vals[i, obs[i]]
    return left, batch.context_length.numpy()


def exact_window_nll(batch: WindowBatch, spec: WindowSpec, phi: np.ndarray) -> np.ndarray:
    """Последовательный эталон в натах на окно: минус лог-плотность горизонта при контексте по цепному правилу."""
    left, lengths = _left_aligned(batch)
    full = log_marginal(left, lengths + spec.horizon, phi)
    ctx = log_marginal(left[:, : max(lengths)], lengths, phi)
    return -(full - ctx)


def pointwise_window_nll(batch: WindowBatch, spec: WindowSpec, phi: np.ndarray, chunk: int = 256) -> np.ndarray:
    r"""Поточечный эталон в натах на окно: каждая точка горизонта при одном контексте с апостериорным усреднением по $\varphi$."""
    left, lengths = _left_aligned(batch)
    h = spec.horizon
    idx = np.arange(len(batch))
    x_last = left[idx, lengths - 1]
    horizon = np.stack([left[i, c : c + h] for i, c in enumerate(lengths)])
    t = np.arange(1, h + 1)
    out = np.empty(len(batch))
    for i in range(0, len(batch), chunk):
        sl = slice(i, i + chunk)
        lw = prefix_loglik(left[sl, : max(lengths)], lengths[sl], phi)
        lw = lw - _logsumexp(lw)[:, None]
        mean = phi[None, :, None] ** t[None, None, :] * x_last[sl, None, None]
        var = 1 - phi[None, :, None] ** (2 * t[None, None, :])
        ll = -0.5 * np.log(2 * math.pi * var) - (horizon[sl, None, :] - mean) ** 2 / (2 * var)
        out[sl] = -_logsumexp(lw[:, :, None] + ll, axis=1).sum(1)
    return out


def model_window_nll(run: Path, batch: WindowBatch, spec: WindowSpec, device: str, use_ema: bool) -> np.ndarray:
    """Лосс окон батча у финального чекпойнта прогона, float32 без autocast, в натах на окно."""
    state = torch.load(run / "ckpt_1.pt", map_location=device, weights_only=False)
    model = MoiraiModule(MoiraiConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    if use_ema:
        with torch.no_grad():
            for p, e in zip(model.parameters(), state["ema"], strict=True):
                p.copy_(e.to(device))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(batch), 1024):
            part = batch.slice(i, i + 1024).to(device)
            out.append(window_nll(model(**part.model_inputs(), patch_size=spec.patch_size), part, spec.patch_size).double().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    """Сравнение лосса прогона на ОУ с точным байесовским на свежих окнах из отложенного диапазона сидов."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-windows", type=int, default=20000)
    parser.add_argument("--n-nodes", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["generator"] == "ou", f"прогон на {cfg['generator']}, а не на ou"
    spec = WindowSpec(**cfg["spec"])
    first = cfg["train"]["heldout_first_seed"]
    seeds = np.arange(first, first + args.n_windows)
    batch = window_batch(OU(length=spec.series_length), seeds, spec)
    phi = phi_nodes(args.n_nodes)
    h = spec.horizon
    sequential = exact_window_nll(batch, spec, phi)
    pointwise = pointwise_window_nll(batch, spec, phi)
    res = {
        "n_windows": args.n_windows,
        "sequential_nats_per_point": float(sequential.mean() / h),
        "pointwise_nats_per_point": float(pointwise.mean() / h),
        "pointwise_se": float(pointwise.std() / math.sqrt(len(pointwise)) / h),
    }
    for name, use_ema in (("raw", False), ("ema", True)):
        nll = model_window_nll(run, batch, spec, args.device, use_ema)
        gap = nll - pointwise
        res[name] = {"nats_per_point": float(nll.mean() / h), "gap_to_pointwise": float(gap.mean() / h), "gap_se": float(gap.std() / math.sqrt(len(gap)) / h)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
