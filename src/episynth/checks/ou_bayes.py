r"""Compare model loss on OU windows with the exact Bayesian reference.

For the `ou` generator, integrate over the prior on :math:`\varphi` using
midpoint quadrature on :math:`u` with `n_nodes` nodes. Both references use
nats per window, with context length :math:`c` and horizon :math:`h`.

- *Joint*: :math:`-\log M_0(x_{1:c+h}) + \log M_0(x_{1:c})`. Each horizon point
  conditions on all preceding points. This is the minimum achievable log loss.
- *Pointwise*: :math:`-\sum_{t=1}^{h} \log \int p(x_{c+t} \mid x_c, \varphi)\, p(\varphi \mid x_{1:c})\, d\varphi`.
  Each horizon point conditions on the context alone, as in the conditional
  independence factorization used by Moirai. This is the minimum for that factorization.

Both references use the model's dequantized windows. Noise at :math:`10^{-3}`
times the scale changes them by less than :math:`10^{-5}` nats per point.
A trained model should approach the pointwise reference from above. The gap
between the two references is the cost of treating horizon points as independent.

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
    r"""Midpoint squaring nodes in :math:`u`, translated into :math:`\varphi = \sin u`."""
    h = (OU.u_max - OU.u_min) / n_nodes
    return np.sin(OU.u_min + h * (np.arange(n_nodes) + 0.5))


def prefix_loglik(x: np.ndarray, lengths: np.ndarray, phi: np.ndarray) -> np.ndarray:
    r""":math:`\log p_\varphi(x_{1:\ell})` for `(B, T)` windows with `(B,)` prefix lengths and nodes :math:`\varphi` `(K,)`: `(B, K)` matrix."""
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
    r""":math:`\log M_0(x_{1:\ell})` for `(B, T)` windows with `(B,)` prefix lengths and :math:`\varphi` `(K,)` nodes with equal weights."""
    return np.concatenate([_logsumexp(prefix_loglik(x[i : i + chunk], lengths[i : i + chunk], phi)) - math.log(len(phi)) for i in range(0, len(x), chunk)])


def _left_aligned(batch: WindowBatch) -> tuple[np.ndarray, np.ndarray]:
    """The window values pressed to the left edge, float64 `(B, c_max + h)`, and the context length of `(B,)`."""
    vals = batch.target.reshape(len(batch), -1).double().numpy()
    obs = batch.observed_mask.reshape(len(batch), -1).numpy()
    left = np.zeros_like(vals)
    for i in range(len(batch)):
        left[i, : obs[i].sum()] = vals[i, obs[i]]
    return left, batch.context_length.numpy()


def exact_window_nll(batch: WindowBatch, spec: WindowSpec, phi: np.ndarray) -> np.ndarray:
    """Sequential standard in nats on the window: minus the log-density of the horizon in the context of the chain rule."""
    left, lengths = _left_aligned(batch)
    full = log_marginal(left, lengths + spec.horizon, phi)
    ctx = log_marginal(left[:, : max(lengths)], lengths, phi)
    return -(full - ctx)


def pointwise_window_nll(batch: WindowBatch, spec: WindowSpec, phi: np.ndarray, chunk: int = 256) -> np.ndarray:
    r"""Point standard in nats per window: each point of the horizon under the same context with a posteriori averaging by :math:`\varphi`."""
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
    """Loss of batch windows at the final checkpoint run, float32 without autocast, in nats at the window."""
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
    """Comparison of the Loss run on the OU with the exact Bayesian on fresh windows from the delayed range of seed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--n-windows", type=int, default=20000)
    parser.add_argument("--n-nodes", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["generator"] == "ou", f"Run uses {cfg['generator']}instead of ou"
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
