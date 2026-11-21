r"""Forecasters to evaluate: seasonal naive and Moirai in two protocols.

The forecaster gets the window histories and the horizon and returns the forecast samples, and if he
can, minus the truth log density. `"ours"` The form of the learning task: the latest `context_max`
points of history leading NaN Discarded as a missing history, the internals are filled with linear
interpolation, the prognosis `horizon_max` From which are taken the first :math:`h`Protocol.
`"notebook"` repeat `MoiraiForecast` notebook GIFT-Eval: History before 4000 Dots, tokens to the
left of the beginning — filling out, out of the spotlight. NaN within the history as unobservable
without filling, time indexes from the first observed patch, :math:`\lceil h / P \rceil` Predictive
tokens.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from ..model.moirai import MoiraiModule


@dataclass
class Forecast:
    """Forecasts for a batch of windows.

    Attributes:
        samples: Samples `(n, B, h)` float32.
        nll: Minus the truth log density of `(B, h)` float32 or `None` if the truth is not transmitted or there is no density.
        context_length: How many points of history are used by `(B,)`?
    """

    samples: np.ndarray
    nll: np.ndarray | None
    context_length: np.ndarray


class Forecaster(ABC):
    """One-dimensional window forecaster."""

    @abstractmethod
    def predict(self, histories: list[np.ndarray], horizon: int, labels: list[np.ndarray] | None = None) -> Forecast:
        """Prediction for a list of histories on `horizon` points; `labels` is only needed for log density."""


class SeasonalNaive(Forecaster):
    """Repeat the last season of history; with a shorter season, the last value.

    The NaN in history is filled with the last observed value. One sample, so all quantiles are
    equal to the point prediction.

    Args:
        season: Seasonal.
    """

    def __init__(self, season: int) -> None:
        self.season = season

    def predict(
        self, histories: list[np.ndarray], horizon: int,
        labels: list[np.ndarray] | None = None,
    ) -> Forecast:
        """`(1, B, h)` forecast."""
        out = np.empty((1, len(histories), horizon), dtype=np.float32)
        for i, h in enumerate(histories):
            y = _forward_fill(h.astype(np.float64))
            if len(y) >= self.season:
                tile = y[-self.season :]
                out[0, i] = np.tile(tile, math.ceil(horizon / self.season))[:horizon]
            else:
                out[0, i] = y[-1]
        return Forecast(out, None, np.array([len(h) for h in histories]))


def _forward_fill(y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    if mask.all():
        return y
    idx = np.where(mask, np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    filled = np.where(idx >= 0, y[np.maximum(idx, 0)], np.nan)
    first = np.flatnonzero(mask)
    if len(first):
        filled[: first[0]] = y[first[0]]
    return np.nan_to_num(filled, nan=0.0)


def _interpolate_nan(y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y)
    if mask.all():
        return y
    t = np.arange(len(y))
    return np.interp(t, t[mask], y[mask])


class MoiraiForecaster(Forecaster):
    """Forecast model Moirai sampled from a predictive mixture.

    Args:
        module: Model.
        patch_size: Patch size from the model configuration.
        protocol: `"ours"` or `"notebook"`.
        context_max: Longest history: 512 in our protocol, 4000 in the notebook.
        horizon_max: The length of the model forecast in our protocol; a multiple of the patch.
        n_samples: Number of samples.
        batch_size: Windows on batch.
        device: The device.
    """

    def __init__(
        self,
        module: MoiraiModule,
        patch_size: int,
        protocol: str = "ours",
        context_max: int = 512,
        horizon_max: int = 64,
        n_samples: int = 100,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        if protocol not in {"ours", "notebook"}:
            raise ValueError("protocol must be 'ours' or 'notebook'")
        for name, value in (("patch_size", patch_size), ("context_max", context_max),
                            ("n_samples", n_samples), ("batch_size", batch_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if patch_size not in module.config.patch_sizes:
            raise ValueError("patch_size is not supported by the model")
        if protocol == "ours":
            if not isinstance(horizon_max, int) or isinstance(horizon_max, bool) or horizon_max <= 0:
                raise ValueError("horizon_max must be a positive integer")
            if context_max % patch_size or horizon_max % patch_size:
                raise ValueError("context_max and horizon_max must be divisible by patch_size")
            if (context_max + horizon_max) // patch_size > module.config.max_seq_len:
                raise ValueError("context and forecast exceed the model's max_seq_len")
        self.module = module.to(device).eval()
        self.patch_size = patch_size
        self.protocol = protocol
        self.context_max = context_max
        self.horizon_max = horizon_max
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = device

    def predict(
        self, histories: list[np.ndarray], horizon: int,
        labels: list[np.ndarray] | None = None,
    ) -> Forecast:
        """Samples of `(n, B, h)` and, at `labels`, minus the `(B, h)` log density."""
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.protocol == "ours" and horizon > self.horizon_max:
            raise ValueError("horizon exceeds horizon_max for protocol 'ours'")
        n_pred = self.horizon_max // self.patch_size if self.protocol == "ours" else math.ceil(horizon / self.patch_size)
        if math.ceil(self.context_max / self.patch_size) + n_pred > self.module.config.max_seq_len:
            raise ValueError("context and forecast exceed the model's max_seq_len")
        samples, nlls, lengths = [], [], []
        for start in range(0, len(histories), self.batch_size):
            hs = histories[start : start + self.batch_size]
            ls = None if labels is None else labels[start : start + self.batch_size]
            s, nll, n = self._predict_batch(hs, horizon, ls)
            samples.append(s)
            nlls.append(nll)
            lengths.append(n)
        nll = None if labels is None else np.concatenate(nlls, axis=0)
        return Forecast(np.concatenate(samples, axis=1), nll, np.concatenate(lengths))

    def _build(self, histories: list[np.ndarray], horizon: int, labels: list[np.ndarray] | None):
        P = self.patch_size
        B = len(histories)
        if self.protocol == "ours":
            ctx_len = self.context_max
            n_pred = self.horizon_max // P
        else:
            ctx_len = math.ceil(self.context_max / P) * P
            n_pred = math.ceil(horizon / P)
        total = ctx_len + n_pred * P
        values = np.zeros((B, total), dtype=np.float32)
        observed = np.zeros((B, total), dtype=bool)
        in_series = np.zeros((B, total), dtype=bool)
        in_series[:, ctx_len:] = True
        lengths = np.zeros(B, dtype=np.int64)
        for i, h in enumerate(histories):
            y = h.astype(np.float64)[-self.context_max :]
            if self.protocol == "ours":
                fin = np.flatnonzero(np.isfinite(y))
                y = _interpolate_nan(y[fin[0] :]) if len(fin) else y[:0]
                obs = np.ones(len(y), dtype=bool)
            else:
                obs = np.isfinite(y)
                y = np.where(obs, y, 0.0)
            lengths[i] = len(y)
            if len(y):
                values[i, ctx_len - len(y) : ctx_len] = y
                observed[i, ctx_len - len(y) : ctx_len] = obs
                in_series[i, ctx_len - len(y) : ctx_len] = True
            if labels is not None:
                lab = labels[i].astype(np.float64)
                values[i, ctx_len : ctx_len + len(lab)] = np.where(np.isfinite(lab), lab, 0.0)
                observed[i, ctx_len : ctx_len + len(lab)] = np.isfinite(lab)
        S = total // P
        target = torch.from_numpy(values.reshape(B, S, P))
        observed_mask = torch.from_numpy(observed.reshape(B, S, P))
        prediction_mask = (torch.arange(S) >= ctx_len // P).expand(B, S).clone()
        sample_id = torch.from_numpy(in_series.reshape(B, S, P).any(-1)).to(torch.int64)
        if self.protocol == "ours":
            time_id = torch.arange(S, dtype=torch.int64).expand(B, S).clone()
        else:
            past_obs = observed_mask[:, : ctx_len // P].any(-1).to(torch.int64)
            past_id = torch.clamp(past_obs.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0)
            future_id = torch.arange(n_pred).expand(B, n_pred) + past_id.max(dim=-1, keepdim=True).values + 1
            time_id = torch.cat([past_id, future_id], dim=-1)
        batch = {
            "target": target,
            "observed_mask": observed_mask,
            "sample_id": sample_id,
            "time_id": time_id,
            "variate_id": torch.zeros(B, S, dtype=torch.int64),
            "prediction_mask": prediction_mask,
        }
        return {k: v.to(self.device) for k, v in batch.items()}, ctx_len // P, lengths

    @torch.no_grad()
    def _predict_batch(self, histories, horizon, labels):
        batch, n_ctx, lengths = self._build(histories, horizon, labels)
        mix = self.module(**batch, patch_size=self.patch_size)
        P = self.patch_size
        draws = mix.sample(self.n_samples)[:, :, n_ctx:, :P]
        samples = draws.reshape(self.n_samples, len(histories), -1)[:, :, :horizon].float().cpu().numpy()
        nll = None
        if labels is not None:
            lp = mix.log_prob(batch["target"])[:, n_ctx:, :P].reshape(len(histories), -1)[:, :horizon]
            nll = (-lp).float().cpu().numpy()
        return samples, nll, lengths
