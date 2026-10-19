r"""Прогнозчики для оценки: сезонный наивный и Moirai в двух протоколах.

Прогнозчик получает истории окон и горизонт и возвращает сэмплы прогноза, а если умеет —
минус лог-плотность правды. Протокол Moirai `"ours"` — форма задачи обучения: последние
`context_max` точек истории, ведущие NaN отбрасываются как отсутствующая история, внутренние
заполняются линейной интерполяцией, прогноз на `horizon_max` точек, из которых берутся первые
$h$. Протокол `"notebook"` повторяет `MoiraiForecast` из тетради GIFT-Eval: история до 4000
точек, токены левее её начала — заполнение вне внимания, NaN внутри истории как
ненаблюдаемые без заполнения, индексы времени от первого наблюдаемого патча,
$\lceil h / P \rceil$ прогнозных токенов.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from ..model.moirai import MoiraiModule


@dataclass
class Forecast:
    """Прогноз батча окон.

    Attributes:
        samples: Сэмплы `(n, B, h)` float32.
        nll: Минус лог-плотность правды `(B, h)` float32 или `None`, если правда не передана или плотности нет.
        context_length: Сколько точек истории использовано `(B,)`.
    """

    samples: np.ndarray
    nll: np.ndarray | None
    context_length: np.ndarray


class Forecaster(ABC):
    """Прогнозчик окон одномерных рядов."""

    @abstractmethod
    def predict(self, histories: list[np.ndarray], horizon: int, labels: list[np.ndarray] | None = None) -> Forecast:
        """Прогноз для списка историй на `horizon` точек; `labels` нужны только для лог-плотности."""


class SeasonalNaive(Forecaster):
    """Повтор последнего сезона истории; при истории короче сезона — последнее значение.

    NaN в истории заполняются последним наблюдённым значением. Один сэмпл, так что все
    квантили равны точечному прогнозу.

    Args:
        season: Сезонность.
    """

    def __init__(self, season: int) -> None:
        self.season = season

    def predict(self, histories, horizon, labels=None) -> Forecast:
        """Прогноз `(1, B, h)`."""
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
    """Прогноз моделью Moirai сэмплированием из предсказательной смеси.

    Args:
        module: Модель.
        patch_size: Размер патча из конфигурации модели.
        protocol: `"ours"` или `"notebook"`.
        context_max: Наибольшая длина истории: 512 в нашем протоколе, 4000 в тетради.
        horizon_max: Длина прогноза модели в нашем протоколе; кратна патчу.
        n_samples: Число сэмплов.
        batch_size: Окон на батч.
        device: Устройство.
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
        assert protocol in {"ours", "notebook"}
        self.module = module.to(device).eval()
        self.patch_size = patch_size
        self.protocol = protocol
        self.context_max = context_max
        self.horizon_max = horizon_max
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.device = device

    def predict(self, histories, horizon, labels=None) -> Forecast:
        """Сэмплы `(n, B, h)` и, при `labels`, минус лог-плотность `(B, h)`."""
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
