"""Оценка прогнозов на GIFT-Eval с сохранением метрик по окнам."""

from .forecasters import Forecast, Forecaster, MoiraiForecaster, SeasonalNaive
from .gift_eval import SHORT_CONFIGS, GiftDataset, TestInstance
from .metrics import aggregate, window_stats

__all__ = [
    "SHORT_CONFIGS",
    "Forecast",
    "Forecaster",
    "GiftDataset",
    "MoiraiForecaster",
    "SeasonalNaive",
    "TestInstance",
    "aggregate",
    "window_stats",
]
