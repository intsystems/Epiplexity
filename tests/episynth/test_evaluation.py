"""Оценка: метрики по окнам, сведение, наив, прогнозчик Moirai, загрузка GIFT-Eval при наличии снимка."""

import os

import numpy as np
import pytest

from episynth.evaluation.forecasters import MoiraiForecaster, SeasonalNaive
from episynth.evaluation.gift_eval import GiftDataset, config_name, seasonality
from episynth.evaluation.metrics import aggregate, crps_from_samples, seasonal_error, window_stats
from episynth.model.moirai import MoiraiConfig
from episynth.training import build_model

DATA = os.environ.get("GIFT_EVAL_DIR")


def test_seasonality_and_config_names():
    """Сезонность по правилу gluonts и имена конфигураций как в лидерборде."""
    assert seasonality("15T") == 96 and seasonality("H") == 24 and seasonality("D") == 1 and seasonality("7T") == 1
    assert config_name("m4_yearly", "A-DEC", "short") == "m4_yearly/A/short"
    assert config_name("saugeenday/D", "D", "short") == "saugeen/D/short"
    assert config_name("bizitobs_application", "10S", "short") == "bizitobs_application/10S/short"


def test_crps_and_seasonal_error_closed_forms():
    """CRPS по одному сэмплу равен модулю ошибки, по двум — средней ошибке минус четверть их разности."""
    y = np.array([1.0, -2.0])
    assert np.allclose(crps_from_samples(np.array([[0.5, 0.0]]), y), [0.5, 2.0])
    two = np.array([[0.0, 0.0], [2.0, 1.0]])
    expected = (np.abs(two - y).mean(0)) - np.abs(two[0] - two[1]) / 4
    assert np.allclose(crps_from_samples(two, y), expected)
    se, se_sq = seasonal_error(np.array([1.0, 2.0, 4.0, 7.0]), 2)
    assert se == 4.0 and se_sq == 17.0
    assert seasonal_error(np.array([1.0, 2.0]), 5)[0] == 1.0


def test_window_stats_and_aggregate_handle_nan_and_zero_seasonal_error():
    """NaN в правде не считаются, окна с нулевой сезонной ошибкой выпадают из масштабированных метрик."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((50, 4)) + 1.0
    label = np.array([1.0, np.nan, 2.0, 0.0])
    r1 = window_stats(label, samples, np.array([0.0, 1.0, 3.0, 2.0]), 1, nll=np.ones(4), context_length=4)
    assert r1["n_valid"] == 3 and r1["sum_nll"] == 3 and r1["seasonal_error"] > 0
    r2 = window_stats(label, samples, np.full(5, 3.0), 1)
    agg = aggregate([r1, r2])
    assert agg["n_excluded"] == 1 and agg["n_windows"] == 2 and agg["n_points"] == 6
    median = np.quantile(samples, 0.5, axis=0)
    manual = np.abs(label - median)[[0, 2, 3]].sum() / r1["seasonal_error"] / 3
    assert np.isclose(agg["MASE[0.5]"], manual)
    assert 0 <= agg["coverage_0.9"] <= 1


def test_seasonal_naive_forecast():
    """Повтор последнего сезона, короткая история даёт последнее значение, NaN заполняются вперёд."""
    fc = SeasonalNaive(2).predict([np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0]), np.array([1.0, np.nan, 3.0, np.nan])], 3)
    assert fc.samples.shape == (1, 3, 3)
    assert fc.samples[0, 0].tolist() == [3.0, 4.0, 3.0] and fc.samples[0, 1].tolist() == [5.0] * 3
    assert fc.samples[0, 2].tolist() == [3.0, 3.0, 3.0]


def test_moirai_forecaster_both_protocols():
    """Прогнозчик даёт сэмплы и лог-плотность нужной формы при NaN и коротких историях в обоих протоколах."""
    model = build_model(MoiraiConfig(d_model=64, num_layers=1, patch_sizes=(16,), components=("student_t", "normal_fixed")), 0)
    hist = [np.sin(np.arange(300) / 5.0), np.array([1.0, 2.0, np.nan, 4.0, 5.0]), np.full(40, np.nan)]
    labels = [np.ones(10), np.array([1.0, np.nan] * 5), np.ones(10)]
    for protocol, ctx in (("ours", 128), ("notebook", 100)):
        fc = MoiraiForecaster(model, 16, protocol, context_max=ctx, horizon_max=32, n_samples=7, batch_size=2).predict(hist, 10, labels)
        assert fc.samples.shape == (7, 3, 10) and np.isfinite(fc.samples).all()
        assert fc.nll.shape == (3, 10) and np.isfinite(fc.nll[0]).all()
        assert fc.context_length.tolist()[:2] == [ctx if protocol == "ours" else 100, 5]


@pytest.mark.skipif(DATA is None, reason="GIFT_EVAL_DIR не задан")
def test_gift_dataset_windows():
    """Снимок читается, число окон и горизонт как у gift_eval.data.Dataset."""
    ds = GiftDataset(DATA, "m4_hourly")
    assert ds.windows == 1 and ds.prediction_length == 48 and ds.season == 24 and len(ds.series) == 414
    inst = next(ds.test_instances())
    assert len(inst.label) == 48 and len(inst.history) + 48 == len(ds.series[0][1])
    ds = GiftDataset(DATA, "bizitobs_l2c/H")
    assert ds.target_dim > 1 and ds.series[0][0].endswith("_dim0") and ds.config == "bizitobs_l2c/H/short"
