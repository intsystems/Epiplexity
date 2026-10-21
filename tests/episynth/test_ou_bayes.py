"""Генератор ОУ и точный байесовский лосс окон."""

import math

import numpy as np

from episynth.checks.ou_bayes import exact_window_nll, log_marginal, phi_nodes
from episynth.generators import GENERATORS
from episynth.windows import WindowSpec, make_windows


def test_ou_generator_is_stationary_ar1():
    """Ряды ОУ стационарны с единичной дисперсией; автокорреляция лага 1 лежит в носителе приора."""
    gen = GENERATORS["ou"](length=1024)
    x = gen.generate(np.arange(400)).astype(np.float64)
    assert abs(x.var() - 1.0) < 0.1
    xc = x - x.mean(1, keepdims=True)
    r1 = (xc[:, 1:] * xc[:, :-1]).mean(1) / xc.var(1)
    assert r1.min() > -0.1 and r1.max() < 1.01 and r1.mean() > 0.3
    assert np.array_equal(gen.generate([5, 7]), gen.generate([7, 5])[::-1])


def test_single_node_matches_closed_form():
    """При одном узле квадратуры точный лосс горизонта равен сумме условных гауссовских лог-плотностей AR(1)."""
    phi = np.array([0.7])
    spec = WindowSpec(series_length=256, context_max=128, horizon=32, patch_size=16, context_lengths=(50, 128), dequantize_width=0.0)
    gen = GENERATORS["ou"](length=256)
    seeds = np.arange(20)
    batch = make_windows(gen.generate(seeds), seeds, spec)
    got = exact_window_nll(batch, spec, phi)
    vals = batch.target.reshape(20, -1).double().numpy()
    obs = batch.observed_mask.reshape(20, -1).numpy()
    v = 1 - phi[0] ** 2
    for i in range(20):
        w = vals[i, obs[i]]
        c = int(batch.context_length[i])
        resid = w[c:] - phi[0] * w[c - 1 : -1]
        expected = 0.5 * 32 * math.log(2 * math.pi * v) + (resid**2).sum() / (2 * v)
        assert abs(got[i] - expected) < 1e-8


def test_log_marginal_prefix_consistency():
    """Маргинал префикса длины 1 есть стандартная нормаль; квадратура даёт конечные значения."""
    x = np.random.default_rng(0).standard_normal((5, 10))
    lm = log_marginal(x, np.ones(5, dtype=int), phi_nodes(50))
    assert np.allclose(lm, -0.5 * math.log(2 * math.pi) - 0.5 * x[:, 0] ** 2)
    assert np.isfinite(log_marginal(x, np.full(5, 10), phi_nodes(50))).all()


def test_pointwise_bound_above_sequential_on_average():
    """Поточечный эталон в среднем выше последовательного; при одном узле он равен сумме гауссовских лог-плотностей точек при контексте."""
    from episynth.checks.ou_bayes import pointwise_window_nll

    spec = WindowSpec(series_length=256, context_max=128, horizon=32, patch_size=16, context_lengths=(50, 128), dequantize_width=0.0)
    gen = GENERATORS["ou"](length=256)
    seeds = np.arange(200)
    batch = make_windows(gen.generate(seeds), seeds, spec)
    phi = phi_nodes(200)
    seq, pw = exact_window_nll(batch, spec, phi), pointwise_window_nll(batch, spec, phi)
    assert pw.mean() > seq.mean() + 1.0
    one = np.array([0.8])
    got = pointwise_window_nll(batch, spec, one)
    vals = batch.target.reshape(200, -1).double().numpy()
    obs = batch.observed_mask.reshape(200, -1).numpy()
    for i in range(0, 200, 7):
        w = vals[i, obs[i]]
        c = int(batch.context_length[i])
        t = np.arange(1, 33)
        var = 1 - one[0] ** (2 * t)
        expected = (0.5 * np.log(2 * math.pi * var) + (w[c:] - one[0] ** t * w[c - 1]) ** 2 / (2 * var)).sum()
        assert abs(got[i] - expected) < 1e-8
