"""Инварианты генераторов: форма, детерминизм, независимость рядов от состава пакета."""

import numpy as np
import pytest

from episynth.generators import GENERATORS

SEEDS = np.array([3, 11, 27, 1_000_003, 2**31 - 1])


@pytest.mark.parametrize("name", list(GENERATORS))
def test_shape_dtype_finite(name):
    """Ряды имеют форму `(n, length)`, тип float32 и конечны."""
    gen = GENERATORS[name](length=256)
    x = gen.generate(SEEDS)
    assert x.shape == (len(SEEDS), 256) and x.dtype == np.float32
    assert np.isfinite(x).all()


@pytest.mark.parametrize("name", list(GENERATORS))
def test_deterministic_and_batch_independent(name):
    """Ряд зависит только от сида: повтор, поодиночке и в другом порядке дают те же ряды."""
    gen = GENERATORS[name](length=256)
    full = gen.generate(SEEDS)
    again = gen.generate(SEEDS)
    assert np.array_equal(full, again)
    single = np.concatenate([gen.generate([s]) for s in SEEDS])
    assert np.array_equal(full, single)
    perm = np.array([4, 0, 3, 1, 2])
    assert np.array_equal(gen.generate(SEEDS[perm]), full[perm])


@pytest.mark.parametrize("name", list(GENERATORS))
def test_exact_mode_runs(name):
    """Режим воспроизведения работает и даёт конечные ряды нужной формы."""
    gen = GENERATORS[name](length=256)
    x = gen.generate(SEEDS[:2], exact=True)
    assert x.shape == (2, 256) and np.isfinite(x).all()


def test_timesynth_batch_matches_rowwise():
    """Пакетные рекурсии TimeSynth совпадают с построчной реализацией до бита при регулярной и нерегулярной сетке."""
    from episynth.generators.timesynth import TimeSynth

    for prob in (0.0, 1.0, 0.2):
        gen = TimeSynth(length=256, irregular_prob=prob)
        seeds = np.arange(900, 960)
        batch = gen.generate(seeds)
        rows = np.stack([gen._one(int(s), prob) for s in seeds]).astype(np.float32)
        assert np.array_equal(batch, rows)
