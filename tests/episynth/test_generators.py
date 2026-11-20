"""Инварианты генераторов: форма, детерминизм, независимость рядов от состава пакета."""

import hashlib

import numpy as np
import pytest

from episynth.generators import GENERATORS
from episynth.generators.arima import ARIMA, _sample_ar, _sample_ma

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


def test_arima_legacy_preserves_seeded_output_bits():
    """The default retains the exact float32 stream recorded before adding corrected mode."""
    default = ARIMA(length=256).generate(SEEDS)
    assert hashlib.sha256(default.astype("<f4").tobytes()).hexdigest() == (
        "d9a35b725e6feb9478d000cafb758dddb109dc99c75be38e676a464a5c1dc0e8"
    )
    legacy = ARIMA(length=256, coefficient_mode="legacy")
    np.testing.assert_array_equal(default, legacy.generate(SEEDS))
    np.testing.assert_array_equal(default, legacy.generate(SEEDS, exact=True))


@pytest.mark.parametrize("sample", [_sample_ar, _sample_ma])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_arima_legacy_uses_small_fallback_distribution(sample, order):
    """The root-product obstruction rejects every legacy candidate before the fallback draw."""
    for seed in range(8):
        expected_rng = np.random.default_rng(seed)
        candidates = expected_rng.uniform(-0.8, 0.8, size=(100, order))
        # A monic polynomial's root-product modulus is its final coefficient's
        # modulus, so all roots cannot exceed 1.05 for any such candidate.
        assert (np.abs(candidates[:, -1]) < 1.05**order).all()
        expected = expected_rng.uniform(-0.3, 0.3, size=order)
        actual_rng = np.random.default_rng(seed)
        actual = sample(order, actual_rng, "legacy")
        np.testing.assert_array_equal(actual, expected)
        assert actual_rng.random() == expected_rng.random()


@pytest.mark.parametrize("sample,sign", [(_sample_ar, -1), (_sample_ma, 1)])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_arima_corrected_coefficients_have_valid_lag_roots(sample, sign, order):
    for seed in range(64):
        coefficients = sample(order, np.random.default_rng(seed), "corrected")
        assert np.isfinite(coefficients).all()
        # Polynomial takes coefficients in ascending lag order, independently
        # checking AR stationarity / MA invertibility and the 1.05 margin.
        roots = np.polynomial.Polynomial(np.r_[1.0, sign * coefficients]).roots()
        assert (np.abs(roots) > 1.05).all()
        np.testing.assert_array_equal(coefficients, sample(order, np.random.default_rng(seed), "corrected"))


def test_arima_corrected_mode_is_finite_deterministic_and_batch_independent():
    generator = ARIMA(length=256, coefficient_mode="corrected")
    full = generator.generate(SEEDS)
    assert full.shape == (len(SEEDS), 256) and np.isfinite(full).all()
    np.testing.assert_array_equal(full, generator.generate(SEEDS))
    np.testing.assert_array_equal(full, generator.generate(SEEDS, exact=True))
    np.testing.assert_array_equal(full[::-1], generator.generate(SEEDS[::-1]))
    assert not np.array_equal(full, ARIMA(length=256).generate(SEEDS))
    with pytest.raises(ValueError, match="coefficient_mode"):
        ARIMA(coefficient_mode="unknown")
