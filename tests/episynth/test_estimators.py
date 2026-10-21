r"""Тесты плагинного интерфейса оценщиков эпиплексии."""

import pytest

from episynth.estimators import EpiplexityEstimator, available, build, register
from episynth.estimators.prequential import PrequentialEstimator
from episynth.estimators.registry import _REGISTRY


def test_builtin_estimators_registered():
    """Встроенные оценщики зарегистрированы под ожидаемыми именами."""
    assert "prequential" in available()
    assert "requential" in available()
    assert "prequential" in _REGISTRY
    assert "requential" in _REGISTRY


def test_build_returns_instance():
    """build создаёт экземпляр запрошенного оценщика."""
    est = build("prequential")
    assert isinstance(est, PrequentialEstimator)
    assert isinstance(est, EpiplexityEstimator)


def test_build_unknown_raises():
    """build неизвестного имени бросает KeyError с перечнем доступных."""
    with pytest.raises(KeyError, match="prequential"):
        build("no_such_estimator")


def test_register_rejects_duplicates():
    """Повторная регистрация того же имени бросает ValueError."""

    class _Dup(EpiplexityEstimator):
        name = "dupe"

    register("dupe")(_Dup)
    with pytest.raises(ValueError, match="already registered"):
        register("dupe")(_Dup)
    _REGISTRY.pop("dupe")


def test_requential_is_skeleton():
    """Requential-оценщик зарегистрирован, но его прогон ещё не реализован."""
    est = build("requential")
    assert est.name == "requential"
    with pytest.raises(NotImplementedError):
        est.run(None, None, None, None, "unused")  # type: ignore[arg-type]


def test_prequential_from_run_dir_uses_heldout(tmp_path):
    """from_run_dir считает площадь через heldout_ema_1.f32, а не через хвост прогона."""
    import numpy as np

    np.asarray([5.0, 4.0, 3.0, 2.0], dtype=np.float32).tofile(tmp_path / "window_loss.f32")
    np.asarray([2.0, 2.0, 1.0, 3.0], dtype=np.float32).tofile(tmp_path / "heldout_ema_1.f32")
    est = build("prequential", variant="ema")
    res = est.from_run_dir(tmp_path)
    # model part = sum_loss - N*l_fin = 14 - 4*2 = 6 нат
    import math

    assert res["n_windows"] == 4.0
    assert abs(res["total_bits"] - 14.0 / math.log(2)) < 1e-9
    assert abs(res["model_part_bits"] - 6.0 / math.log(2)) < 1e-9
    assert abs(res["data_part_bits"] - 8.0 / math.log(2)) < 1e-9
    with pytest.raises(ValueError):
        build("prequential", variant="bad")
