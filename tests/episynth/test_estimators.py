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
