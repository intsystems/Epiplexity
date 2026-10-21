r"""Реестр оценщиков эпиплексии.

Оценщики регистрируются по имени декоратором :func:`register` и создаются фабрикой
:func:`build`. Пользователь добавляет свой алгоритм одним модулем: определяет класс, реализующий
:class:`~episynth.estimators.base.EpiplexityEstimator`, и помечает его ``@register("name")``.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import EpiplexityEstimator

_REGISTRY: dict[str, type[EpiplexityEstimator]] = {}


def register(name: str) -> Callable[[type[EpiplexityEstimator]], type[EpiplexityEstimator]]:
    """Регистрирует класс оценщика под `name`.

    Args:
        name: Ключ, по которому алгоритм будет доступен через :func:`build` и CLI.

    Returns:
        Декоратор, возвращающий класс без изменений.
    """

    def _decorator(cls: type[EpiplexityEstimator]) -> type[EpiplexityEstimator]:
        if name in _REGISTRY:
            raise ValueError(f"Estimator {name!r} is already registered.")
        cls.name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build(name: str, **config: object) -> EpiplexityEstimator:
    """Создаёт оценщик по имени.

    Args:
        name: Ключ в реестре.
        config: Аргументы конструктора алгоритма (``**config``).

    Returns:
        Экземпляр оценщика.

    Raises:
        KeyError: Если алгоритм с таким именем не зарегистрирован.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No epiplexity estimator registered under {name!r}. Available: {sorted(_REGISTRY)}.")
    return _REGISTRY[name](**config)


def available() -> list[str]:
    """Отсортированный список зарегистрированных имён алгоритмов."""
    return sorted(_REGISTRY)


def _import_estimators() -> None:
    """Импортирует встроенные оценщики, чтобы они зарегистрировались."""
    from . import (
        prequential,  # noqa: F401
        requential,  # noqa: F401
    )


_import_estimators()
