r"""Register of Epiplexity Estimators.

The estimators are registered by name by the decorator :func:`register` and
created by the factory
:func:`build`. The user adds his algorithm with one module: determines the
class that implements
:class:`~episynth.estimators.base.EpiplexityEstimator`, and marks it with
``@register("name")``.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import EpiplexityEstimator

_REGISTRY: dict[str, type[EpiplexityEstimator]] = {}


def register(
    name: str,
) -> Callable[[type[EpiplexityEstimator]], type[EpiplexityEstimator]]:
    """Registers an estimator class under `name`.

    Args:
        name: The key to access the algorithm is :func:`build` and CLI.

    Returns:
        Decorator, returning the class unchanged.
    """

    def _decorator(
        cls: type[EpiplexityEstimator],
    ) -> type[EpiplexityEstimator]:
        if name in _REGISTRY:
            raise ValueError(f"Estimator {name!r} is already registered.")
        cls.name = name  # type: ignore[attr-defined]
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build(name: str, **config: object) -> EpiplexityEstimator:
    """Creates an estimator by name.

    Args:
        name: The key's in the register.
        config: Algorithm Designer Arguments (``**config``)

    Returns:
        An instance of the estimator.

    Raises:
        KeyError: If the algorithm with that name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No epiplexity estimator registered under {name!r}. Available:"
            f" {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name](**config)


def available() -> list[str]:
    """A sorted list of registered algorithm names."""
    return sorted(_REGISTRY)


def _import_estimators() -> None:
    """Imports built-in estimators so they register."""
    from . import (
        prequential,  # noqa: F401
        requential,  # noqa: F401
        v_information,  # noqa: F401
    )


_import_estimators()
