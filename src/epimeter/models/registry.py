"""Registry for observer models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_TYPE = type[Any]

_REGISTRY: dict[str, _TYPE] = {}


def register(name: str) -> Callable[[_TYPE], _TYPE]:
    """Decorator registering an observer class under ``name``."""

    def _decorator(cls: _TYPE) -> _TYPE:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build(name: str, **config: Any) -> Any:
    """Instantiate an observer by name.

    Raises ``NotImplementedError`` while the registry has no observers (later
    stages register MLP/CNN/DiT through ``@register``).
    """
    if name not in _REGISTRY:
        raise NotImplementedError(f"No observer registered under {name!r}.")
    return _REGISTRY[name](**config)
