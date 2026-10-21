r"""Оценщики эпиплексии: плагинный интерфейс выбора и добавления алгоритмов.

Пользователь выбирает алгоритм по имени через :func:`build` или CLI
``python -m episynth.estimators.run --estimator <name>``. Новый алгоритм добавляется одним модулем:
определить класс, реализующий :class:`~episynth.estimators.base.EpiplexityEstimator`, и пометить его
``@register("name")``; остальные модули библиотеки менять не нужно.
"""

from .base import EpiplexityEstimator
from .registry import available, build, register

__all__ = ["EpiplexityEstimator", "register", "build", "available"]
