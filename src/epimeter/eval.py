"""Evaluation interface for epiplexity estimators."""

from typing import Any


def evaluate(model: Any, eval_loader: Any, estimator: Any, **kwargs: Any) -> Any:
    """Compute ``estimator`` (model/data parts in bits) on ``eval_loader``.

    Raises ``NotImplementedError`` until evaluation is implemented.
    """
    raise NotImplementedError
