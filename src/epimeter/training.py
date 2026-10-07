"""Training interface for epiplexity estimators.

Stage 4 fills this in: a train loop with Adam, a learning rate scaled by
1/fan-in, a constant rate and an EMA of the weights (shared protocol in
``plan/README.md``).
"""

from typing import Any


def train(model: Any, train_loader: Any, **kwargs: Any) -> Any:
    """Train ``model`` on ``train_loader``; returns the trained model.

    Raises ``NotImplementedError`` until the training loop is implemented.
    """
    raise NotImplementedError
