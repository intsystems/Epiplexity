"""Example observer wired through the registry.

Demonstrates the pluggability contract (stage 3): a network is defined outside
the package core and becomes available via ``epimeter.build("mlp", ...)`` after
``import`` of this module. Real MLP/CNN/DiT observers live in ``experiments/``
and ``notebooks/``; nothing in ``src/epimeter`` has to change to add one.
"""

import torch

from epimeter.models.base import BaseModel
from epimeter.models.registry import register


@register("mlp")
class MLP(BaseModel):
    """A minimal two-layer MLP observer for 784-d inputs and 10 classes."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
