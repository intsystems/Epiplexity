"""Pluggability test (stage 3): a network registered through the registry is
usable through the public interface without touching the package core.
"""

import torch

import epimeter
from epimeter.models.base import BaseModel
from epimeter.models.registry import build


def _make_mlp_class():
    @epimeter.register("_test_mlp")
    class MLP(BaseModel):
        def __init__(self, hidden=128):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(784, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 10),
            )

        def forward(self, x):
            return self.net(x)

    return MLP


def test_registered_mlp_is_buildable_and_forward_shape():
    _make_mlp_class()
    net = build("_test_mlp", hidden=64)
    assert isinstance(net, BaseModel)
    out = net.forward(torch.zeros(4, 784))
    assert out.shape == (4, 10)


def test_count_parameters_positive():
    _make_mlp_class()
    net = build("_test_mlp", hidden=16)
    assert net.count_parameters() > 0
