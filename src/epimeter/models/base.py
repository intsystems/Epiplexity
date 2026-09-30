"""Observer model interface."""

from abc import ABC
from pathlib import Path
from typing import Any

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """An observer: any network (MLP, CNN, DiT) used to estimate epiplexity.

    Subclasses implement ``forward``; ``save``/``load`` persist weights and
    ``count_parameters`` reports the number of trainable parameters.
    """

    def count_parameters(self) -> int:
        """Number of trainable parameters in the network."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        """Persist state dict to ``path``."""
        torch.save(self.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        """Load state dict from ``path`` in place."""
        state: dict[str, Any] = torch.load(str(path), map_location="cpu")
        self.load_state_dict(state)
