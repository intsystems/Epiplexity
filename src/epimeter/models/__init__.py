"""Observer models (MLP, CNN, DiT)."""

from .base import BaseModel
from .registry import build, register

__all__ = ["BaseModel", "build", "register"]
