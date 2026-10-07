"""Logging helpers (skeleton)."""

import logging

logger = logging.getLogger("epimeter")


def configure(level: int = logging.INFO) -> None:
    """Configure the epimeter logger on first use."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
