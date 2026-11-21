"""Equal mixture of generators: the seed selects a component and a series."""

from collections.abc import Sequence

import numpy as np

from .base import Generator


class Mixture(Generator):
    """Uniform mixture of generators with the same series length.

    For `k` components, seed `s` selects component `s % k` with seed `s // k`.
    Consecutive seeds cycle through the components without reusing a series.
    A mixture range `[a, b)` requires component seeds within `[a // k, b // k]`.
    For corpus-backed mixtures, a held-out split starting at seed `h` reads
    from the component corpora starting at `h // k`. Reproduction mode is
    forwarded to the selected component.

    Args:
        components: Generators in assignment order.

    Attributes:
        name: `mixture(` followed by the comma-separated component names.
        components: Generators in assignment order.
    """

    def __init__(self, components: Sequence[Generator]) -> None:
        if not components:
            raise ValueError("A mixture needs at least one component")
        if len({g.length for g in components}) != 1:
            raise ValueError("Mixture components have different series lengths")
        super().__init__(components[0].length)
        self.components = list(components)
        self.name = "mixture(" + ",".join(g.name for g in self.components) + ")"

    def _batch(self, seeds: np.ndarray) -> np.ndarray:
        k = len(self.components)
        out = np.empty((len(seeds), self.length), dtype=np.float32)
        for j, g in enumerate(self.components):
            mask = seeds % k == j
            if mask.any():
                out[mask] = g.generate(seeds[mask] // k)
        return out

    def _exact(self, seed: int) -> np.ndarray:
        k = len(self.components)
        return self.components[seed % k].generate([seed // k], exact=True)[0]
