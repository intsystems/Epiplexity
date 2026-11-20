"""Source identity, seed coverage and split isolation, including nested mixtures."""

import inspect
import json
from collections.abc import Iterator
from pathlib import Path

from .corpus import Corpus
from .generators import Generator, Mixture


def source_description(generator: Generator) -> dict:
    """Record constructor settings, corpus manifests and mixture composition."""
    description = {"class": f"{type(generator).__module__}.{type(generator).__qualname__}",
                   "name": generator.name, "length": generator.length}
    if isinstance(generator, Mixture):
        description["components"] = [source_description(g) for g in generator.components]
    elif isinstance(generator, Corpus):
        description["corpora"] = [
            {"path": str(Path(p).resolve()), "manifest": m.model_dump(mode="json")}
            for p, m in zip(generator._paths, generator.manifests, strict=True)
        ]
    else:
        description["parameters"] = {
            name: getattr(generator, name, parameter.default)
            for name, parameter in inspect.signature(type(generator)).parameters.items()
            if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
    # Constructor parameters of built-in generators are JSON values.
    return json.loads(json.dumps(description, allow_nan=False))


def leaf_ranges(generator: Generator, first: int, n: int) -> Iterator[tuple[Generator, int, int]]:
    """Project a half-open range to the exact ranges of the underlying sources."""
    if n <= 0:
        return
    if isinstance(generator, Mixture):
        k = len(generator.components)
        for j, component in enumerate(generator.components):
            lo = (first - j + k - 1) // k
            hi = (first + n - j + k - 1) // k
            yield from leaf_ranges(component, lo, hi - lo)
    else:
        yield generator, first, first + n


def seeds_available(generator: Generator, first_seed: int, n: int) -> bool:
    """Whether all requested seeds are covered (adjacent corpus parts are allowed)."""
    for source, lo, hi in leaf_ranges(generator, first_seed, n):
        if isinstance(source, Corpus):
            cursor = lo
            for a, b in sorted(source.seed_ranges):
                if a > cursor:
                    break
                cursor = max(cursor, b)
                if cursor >= hi:
                    break
            if cursor < hi:
                return False
    return True


def validate_splits(generator: Generator, splits: dict[str, dict[str, int]]) -> None:
    """Reject missing/overlapping seeds before generation, down to mixture sources.

    Sources with the same generator name and series length conservatively share
    a seed namespace, including corpus/live or repeated components of a mixture.
    """
    previous = []
    for name, split in splits.items():
        first, n = split["first_seed"], split["n_windows"]
        if first < 0 or n <= 0 or first + n > 2**31:
            raise ValueError(f"invalid {name} seed range: [{first}, {first + n})")
        if not seeds_available(generator, first, n):
            raise ValueError(f"{name} seeds are not covered by source {generator.name}")
        current = list(leaf_ranges(generator, first, n))
        for source, lo, hi in current:
            key = (source.name, source.length)
            for old_name, old_key, a, b in previous:
                if key == old_key and max(lo, a) < min(hi, b):
                    raise ValueError(f"{name} and {old_name} splits overlap in source {source.name}")
        previous.extend((name, (s.name, s.length), a, b) for s, a, b in current)
