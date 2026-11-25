r"""Protocol for epiplexity estimator plugins.

An estimator receives a model, series generator, task shape, and run
settings. Its JSON-compatible result uses method-specific fields. The
prequential estimator decomposes code length, and V-information measures
predictive loss reduction. Built-in and custom estimators implement
:meth:`EpiplexityEstimator.run` and register under a name used to select them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, runtime_checkable

from ..generators.base import Generator
from ..model.moirai import MoiraiModule
from ..training import TrainConfig
from ..windows import WindowSpec


def write_estimator_result(out_dir: str | Path, result: Mapping[str, Any]) -> None:
    """Publish a complete JSON result atomically on the run filesystem."""
    out = Path(out_dir)
    with TemporaryDirectory(dir=out) as staging:
        path = Path(staging) / "estimator_result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        path.replace(out / path.name)


@runtime_checkable
class EpiplexityEstimator(Protocol):
    """A contract for an epiplexity evaluation algorithm.

    Each estimator is a class with the `name` attribute (key in the registry) and the method
    :meth:`run`, which performs a run of training and writes the result in `out_dir`.

    Args:
        name: The name of the algorithm under which it is registered in the registry.

    Attributes:
        name: An algorithm key in the registry; used for selection via CLI and `build`.
    """

    name: str

    def run(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        out_dir: str | Path,
    ) -> Mapping[str, Any]:
        """Performs a run of training and returns an epiplexity assessment.

        Args:
            model: Observer model; initialized externally (e.g. via build_model).
            generator: Synthetic series generator.
            spec: Form of the forecast problem.
            config: Parameters run.
            out_dir: Catalog, where the artifacts of the run and the result are written.

        Returns:
            JSON-compatible dictionary. Prequential returns ``model_part_bits``, ``data_part_bits``
            and ``total_bits``; V-information predictive entropies, their bit difference per window
            and protocol metadata.
        """
        ...
