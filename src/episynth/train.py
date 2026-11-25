"""Start a training run from a configuration file.

The JSON file has `data` (series source), `spec` (`WindowSpec`), `model`
(`MoiraiConfig`), and `train` (`TrainConfig`) sections. The `data` section
selects a registered generator, a saved corpus of that generator through
`corpus`, or an equally weighted `mixture` of sources. `context_table` points
to a JSON file with `lengths` and `probs` for the context-length distribution.
Override fields with `--set section.field=value`. Values are parsed as JSON
when possible and otherwise kept as strings. The resolved config is written
to `run.json` while an existing file is retained. A run with `last.pt` resumes.

`estimator` selects the algorithm and `estimator_config` supplies its
constructor arguments. V-information uses two run subdirectories and
separate training, validation, and test splits.

    python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth --set train.lr=3e-4
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .corpus import Corpus
from .estimators import available as estimators_available
from .estimators import build as build_estimator
from .estimators.base import write_estimator_result
from .generators import GENERATORS, Generator, Mixture
from .model.moirai import MoiraiConfig, small
from .splits import seeds_available, validate_splits
from .training import RESUME_RUNTIME_FIELDS, TrainConfig, build_model
from .windows import WindowSpec


class DataConfig(BaseModel):
    """Source of time series.

    Attributes:
        generator: Registry name, also used to validate corpora. Mixtures use their component names.
        corpus: Corpus directories for this generator. Empty means on-the-fly generation.
        generator_args: Constructor arguments for on-the-fly generation.
        mixture: Components of an equal mixture. A nonempty list takes precedence.
    """

    generator: str = ""
    corpus: list[str] = []
    generator_args: dict = {}
    mixture: list["DataConfig"] = []


class RunConfig(BaseModel):
    """Complete run configuration.

    Attributes:
        data: Source of time series.
        spec: Forecasting task shape.
        model: Model configuration, defaulting to Moirai-Small with patch size 32
            and three continuous mixture components.
        train: Training settings.
        context_table: Path to JSON with `lengths` and `probs` for `spec`.
        estimator: Algorithm name in the `episynth.estimators` registry, defaulting to `prequential`.
        estimator_config: Constructor arguments for the selected estimator.
    """

    data: DataConfig
    spec: WindowSpec = WindowSpec()
    model: MoiraiConfig = Field(default_factory=lambda: small(patch_sizes=(32,), components=("student_t", "normal_fixed", "log_normal")))
    train: TrainConfig
    context_table: str | None = None
    estimator: str = "prequential"
    estimator_config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "RunConfig":
        if self.spec.patch_size not in self.model.patch_sizes:
            raise ValueError(f"Patch size {self.spec.patch_size} is not in {self.model.patch_sizes}")
        if self.spec.n_tokens > self.model.max_seq_len:
            raise ValueError("Window exceeds the model max_seq_len")
        if self.estimator not in estimators_available():
            raise ValueError(f"Unknown estimator {self.estimator!r}. Available estimators: {sorted(estimators_available())}")
        return self


def resolve_context_table(raw: dict) -> dict:
    """Substitutes the context length table from the `context_table` file to the `spec` section of the raw configuration."""
    path = raw.get("context_table")
    if path:
        table = json.loads(Path(path).read_text())
        raw.setdefault("spec", {}).update(context_lengths=list(table["lengths"]), context_probs=list(table["probs"]))
    return raw


def build_generator(data: DataConfig, length: int) -> Generator:
    """Build the generator, corpus, or mixture specified in `data`.

    Args:
        data: Configuration section.
        length: The length of the series from the shape of the task.

    Returns:
        Generator: Source of rows.
    """
    if data.mixture:
        return Mixture([build_generator(d, length) for d in data.mixture])
    if data.corpus:
        corpus = Corpus(*data.corpus)
        if data.generator and corpus.name != data.generator:
            raise ValueError(f"Corpus generator {corpus.name} does not match configured generator {data.generator}")
        if corpus.length != length:
            raise ValueError(f"Corpus series length {corpus.length} does not match task series length {length}")
        return corpus
    return GENERATORS[data.generator](length=length, **data.generator_args)


def check_seed_ranges(generator: Generator, train: TrainConfig) -> None:
    """Check that training and held-out seed ranges are disjoint and covered by the source."""
    for name, first, n in (("training", train.first_seed, train.n_windows), ("held-out", train.heldout_first_seed, train.heldout_windows)):
        if not seeds_available(generator, first, n):
            raise ValueError(f"{name} seeds [{first}, {first + n}) are outside the coverage of source {generator.name}")
    validate_splits(generator, {
        "train": {"first_seed": train.first_seed, "n_windows": train.n_windows},
        "heldout": {"first_seed": train.heldout_first_seed, "n_windows": train.heldout_windows},
    })


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply `section.field=value` overrides to a raw configuration.

    Parse each value as JSON when possible and otherwise keep it as a string.
    """
    for item in overrides:
        key, _, value = item.partition("=")
        node = raw
        *path, last = key.split(".")
        for part in path:
            node = node.setdefault(part, {})
        try:
            node[last] = json.loads(value)
        except json.JSONDecodeError:
            node[last] = value
    return raw


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Configuration from a redefinition file, with a context length table."""
    raw = resolve_context_table(apply_overrides(json.loads(Path(path).read_text()), overrides or []))
    return RunConfig.model_validate(raw)


def main() -> None:
    """Command line run."""
    parser = argparse.ArgumentParser(description="Start or resume a training run from a configuration file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="section.field=value")
    args = parser.parse_args()
    config = load_run_config(args.config, args.set)
    out = Path(args.out)
    if (out / "run.json").exists():
        previous = RunConfig.model_validate_json((out / "run.json").read_text())
        saved, requested = previous.model_dump(mode="json"), config.model_dump(mode="json")
        if previous.estimator == config.estimator == "prequential":
            for field in RESUME_RUNTIME_FIELDS:
                saved["train"].pop(field, None)
                requested["train"].pop(field, None)
        if saved != requested:
            raise ValueError("incompatible run configuration; use a different output directory")
    generator = build_generator(config.data, config.spec.series_length)
    check_seed_ranges(generator, config.train)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "run.json").exists():
        (out / "run.json").write_text(config.model_dump_json(indent=1))
    model = build_model(config.model, config.train.init_seed)
    result = build_estimator(config.estimator, **config.estimator_config).run(model, generator, config.spec, config.train, out)
    write_estimator_result(out, result)
    print(json.dumps({"estimator": config.estimator, "result": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
