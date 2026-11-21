r"""Running an epiplexity evaluator from the command line.

    python -m episynth.estimators.run --estimator <name> --generator <name>
        [--config-json runs/<run>/config.json] [--out <catalog>] [-n-windows N] \
        [--batch-size B] [--device cuda] ...

By default, the model is built using :func:`episynth.training.build_model` from the `config.json`
configuration (`model` and `train` fields). If `--config-json` is not specified, the
`--n-windows`/`--batch-size` arguments (mandatory) and the default `TrainConfig` values are used.

`--estimator-config` is a JSON parameter object, such as `{"variant": "raw", "test_windows": 1000}`
for V-information. The result is stored in `estimator_result.json`.

The names of the available algorithms are :func:`episynth.estimators.available`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..generators import GENERATORS
from ..model.moirai import MoiraiConfig
from ..training import TrainConfig, build_model
from ..windows import WindowSpec
from .registry import available, build


def _build_from_config_json(path: Path) -> tuple[TrainConfig, WindowSpec, MoiraiConfig]:
    """Restores configurations from `config.json` recorded by `Trainer.run`."""
    data = json.loads(path.read_text())
    return (
        TrainConfig(**data["train"]),
        WindowSpec(**data["spec"]),
        MoiraiConfig(**data["model"]),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: Launches the selected estimator."""
    parser = argparse.ArgumentParser(description="Run an epiplexity estimator.")
    parser.add_argument("--estimator", required=True, choices=available(), help="Estimator name.")
    parser.add_argument("--generator", required=True, help="Series generator name.")
    parser.add_argument("--out", default="runs/epi", help="Output directory.")
    parser.add_argument("--config-json", help="Path to a run's config.json; overrides the defaults.")
    parser.add_argument("--n-windows", type=int, help="Training budget in windows (required without --config-json).")
    parser.add_argument("--batch-size", type=int, help="Windows per step (required without --config-json).")
    parser.add_argument("--device", help="Device, e.g. cuda or cpu; overrides the saved config when supplied.")
    parser.add_argument("--init-seed", type=int, help="Weight initialization seed; overrides config train.init_seed.")
    parser.add_argument("--estimator-config", default="{}", help="JSON object of estimator constructor arguments.")
    args = parser.parse_args(argv)
    try:
        estimator_config = json.loads(args.estimator_config)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --estimator-config: {exc}")
    if not isinstance(estimator_config, dict):
        parser.error("--estimator-config must be a JSON object")

    if args.config_json:
        config, spec, model_cfg = _build_from_config_json(Path(args.config_json))
    else:
        if args.n_windows is None or args.batch_size is None:
            parser.error("--n-windows and --batch-size are required unless --config-json is given.")
        spec = WindowSpec()
        config = TrainConfig(n_windows=args.n_windows, batch_size=args.batch_size, init_seed=args.init_seed if args.init_seed is not None else 0)
        model_cfg = MoiraiConfig(patch_sizes=(spec.patch_size,), components=("student_t", "normal_fixed", "log_normal"))

    if args.device is not None:
        config = config.model_copy(update={"device": args.device})
    if args.init_seed is not None:
        config = config.model_copy(update={"init_seed": args.init_seed})
    generator = GENERATORS[args.generator](length=spec.series_length)
    model = build_model(model_cfg, config.init_seed)
    estimator = build(args.estimator, **estimator_config)
    result = estimator.run(model, generator, spec, config, args.out)
    Path(args.out, "estimator_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"estimator": args.estimator, "generator": args.generator, "result": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
