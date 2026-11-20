r"""Запуск оценщика эпиплексии из командной строки.

    python -m episynth.estimators.run --estimator <name> --generator <имя> \\
        [--config-json runs/<прогон>/config.json] [--out <каталог>] [--n-windows N] \\
        [--batch-size B] [--device cuda] ...

По умолчанию модель строится через :func:`episynth.training.build_model` из конфигурации
`config.json` (поля `model` и `train`). Если `--config-json` не задан, используются аргументы
`--n-windows`/`--batch-size` (обязательные) и значения `TrainConfig` по умолчанию.

`--estimator-config` — JSON-объект параметров конструктора, например
`{"variant": "raw", "test_windows": 1000}` для V-information.
Результат сохраняется в `estimator_result.json`.

Имена доступных алгоритмов — :func:`episynth.estimators.available`.
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
    """Восстанавливает конфигурации из `config.json`, записанного `Trainer.run`."""
    data = json.loads(path.read_text())
    return (
        TrainConfig(**data["train"]),
        WindowSpec(**data["spec"]),
        MoiraiConfig(**data["model"]),
    )


def main(argv: list[str] | None = None) -> None:
    """Точка входа CLI: запускает выбранный оценщик."""
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
