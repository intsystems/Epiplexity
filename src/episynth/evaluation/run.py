r"""Прогон прогнозчика по конфигурациям GIFT-Eval с записью метрик по окнам.

    python -m episynth.evaluation.run --data <снимок GiftEval> --out <каталог> \\
        --model seasonal_naive | moirai [--checkpoint <model.pt|model.safetensors>] [--patch-size 32] \\
        [--protocol ours|notebook] [--configs electricity/H m4_hourly ...] [--device cuda]

В каталог пишутся `windows/<конфигурация>.csv` (строка на окно: ряд, номер окна и все суммы
`window_stats`) и `summary.csv` (строка на конфигурацию: метрики `aggregate`). Готовые
конфигурации при повторном запуске пропускаются только при совпадении `evaluation.json`:
настроек оценки, SHA-256 чекпойнта и абсолютного пути снимка данных. Содержимое снимка
не хешируется; для изменённых данных или настроек нужен новый выходной каталог.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch

from ..model import MoiraiModule, load_uni2ts_checkpoint
from ..model.moirai import MoiraiConfig, small
from .forecasters import Forecaster, MoiraiForecaster, SeasonalNaive
from .gift_eval import SHORT_CONFIGS, GiftDataset
from .metrics import aggregate, window_stats


def evaluate_config(dataset: GiftDataset, forecaster: Forecaster, chunk: int = 2048) -> tuple[list[dict], dict]:
    """Записи окон и метрики датасета для прогнозчика."""
    instances = list(dataset.test_instances())
    rows = []
    for start in range(0, len(instances), chunk):
        part = instances[start : start + chunk]
        fc = forecaster.predict([p.history for p in part], dataset.prediction_length, [p.label for p in part])
        for i, inst in enumerate(part):
            nll = None if fc.nll is None else fc.nll[i]
            rec = window_stats(inst.label, fc.samples[:, i], inst.history, dataset.season, nll, int(fc.context_length[i]))
            rows.append({"item_id": inst.item_id, "window": inst.window, **rec})
    return rows, aggregate(rows)


def _load_model(args: argparse.Namespace) -> MoiraiModule:
    """Восстанавливает native-модель по чекпойнту, uni2ts — по явной или стандартной конфигурации."""
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --model moirai")
    native = Path(args.checkpoint).suffix != ".safetensors"
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if native else None
    explicit_cfg = None
    if args.config_json:
        data = json.loads(Path(args.config_json).read_text())
        explicit_cfg = MoiraiConfig(**data.get("model", data))
    if native:
        if "model_config" not in state:
            raise ValueError("Native checkpoint is missing model_config")
        cfg = MoiraiConfig(**state["model_config"])
        if explicit_cfg is not None and explicit_cfg != cfg:
            raise ValueError("--config-json does not match the checkpoint model_config")
    else:
        cfg = explicit_cfg or (small() if args.protocol == "notebook" else small(patch_sizes=(args.patch_size,), components=tuple(args.components)))
    module = MoiraiModule(cfg)
    if native:
        module.load_state_dict(state["model"])
        if args.weights == "ema":
            with torch.no_grad():
                for p, e in zip(module.parameters(), state["ema"], strict=True):
                    p.copy_(e)
    else:
        load_uni2ts_checkpoint(module, args.checkpoint)
    return module


def build_forecaster(args: argparse.Namespace, dataset: GiftDataset, module: MoiraiModule | None = None) -> Forecaster:
    """Прогнозчик по аргументам командной строки; готовая модель позволяет не читать чекпойнт повторно."""
    if args.model == "seasonal_naive":
        return SeasonalNaive(dataset.season)
    if args.forecaster is None:
        module = _load_model(args) if module is None else module
        context_max = 4000 if args.protocol == "notebook" else args.context_max
        args.forecaster = MoiraiForecaster(
            module, args.patch_size, args.protocol, context_max, args.horizon_max, args.n_samples, args.batch_size, args.device
        )
    return args.forecaster


def _prepare_output(args: argparse.Namespace, cfg: MoiraiConfig | None) -> None:
    """Проверяет идентичность оценки до чтения готовых строк и любых изменений старого вывода."""
    identity = {"version": 1, "data": str(args.data.resolve()), "model": args.model, "term": args.term}
    if cfg is not None:
        digest = hashlib.sha256()
        with open(args.checkpoint, "rb") as f:
            for part in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(part)
        identity.update(
            checkpoint_sha256=digest.hexdigest(), model_config=cfg.model_dump(mode="json"),
            weights=args.weights, patch_size=args.patch_size, protocol=args.protocol,
            context_max=4000 if args.protocol == "notebook" else args.context_max,
            horizon_max=args.horizon_max, n_samples=args.n_samples, batch_size=args.batch_size, device=args.device,
        )
    manifest = args.out / "evaluation.json"
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read evaluation.json; use a new --out directory") from exc
        if previous != identity:
            raise ValueError("Evaluation settings do not match evaluation.json; use a new --out directory")
    elif (args.out / "summary.csv").exists():
        raise ValueError("Existing summary.csv has no evaluation.json; use a new --out directory")
    (args.out / "windows").mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Разбор аргументов и прогон по конфигурациям."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=["seasonal_naive", "moirai"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config-json", default=None, help="config.json прогона обучения или конфигурация модели")
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--components", nargs="*", default=["student_t", "normal_fixed", "log_normal"])
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--protocol", choices=["ours", "notebook"], default="ours")
    parser.add_argument("--context-max", type=int, default=512)
    parser.add_argument("--horizon-max", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--configs", nargs="*", default=list(SHORT_CONFIGS))
    parser.add_argument("--term", default="short")
    args = parser.parse_args()
    args.forecaster = None
    module = _load_model(args) if args.model == "moirai" else None
    _prepare_output(args, None if module is None else module.config)
    summary_path = args.out / "summary.csv"
    done = set()
    if summary_path.exists():
        with open(summary_path) as f:
            done = {r["config"] for r in csv.DictReader(f)}
    for name in args.configs:
        dataset = GiftDataset(args.data, name, args.term)
        if dataset.config in done:
            continue
        t0 = time.perf_counter()
        rows, agg = evaluate_config(dataset, build_forecaster(args, dataset, module))
        with open(args.out / "windows" / (dataset.config.replace("/", "__") + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        rec = {"config": dataset.config, "model": args.model, "seconds": round(time.perf_counter() - t0, 1), **agg}
        new = not summary_path.exists()
        with open(summary_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"{dataset.config:40s} MASE {agg['MASE[0.5]']:.4f} CRPS {agg['mean_weighted_sum_quantile_loss']:.4f} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
