"""Measure cumulative requential KL at the two archived transfer budgets."""

import argparse
import csv
import fcntl
import json
import math
import time
from contextlib import ExitStack, closing
from pathlib import Path

import torch
from requential_plot import render_figure
from v_information_transfer import LARGE, ROOT, SMALL, archive, config_for

from episynth.estimators import build
from episynth.estimators.requential import _Learner, _projection_schedule, _public
from episynth.splits import source_description
from episynth.train import RunConfig, build_generator
from episynth.training import build_model, weights_digest, window_nll
from episynth.windows import WindowStream


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_pt(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def learner_state(learner: _Learner) -> dict:
    return {
        "raw": learner.raw.state_dict(),
        "ema": learner.ema.state_dict(),
        "optimizer": learner.optimizer.state_dict(),
        "clock": learner.clock,
    }


def restore(learner: _Learner, state: dict) -> None:
    learner.raw.load_state_dict(state["raw"])
    learner.ema.load_state_dict(state["ema"])
    learner.optimizer.load_state_dict(state["optimizer"])
    learner.clock = state["clock"]


def measure(config: RunConfig, out: Path, budgets: list[int], workers: int) -> None:
    """Continue the same protocol across real-window milestones, including during recovery."""
    cfg, spec = config.train, config.spec
    if (
        config.estimator != "requential"
        or config.estimator_config.get("mode", "direct") != "direct"
    ):
        raise ValueError("the transfer measurement uses direct EMA-teacher sampling")
    if not budgets or any(n <= 0 or n > cfg.n_windows or n % cfg.batch_size for n in budgets):
        raise ValueError("invalid real-window milestone")
    source = build_generator(config.data, spec.series_length)
    model = build_model(config.model, cfg.init_seed)
    estimator = build("requential", **config.estimator_config)
    splits = estimator._splits(model, source, spec, cfg)
    schedule = _projection_schedule(estimator.projection_steps, cfg.n_steps)
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((out / "fit.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = {
            "config": config.model_dump(mode="json"),
            "splits": splits,
            "source": source_description(source),
            "initial_weights_sha256": weights_digest(model),
        }
        manifest = out / "protocol.json"
        if manifest.exists() and json.loads(manifest.read_text()) != signature:
            raise ValueError(f"incompatible measurement at {out}")
        if not manifest.exists():
            write_json(manifest, signature)
        if all((out / str(n) / "measurement.json").is_file() for n in budgets):
            return
        teacher = _Learner(model, spec, cfg, estimator.teacher_ema, teacher=True)
        student = _Learner(model, spec, cfg, estimator.student_ema)
        real_steps, steps, events, active = 0, [], [], None
        last = out / "resume.pt"
        if last.exists():
            state = torch.load(last, map_location="cpu", weights_only=True)
            restore(teacher, state["teacher"])
            restore(student, state["student"])
            real_steps, steps, events, active = (
                state[k] for k in ("real_steps", "steps", "events", "active")
            )
            del state
        for n in budgets:
            if (
                n < real_steps * cfg.batch_size
                and not (out / str(n) / "measurement.json").is_file()
            ):
                raise ValueError(f"missing earlier milestone {n} at {out}")

        def stream(split: str, first_batch: int):
            start = splits[split]["first_seed"]
            return stack.enter_context(
                closing(
                    iter(
                        WindowStream(
                            source,
                            spec,
                            cfg.batch_size,
                            start,
                            max(budgets) // cfg.batch_size,
                            first_batch=first_batch,
                            workers=workers,
                        )
                    )
                )
            )

        real = stream("teacher_train", real_steps)
        public = stream("public_context", student.clock)
        cached = {}

        @torch.no_grad()
        def evaluate(learner: _Learner, split: str) -> float:
            if split not in cached:
                first, n = splits[split]["first_seed"], splits[split]["n_windows"]
                batches = WindowStream(
                    source,
                    spec,
                    cfg.eval_batch_size,
                    first,
                    math.ceil(n / cfg.eval_batch_size),
                    workers=workers,
                    seed_limit=first + n,
                )
                cached[split] = [b.to(cfg.device) for b in batches]
            total = torch.zeros((), device=cfg.device, dtype=torch.float64)
            for b in cached[split]:
                total += window_nll(learner.predict(b), b, spec.patch_size).double().sum()
            mean = total.item() / splits[split]["n_windows"]
            if not math.isfinite(mean):
                raise FloatingPointError(f"nonfinite {split} loss at real step {real_steps}")
            return mean

        t0 = time.monotonic()
        initial_steps = real_steps
        while real_steps * cfg.batch_size < max(budgets):
            step = student.clock
            if (
                active is None
                and step in schedule
                and (not events or events[-1]["student_step"] != step)
            ):
                previous = evaluate(teacher, "projection_validation")
                teacher.project(student)
                current = evaluate(teacher, "projection_validation")
                active = {
                    "student_step": step,
                    "message_count": step * cfg.batch_size,
                    "real_windows_before": real_steps * cfg.batch_size,
                    "recovery_windows": 0,
                    "previous_nll": previous,
                    "recovered_nll": current,
                    "complete": current <= previous,
                }
                if active["complete"]:
                    events.append(active)
                    active = None
            if active is None:
                batch = _public(next(public).to(cfg.device))
                synthetic, record, _ = estimator._message_batch(student, teacher, batch, step)
                student.update(synthetic, step + 1)
                steps.append(record)
            teacher.update(next(real).to(cfg.device), real_steps + 1)
            real_steps += 1
            if active is not None:
                active["recovery_windows"] += cfg.batch_size
                active["recovered_nll"] = evaluate(teacher, "projection_validation")
                active["complete"] = active["recovered_nll"] <= active["previous_nll"]
                if active["complete"]:
                    events.append(active)
                    active = None
            n = real_steps * cfg.batch_size
            if real_steps % 50 == 0 or n in budgets or real_steps == initial_steps + 1:
                progress = {
                    "real_windows": n,
                    "student_steps": student.clock,
                    "recovery": active,
                    "real_windows_per_second": round(
                        (real_steps - initial_steps) * cfg.batch_size / (time.monotonic() - t0), 1
                    ),
                }
                write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if n in budgets:
                folder = out / str(n)
                folder.mkdir(exist_ok=True)
                totals = {
                    key: math.fsum(s[key] for s in steps)
                    for key in (
                        "estimated_kl_bits",
                        "estimated_overhead_bits",
                        "plugin_expected_length_bound_bits",
                    )
                }
                measurement = {
                    **totals,
                    "real_windows": n,
                    "first_seed": cfg.first_seed,
                    "message_count": student.clock * cfg.batch_size,
                    "student_steps": student.clock,
                    "estimated_kl_mc_se_bits": math.sqrt(
                        math.fsum(s["conditional_mc_variance_bits2"] for s in steps)
                    ),
                    "projections": events + ([active] if active else []),
                    "heldout_nats_per_window": {
                        key: evaluate(learner, "final_eval")
                        for key, learner in (("student", student), ("teacher", teacher))
                    },
                    "generator": source.name,
                    "init_seed": cfg.init_seed,
                    "mc_repeats": estimator.mc_repeats,
                    "initial_weights_sha256": signature["initial_weights_sha256"],
                }
                save_pt(
                    folder / "student.pt",
                    {
                        "model": student.raw.state_dict(),
                        "ema": [p.detach().cpu() for p in student.ema.parameters()],
                        "model_config": config.model.model_dump(mode="json"),
                        "step": student.clock,
                    },
                )
                write_json(folder / "measurement.json", measurement)
            if real_steps % 512 == 0 or n in budgets:
                save_pt(
                    last,
                    {
                        "student": learner_state(student),
                        "teacher": learner_state(teacher),
                        "steps": steps,
                        "real_steps": real_steps,
                        "events": events,
                        "active": active,
                    },
                )


def fit(root: Path, source_root: Path, name: str, through: int, workers: int) -> None:
    rows = archive()
    row = next(r for r in rows if r["run"] == name)
    config = config_for(source_root, row, workers=0)
    if int(row["series"]) == 2 and int(row["init_seed"]) == 0:
        small_row = next(
            r for r in rows if r["generator"] == row["generator"] and int(r["series"]) == 1
        )
        small = config_for(source_root, small_row, workers=0)
        if (
            small["model"] != config["model"]
            or small["spec"] != config["spec"]
            or any(v != small["train"][k] for k, v in config["train"].items() if k != "n_windows")
            or any(v != small["data"][k] for k, v in config["data"].items() if k != "corpus")
        ):
            raise ValueError("archived runs do not share a training prefix")
    config["estimator"] = "requential"
    config["estimator_config"] = {
        "public_first_seed": 100_000_000,
        "validation_first_seed": 10**9,
        "validation_windows": 100_000,
        "mc_repeats": 8,
    }
    config["train"]["heldout_first_seed"] = 2 * 10**9
    if row["generator"] == "kernelsynth":
        config["data"]["corpus"].extend(
            str(p.parent)
            for p in sorted((root / "data/kernelsynth").glob("*/manifest.json"))
            if int(p.parent.name) < 100_000_000 + config["train"]["n_windows"]
        )
    cfg = RunConfig.model_validate(config)
    budgets = [
        n for n in [SMALL, LARGE] if n <= through and (n == LARGE or int(row["init_seed"]) == 0)
    ]
    if name == "s2_kernelsynth" and through != LARGE:
        raise ValueError("use series_kernelsynth for the smaller archived run")
    if name == "s2_kernelsynth":
        budgets = [LARGE]
    measure(cfg, root / "runs" / name, budgets, workers)


def plot(root: Path, through: int = LARGE, *, partial: bool = False) -> None:
    rows = []
    for archived in archive():
        if int(archived["n_windows"]) > through:
            continue
        generator = archived["generator"]
        seed = int(archived["init_seed"])
        budget = int(archived["n_windows"])
        name = "s2_" + generator + ("_seed1" if seed else "")
        if generator == "kernelsynth" and budget == SMALL:
            name = "series_kernelsynth"
        path = root / "runs" / name / str(budget) / "measurement.json"
        if partial and not path.is_file():
            continue
        result = json.loads(path.read_text())
        if (result["generator"], result["init_seed"], result["real_windows"]) != (
            generator,
            seed,
            budget,
        ):
            raise ValueError(f"measurement does not match the archive at {path}")
        recovery = sum(e["recovery_windows"] for e in result["projections"])
        if result["message_count"] + recovery != budget:
            raise ValueError(f"inconsistent coding and recovery counts at {path}")
        row = {
            k: archived[k]
            for k in ("series", "run", "generator", "init_seed", "n_windows", "crps_geo")
        }
        row.update(
            {
                k: result[k]
                for k in (
                    "estimated_kl_bits",
                    "estimated_kl_mc_se_bits",
                    "estimated_overhead_bits",
                    "plugin_expected_length_bound_bits",
                    "student_steps",
                    "message_count",
                )
            }
        )
        row.update(recovery_windows=recovery, measurement=str(path.relative_to(root)))
        rows.append(row)
    if rows:
        render(rows, root / "results/progress" if partial else root / "results")


def render(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    panels = len({int(r["series"]) for r in rows})
    stem = out / ("requential_vs_crps" if panels == 2 else "requential_vs_crps_2m")
    render_figure(rows, stem)
    with stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run")
    action.add_argument("--plot", action="store_true")
    parser.add_argument("--through", type=int, choices=[SMALL, LARGE], default=LARGE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.plot:
        plot(args.root.resolve(), args.through)
    else:
        fit(args.root.resolve(), args.source_root, args.run, args.through, args.workers)


if __name__ == "__main__":
    main()
