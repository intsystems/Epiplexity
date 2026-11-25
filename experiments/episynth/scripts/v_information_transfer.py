"""Prepare the Figure 4 V-information sweep and plot verified test measurements."""

import argparse
import csv
import fcntl
import json
import math
import os
import sys
from contextlib import suppress
from pathlib import Path
from statistics import geometric_mean

REPO = Path(__file__).resolve().parents[3]
ROOT = Path("/home/jovyan/shares/SR008.fs2/kazachkovdi/side-projects/epiplexity-v-information")
SMALL = 1_999_872
LARGE = 39_999_488
NAMES = {
    "kernelsynth": "KernelSynth", "ets": "ETS", "waveform": "Waveform", "sde": "SDE",
    "stepfunction": "StepFunction", "tsi": "TSI", "arima": "ARIMA", "fbm": "fBm",
    "garch": "GARCH", "chaotic": "Chaotic", "timesynth": "TimeSynth",
}

# Label positions follow the article's convention of coordinates within each panel.
LABELS = {
    1: {"kernelsynth": (.51, .05), "ets": (.91, .09), "waveform": (.30, .15),
        "sde": (.13, .22), "stepfunction": (.66, .93), "tsi": (.69, .17),
        "arima": (.58, .43), "fbm": (.40, .33), "garch": (.76, .34),
        "chaotic": (.83, .60), "timesynth": (.13, .44)},
    2: {"kernelsynth": (.47, .035), "ets": (.93, .07), "waveform": (.39, .79),
        "sde": (.11, .41), "stepfunction": (.43, .15), "tsi": (.62, .13),
        "arima": (.81, .43), "fbm": (.31, .30), "garch": (.67, .32),
        "chaotic": (.87, .70), "timesynth": (.12, .18), "repeat": (.32, .47)},
}


def kernel_parts(budget: int) -> list[tuple[int, int]]:
    """Split the shared prefix and extension into independently generated seed ranges."""
    if budget not in (SMALL, LARGE):
        raise ValueError(f"Unexpected training budget {budget}")
    parts = [(0, 65_536)]
    cursor = parts[0][1]
    for end in sorted({SMALL, budget}):
        while cursor < end:
            size = min(1_000_000, end - cursor)
            parts.append((cursor, size))
            cursor += size
    return [*parts, (10**9, 100_000), (2 * 10**9, 100_000)]


def archive() -> list[dict]:
    folder = REPO / "publications/article/data"
    with (folder / "run_statistics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {(s, g, 0) for s in (1, 2) for g in NAMES} | {(2, "waveform", 1)}
    keys = {(int(r["series"]), r["generator"], int(r["init_seed"])) for r in rows}
    if keys != expected or len(rows) != len(keys):
        raise ValueError("Unexpected archive run matrix")
    with (folder / "task_ratios.csv").open() as handle:
        tasks = list(csv.DictReader(handle))
    for row in rows:
        subset = [t for t in tasks if t["run"] == row["run"]]
        if len(subset) != 55 or len({t["config"] for t in subset}) != 55:
            raise ValueError(f"Unexpected task count for {row['run']}")
        if not math.isclose(geometric_mean(float(t["crps_ratio"]) for t in subset),
                            float(row["crps_geo"]), rel_tol=1e-12):
            raise ValueError(f"Invalid archived CRPS for {row['run']}")
    return rows


def config_for(root: Path, row: dict, workers: int) -> dict:
    folder = REPO / f"experiments/episynth/series{row['series']}_2026-10-19/runs"
    config = json.loads((folder / row["run"] / "run.json").read_text())
    train = config["train"]
    if (config["data"]["generator"], train["init_seed"], train["n_windows"]) != (
        row["generator"], int(row["init_seed"]), int(row["n_windows"])
    ):
        raise ValueError(f"Mismatched run {row['run']}")
    config["context_table"] = None
    config["data"]["corpus"] = []
    train["stream_workers"] = workers
    if row["generator"] == "kernelsynth":
        config["data"]["corpus"] = [
            str(root / "data/kernelsynth" / str(start)) for start, _ in kernel_parts(train["n_windows"])
        ]
        train["stream_workers"] = 0
    config["estimator"] = "v_information"
    config["estimator_config"] = {"variant": "ema", "test_windows": 100_000, "test_first_seed": 2 * 10**9}
    return config


def prepare(root: Path, workers: int) -> dict:
    configs = {}
    (root / "configs").mkdir(parents=True, exist_ok=True)
    for row in archive():
        run = row["run"]
        config = config_for(root, row, workers)
        path = root / "configs" / f"{run}.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Existing config differs at {path}")
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(config, indent=2))
            tmp.replace(path)
        configs[run] = config
    return configs


def warm(root: Path, run: str) -> None:
    """Publish train normalization without overwriting a live estimator manifest."""
    from episynth.estimators.v_information import VInformationEstimator
    from episynth.train import build_generator, load_run_config
    from episynth.training import build_model

    out = root / "runs" / run
    target = out / "v_information.json"
    if target.exists():
        return
    if any((out / branch).exists() for branch in ("conditional", "unconditional")):
        raise ValueError(f"Branch without estimator manifest at {out}")
    config = load_run_config(root / "configs" / f"{run}.json")
    staging = root / "cache/normalization" / run
    staging.mkdir(parents=True, exist_ok=True)
    with (staging / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return
        generator = build_generator(config.data, config.spec.series_length)
        model = build_model(config.model, config.train.init_seed)
        VInformationEstimator(**config.estimator_config).prepare(model, generator, config.spec,
                                                                  config.train, staging)
        out.mkdir(parents=True, exist_ok=True)
        # A GPU job can finish its own normalization while the CPU job is running.
        with suppress(FileExistsError):
            os.link(staging / target.name, target)


def fit_conditional(root: Path, run: str, workers: int) -> None:
    """Fit on a spare GPU and publish the completed branch if the main fit still needs it."""
    from episynth.estimators.v_information import VInformationEstimator
    from episynth.model.moirai import MoiraiConfig
    from episynth.train import build_generator, load_run_config
    from episynth.training import Trainer, build_model

    out = root / "runs" / run
    destination = out / "conditional"
    staging = root / "runs/.prepared-branches" / run / "conditional"
    staging.parent.mkdir(parents=True, exist_ok=True)
    with (out / "conditional.prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if destination.exists():
            return
        config = load_run_config(root / "configs" / f"{run}.json")
        generator = build_generator(config.data, config.spec.series_length)
        model = build_model(config.model, config.train.init_seed)
        estimator = VInformationEstimator(**config.estimator_config)
        if not (out / "v_information.json").is_file():
            raise ValueError(f"Wait for train normalization at {out}")
        manifest = estimator.prepare(model, generator, config.spec, config.train, out)
        model.config = MoiraiConfig.model_validate(manifest["models"]["conditional"])
        runtime = config.train.model_copy(update={"stream_workers": workers})
        Trainer(model, generator, config.spec, runtime, staging).run()
        # Exclusive publication leaves a concurrently started main branch intact.
        with suppress(FileExistsError):
            destination.symlink_to(os.path.relpath(staging, out), target_is_directory=True)


def plot(root: Path) -> None:
    from episynth.estimators.v_information import VInformationEstimator
    from episynth.train import RunConfig

    rows = []
    for row in archive():
        run = row["run"]
        result = VInformationEstimator().from_run_dir(root / "runs" / run)
        request = result["metadata"]["request"]
        config = RunConfig.model_validate(config_for(root, row, request["train"]["stream_workers"]))
        if (request["train"] != config.train.model_dump(mode="json")
                or request["spec"] != config.spec.model_dump(mode="json")
                or request["model"] != config.model.model_dump(mode="json")
                or request["source"]["name"] != row["generator"]
                or request["splits"]["test"] != {"first_seed": 2 * 10**9, "n_windows": 100_000}
                or request["variant"] != "ema"):
            raise ValueError(f"Unexpected measurement protocol for {run}")
        point = {k: row[k] for k in ("series", "run", "generator", "init_seed", "n_windows", "crps_geo")}
        point.update({k: result[k] for k in ("v_information_bits_per_window",
                                             "standard_error_bits_per_window", "n_test_windows")})
        rows.append(point)
    render(root, rows)


def render(root: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = root / "results"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / "v_information_vs_crps"
    ink, point, leader = "#203245", "#17688a", "#92a3b2"
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.95), sharey=True)
    fig.subplots_adjust(left=0.077, right=0.985, bottom=0.21, top=0.87, wspace=0.12)
    for series, ax in enumerate(axes, 1):
        ax.spines[["right", "top"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#818891")
        ax.tick_params(colors="#525b66", labelsize=8, length=2.5, width=0.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e9ee", lw=0.65)
        ax.axhline(1, color="#647589", lw=0.8, ls=(0, (4, 3)))
        ax.set_title(f"{2 if series == 1 else 40}M training windows", loc="left",
                     color=ink, fontsize=10, weight="semibold")
        ax.set_ylim(0.55, 1.72)
        ax.set_yticks((0.6, 0.8, 1.0, 1.2, 1.4, 1.6))
        ax.margins(x=0.08)
        for row in rows:
            if int(row["series"]) != series:
                continue
            x, y = row["v_information_bits_per_window"], float(row["crps_geo"])
            repeat = int(row["init_seed"]) == 1
            label = NAMES[row["generator"]] + (" (seed 1)" if repeat else "")
            key = "repeat" if repeat else row["generator"]
            ax.scatter(x, y, s=44 if repeat else 20, facecolors="none" if repeat else point,
                       edgecolors="#bb682c" if repeat else "white", lw=1.2 if repeat else 0.45, zorder=3)
            ax.annotate(label, (x, y), xytext=LABELS[series][key], textcoords="axes fraction",
                        ha="center", va="center", fontsize=7.8, color=ink,
                        bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "linewidth": 0},
                        arrowprops={"arrowstyle": "-", "color": leader, "lw": 0.65,
                                    "shrinkA": 2, "shrinkB": 3.5 if repeat else 2.5})
    axes[0].set_ylabel("CRPS / seasonal naive", fontsize=9, color=ink)
    fig.supxlabel("Predictive V-information (bits per forecast window)", x=0.53, y=0.025, fontsize=9)
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext), dpi=200)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "warm", "measure", "fit-conditional", "plot"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run", help="Archived run ID for preparation or a branch fit")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.workers)
    elif args.action in {"warm", "measure", "fit-conditional"}:
        if args.run not in {row["run"] for row in archive()}:
            parser.error("Select an archived run ID with --run")
        if args.action == "warm":
            os.nice(5)
            warm(args.root.resolve(), args.run)
        elif args.action == "measure":
            root = args.root.resolve()
            warm(root, args.run)
            os.execv(sys.executable, [sys.executable, "-u", "-m", "episynth.train",
                     "--config", str(root / "configs" / f"{args.run}.json"),
                     "--out", str(root / "runs" / args.run)])
        elif args.workers >= 0:
            fit_conditional(args.root.resolve(), args.run, args.workers)
        else:
            parser.error("Use a nonnegative worker count")
    else:
        plot(args.root.resolve())
