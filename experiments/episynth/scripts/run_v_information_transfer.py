"""Run the Figure 4 sweep on an allocated GPU node. Relaunch to resume."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from v_information_transfer import (
    LARGE,
    REPO,
    ROOT,
    SMALL,
    kernel_parts,
    plot,
    prepare,
)


def ready_shard(root: Path, first: int, size: int) -> bool:
    import numpy as np

    folder = root / "data/kernelsynth" / str(first)
    path = folder / "manifest.json"
    if not path.exists():
        return False
    m = json.loads(path.read_text())
    if (
        m["generator"],
        m["first_seed"],
        m["n"],
        m["length"],
        m["nonfinite_rows"],
        m["generator_args"],
    ) != (
        "kernelsynth",
        first,
        size,
        1024,
        0,
        {"max_kernels": 5, "chunk": 64, "device": "cuda"},
    ):
        raise ValueError(f"Invalid shard manifest at {path}")
    data = np.load(folder / "series.npy", mmap_mode="r")
    if data.shape != (size, 1024) or data.dtype != np.float32:
        raise ValueError(f"Invalid corpus array at {folder}")
    return True


def run(
    root: Path,
    gpus: list[str],
    workers: int,
    hours: float,
    *,
    ongoing=None,
    cpu_prefetch: bool = True,
) -> None:
    from episynth.corpus import available_cpus
    from episynth.estimators.v_information import VInformationEstimator

    configs = prepare(root, workers)
    parts = kernel_parts(LARGE)
    verified = {
        str(root / "data/kernelsynth" / str(s))
        for s, n in parts
        if ready_shard(root, s, n)
    }
    active = dict(ongoing or {})
    in_flight = {tag for tag, _, _ in active.values()}
    shards = deque(
        (s, n)
        for s, n in parts
        if str(root / "data/kernelsynth" / str(s)) not in verified
        and f"kernel-{s}" not in in_flight
    )
    shard_jobs = {f"kernel-{s}": (s, n) for s, n in parts}
    # Produce both holdouts and the short prefix early, then extend the same
    # corpus.
    shards = deque(
        sorted(shards, key=lambda p: (p[0] < 10**9 and p[0] >= SMALL, p[0]))
    )
    done = set()
    for name in configs:
        if (root / "runs" / name / "estimator_result.json").exists():
            VInformationEstimator().from_run_dir(root / "runs" / name)
            done.add(name)
    runs = [name for name in configs if name not in done | in_flight]
    runs.sort(key=lambda name: configs[name]["train"]["n_windows"])
    failed = []
    staged_fbm = False
    cpu_slots = min(
        2,
        max(
            0, (available_cpus() - len(gpus) * (workers + 1)) // (workers + 1)
        ),
    )
    cpu_keys = [f"cpu-{i}" for i in range(cpu_slots if cpu_prefetch else 0)]
    # Prepare later runs first so their CPU passes overlap earlier GPU fits.
    warmups = [
        n
        for n in reversed(runs)
        if n.startswith("s2_") and not configs[n]["data"]["corpus"]
    ]
    deadline = time.monotonic() + hours * 3600

    def launch(slot, gpu, tag, argv):
        # The accelerator loads compiled libraries by path. Keep each writer on
        # POSIX storage.
        cache_home = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        )
        compiler_cache = cache_home / "episynth-compile" / f"gpu-{gpu}"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(REPO / "src"), OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   TORCHINDUCTOR_COMPILE_THREADS="2",
                   TORCHINDUCTOR_CACHE_DIR=str(compiler_cache / "inductor"),
                   TRITON_CACHE_DIR=str(compiler_cache / "triton"),
                   MPLCONFIGDIR=str(root / "cache/matplotlib"))
        with (root / "logs" / f"{tag}.log").open("a") as log:
            proc = subprocess.Popen(
                [sys.executable, *map(str, argv)],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        active[slot] = (tag, proc, time.monotonic())
        print(f"Slot {slot} started {tag} pid={proc.pid}", flush=True)

    def launch_shard(slot, gpu):
        first, size = shards.popleft()
        launch(
            slot,
            gpu,
            f"kernel-{first}",
            [
                "-m",
                "episynth.corpus",
                "--generator",
                "kernelsynth",
                "--first-seed",
                first,
                "--n",
                size,
                "--workers",
                1,
                "--generator-arg",
                "device=cuda",
                "--out",
                root / "data/kernelsynth" / str(first),
            ],
        )

    def stage_fbm(slot, gpu):
        nonlocal staged_fbm
        fbm = root / "runs/s2_fbm"
        if (
            staged_fbm
            or runs
            or shards
            or not (fbm / "unconditional/config.json").exists()
        ):
            return
        if (fbm / "conditional").exists():
            return
        # Leave cores for active fits and use spare capacity for the slowest
        # branch.
        spare = available_cpus() - len(active) * (workers + 1)
        branch_workers = min(24, max(workers, spare - 1))
        launch(
            slot,
            gpu,
            "conditional-s2_fbm",
            [
                Path(__file__).with_name("v_information_transfer.py"),
                "fit-conditional",
                "--root",
                root,
                "--run",
                "s2_fbm",
                "--workers",
                branch_workers,
            ],
        )
        staged_fbm = True

    try:
        while active or runs or shards:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Allocation time exhausted. Relaunch to resume from"
                    " checkpoints."
                )
            for slot, (tag, proc, start) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                print(
                    f"Slot {slot} finished {tag} exit={code}"
                    f" seconds={time.monotonic()-start:.1f}",
                    flush=True,
                )
                del active[slot]
                if code == 0 and tag in shard_jobs:
                    first, size = shard_jobs[tag]
                    if not ready_shard(root, first, size):
                        raise RuntimeError(
                            f"Missing completed corpus shard {tag}"
                        )
                    verified.add(str(root / "data/kernelsynth" / str(first)))
                if (
                    code
                    and slot not in cpu_keys
                    and tag != "conditional-s2_fbm"
                ):
                    failed.append(tag)
            for gpu in [*gpus, *cpu_keys]:
                auxiliary = f"corpus-{gpu}"
                if auxiliary in active:
                    continue
                if gpu in active:
                    tag = active[gpu][0]
                    # Normalization runs on CPU. A shard may overlap the first
                    # few fit steps.
                    if (
                        tag in configs
                        and not (
                            root / "runs" / tag / "v_information.json"
                        ).exists()
                    ):
                        if shards:
                            launch_shard(auxiliary, gpu)
                        else:
                            stage_fbm(auxiliary, gpu)
                    continue
                if gpu in cpu_keys:
                    while warmups:
                        name = warmups.pop(0)
                        if (
                            name in runs
                            and not (
                                root / "runs" / name / "v_information.json"
                            ).exists()
                        ):
                            launch(
                                gpu,
                                "",
                                f"normalize-{name}",
                                [
                                    Path(__file__).with_name(
                                        "v_information_transfer.py"
                                    ),
                                    "warm",
                                    "--root",
                                    root,
                                    "--run",
                                    name,
                                ],
                            )
                            break
                    continue
                candidates = [
                    n
                    for n in runs
                    if verified.issuperset(configs[n]["data"]["corpus"])
                ]
                prepared = next(
                    (
                        n
                        for n in candidates
                        if (root / "runs" / n / "v_information.json").is_file()
                    ),
                    None,
                )
                eligible = prepared or next(iter(candidates), None)
                if shards and (
                    eligible is None
                    or (
                        prepared is None
                        and gpu in gpus[: max(1, len(gpus) // 4)]
                    )
                ):
                    launch_shard(gpu, gpu)
                elif eligible is not None:
                    runs.remove(eligible)
                    launch(
                        gpu,
                        gpu,
                        eligible,
                        [
                            Path(__file__).with_name(
                                "v_information_transfer.py"
                            ),
                            "measure",
                            "--root",
                            root,
                            "--run",
                            eligible,
                        ],
                    )
                else:
                    stage_fbm(gpu, gpu)
            if not active and runs and not shards:
                raise RuntimeError(
                    f"Missing corpus shards for {runs}. Failed jobs: {failed}"
                )
            if active:
                time.sleep(3)
    finally:
        for _, proc, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for _, proc, _ in active.values():
            proc.wait()
    if failed:
        raise RuntimeError(f"Failed jobs: {failed}")
    plot(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--gpus", required=True, help="Comma-separated allocated GPU indices"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error(
            "The V-information GPU sweep requires POSIX file locking and process groups."
        )
    import fcntl

    if args.workers < 0 or not math.isfinite(args.hours) or args.hours <= 0:
        parser.error(
            "Use a nonnegative worker count and a positive wall-time allowance"
        )

    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or not all(g.isdecimal() for g in gpus):
        parser.error("GPU indices must be distinct nonnegative integers")
    (args.root / "logs").mkdir(parents=True, exist_ok=True)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    with (args.root / "logs/sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.root.resolve(), gpus, args.workers, args.hours)
