"""Check the two-panel measurement against the estimator across a restart."""

import json
import sys
from pathlib import Path

import pytest

from episynth.estimators import build
from episynth.train import DataConfig, RunConfig, build_generator
from episynth.training import build_model

SCRIPTS = Path(__file__).resolve().parents[2] / "experiments/episynth/scripts"
sys.path.insert(0, str(SCRIPTS))


def test_restart_matches_estimator(tmp_path):
    from requential_transfer import measure

    base = json.loads(
        (SCRIPTS.parents[1] / "demos/requential.json").read_text()
    )
    cfg = RunConfig.model_validate(base)
    cfg.estimator_config.update(
        projection_steps=[2], mc_repeats=3, sampling_seed=0
    )
    model = build_model(cfg.model, cfg.train.init_seed)
    source = build_generator(
        DataConfig(generator="waveform"), cfg.spec.series_length
    )
    expected = build("requential", **cfg.estimator_config).run(
        model, source, cfg.spec, cfg.train, tmp_path / "reference"
    )
    out = tmp_path / "sweep"
    measure(cfg, out, [12], 0)
    first = json.loads((out / "12/measurement.json").read_text())
    assert (
        first["student_steps"] == 2 and not first["projections"][0]["complete"]
    )
    measure(cfg, out, [12, 32], 0)
    actual = json.loads((out / "32/measurement.json").read_text())
    assert actual["first_seed"] == 0
    for key in (
        "estimated_kl_bits",
        "estimated_overhead_bits",
        "plugin_expected_length_bound_bits",
        "estimated_kl_mc_se_bits",
        "real_windows",
        "student_steps",
        "message_count",
    ):
        assert actual[key] == pytest.approx(expected[key], abs=1e-9)
    assert actual["heldout_nats_per_window"] == pytest.approx(
        expected["heldout_nats_per_window"]
    )
    assert actual["projections"] == expected["projections"]


def test_plot_joins_archived_crps_and_preserves_units(tmp_path):
    import csv

    pytest.importorskip("matplotlib")
    import requential_transfer as sweep

    archived = sweep.archive()
    for idx, row in enumerate(archived):
        n, seed = int(row["n_windows"]), int(row["init_seed"])
        name = "s2_" + row["generator"] + ("_seed1" if seed else "")
        if row["generator"] == "kernelsynth" and n == sweep.SMALL:
            name = "series_kernelsynth"
        folder = tmp_path / "runs" / name / str(n)
        folder.mkdir(parents=True)
        point = {
            "generator": row["generator"],
            "init_seed": seed,
            "real_windows": n,
            "estimated_kl_bits": (idx + 1) * 1e6,
            "estimated_kl_mc_se_bits": 200.0,
            "estimated_overhead_bits": 10.0,
            "plugin_expected_length_bound_bits": 20.0,
            "student_steps": n // 1024,
            "message_count": n,
            "projections": [],
        }
        (folder / "measurement.json").write_text(json.dumps(point))
    sweep.plot(tmp_path)
    stem = tmp_path / "results/requential_vs_crps"
    assert stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with stem.with_suffix(".csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["crps_geo"] for r in rows] == [r["crps_geo"] for r in archived]
    assert [float(r["estimated_kl_bits"]) for r in rows] == [
        (i + 1) * 1e6 for i in range(23)
    ]
    sweep.plot(tmp_path, sweep.SMALL)
    with (tmp_path / "results/requential_vs_crps_2m.csv").open() as f:
        small = list(csv.DictReader(f))
    assert small == [row for row in rows if row["series"] == "1"]
