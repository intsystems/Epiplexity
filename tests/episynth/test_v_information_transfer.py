import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "experiments/episynth/scripts/v_information_transfer.py"


def test_production_matrix(tmp_path):
    workflow = runpy.run_path(str(SCRIPT))
    configs = workflow["prepare"](tmp_path, workers=4)
    assert len(configs) == 23
    assert sum(2 * c["train"]["n_windows"] // 1024 for c in configs.values()) == 980454
    assert configs["s2_waveform_seed1"]["train"]["init_seed"] == 1
    for name, config in configs.items():
        assert config["context_table"] is None
        assert len(config["spec"]["context_probs"]) == len(config["spec"]["context_lengths"])
        assert config["estimator"] == "v_information"
        assert config["estimator_config"]["test_windows"] == 100000
        assert config["train"]["stream_workers"] == (0 if "kernelsynth" in name else 4)
    for name in ("series_kernelsynth", "s2_kernelsynth"):
        parts = configs[name]["data"]["corpus"]
        assert all(Path(p).is_relative_to(tmp_path / "data") for p in parts)
        assert str(tmp_path / "data/kernelsynth/2000000000") in parts


def test_corpus_coverage():
    workflow = runpy.run_path(str(SCRIPT))
    for budget in (1999872, 39999488):
        parts = workflow["kernel_parts"](budget)
        assert parts[-2:] == [(10**9, 100000), (2 * 10**9, 100000)]
        cursor = 0
        for start, size in parts[:-2]:
            assert start == cursor
            cursor += size
        assert cursor == budget


def test_plot_rejects_missing_measurements(tmp_path):
    workflow = runpy.run_path(str(SCRIPT))
    with pytest.raises(FileNotFoundError):
        workflow["plot"](tmp_path)
    assert not (tmp_path / "results/v_information_vs_crps.csv").exists()


def test_render_signed_information_and_repeat(tmp_path):
    import csv

    workflow = runpy.run_path(str(SCRIPT))
    rows = [
        {**{k: r[k] for k in ("series", "run", "generator", "init_seed", "n_windows", "crps_geo")},
         "v_information_bits_per_window": float(i - 1),
         "standard_error_bits_per_window": 0.25, "n_test_windows": 100_000}
        for i, r in enumerate(workflow["archive"]())
    ]
    workflow["render"](tmp_path, rows)
    base = tmp_path / "results/v_information_vs_crps"
    with base.with_suffix(".csv").open() as handle:
        points = list(csv.DictReader(handle))
    assert len(points) == 23
    assert float(points[0]["v_information_bits_per_window"]) == -1.0
    repeat = [p for p in points if int(p["init_seed"]) == 1]
    assert len(repeat) == 1 and float(repeat[0]["crps_geo"]) == pytest.approx(0.97151083774)
    assert base.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert base.with_suffix(".png").read_bytes().startswith(b"\x89PNG")


def test_sweep_dispatches_each_fit_once(monkeypatch, tmp_path):
    from unittest.mock import Mock

    import episynth.corpus

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import run_v_information_transfer as launcher

    monkeypatch.setattr(episynth.corpus, "available_cpus", lambda: 96)
    (tmp_path / "logs").mkdir()
    shards = set()
    trained = []
    warmed = []
    monkeypatch.setattr(launcher, "ready_shard", lambda root, start, n: start in shards)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    plot = Mock()
    monkeypatch.setattr(launcher, "plot", plot)

    def popen(cmd, *, env, **_kwargs):
        if "episynth.corpus" in cmd:
            assert env["CUDA_VISIBLE_DEVICES"]
            first = int(cmd[cmd.index("--first-seed") + 1])
            assert first not in shards
            shards.add(first)
        elif "measure" in cmd:
            assert env["CUDA_VISIBLE_DEVICES"]
            run = cmd[cmd.index("--run") + 1]
            if "kernelsynth" in run:
                budget = launcher.LARGE if run.startswith("s2_") else launcher.SMALL
                assert shards.issuperset(start for start, _ in launcher.kernel_parts(budget))
            trained.append(run)
        else:
            assert env["CUDA_VISIBLE_DEVICES"] == "" and "warm" in cmd
            warmed.append(cmd[-1])
        return Mock(pid=123, poll=Mock(side_effect=[None, 0]))

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    launcher.run(tmp_path, list(map(str, range(8))), workers=8, hours=1)
    assert len(trained) == len(set(trained)) == 23
    assert len(shards) == len(launcher.kernel_parts(launcher.LARGE))
    assert warmed and len(warmed) == len(set(warmed))
    plot.assert_called_once_with(tmp_path)


def test_staged_conditional_matches_serial_fit(tmp_path):
    import json

    from episynth.estimators.v_information import VInformationEstimator
    from episynth.train import RunConfig, build_generator
    from episynth.training import build_model

    workflow = runpy.run_path(str(SCRIPT))
    raw = json.loads((workflow["REPO"] / "experiments/demos/v_information.json").read_text())
    config = RunConfig.model_validate(raw)
    root = tmp_path / "experiment"
    run = "series_waveform"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / f"{run}.json").write_text(json.dumps(raw))
    generator = build_generator(config.data, config.spec.series_length)
    model = build_model(config.model, config.train.init_seed)
    estimator = VInformationEstimator(**config.estimator_config)
    reference = estimator.run(model, generator, config.spec, config.train, tmp_path / "reference")
    out = root / "runs" / run
    manifest = estimator.prepare(model, generator, config.spec, config.train, out)
    manifest["models"] = reference["metadata"]["models"]
    (out / "v_information.json").write_text(json.dumps(manifest))
    workflow["fit_conditional"](root, run, workers=0)
    assert (out / "conditional").is_symlink()
    result = estimator.run(model, generator, config.spec, config.train, out)
    assert result["v_information_bits_per_window"] == reference["v_information_bits_per_window"]
    assert result["metadata"]["selection"] == reference["metadata"]["selection"]
