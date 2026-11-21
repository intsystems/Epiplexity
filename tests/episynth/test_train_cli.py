"""Mix generators and run through the configuration file."""

import json

import numpy as np
import pytest

from episynth.corpus import generate_corpus
from episynth.generators import GENERATORS, Mixture
from episynth.train import (
    DataConfig,
    RunConfig,
    apply_overrides,
    build_generator,
    load_run_config,
    main,
)


def test_mixture_routes_seeds_round_robin():
    """Side `s` gives a number of components `s % k` with `s // k`; the mixture and components are consistent in length."""
    a, b = GENERATORS["waveform"](length=128), GENERATORS["stepfunction"](length=128)
    mix = Mixture([a, b])
    assert mix.name == "mixture(waveform,stepfunction)" and mix.length == 128
    x = mix.generate([0, 1, 2, 7])
    assert np.array_equal(x[0], a.generate([0])[0]) and np.array_equal(x[1], b.generate([0])[0])
    assert np.array_equal(x[2], a.generate([1])[0]) and np.array_equal(x[3], b.generate([3])[0])
    assert np.array_equal(mix.generate([7], exact=True)[0], b.generate([3], exact=True)[0])
    with pytest.raises(ValueError):
        Mixture([a, GENERATORS["waveform"](length=64)])


def test_overrides_and_resolution(tmp_path):
    """Overdefinitions of nested fields are read as JSON; a table of context lengths is inserted into the form of a problem."""
    raw = apply_overrides({"train": {"lr": 1e-3}}, ["train.lr=3e-4", "data.generator=sde", "model.patch_sizes=[16]"])
    assert raw == {"train": {"lr": 3e-4}, "data": {"generator": "sde"}, "model": {"patch_sizes": [16]}}
    table = tmp_path / "ctx.json"
    table.write_text(json.dumps({"lengths": [64, 128], "probs": [0.25, 0.75]}))
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({"data": {"generator": "sde"}, "train": {"n_windows": 8, "batch_size": 4}, "context_table": str(table)}))
    cfg = load_run_config(cfg_path, ["spec.context_max=128", "spec.horizon=32", "spec.series_length=256"])
    assert cfg.spec.context_lengths == (64, 128) and cfg.spec.context_probs == (0.25, 0.75)
    with pytest.raises(ValueError, match="Patch size"):
        RunConfig(data=DataConfig(generator="sde"), train={"n_windows": 8, "batch_size": 4}, spec={"patch_size": 16, "context_max": 128, "horizon": 32, "series_length": 256, "context_lengths": (128,)})


def test_build_generator_from_corpus_and_mixture(tmp_path):
    """The body is built according to catalogs with a name and length check; the mixture is recursive from the components."""
    out = tmp_path / "waveform"
    generate_corpus("waveform", first_seed=0, n=8, out=out, length=128, workers=1)
    gen = build_generator(DataConfig(generator="waveform", corpus=[str(out)]), length=128)
    assert gen.name == "waveform" and gen.generate([3]).shape == (1, 128)
    with pytest.raises(ValueError, match="Corpus generator"):
        build_generator(DataConfig(generator="sde", corpus=[str(out)]), length=128)
    with pytest.raises(ValueError, match="Corpus series length"):
        build_generator(DataConfig(generator="waveform", corpus=[str(out)]), length=256)
    mix = build_generator(DataConfig(mixture=[DataConfig(generator="waveform", corpus=[str(out)]), DataConfig(generator="sde")]), length=128)
    assert mix.name == "mixture(waveform,sde)" and mix.generate([0, 1]).shape == (2, 128)


def test_cli_runs_tiny_training(tmp_path, monkeypatch):
    """Run from the configuration file on CPU writes `run.json` and learning artifacts."""
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data": {"generator": "waveform"},
                "spec": {"series_length": 256, "context_max": 128, "horizon": 32, "patch_size": 16, "context_lengths": [64, 128]},
                "model": {"d_model": 64, "num_layers": 1, "patch_sizes": [16], "max_seq_len": 32, "components": ["student_t", "normal_fixed", "log_normal"]},
                "train": {"n_windows": 8, "batch_size": 4, "warmup_steps": 1, "heldout_windows": 4, "device": "cpu", "precision": "fp32", "fused_adam": False, "log_every": 1},
            }
        )
    )
    out = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["train", "--config", str(cfg_path), "--out", str(out), "--set", "train.lr=5e-4"])
    main()
    assert json.loads((out / "run.json").read_text())["train"]["lr"] == 5e-4
    assert (out / "config.json").exists() and (out / "steps.csv").exists() and (out / "last.pt").exists()
    assert np.fromfile(out / "window_loss.f32", dtype=np.float32).shape == (8,)
    # the default evaluator writes the result and the final loss on the held-out set
    result = json.loads((out / "estimator_result.json").read_text())
    assert {"model_part_bits", "data_part_bits", "total_bits"} <= set(result)
    assert (out / "heldout_ema_1.f32").exists()
    # Operational overrides can resume the same experiment without changing losses.
    losses = (out / "window_loss.f32").read_bytes()
    monkeypatch.setattr("sys.argv", ["train", "--config", str(cfg_path), "--out", str(out),
                                    "--set", "train.lr=5e-4", "--set", "train.eval_batch_size=2"])
    main()
    assert (out / "window_loss.f32").read_bytes() == losses


@pytest.mark.parametrize("override", ["data.generator_args.irregular_prob=1.0", "train.first_seed=100"])
def test_cli_rejects_changed_experiment_before_writing(tmp_path, monkeypatch, override):
    """A reused directory cannot silently represent different data or seed ranges."""
    config = RunConfig(data=DataConfig(generator="timesynth", generator_args={"irregular_prob": 0.0}),
                       train={"n_windows": 8, "batch_size": 4})
    config_path = tmp_path / "request.json"
    config_path.write_text(config.model_dump_json())
    out = tmp_path / "existing"
    out.mkdir()
    (out / "run.json").write_text(config.model_dump_json())
    (out / "window_loss.f32").write_bytes(b"existing measurements")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config_path), "--out", str(out), "--set", override])
    with pytest.raises(ValueError, match="incompatible run configuration"):
        main()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_cli_rejects_overlapping_splits_before_creating_run(tmp_path, monkeypatch):
    """Split validation precedes model construction and writing run metadata."""
    config = RunConfig(data=DataConfig(generator="waveform"),
                       train={"n_windows": 8, "batch_size": 4, "heldout_first_seed": 4, "heldout_windows": 4})
    config_path = tmp_path / "request.json"
    config_path.write_text(config.model_dump_json())
    out = tmp_path / "overlap"
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config_path), "--out", str(out)])
    with pytest.raises(ValueError, match="overlap"):
        main()
    assert not out.exists()


def test_seed_range_check(tmp_path):
    """Sids outside the housing are found before starting, in the mixture - taking into account the division of the seats between the components."""
    from episynth.train import check_seed_ranges, seeds_available
    from episynth.training import TrainConfig

    out = tmp_path / "waveform"
    generate_corpus("waveform", first_seed=0, n=16, out=out, length=128, workers=1)
    held = tmp_path / "heldout"
    generate_corpus("waveform", first_seed=1000, n=8, out=held, length=128, workers=1)
    corpus = build_generator(DataConfig(generator="waveform", corpus=[str(out), str(held)]), length=128)
    assert seeds_available(corpus, 0, 16) and not seeds_available(corpus, 0, 17) and seeds_available(corpus, 1000, 8)
    check_seed_ranges(corpus, TrainConfig(n_windows=16, batch_size=4, heldout_first_seed=1000, heldout_windows=8))
    with pytest.raises(ValueError, match="training"):
        check_seed_ranges(corpus, TrainConfig(n_windows=20, batch_size=4, heldout_first_seed=1000, heldout_windows=8))
    with pytest.raises(ValueError, match="held-out"):
        check_seed_ranges(corpus, TrainConfig(n_windows=16, batch_size=4, heldout_first_seed=1000, heldout_windows=9))
    mix = build_generator(DataConfig(mixture=[DataConfig(generator="waveform", corpus=[str(out)]), DataConfig(generator="sde")]), length=128)
    assert seeds_available(mix, 0, 32) and not seeds_available(mix, 0, 33)
