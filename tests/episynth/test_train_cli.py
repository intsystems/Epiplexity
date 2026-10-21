"""Смесь генераторов и запуск прогона по файлу конфигурации."""

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
    """Сид `s` даёт ряд компоненты `s % k` с сидом `s // k`; смесь и компоненты согласованы по длине."""
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
    """Переопределения вложенных полей читаются как JSON; таблица длин контекста подставляется в форму задачи."""
    raw = apply_overrides({"train": {"lr": 1e-3}}, ["train.lr=3e-4", "data.generator=sde", "model.patch_sizes=[16]"])
    assert raw == {"train": {"lr": 3e-4}, "data": {"generator": "sde"}, "model": {"patch_sizes": [16]}}
    table = tmp_path / "ctx.json"
    table.write_text(json.dumps({"lengths": [64, 128], "probs": [0.25, 0.75]}))
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps({"data": {"generator": "sde"}, "train": {"n_windows": 8, "batch_size": 4}, "context_table": str(table)}))
    cfg = load_run_config(cfg_path, ["spec.context_max=128", "spec.horizon=32", "spec.series_length=256"])
    assert cfg.spec.context_lengths == (64, 128) and cfg.spec.context_probs == (0.25, 0.75)
    with pytest.raises(ValueError, match="патча"):
        RunConfig(data=DataConfig(generator="sde"), train={"n_windows": 8, "batch_size": 4}, spec={"patch_size": 16, "context_max": 128, "horizon": 32, "series_length": 256, "context_lengths": (128,)})


def test_build_generator_from_corpus_and_mixture(tmp_path):
    """Корпус строится по каталогам с проверкой имени и длины; смесь — рекурсивно из компонент."""
    out = tmp_path / "waveform"
    generate_corpus("waveform", first_seed=0, n=8, out=out, length=128, workers=1)
    gen = build_generator(DataConfig(generator="waveform", corpus=[str(out)]), length=128)
    assert gen.name == "waveform" and gen.generate([3]).shape == (1, 128)
    with pytest.raises(ValueError, match="корпус генератора"):
        build_generator(DataConfig(generator="sde", corpus=[str(out)]), length=128)
    with pytest.raises(ValueError, match="длина ряда"):
        build_generator(DataConfig(generator="waveform", corpus=[str(out)]), length=256)
    mix = build_generator(DataConfig(mixture=[DataConfig(generator="waveform", corpus=[str(out)]), DataConfig(generator="sde")]), length=128)
    assert mix.name == "mixture(waveform,sde)" and mix.generate([0, 1]).shape == (2, 128)


def test_cli_runs_tiny_training(tmp_path, monkeypatch):
    """Прогон из файла конфигурации на CPU пишет `run.json` и артефакты обучения."""
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
    # выбранный по умолчанию оценщик пишет результат и финальный лосс на отложенном наборе
    result = json.loads((out / "estimator_result.json").read_text())
    assert {"model_part_bits", "data_part_bits", "total_bits"} <= set(result)
    assert (out / "heldout_ema_1.f32").exists()


def test_seed_range_check(tmp_path):
    """Сиды за пределами корпуса обнаруживаются до запуска, у смеси — с учётом деления сидов между компонентами."""
    from episynth.train import check_seed_ranges, seeds_available
    from episynth.training import TrainConfig

    out = tmp_path / "waveform"
    generate_corpus("waveform", first_seed=0, n=16, out=out, length=128, workers=1)
    held = tmp_path / "heldout"
    generate_corpus("waveform", first_seed=1000, n=8, out=held, length=128, workers=1)
    corpus = build_generator(DataConfig(generator="waveform", corpus=[str(out), str(held)]), length=128)
    assert seeds_available(corpus, 0, 16) and not seeds_available(corpus, 0, 17) and seeds_available(corpus, 1000, 8)
    check_seed_ranges(corpus, TrainConfig(n_windows=16, batch_size=4, heldout_first_seed=1000, heldout_windows=8))
    with pytest.raises(ValueError, match="обучающие"):
        check_seed_ranges(corpus, TrainConfig(n_windows=20, batch_size=4, heldout_first_seed=1000, heldout_windows=8))
    with pytest.raises(ValueError, match="отложенные"):
        check_seed_ranges(corpus, TrainConfig(n_windows=16, batch_size=4, heldout_first_seed=1000, heldout_windows=9))
    mix = build_generator(DataConfig(mixture=[DataConfig(generator="waveform", corpus=[str(out)]), DataConfig(generator="sde")]), length=128)
    assert seeds_available(mix, 0, 32) and not seeds_available(mix, 0, 33)
