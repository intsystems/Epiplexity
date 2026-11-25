"""Scientific definition, split isolation, leakage, CLI and resume checks."""

import copy
import json
import math

import numpy as np
import pytest
import torch

from episynth.corpus import Corpus, CorpusManifest
from episynth.estimators import EpiplexityEstimator, available, build
from episynth.estimators.run import main as estimator_main
from episynth.estimators.v_information import (
    VInformationEstimator,
    _choose,
    _normalization,
    _weights_digest,
)
from episynth.generators import GENERATORS, Generator, Mixture
from episynth.model.moirai import MoiraiConfig, MoiraiModule
from episynth.splits import seeds_available, validate_splits
from episynth.train import main as train_main
from episynth.training import TrainConfig, build_model
from episynth.windows import WindowSpec, window_batch


@pytest.fixture
def setup():
    spec = WindowSpec(
        series_length=32,
        context_max=16,
        horizon=8,
        patch_size=4,
        context_lengths=(8, 16),
    )
    config = TrainConfig(
        n_windows=16,
        batch_size=4,
        warmup_steps=1,
        heldout_windows=8,
        heldout_first_seed=100,
        eval_batch_size=3,
        precision="fp32",
        device="cpu",
        fused_adam=False,
        checkpoint_fractions=(0.5, 1.0),
    )
    model = build_model(
        MoiraiConfig(
            d_model=64,
            num_layers=1,
            patch_sizes=(4,),
            max_seq_len=32,
            components=("student_t", "normal_fixed"),
        ),
        4,
    )
    return model, GENERATORS["waveform"](length=32), spec, config


def _save_measurement(out, unconditional, conditional):
    out.mkdir(exist_ok=True)
    unconditional = np.asarray(unconditional, dtype=np.float32)
    conditional = np.asarray(conditional, dtype=np.float32)
    unconditional.tofile(out / "test_unconditional.f32")
    conditional.tofile(out / "test_conditional.f32")
    (
        (unconditional.astype(float) - conditional.astype(float)) / math.log(2)
    ).tofile(out / "pvi_bits.f64")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "request": {
            "splits": {
                "test": {"first_seed": 0, "n_windows": len(unconditional)}
            }
        },
        "normalization": {"loc": 0.0, "scale": 1.0},
        "models": {},
        "selection": {},
        "artifacts": {
            "unconditional": "test_unconditional.f32",
            "conditional": "test_conditional.f32",
            "pvi": "pvi_bits.f64",
        },
    }
    (out / "v_information.json").write_text(json.dumps(manifest))


def test_signed_information_and_paired_error(tmp_path):
    _save_measurement(tmp_path, [1, 2, 3], [2, 4, 5])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = VInformationEstimator().from_run_dir(tmp_path)
    differences = np.array([-1, -2, -2]) / math.log(2)
    assert (
        result["method"] == "v_information"
        and result["units"] == "bits_per_window"
    )
    assert result["v_information_bits_per_window"] == pytest.approx(
        differences.mean()
    )
    assert result["standard_error_bits_per_window"] == pytest.approx(
        differences.std(ddof=1) / math.sqrt(3)
    )
    assert result["unconditional_entropy_bits_per_window"] == pytest.approx(
        2 / math.log(2)
    )
    assert result["conditional_entropy_bits_per_window"] == pytest.approx(
        11 / 3 / math.log(2)
    )
    assert (
        not {"model_part_bits", "data_part_bits", "total_bits"} & result.keys()
    )
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("rho", [0.0, 0.7])
def test_gaussian_information_matches_analytic_value(tmp_path, rho):
    rng = np.random.default_rng(7)
    x, noise = rng.standard_normal((2, 100_000))
    y = rho * x + math.sqrt(1 - rho**2) * noise
    unconditional = 0.5 * (math.log(2 * math.pi) + y**2)
    conditional = 0.5 * (math.log(2 * math.pi * (1 - rho**2)) + noise**2)
    _save_measurement(tmp_path, unconditional, conditional)
    result = VInformationEstimator().from_run_dir(tmp_path)
    expected = -0.5 * math.log2(1 - rho**2)
    assert (
        abs(result["v_information_bits_per_window"] - expected)
        <= 5 * result["standard_error_bits_per_window"] + 1e-7
    )


@pytest.mark.parametrize(
    "artifact",
    ["test_unconditional.f32", "test_conditional.f32", "pvi_bits.f64"],
)
@pytest.mark.parametrize(
    "damage", ["missing", "truncated", "nonfinite", "modified"]
)
def test_recompute_rejects_invalid_artifacts(tmp_path, artifact, damage):
    _save_measurement(tmp_path, [1, 2, 3], [2, 4, 5])
    path = tmp_path / artifact
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        dtype = np.float64 if artifact.endswith("f64") else np.float32
        values = np.fromfile(path, dtype=dtype)
        values[0] = float("nan") if damage == "nonfinite" else 100
        values.tofile(path)
    with pytest.raises(ValueError):
        VInformationEstimator().from_run_dir(tmp_path)


@pytest.mark.parametrize(
    "fixed", [(0, 0), (0, -1), (float("inf"), 1), (0, float("nan"))]
)
def test_invalid_fixed_normalization(fixed):
    with pytest.raises(ValueError, match="fixed_scaling"):
        MoiraiConfig(fixed_scaling=fixed)


def test_fixed_normalization_original_units_and_roundtrip(setup):
    model, generator, spec, _ = setup
    with pytest.raises(ValueError, match="fixed_scaling"):
        MoiraiConfig(scaling=False, fixed_scaling=(0, 1))
    config = model.config.model_dump()
    config.update(fixed_scaling=(2.0, 3.0))
    normalized = MoiraiModule(MoiraiConfig.model_validate(config)).eval()
    normalized.load_state_dict(model.state_dict())
    config.update(fixed_scaling=None, scaling=False)
    identity = MoiraiModule(MoiraiConfig.model_validate(config)).eval()
    identity.load_state_dict(model.state_dict())
    batch = window_batch(generator, [0, 1, 2], spec)
    z = copy.deepcopy(batch)
    z.target = (z.target - 2) / 3
    with torch.no_grad():
        actual = normalized(**batch.model_inputs(), patch_size=4)
        reference = identity(**z.model_inputs(), patch_size=4)
    assert torch.allclose(
        actual.log_prob(batch.target),
        reference.log_prob(z.target) - math.log(3),
        atol=1e-5,
    )
    restored = MoiraiModule(
        MoiraiConfig.model_validate_json(normalized.config.model_dump_json())
    ).eval()
    restored.load_state_dict(normalized.state_dict())
    with torch.no_grad():
        assert torch.equal(
            restored(**batch.model_inputs(), patch_size=4).logits,
            actual.logits,
        )


def test_null_predictor_ignores_context_layout_values_and_targets(setup):
    model, generator, spec, _ = setup
    model.config = model.config.model_copy(
        update={"ignore_context": True, "fixed_scaling": (1.0, 2.0)}
    )
    model.eval()
    batch = window_batch(generator, [0, 1, 2], spec)
    changed = copy.deepcopy(batch)
    changed.target.normal_(mean=100, std=50)
    changed.observed_mask.logical_not_()
    changed.sample_id.fill_(123)
    changed.time_id.zero_()
    changed.variate_id.fill_(17)
    changed.context_length.zero_()
    with torch.no_grad():
        a = model(**batch.model_inputs(), patch_size=4)
        b = model(**changed.model_inputs(), patch_size=4)
    mask = batch.prediction_mask
    assert torch.equal(a.logits[mask], b.logits[mask])
    assert torch.equal(
        a.log_prob(batch.target)[mask], b.log_prob(batch.target)[mask]
    )
    # Adding/removing padding tokens leaves the canonical horizon unchanged.
    inputs = {
        k: v[:, spec.n_context_tokens:]
        for k, v in batch.model_inputs().items()
    }
    with torch.no_grad():
        cropped = model(**inputs, patch_size=4)
    assert torch.allclose(
        a.logits[:, spec.n_context_tokens:], cropped.logits, atol=1e-6
    )


def test_conditional_predictions_do_not_observe_targets(setup):
    model, generator, spec, _ = setup
    batch = window_batch(generator, [0, 1, 2], spec)
    changed = copy.deepcopy(batch)
    changed.target[changed.prediction_mask] = 1e5
    with torch.no_grad():
        a = model(**batch.model_inputs(), patch_size=4)
        b = model(**changed.model_inputs(), patch_size=4)
    assert torch.equal(a.logits, b.logits)


def test_normalization_reads_train_horizons_only(setup):
    _, generator, spec, config = setup

    class Recorded(Generator):
        name = "recorded"

        def __init__(self, length=32):
            super().__init__(length)
            self.seeds = []

        def _batch(self, seeds):
            self.seeds.extend(seeds.tolist())
            return generator.generate(seeds)

    recorded = Recorded()
    loc, scale = _normalization(recorded, spec, config)
    assert recorded.seeds == list(range(config.n_windows))
    batch = window_batch(generator, np.arange(config.n_windows), spec)
    values = (
        batch.target[batch.prediction_mask.unsqueeze(-1) & batch.observed_mask]
        .double()
        .numpy()
    )
    assert loc == pytest.approx(values.mean(), abs=1e-12)
    assert scale == pytest.approx(math.sqrt(values.var() + 1e-5), abs=1e-12)


def _corpus(path, first, n, name="waveform"):
    path.mkdir()
    (path / "manifest.json").write_text(
        CorpusManifest(
            generator=name, length=32, first_seed=first, n=n
        ).model_dump_json()
    )
    np.save(path / "series.npy", np.zeros((n, 32), dtype=np.float32))
    return path


def test_split_coverage_adjacent_corpora_and_exact_mixture_projection(
    tmp_path,
):
    a = _corpus(tmp_path / "a", 0, 4)
    b = _corpus(tmp_path / "b", 4, 4)
    corpus = Corpus(a, b)
    assert seeds_available(corpus, 1, 7)
    assert not seeds_available(corpus, 1, 8)
    other = Corpus(_corpus(tmp_path / "other", 0, 1, "sde"))
    mixture = Mixture([corpus, other])
    assert seeds_available(mixture, 2, 1)  # Only component 0, seed 1.
    assert not seeds_available(mixture, 3, 1)
    with pytest.raises(ValueError, match="not covered"):
        validate_splits(mixture, {"test": {"first_seed": 3, "n_windows": 1}})


def test_split_overlap_after_mixture_projection(setup):
    _, generator, _, _ = setup
    mixture = Mixture([generator, generator])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(mixture, {"train": {"first_seed": 0, "n_windows": 1},
                                  "test": {"first_seed": 1, "n_windows": 1}})
    nested = Mixture([mixture, GENERATORS["sde"](length=32)])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(nested, {"train": {"first_seed": 0, "n_windows": 1},
                                 "test": {"first_seed": 2, "n_windows": 1}})


@pytest.mark.parametrize(
    "change",
    [
        {"heldout_first_seed": 0},
        {"heldout_windows": 0},
        {"n_windows": 0},
        {"checkpoint_fractions": ()},
        {"checkpoint_fractions": (0.5,)},
        {"eval_batch_size": 0},
    ],
)
def test_invalid_runs_fail_before_generation(
    tmp_path, setup, monkeypatch, change
):
    model, generator, spec, config = setup
    monkeypatch.setattr(
        generator,
        "generate",
        lambda *a, **k: pytest.fail("must validate before generation"),
    )
    with pytest.raises(ValueError):
        VInformationEstimator(test_windows=7, test_first_seed=200).run(
            model, generator, spec, config.model_copy(update=change), tmp_path)


def test_selection_prefers_validation_then_baseline_then_earliest():
    def candidate(branch, step, loss):
        return {
            "branch": branch,
            "step": step,
            "validation_nats_per_window": loss,
        }

    baseline = candidate("unconditional", 10, 1.0)
    early = candidate("conditional", 2, 1.0)
    assert _choose([early, baseline]) is baseline
    better = candidate("conditional", 8, 0.9)
    assert _choose([baseline, better, early]) is better
    assert _choose([candidate("conditional", 8, 1.0), early]) is early


@pytest.mark.parametrize("variant", ["raw", "ema"])
def test_end_to_end_resume_selection_and_recompute(
    tmp_path, setup, monkeypatch, variant
):
    model, generator, spec, config = setup
    initial_digest = _weights_digest(model)
    estimator = build(
        "v_information", variant=variant, test_first_seed=200, test_windows=7
    )
    assert (
        isinstance(estimator, EpiplexityEstimator)
        and "v_information" in available()
    )
    result = estimator.run(model, generator, spec, config, tmp_path)
    assert (
        _weights_digest(model) == initial_digest
        and model.config.fixed_scaling is None
    )
    assert result == estimator.from_run_dir(tmp_path)
    assert result == json.loads(
        (tmp_path / "estimator_result.json").read_text()
    )
    manifest = json.loads((tmp_path / "v_information.json").read_text())
    for branch in ("unconditional", "conditional"):
        assert (tmp_path / branch / "last.pt").is_file()
        assert (
            np.fromfile(tmp_path / f"test_{branch}.f32", dtype=np.float32).size
            == 7
        )
        state = torch.load(tmp_path / branch / "ckpt_1.pt", weights_only=False)
        restored = MoiraiModule(
            MoiraiConfig.model_validate(state["model_config"])
        )
        restored.load_state_dict(state["model"])
        assert restored.config.ignore_context == (branch == "unconditional")
        assert restored.config.fixed_scaling == (
            manifest["normalization"]["loc"],
            manifest["normalization"]["scale"],
        )
    baseline = manifest["selection"]["unconditional"]
    conditional = manifest["selection"]["conditional"]
    for branch in ("unconditional", "conditional"):
        for fraction in (0.5, 1.0):
            losses = np.fromfile(
                tmp_path / branch / f"heldout_{variant}_{fraction:g}.f32",
                dtype=np.float32,
            )
            assert (
                conditional["validation_nats_per_window"]
                <= losses.astype(float).mean()
            )
    if conditional == baseline:
        assert result["v_information_bits_per_window"] == 0
        assert result["standard_error_bits_per_window"] == 0
    contents = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        generator,
        "generate",
        lambda *a, **k: pytest.fail("completed resume must not generate data"),
    )
    assert estimator.run(model, generator, spec, config, tmp_path) == result
    assert contents == {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }


def test_cli_both_entrypoints_and_estimator_options(
    tmp_path, setup, monkeypatch
):
    model, _, spec, config = setup
    raw = {
        "data": {"generator": "waveform"},
        "model": model.config.model_dump(mode="json"),
        "spec": spec.model_dump(mode="json"),
        "train": config.model_dump(mode="json"),
        "estimator": "v_information",
        "estimator_config": {"test_first_seed": 200, "test_windows": 7},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    out = tmp_path / "train"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--config",
            str(path),
            "--out",
            str(out),
            "--set",
            "estimator_config.variant=raw",
        ],
    )
    train_main()
    result = json.loads((out / "estimator_result.json").read_text())
    assert result["metadata"]["request"]["variant"] == "raw"
    with pytest.raises(ValueError, match="incompatible"):
        monkeypatch.setattr(
            "sys.argv", ["train", "--config", str(path), "--out", str(out)]
        )
        train_main()
    other = tmp_path / "estimator"
    estimator_main(
        [
            "--estimator",
            "v_information",
            "--generator",
            "waveform",
            "--out",
            str(other),
            "--config-json",
            str(out / "conditional" / "config.json"),
            "--estimator-config",
            json.dumps(raw["estimator_config"]),
            "--init-seed",
            "9",
        ]
    )
    other_result = json.loads((other / "estimator_result.json").read_text())
    assert other_result["n_test_windows"] == 7
    assert other_result["metadata"]["request"]["train"]["init_seed"] == 9
    assert other_result["metadata"]["request"][
        "initial_weights_sha256"
    ] == _weights_digest(build_model(model.config, 9))
    with pytest.raises(SystemExit):
        estimator_main(
            [
                "--estimator",
                "v_information",
                "--generator",
                "waveform",
                "--estimator-config",
                "[]",
            ]
        )


def test_resume_after_interruption_before_validation(
    tmp_path, setup, monkeypatch
):
    """Recover even when the final checkpoint exists but validation did not
    finish.
    """
    from episynth.training import Trainer

    model, generator, spec, config = setup
    estimator = VInformationEstimator(test_first_seed=200, test_windows=7)
    original_evaluate = Trainer.evaluate
    original_run = Trainer.run
    calls = []

    def interrupted_evaluate(self, *args, **kwargs):
        if self.out_dir.name == "conditional" and self.step == config.n_steps:
            raise RuntimeError("simulated interruption after final checkpoint")
        return original_evaluate(self, *args, **kwargs)

    def recorded_run(self):
        calls.append(self.out_dir.name)
        return original_run(self)

    monkeypatch.setattr(Trainer, "evaluate", interrupted_evaluate)
    monkeypatch.setattr(Trainer, "run", recorded_run)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        estimator.run(model, generator, spec, config, tmp_path)
    assert calls == ["unconditional", "conditional"]
    baseline_bytes = (tmp_path / "unconditional" / "last.pt").read_bytes()
    assert not json.loads((tmp_path / "v_information.json").read_text())[
        "complete"
    ]
    with pytest.raises(ValueError, match="incomplete"):
        estimator.from_run_dir(tmp_path)
    monkeypatch.setattr(Trainer, "evaluate", original_evaluate)
    result = estimator.run(model, generator, spec, config, tmp_path)
    assert result == estimator.from_run_dir(tmp_path)
    assert (
        baseline_bytes == (tmp_path / "unconditional" / "last.pt").read_bytes()
    )
    # Compare against uninterrupted fitting, including selected checkpoints and
    # test NLL.
    clean = tmp_path / "clean"
    assert estimator.run(model, generator, spec, config, clean) == result


@pytest.mark.parametrize("damage", ["truncated", "nonfinite"])
def test_resume_repairs_damaged_validation_losses(
    tmp_path, setup, monkeypatch, damage
):
    """An interrupted intermediate evaluation must not permanently block
    selection.
    """
    from episynth.training import Trainer

    model, generator, spec, config = setup
    estimator = VInformationEstimator(test_first_seed=200, test_windows=7)
    original_evaluate = Trainer.evaluate

    def interrupted_evaluate(
        self, batch, use_ema, precision=None, save_to=None
    ):
        record = original_evaluate(self, batch, use_ema, precision, save_to)
        if (
            self.out_dir.name == "conditional"
            and self.step == config.n_steps // 2
            and use_ema
        ):
            if damage == "truncated":
                save_to.write_bytes(save_to.read_bytes()[:-4])
            else:
                values = np.fromfile(save_to, dtype=np.float32)
                values[0] = np.nan
                values.tofile(save_to)
            raise RuntimeError("interrupted validation write")
        return record

    monkeypatch.setattr(Trainer, "evaluate", interrupted_evaluate)
    with pytest.raises(RuntimeError, match="interrupted validation write"):
        estimator.run(model, generator, spec, config, tmp_path)
    monkeypatch.setattr(Trainer, "evaluate", original_evaluate)
    result = estimator.run(model, generator, spec, config, tmp_path)
    clean = tmp_path / "clean"
    assert estimator.run(model, generator, spec, config, clean) == result
    artifact = "conditional/heldout_ema_0.5.f32"
    assert (tmp_path / artifact).read_bytes() == (
        clean / artifact
    ).read_bytes()


def test_resume_rejects_changed_initial_weights_normalization_and_budget(
    tmp_path, setup
):
    model, generator, spec, config = setup
    estimator = VInformationEstimator(test_first_seed=200, test_windows=7)
    estimator.run(model, generator, spec, config, tmp_path)
    changed = copy.deepcopy(model)
    with torch.no_grad():
        next(changed.parameters()).add_(1)
    with pytest.raises(ValueError, match="incompatible"):
        estimator.run(changed, generator, spec, config, tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        estimator.run(
            model,
            generator,
            spec,
            config.model_copy(update={"lr": config.lr * 2}),
            tmp_path,
        )
    path = tmp_path / "v_information.json"
    manifest = json.loads(path.read_text())
    manifest["normalization"]["scale"] *= 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="normalization"):
        estimator.run(model, generator, spec, config, tmp_path)


def test_validation_selection_is_saved_before_any_test_generation(
    tmp_path, setup, monkeypatch
):
    model, generator, spec, config = setup
    generate = generator.generate

    def checked_generate(seeds, **kwargs):
        if any(s >= 200 for s in seeds):
            manifest = json.loads(
                (tmp_path / "v_information.json").read_text()
            )
            assert set(manifest["selection"]) == {
                "unconditional",
                "conditional",
            }
            assert not manifest["complete"]
        return generate(seeds, **kwargs)

    monkeypatch.setattr(generator, "generate", checked_generate)
    estimator = VInformationEstimator(test_first_seed=200, test_windows=7)
    estimator.run(model, generator, spec, config, tmp_path)


def test_validation_uses_eval_precision(tmp_path, setup):
    model, generator, spec, config = setup
    config = config.model_copy(
        update={"precision": "bf16", "eval_precision": "fp32"}
    )
    result = VInformationEstimator(test_first_seed=200, test_windows=7).run(
        model, generator, spec, config, tmp_path
    )
    for candidate in result["metadata"]["selection"].values():
        fraction = 0.5 if candidate["step"] == 2 else 1
        losses = np.fromfile(
            tmp_path
            / candidate["branch"]
            / f"heldout_ema_fp32_{fraction:g}.f32",
            dtype=np.float32,
        )
        assert (
            candidate["validation_nats_per_window"]
            == losses.astype(float).mean()
        )


@pytest.mark.parametrize("preferred_branch", ["unconditional", "conditional"])
def test_validation_can_select_either_trained_predictor(
    tmp_path, setup, monkeypatch, preferred_branch
):
    """Simulate either validation winner; test still uses actual model
    densities.
    """
    from episynth.training import Trainer

    model, generator, spec, config = setup
    original = Trainer.evaluate

    def controlled_validation(
        self, batch, use_ema, precision=None, save_to=None
    ):
        record = original(self, batch, use_ema, precision, save_to)
        if self.out_dir.name == preferred_branch:
            losses = np.fromfile(save_to, dtype=np.float32) - np.float32(
                10_000
            )
            losses.tofile(save_to)
            record["nats_per_window"] = float(losses.astype(float).mean())
            record["nats_per_point"] = record["nats_per_window"] / spec.horizon
        return record

    monkeypatch.setattr(Trainer, "evaluate", controlled_validation)
    estimator = VInformationEstimator(test_first_seed=200, test_windows=7)
    result = estimator.run(model, generator, spec, config, tmp_path)
    assert (
        result["metadata"]["selection"]["conditional"]["branch"]
        == preferred_branch
    )
    u = np.fromfile(
        tmp_path / "test_unconditional.f32", dtype=np.float32
    ).astype(float)
    c = np.fromfile(
        tmp_path / "test_conditional.f32", dtype=np.float32
    ).astype(float)
    assert result["v_information_bits_per_window"] == pytest.approx(
        ((u - c) / math.log(2)).mean()
    )
    if preferred_branch == "unconditional":
        assert np.array_equal(u, c)
    else:
        assert not np.array_equal(u, c)
    assert estimator.from_run_dir(tmp_path) == result


def test_prepared_normalization_is_reused(setup, tmp_path, monkeypatch):
    import episynth.estimators.v_information as module

    model, generator, spec, train = setup
    estimator = VInformationEstimator(test_first_seed=200, test_windows=4)
    manifest = estimator.prepare(model, generator, spec, train, tmp_path)
    assert manifest["complete"] is False
    assert not (tmp_path / "conditional").exists()
    monkeypatch.setattr(
        module,
        "_normalization",
        lambda *args: pytest.fail("repeated normalization"),
    )
    result = estimator.run(model, generator, spec, train, tmp_path)
    assert result["metadata"]["normalization"] == manifest["normalization"]


def test_wait_for_reserved_conditional_fit(setup, tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from episynth.training import Trainer

    fcntl = pytest.importorskip("fcntl")
    unconditional_done = threading.Event()
    conditional_started = threading.Event()
    original = Trainer.run

    def observe(trainer):
        if trainer.out_dir.name == "conditional":
            conditional_started.set()
        original(trainer)
        if trainer.out_dir.name == "unconditional":
            unconditional_done.set()

    monkeypatch.setattr(Trainer, "run", observe)
    estimator = VInformationEstimator(test_first_seed=200, test_windows=4)
    with ThreadPoolExecutor(1) as pool, (
        tmp_path / "conditional.prepare.lock"
    ).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        future = pool.submit(estimator.run, *setup, tmp_path)
        assert unconditional_done.wait(timeout=30)
        try:
            assert not conditional_started.wait(timeout=0.4)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        assert future.result(timeout=30)["n_test_windows"] == 4
