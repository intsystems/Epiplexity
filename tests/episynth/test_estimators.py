r"""Plugin interface tests of epiplexity evaluators."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from episynth.estimators import EpiplexityEstimator, available, build, register
from episynth.estimators.prequential import PrequentialEstimator
from episynth.estimators.registry import _REGISTRY


def test_builtin_estimators_registered():
    """Built-in estimators are registered under expected names."""
    assert "prequential" in available()
    assert "requential" in available()
    assert "prequential" in _REGISTRY
    assert "requential" in _REGISTRY


def test_build_returns_instance():
    """build creates a copy of the requested estimator."""
    est = build("prequential")
    assert isinstance(est, PrequentialEstimator)
    assert isinstance(est, EpiplexityEstimator)


def test_build_unknown_raises():
    """An unknown name builder drops a KeyError with a list of available ones.
    """
    with pytest.raises(KeyError, match="prequential"):
        build("no_such_estimator")


def test_register_rejects_duplicates():
    """Re-registration of the same name is dropped by ValueError."""

    class _Dup(EpiplexityEstimator):
        name = "dupe"

    register("dupe")(_Dup)
    with pytest.raises(ValueError, match="already registered"):
        register("dupe")(_Dup)
    _REGISTRY.pop("dupe")


def test_requential_factory():
    est = build("requential", projection_steps=[], mode="orc", candidates=4)
    assert isinstance(est, EpiplexityEstimator)
    assert est.name == "requential"


def test_prequential_from_run_dir_uses_heldout(tmp_path):
    """The from_run_dir counts the area through the heldout_ema_1.f32 rather
    than through the tail of the run.
    """
    import numpy as np

    np.asarray([5.0, 4.0, 3.0, 2.0], dtype=np.float32).tofile(
        tmp_path / "window_loss.f32"
    )
    np.asarray([2.0, 2.0, 1.0, 3.0], dtype=np.float32).tofile(
        tmp_path / "heldout_ema_1.f32"
    )
    est = build("prequential", variant="ema")
    res = est.from_run_dir(tmp_path)
    # model part = sum_loss - N*l_fin = 14 - 4*2 = 6 nats
    import math

    assert res["n_windows"] == 4.0
    assert abs(res["total_bits"] - 14.0 / math.log(2)) < 1e-9
    assert abs(res["model_part_bits"] - 6.0 / math.log(2)) < 1e-9
    assert abs(res["data_part_bits"] - 8.0 / math.log(2)) < 1e-9
    with pytest.raises(ValueError):
        build("prequential", variant="bad")


@pytest.mark.parametrize("device", [None, "cpu"])
def test_estimator_cli_applies_only_explicit_device_override(
    tmp_path, monkeypatch, device
):
    """An explicit CPU request overrides CUDA in a config; omission preserves
    it.
    """
    from episynth.estimators import run as estimator_cli
    from episynth.model.moirai import MoiraiConfig
    from episynth.training import TrainConfig
    from episynth.windows import WindowSpec

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "train": (
                    TrainConfig(
                        n_windows=8, batch_size=4, device="cuda"
                    ).model_dump(mode="json")
                ),
                "spec": WindowSpec().model_dump(mode="json"),
                "model": MoiraiConfig().model_dump(mode="json"),
            }
        )
    )
    received = []

    def capture(model, generator, spec, config, out_dir):
        received.append(config.device)
        Path(out_dir).mkdir()
        return {"n_windows": config.n_windows}

    monkeypatch.setattr(
        estimator_cli,
        "build",
        lambda *args, **kwargs: SimpleNamespace(run=capture),
    )
    monkeypatch.setattr(estimator_cli, "build_model", lambda *args: object())
    args = [
        "--estimator",
        "prequential",
        "--generator",
        "waveform",
        "--config-json",
        str(config_path),
        "--out",
        str(tmp_path / "out"),
    ]
    if device is not None:
        args.extend(["--device", device])
    estimator_cli.main(args)
    assert received == [device or "cuda"]
