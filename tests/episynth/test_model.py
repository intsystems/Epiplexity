"""Перенос Moirai: число параметров, загрузка чекпойнта, формулы компонент смеси, прямой проход."""

import os

import numpy as np
import pytest
import torch

from episynth.generators import GENERATORS
from episynth.model import MoiraiModule, load_uni2ts_checkpoint
from episynth.model.moirai import Mixture, small
from episynth.windows import WindowSpec, make_windows

CKPT = os.environ.get("MOIRAI_SMALL_DIR")


def test_param_count_matches_published_small():
    """Конфигурация small с четырьмя компонентами и пятью патчами даёт 13 827 528 параметров."""
    model = MoiraiModule(small())
    assert sum(p.numel() for p in model.parameters()) == 13_827_528


def test_component_log_probs_match_torch_distributions():
    """Формулы компонент совпадают с `torch.distributions` на случайных параметрах."""
    g = torch.Generator().manual_seed(0)
    shape = (2, 3, 5)
    df = 2 + torch.rand(shape, generator=g) * 5
    loc = torch.randn(shape, generator=g)
    scale = 0.3 + torch.rand(shape, generator=g)
    r = 0.5 + torch.rand(shape, generator=g) * 3
    logits = torch.randn(shape, generator=g)
    z = torch.rand(shape, generator=g) * 3 + 0.1
    params = [
        {"df": df, "loc": loc, "scale": scale},
        {"loc": loc},
        {"total_count": r, "logits": logits},
        {"loc": loc, "scale": scale},
    ]
    mix = Mixture(
        torch.zeros(*shape, 4), params, small().components, 1e-3, torch.zeros(2, 3, 1), torch.ones(2, 3, 1)
    )
    ref = [
        torch.distributions.StudentT(df, loc, scale).log_prob(z),
        torch.distributions.Normal(loc, 1e-3).log_prob(z),
        None,
        torch.distributions.LogNormal(loc, scale).log_prob(z),
    ]
    for name, p, expected in zip(mix.names, params, ref, strict=True):
        got = mix._component_log_prob(name, p, z)
        if expected is not None:
            assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), name
    nb = mix._component_log_prob("negative_binomial", params[2], z)
    k = torch.round(z)
    nb_int = mix._component_log_prob("negative_binomial", params[2], k)
    ref_nb = torch.distributions.NegativeBinomial(r, logits=logits).log_prob(k)
    assert torch.allclose(nb_int, ref_nb, atol=1e-5) and torch.isfinite(nb).all()
    neg = mix._component_log_prob("negative_binomial", params[2], -z)
    assert torch.isinf(neg).all()
    total = mix.log_prob(z)
    assert torch.isfinite(total).all()


def test_forward_shapes_and_finite_nll():
    """Прямой проход на окнах генератора даёт конечную лог-плотность нужной формы."""
    spec = WindowSpec(series_length=256, context_max=128, horizon=32, patch_size=16, context_lengths=(64, 128))
    gen = GENERATORS["sde"](length=256)
    seeds = np.arange(4)
    batch = make_windows(gen.generate(seeds), seeds, spec)
    model = MoiraiModule(small(patch_sizes=(16,), components=("student_t", "normal_fixed", "log_normal")))
    mix = model(**batch.model_inputs(), patch_size=16)
    lp = mix.log_prob(batch.target)
    assert lp.shape == (4, spec.n_tokens, 16) and torch.isfinite(lp[batch.sample_id.bool()]).all()
    samples = mix.sample(3)
    assert samples.shape == (3, 4, spec.n_tokens, 16)


@pytest.mark.skipif(CKPT is None, reason="MOIRAI_SMALL_DIR не задан")
def test_pretrained_checkpoint_loads_and_fits_sine_better_than_noise():
    """Опубликованный чекпойнт грузится и даёт синусу заметно меньший лосс, чем белому шуму.

    Точная сверка переноса с оригиналом — `episynth.checks.moirai`.
    """
    model = MoiraiModule(small())
    load_uni2ts_checkpoint(model, os.path.join(CKPT, "model.safetensors"))
    model.eval()
    spec = WindowSpec(series_length=1024, context_max=512, horizon=64, patch_size=32, dequantize_width=0.0)
    t = np.arange(1024, dtype=np.float32)
    sine = np.sin(2 * np.pi * t / 48)[None]
    noise = np.random.default_rng(0).standard_normal((1, 1024)).astype(np.float32)
    out = {}
    for name, series in (("sine", sine), ("noise", noise)):
        batch = make_windows(series, [1], spec)
        with torch.no_grad():
            mix = model(**batch.model_inputs(), patch_size=32)
            nll = -mix.log_prob(batch.target)[..., :32]
            mask = batch.prediction_mask.unsqueeze(-1) & batch.observed_mask[..., :32]
            out[name] = float((nll * mask).sum() / mask.sum())
    assert out["noise"] - out["sine"] > 0.8 and 1.0 < out["noise"] < 2.0, out
