"""Слой окон: раскладка, маски, детерминизм, деквантование."""

import numpy as np
import torch

from episynth.generators import GENERATORS
from episynth.windows import WindowSpec, WindowStream, make_windows

SPEC = WindowSpec(series_length=256, context_max=128, horizon=32, patch_size=16, context_lengths=(40, 128))


def test_layout_and_masks():
    """Контекст выровнен по правому краю, горизонт в конце, заполнение слева без наблюдений."""
    series = np.arange(4 * 256, dtype=np.float32).reshape(4, 256)
    batch = make_windows(series, [0, 1, 2, 3], WindowSpec(**{**SPEC.model_dump(), "dequantize_width": 0.0}))
    S, P = SPEC.n_tokens, SPEC.patch_size
    assert batch.target.shape == (4, S, P) and batch.prediction_mask.sum(1).tolist() == [2] * 4
    for i in range(4):
        c = int(batch.context_length[i])
        flat_obs = batch.observed_mask[i].reshape(-1)
        assert flat_obs[: 128 - c].sum() == 0 and flat_obs[128 - c :].all()
        vals = batch.target[i].reshape(-1)[128 - c :]
        assert torch.all(vals[1:] - vals[:-1] == 1)
        assert (batch.sample_id[i] == 0).sum() == (128 - c) // P
    assert (batch.time_id == torch.arange(S)).all() and (batch.variate_id == 0).all()


def test_deterministic_by_seed_and_dequantization_scale():
    """Окно зависит только от сида ряда; шум деквантования мал относительно масштаба контекста."""
    gen = GENERATORS["waveform"](length=256)
    seeds = np.array([5, 6, 7])
    a = make_windows(gen.generate(seeds), seeds, SPEC)
    b = make_windows(gen.generate(seeds[::-1]), seeds[::-1], SPEC)
    assert torch.equal(a.target.flip(0), b.target)
    clean = make_windows(gen.generate(seeds), seeds, WindowSpec(**{**SPEC.model_dump(), "dequantize_width": 0.0}))
    diff = (a.target - clean.target)[a.observed_mask]
    assert diff.abs().max() < 1e-3 * (clean.target[clean.observed_mask].std() + 1) and diff.abs().max() > 0


def test_stream_is_resumable():
    """Поток батчей по номеру батча даёт те же окна, что и с начала."""
    gen = GENERATORS["sde"](length=256)
    full = list(WindowStream(gen, SPEC, batch_size=4, first_seed=100, n_batches=3))
    tail = list(WindowStream(gen, SPEC, batch_size=4, first_seed=100, n_batches=3, first_batch=2))
    assert len(full) == 3 and len(tail) == 1
    assert torch.equal(full[2].target, tail[0].target) and torch.equal(full[2].seeds, tail[0].seeds)
