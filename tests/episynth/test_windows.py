"""Слой окон: раскладка, маски, детерминизм, деквантование."""

import threading
import time

import numpy as np
import pytest
import torch

from episynth.generators import GENERATORS, Generator
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


def test_context_length_distribution_follows_probs():
    """Частоты длин контекста по многим сидам близки к заданным вероятностям; начало окна допустимо."""
    spec = WindowSpec(**{**SPEC.model_dump(), "context_lengths": (40, 64, 128), "context_probs": (0.2, 0.3, 0.5)})
    n = 20000
    batch = make_windows(np.zeros((n, 256), dtype=np.float32), np.arange(n), spec)
    freq = np.array([(batch.context_length.numpy() == c).mean() for c in spec.context_lengths])
    assert np.abs(freq - np.array(spec.context_probs)).max() < 0.015
    assert batch.observed_mask.reshape(n, -1).sum(1).numpy().tolist() == (batch.context_length.numpy() + 32).tolist()


@pytest.mark.parametrize("changes", [
    {"series_length": 0}, {"context_max": 0}, {"horizon": 0}, {"patch_size": 0},
    {"horizon": -16}, {"patch_size": -16}, {"context_max": 127}, {"horizon": 31},
    {"context_lengths": ()}, {"context_lengths": (0, 128)}, {"context_lengths": (129,)},
    {"series_length": 159}, {"context_probs": (1.0,)}, {"context_probs": (0.0, 0.0)},
    {"context_probs": (-1.0, 2.0)}, {"context_probs": (float("nan"), 1.0)},
    {"context_probs": (float("inf"), 1.0)}, {"context_probs": (1e308, 1e308)},
    {"dequantize_width": -1.0}, {"dequantize_width": float("nan")}, {"dequantize_width": float("inf")},
])
def test_window_spec_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        WindowSpec(**{**SPEC.model_dump(), **changes})


def test_context_probabilities_allow_zero_entries_and_unnormalized_weights():
    spec = WindowSpec(**{**SPEC.model_dump(), "context_probs": (0.0, 2.0), "dequantize_width": 0.0})
    batch = make_windows(np.zeros((8, 256)), np.arange(8), spec)
    assert batch.context_length.tolist() == [128] * 8


class _ThreadGenerator(Generator):
    name = "thread_test"

    def __init__(self, on_generate=None):
        super().__init__(SPEC.series_length)
        self.on_generate = on_generate
        self.calls = 0
        self.thread = None

    def _batch(self, seeds):
        self.thread = threading.current_thread()
        self.calls += 1
        if self.on_generate is not None:
            self.on_generate(self.calls)
        return np.zeros((len(seeds), self.length))


def test_closing_stream_with_full_queue_stops_producer():
    third_batch = threading.Event()

    def on_generate(n):
        if n == 3:
            third_batch.set()

    gen = _ThreadGenerator(on_generate)
    stream = iter(WindowStream(gen, SPEC, batch_size=2, prefetch=1))
    try:
        next(stream)
        # The second batch is already queued when generation of the third begins.
        assert third_batch.wait(timeout=2)
        stream.close()
        assert not gen.thread.is_alive()
    finally:
        stream.close()


@pytest.mark.parametrize("fail", [False, True])
def test_stream_joins_producer_on_exhaustion_or_generation_error(fail):
    def on_generate(n):
        if fail and n == 2:
            raise RuntimeError("generation failed")

    gen = _ThreadGenerator(on_generate)
    stream = WindowStream(gen, SPEC, batch_size=2, n_batches=2, prefetch=1)
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            list(stream)
    else:
        batches = list(stream)
        assert [batch.seeds.tolist() for batch in batches] == [[0, 1], [2, 3]]
    assert not gen.thread.is_alive()


def test_close_does_not_wait_indefinitely_for_in_progress_generation():
    generating, release = threading.Event(), threading.Event()

    def on_generate(n):
        if n == 2:
            generating.set()
            release.wait(timeout=5)

    gen = _ThreadGenerator(on_generate)
    stream = iter(WindowStream(gen, SPEC, batch_size=2, prefetch=1))
    try:
        next(stream)
        assert generating.wait(timeout=2)
        start = time.monotonic()
        stream.close()
        assert time.monotonic() - start < 2
    finally:
        release.set()
        stream.close()
        gen.thread.join(timeout=2)
    assert not gen.thread.is_alive()


def test_pool_stream_matches_thread_stream():
    """Поток из процессов даёт те же батчи в том же порядке, что поток из одного фонового потока."""
    from episynth.windows import collect_windows

    gen = GENERATORS["waveform"](length=256)
    a = list(WindowStream(gen, SPEC, batch_size=8, first_seed=50, n_batches=4))
    b = list(WindowStream(gen, SPEC, batch_size=8, first_seed=50, n_batches=4, workers=2, prefetch=1))
    assert len(b) == 4 and all(torch.equal(x.target, y.target) and torch.equal(x.seeds, y.seeds) for x, y in zip(a, b, strict=True))
    big = collect_windows(gen, SPEC, np.arange(50, 82), workers=2, batch_size=8)
    assert torch.equal(big.target, torch.cat([x.target for x in a])) and len(big) == 32
    tail = list(WindowStream(gen, SPEC, batch_size=8, first_seed=50, n_batches=4, first_batch=3, workers=2))
    assert len(tail) == 1 and torch.equal(tail[0].target, a[3].target)


def test_generators_pickle_for_workers(tmp_path):
    """Корпус сериализуется путями, KernelSynth без тензора на устройстве; после десериализации ряды те же."""
    import pickle

    from episynth.corpus import Corpus, generate_corpus
    from episynth.generators import Mixture

    out = tmp_path / "sde"
    generate_corpus("sde", first_seed=0, n=8, out=out, length=128, workers=1)
    corpus = Corpus(out)
    again = pickle.loads(pickle.dumps(corpus))
    assert np.array_equal(again.generate([3, 5]), corpus.generate([3, 5])) and again.name == "sde"
    ks = GENERATORS["kernelsynth"](length=128)
    ks2 = pickle.loads(pickle.dumps(ks))
    assert np.array_equal(ks2.generate([1, 2]), ks.generate([1, 2]))
    mix = pickle.loads(pickle.dumps(Mixture([corpus, ks])))
    assert mix.name == "mixture(sde,kernelsynth)" and mix.generate([0, 1]).shape == (2, 128)


def test_collect_windows_respects_corpus_bounds(tmp_path):
    """Сбор отложенного набора не запрашивает сиды за концом корпуса: последний батч короче."""
    from episynth.corpus import Corpus, generate_corpus
    from episynth.windows import collect_windows

    out = tmp_path / "held"
    generate_corpus("waveform", first_seed=1000, n=10, out=out, length=256, workers=1)
    corpus = Corpus(out)
    batch = collect_windows(corpus, SPEC, np.arange(1000, 1010), workers=0, batch_size=4)
    assert len(batch) == 10 and batch.seeds.tolist() == list(range(1000, 1010))
    batch2 = collect_windows(corpus, SPEC, np.arange(1000, 1010), workers=2, batch_size=4)
    assert torch.equal(batch.target, batch2.target)
