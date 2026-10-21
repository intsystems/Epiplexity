"""Слой корпуса: совпадение с генератором, объединение каталогов, ошибки."""

import numpy as np
import pytest

from episynth.corpus import Corpus, CorpusManifest, generate_corpus
from episynth.generators import GENERATORS
from episynth.windows import WindowSpec, window_batch


def test_corpus_rows_equal_generator(tmp_path):
    """Строки корпуса равны рядам генератора по тем же сидам, манифест полон, окна совпадают."""
    out = tmp_path / "waveform" / "train"
    manifest = generate_corpus("waveform", first_seed=100, n=50, out=out, length=256, workers=2, chunk=16)
    assert manifest == CorpusManifest.model_validate_json((out / "manifest.json").read_text())
    assert (manifest.n, manifest.first_seed, manifest.nonfinite_rows, manifest.workers) == (50, 100, 0, 2)
    corpus = Corpus(out)
    seeds = np.array([149, 100, 120])
    expected = GENERATORS["waveform"](length=256).generate(seeds)
    assert corpus.name == "waveform" and corpus.length == 256
    assert np.array_equal(corpus.generate(seeds), expected)
    spec = WindowSpec(series_length=256, context_max=128, horizon=32, patch_size=16, context_lengths=(64, 128))
    a, b = window_batch(corpus, seeds, spec), window_batch(GENERATORS["waveform"](length=256), seeds, spec)
    assert (a.target == b.target).all() and (a.context_length == b.context_length).all()


def test_corpus_parts_and_errors(tmp_path):
    """Несколько каталогов отдают ряды по своим диапазонам; чужие сиды, пересечения и перезапись — ошибки."""
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    generate_corpus("stepfunction", first_seed=0, n=10, out=train, length=128, workers=1, chunk=4)
    generate_corpus("stepfunction", first_seed=1000, n=10, out=heldout, length=128, workers=1, chunk=4)
    corpus = Corpus(train, heldout)
    assert corpus.seed_ranges == [(0, 10), (1000, 1010)]
    gen = GENERATORS["stepfunction"](length=128)
    assert np.array_equal(corpus.generate([1003, 5]), gen.generate([1003, 5]))
    with pytest.raises(KeyError):
        corpus.generate([10])
    with pytest.raises(NotImplementedError):
        corpus.generate([5], exact=True)
    with pytest.raises(FileExistsError):
        generate_corpus("stepfunction", first_seed=0, n=10, out=train, length=128, workers=1)
    overlap = tmp_path / "overlap"
    generate_corpus("stepfunction", first_seed=5, n=10, out=overlap, length=128, workers=1, chunk=4)
    with pytest.raises(ValueError, match="пересекаются"):
        Corpus(train, overlap)
    other = tmp_path / "other"
    generate_corpus("waveform", first_seed=0, n=4, out=other, length=128, workers=1)
    with pytest.raises(ValueError, match="разных генераторов"):
        Corpus(train, other)
