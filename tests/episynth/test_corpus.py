"""Layer of the case: coincidence with the generator, combining directories,
errors.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from episynth import corpus as corpus_module
from episynth.corpus import Corpus, CorpusManifest, generate_corpus
from episynth.generators import GENERATORS
from episynth.windows import WindowSpec, window_batch


def test_corpus_rows_equal_generator(tmp_path):
    """The lines of the case are equal to the rows of the generator on the same
    seeds, the manifesto is full, the windows coincide.
    """
    out = tmp_path / "waveform" / "train"
    manifest = generate_corpus(
        "waveform",
        first_seed=100,
        n=50,
        out=out,
        length=256,
        workers=2,
        chunk=16,
    )
    assert manifest == CorpusManifest.model_validate_json(
        (out / "manifest.json").read_text()
    )
    assert (
        manifest.n,
        manifest.first_seed,
        manifest.nonfinite_rows,
        manifest.workers,
    ) == (50, 100, 0, 2)
    corpus = Corpus(out)
    seeds = np.array([149, 100, 120])
    expected = GENERATORS["waveform"](length=256).generate(seeds)
    assert corpus.name == "waveform" and corpus.length == 256
    assert np.array_equal(corpus.generate(seeds), expected)
    spec = WindowSpec(
        series_length=256,
        context_max=128,
        horizon=32,
        patch_size=16,
        context_lengths=(64, 128),
    )
    a, b = window_batch(corpus, seeds, spec), window_batch(
        GENERATORS["waveform"](length=256), seeds, spec
    )
    assert (a.target == b.target).all() and (
        a.context_length == b.context_length
    ).all()


def test_corpus_parts_and_errors(tmp_path):
    """Several directories give rows according to their ranges; other people's
    seats, intersections, and overwriting are errors.
    """
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    generate_corpus(
        "stepfunction",
        first_seed=0,
        n=10,
        out=train,
        length=128,
        workers=1,
        chunk=4,
    )
    generate_corpus(
        "stepfunction",
        first_seed=1000,
        n=10,
        out=heldout,
        length=128,
        workers=1,
        chunk=4,
    )
    corpus = Corpus(train, heldout)
    assert corpus.seed_ranges == [(0, 10), (1000, 1010)]
    gen = GENERATORS["stepfunction"](length=128)
    assert np.array_equal(corpus.generate([1003, 5]), gen.generate([1003, 5]))
    with pytest.raises(KeyError):
        corpus.generate([10])
    with pytest.raises(NotImplementedError):
        corpus.generate([5], exact=True)
    with pytest.raises(FileExistsError):
        generate_corpus(
            "stepfunction",
            first_seed=0,
            n=10,
            out=train,
            length=128,
            workers=1,
        )
    overlap = tmp_path / "overlap"
    generate_corpus(
        "stepfunction",
        first_seed=5,
        n=10,
        out=overlap,
        length=128,
        workers=1,
        chunk=4,
    )
    with pytest.raises(ValueError, match="overlap"):
        Corpus(train, overlap)
    other = tmp_path / "other"
    generate_corpus(
        "waveform", first_seed=0, n=4, out=other, length=128, workers=1
    )
    with pytest.raises(ValueError, match="different generators"):
        Corpus(train, other)


@pytest.mark.parametrize("generator, args, compatible", [
    ("timesynth", {"irregular_prob": 0.2}, True),
    ("timesynth", {"irregular_prob": 0.8}, False),
    ("kernelsynth", {"max_kernels": 5, "chunk": 64, "device": None}, True),
    ("kernelsynth", {"max_kernels": 2}, False),
])
def test_corpus_parts_require_matching_generator_arguments(
    tmp_path, generator, args, compatible
):
    """Explicit default values are compatible with older manifestos without
    them; other settings are not.
    """
    parts = [tmp_path / "first", tmp_path / "second"]
    for i, (part, kwargs) in enumerate(zip(parts, ({}, args), strict=True)):
        part.mkdir()
        np.save(part / "series.npy", np.zeros((2, 64), dtype=np.float32))
        manifest = CorpusManifest(
            generator=generator,
            length=64,
            first_seed=2 * i,
            n=2,
            generator_args=kwargs,
        )
        (part / "manifest.json").write_text(manifest.model_dump_json())
    if compatible:
        assert Corpus(*parts).seed_ranges == [(0, 2), (2, 4)]
    else:
        with pytest.raises(ValueError, match="generator_args"):
            Corpus(*parts)


@pytest.mark.parametrize("changes", [
    {"n": 0}, {"n": -1}, {"n": 1.5}, {"n": True},
    {"length": 0}, {"length": -1}, {"length": 2.5},
    {"chunk": 0}, {"chunk": -1}, {"chunk": 1.5},
    {"workers": 0}, {"workers": -1}, {"workers": 1.5},
    {"first_seed": -1}, {"first_seed": 1.5}, {"first_seed": True},
    {"first_seed": 2**31}, {"first_seed": 2**31 - 1}, {"n": 2**31 + 1},
])
def test_invalid_generation_arguments_create_no_files(tmp_path, changes):
    out = tmp_path / "invalid"
    kwargs = {
        "generator": "waveform",
        "first_seed": 0,
        "n": 4,
        "length": 64,
        "workers": 1,
        "chunk": 2,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError):
        generate_corpus(out=out, **kwargs)
    assert not out.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_incomplete_generation_never_writes_completion_manifest(
    tmp_path, monkeypatch, fail
):
    class InterruptedPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, *args):
            yield 2, 0
            if fail:
                raise RuntimeError("generation failed")

    monkeypatch.setattr(
        corpus_module.mp,
        "get_context",
        lambda _: SimpleNamespace(Pool=InterruptedPool),
    )
    out = tmp_path / "incomplete"
    message = "generation failed" if fail else "incomplete corpus generation"
    with pytest.raises(RuntimeError, match=message):
        generate_corpus(
            "waveform",
            first_seed=0,
            n=4,
            out=out,
            length=64,
            workers=1,
            chunk=2,
        )
    assert (out / "series.npy").exists()
    assert not (out / "manifest.json").exists()
