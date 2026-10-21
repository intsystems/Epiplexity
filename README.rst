|test| |docs|

.. |test| image:: https://github.com/intsystems/Epiplexity/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/intsystems/Epiplexity/actions/workflows/test.yml
    :alt: Test status

.. |docs| image:: https://github.com/intsystems/Epiplexity/actions/workflows/docs.yaml/badge.svg?branch=main
    :target: https://github.com/intsystems/Epiplexity/actions/workflows/docs.yaml
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Epiplexity on one GPU: estimators for small networks and when they agree
    :Тип научной работы: CoIS, BMM
    :Авторы: Алексей Кравацкий, Марк Иконников, Даниил Казачков, Данила Черноусов
    :Научный руководитель: уточняется
    :Научный консультант(при наличии): уточняется

Abstract
========

Epiplexity (`Finzi, Qiu et al., 2026 <https://arxiv.org/abs/2601.03220>`_) measures the structure
that a computationally bounded learner can extract from data. This project estimates epiplexity on
synthetic time series: a small model (Moirai-Small) is trained on a stream of windows from a
generator, the loss of each window is logged before the step on it, and the area under the loss
curve (the prequential code length) is compared with the model's quality on the short-horizon
GIFT-Eval tasks. The ``episynth`` package implements the generators, task windows, Moirai port,
training with per-window loss logging, the analysis layer and pluggable estimators of epiplexity.

Research publications
=====================
1.

Presentations at conferences on the topic of research
=====================================================
1.

Software modules developed as part of the study
===============================================
1. The python package *episynth* with all implementation `here <https://github.com/intsystems/Epiplexity/tree/main/src/episynth>`_.
2. Runnable demo notebook `here <https://github.com/intsystems/Epiplexity/tree/main/experiments/demos>`_.

Repository layout
=================

.. code-block:: text

    src/episynth/      the package: generators, task windows, Moirai port, training with per-window
                       loss logging, analysis, evaluation on GIFT-Eval, pluggable estimators of
                       epiplexity (overview in src/episynth/README.md)
    tests/             pytest
    experiments/       experiment code and demo (demos/) plus saved outputs (episynth/): two training
                       series (configs, training curves, held-out losses, GIFT-Eval summaries,
                       analysis tables, figures, scripts)
    doc/               Sphinx documentation (autodoc); deploys to GitHub Pages on push to main
    publications/      non-code material for the article and project: paper/ (arXiv manuscript),
                       blogpost/, notes/ (literature review), research_notes/, slides/, plan/,
                       figures/, description.md
    pyproject.toml     package metadata (name = episynth)
    README.rst         this file

Development
===========

Requires Python 3.10+ and `uv <https://docs.astral.sh/uv/>`_. On a network that needs system TLS
certificates, add ``--system-certs`` to the uv commands.

.. code-block:: bash

    uv sync --extra dev
    .venv/bin/pytest tests/
    .venv/bin/ruff check src/ tests/ experiments/demos/

Documentation
=============

The Sphinx documentation (with autodoc) is built and deployed to GitHub Pages automatically on
every push to ``main`` by `.github/workflows/docs.yaml`. To build it locally:

.. code-block:: bash

    .venv/bin/sphinx-build -b html doc/source doc/_build
    # open doc/_build/index.html