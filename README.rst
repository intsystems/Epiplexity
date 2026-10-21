|test| |docs|

`Documentation website <https://intsystems.github.io/Epiplexity/>`_

.. |test| image:: https://github.com/intsystems/Epiplexity/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/intsystems/Epiplexity/actions/workflows/test.yml
    :alt: Test status

.. |docs| image:: https://github.com/intsystems/Epiplexity/actions/workflows/docs.yaml/badge.svg?branch=main
    :target: https://github.com/intsystems/Epiplexity/actions/workflows/docs.yaml
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Epiplexity and transfer in synthetic time-series pretraining
    :Тип научной работы: CoIS, BMM
    :Авторы: Danila Chernousov, Mark Ikonnikov, Daniil Kazachkov, Alexey Kravatskiy
    :Научный руководитель: уточняется
    :Научный консультант(при наличии): уточняется

Abstract
========

The practical goal is to train a time-series foundation model on high-quality synthetic
data and select productive generators for that data. A productive generator supplies
pretraining data that improve forecasting on real datasets under a fixed training-data
budget. We study epiplexity-based measurements as possible signals for this selection.
The implemented ``episynth`` package generates forecasting tasks, trains
a probabilistic Moirai transformer with pre-update loss logging and EMA weights, estimates
signed excess prequential loss, and evaluates transfer on GIFT-Eval. Saved experiments cover
eleven priors at approximately 2 million and 40 million training windows, plus one additional
Waveform initialization. KernelSynth transfers best at both budgets; the full-set association
between excess area and aggregate CRPS is weak. The technical report documents the evidence,
endpoint sensitivity, forecast failures, and limits of interpreting the area as epiplexity.

The earlier research roadmap in ``publications/plan/`` is **outdated**.
Current implementation details are in ``src/episynth/README.md`` and
``publications/description.md``; the report documents the saved experiments.

Technical report and measurement methods
========================================

Read the `technical report PDF <publications/technical_report/main.pdf>`_ and its
`reproduction instructions <publications/technical_report/README.md>`_. The proposed
four-method library scope is described in the
`measurement contracts <doc/source/measurements.rst>`_:

1. **Prequential coding (implemented):** measure the predictive cost of the training
   sequence; return total, residual-data and signed model-proxy costs in bits.
2. **Requential coding (registered stub):** describe a student's learned predictions
   through teacher-selected synthetic training samples and their coding costs.
3. **Predictive V-information (proposed):** fit predictors with and without context;
   estimate the held-out log-loss reduction in bits per window.
4. **Bayesian predictive log-loss (OU reference implemented):** marginalize the known
   AR(1) prior to obtain joint and pointwise forecast-loss references. A common public
   wrapper remains proposed.

Seeded task generation, AdamW/EMA training, and GIFT-Eval scoring provide shared
infrastructure. The saved transfer experiments use the prequential method.

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
    doc/               Sphinx documentation (autodoc); GitHub Pages publication is manual
    publications/      non-code material for the article and project: paper/ (arXiv manuscript),
                       blogpost/, notes/ (literature review), research_notes/, slides/, plan/,
                       figures/, technical_report/ (article and reproducible analysis), description.md
    pyproject.toml     package metadata (name = episynth)
    README.rst         this file

Development
===========

Requires Python 3.10+ and `uv <https://docs.astral.sh/uv/>`_. On a network that needs system TLS
certificates, add ``--system-certs`` to the uv commands.

.. code-block:: bash

    uv sync --locked --extra dev
    uv run --no-sync pytest tests/
    uv run --no-sync ruff check src/ tests/ experiments/demos/ doc/source/conf.py

Documentation
=============

The Sphinx documentation (with autodoc) is built on pushes and pull requests to ``main``.
Automatic publication is disabled as required by the course. To update GitHub Pages, open
**Actions → docs → Run workflow**, select ``main``, and start the workflow manually.
To build it locally:

.. code-block:: bash

    uv run --no-sync sphinx-build -b html -W --keep-going doc/source doc/_build
    # open doc/_build/index.html

The generated API example is ``doc/_build/api/estimators.html``. See
``doc/source/installation.rst`` for standalone hosting and GitHub Pages setup.
The Pages source must be configured as **GitHub Actions**. The expected URL is
https://intsystems.github.io/Epiplexity/; a successful deployment is required to
update the online site. Pull requests build documentation without publishing.
