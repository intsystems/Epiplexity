|test| |docs|

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

This project studies whether structure learned from synthetic time-series priors predicts
forecasting transfer. The implemented ``episynth`` package generates forecasting tasks, trains
a probabilistic Moirai transformer with pre-update loss logging and EMA weights, estimates
signed excess prequential loss, and evaluates transfer on GIFT-Eval. Saved experiments cover
eleven priors at approximately 2 million and 40 million training windows, plus one additional
Waveform initialization. KernelSynth transfers best at both budgets; the full-set association
between excess area and aggregate CRPS is weak. The technical report documents the evidence,
endpoint sensitivity, forecast failures, and limits of interpreting the area as epiplexity.

The earlier estimator roadmap in ``plan/`` is **outdated**. The ``epimeter`` estimator classes
remain stubs. Current scope and results are documented in ``src/episynth/README.md`` and
``technical_report/README.md``.

Technical report and four algorithms
====================================

Read the `technical report PDF <technical_report/main.pdf>`_ and its
`reproduction instructions <technical_report/README.md>`_. The methodology specifies:

1. Seeded synthetic task generation: reproducible context/target windows, dequantization and masks.
2. Probabilistic training with AdamW and EMA: checkpoints and pre-update/held-out NLL logs.
3. Excess prequential-loss estimation: signed area above the held-out endpoint, with stated units.
4. Transfer evaluation and ranking uncertainty: GIFT-Eval scores, baseline ratios and paired bootstrap intervals.

Research publications
===============================
1.

Presentations at conferences on the topic of research
=====================================================
1.

Software modules developed as part of the study
======================================================
1. The implemented time-series package `episynth <src/episynth/README.md>`_:
   generators, task windows, Moirai model, training, evaluation and analysis.
2. The ``epimeter`` interfaces and registries, with runnable extension examples in ``code/``;
   its estimator and training routines remain stubs.

Repository layout
=================

.. code-block:: text

    src/epimeter/   the package: estimators, models, samplers, training, eval
    src/episynth/   time-series track: synthetic generators, task windows, Moirai port, training with
                    per-window loss logging, GIFT-Eval evaluation (overview in src/episynth/README.md)
    tests/          pytest
    code/           experiment code and demo
    experiments/    saved experiment outputs: episynth/ — two training series (configs, training
                    curves, held-out losses, GIFT-Eval summaries, analysis tables, figures, scripts);
                    per-window losses are release assets (see experiments/episynth/README.md)
    doc/            Sphinx documentation
    technical_report/ report PDF and source; reproducible analysis of saved episynth experiments
    paper/          earlier arXiv-style manuscript scaffold
    figures/        figures for the paper
    slides/         presentations: project/ (project plan), talk/ (talk on the epiplexity paper)
    blogpost/       blog post and its preview script
    notes/          literature review (pdf + tex + refs)
    research_notes/ source notes behind the literature review
    plan/           outdated plan, roles, protocol and calendar (historical)
    description.md  historical epimeter architecture and proposed public API

Development
===========

Requires Python 3.10+ and `uv <https://docs.astral.sh/uv/>`_. On a network that needs system TLS
certificates, add ``--system-certs`` to the uv commands.

.. code-block:: bash

    uv sync --extra dev
    .venv/bin/pytest tests/
    .venv/bin/ruff check src/ tests/ code/
