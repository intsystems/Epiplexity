|test| |docs|

`Documentation website <https://intsystems.github.io/Epiplexity/>`_

.. |test| image:: https://github.com/intsystems/Epiplexity/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/intsystems/Epiplexity/actions/workflows/test.yml
    :alt: Test status

.. |docs| image:: https://github.com/intsystems/Epiplexity/actions/workflows/docs.yaml/badge.svg?branch=main
    :target: https://github.com/intsystems/Epiplexity/actions/workflows/docs.yaml
    :alt: Docs status


.. class:: center

    :Research topic: Epiplexity and transfer in synthetic time-series pretraining
    :Research category: CoIS, BMM
    :Authors: Danila Chernousov, Mark Ikonnikov, Daniil Kazachkov, Alexey Kravatskiy
    :Research supervisor: To be confirmed
    :Research adviser (if applicable): To be confirmed

Abstract
========

We develop ``episynth``, a Python library for studying what forecasting models
learn from synthetic time series. It connects seeded generation, probabilistic
training, information measurements, and evaluation on real forecasting tasks.
Our research examines whether measurements collected during pretraining help
select synthetic data sources. In the completed eleven-generator comparison,
KernelSynth gives the best aggregate transfer and excess prequential loss has
a weak association with forecasting quality across the full set.

The earlier research roadmap in ``publications/plan/`` is **outdated**.
Current implementation details are in ``src/episynth/README.md`` and
``publications/description.md``. The research article analyzes the saved experiments.

Technical report and measurement methods
========================================

Read the `technical report PDF <publications/technical_report/main.pdf>`_ for
architecture, interfaces, and examples, with
`reproduction instructions <publications/technical_report/README.md>`_. The
`research article <publications/article/main.pdf>`_ covers coding theory and
experiments, with its own `analysis scripts <publications/article/README.md>`_.
The four-method library scope is described in the
`measurement contracts <doc/source/measurements.rst>`_:

1. **Prequential coding (implemented):** measure the predictive cost of the training
   sequence; return total, residual-data and signed model-proxy costs in bits.
2. **Requential coding (implemented):** estimate the bits needed to teach a student from conditional forecast samples.
   Set ``RUN_ROOT`` to your experiment storage directory and run ``python -m episynth.train --config experiments/demos/requential.json --out "$RUN_ROOT/requential_cpu"``.
3. **Predictive V-information (implemented):** fit predictors with and without context,
   select on validation, then estimate paired test log-loss reduction in bits per window.
   CPU example: ``python -m episynth.train --config experiments/demos/v_information.json --out runs/v_information_cpu``.
4. **Bayesian predictive log-loss (OU reference implemented):** marginalize the known
   AR(1) prior to obtain joint and pointwise forecast-loss references. A common public
   wrapper remains proposed.

Seeded task generation, AdamW/EMA training, and GIFT-Eval scoring provide shared
infrastructure.
The `meeting slides <publications/slides/poc_prequential/episynth_poc_prequential.pdf>`_
and `CPU notebook <experiments/demos/demo_estimators.ipynb>`_ show all three implemented
estimators and compare their measurements with the original forecasters’ archived CRPS.

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
                       figures/, technical_report/ (article and reproducible analysis), journal/ (research
                       journal of the time-series track: landscape and experiment, journal.pdf), description.md
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
If the site returns 404 and Actions shows a separate **pages build and deployment**
run using Jekyll, change **Settings → Pages → Source** from branch publishing to
**GitHub Actions**, then run ``docs`` manually. Publishing the repository root does
not build the Sphinx site.
