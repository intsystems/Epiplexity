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

Read the `technical report PDF <https://github.com/intsystems/Epiplexity/blob/main/publications/technical_report/main.pdf>`_ for
architecture, interfaces, and examples, with
`reproduction instructions <https://github.com/intsystems/Epiplexity/tree/main/publications/technical_report>`_.
The `research article <https://github.com/intsystems/Epiplexity/blob/main/publications/article/main.pdf>`_
covers coding theory and experiments, with its own
`analysis scripts <https://github.com/intsystems/Epiplexity/tree/main/publications/article>`_.

The `generator catalog <https://github.com/intsystems/Epiplexity/blob/main/doc/source/generators.rst>`_
includes Chaotic, which produces trajectories from Lorenz and Mackey–Glass
dynamics. The library implements three estimators.

* **Prequential coding** measures how much extra log-loss accumulates during
  training relative to the final predictor's loss rate.
* **Requential coding** estimates the cost of teaching a student from
  conditional forecasts sampled from a teacher.
* **Predictive V-information** estimates the reduction in test log-loss
  when the predictor has access to the forecast context.

The library also provides a Bayesian log-loss reference for the OU generator.
See the `measurement guide <https://intsystems.github.io/Epiplexity/measurements.html>`_
for units, inputs, and commands. The
`CPU notebook <https://github.com/intsystems/Epiplexity/blob/main/experiments/demos/demo_estimators.ipynb>`_
and `meeting slides <https://github.com/intsystems/Epiplexity/blob/main/publications/slides/poc_prequential/episynth_poc_prequential.pdf>`_
show the three estimators and compare their measurements with archived CRPS.

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
    publications/      article/ and technical_report/ with sources and reproduction scripts,
                       plus blogpost/, slides/, journal/, and research_notes/
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
