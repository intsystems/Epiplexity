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
that a computationally bounded learner can extract from data. This project estimates epiplexity
with small models (MLPs and convolutional networks) on small datasets on a single GPU. The
``epimeter`` library gives one interface that returns the model part and the data part of a code
length in bits for the prequential, requential, variational and Laplace codes, the classical
criteria, EDL and the reservoir score. With it we estimate epiplexity on elementary cellular
automata, label noise, permuted pixels and a planted teacher–student task, measure the spread
over seeds under a fixed protocol, and find when Bayesian model parts agree with epiplexity.

Research publications
===============================
1.

Presentations at conferences on the topic of research
=====================================================
1.

Software modules developed as part of the study
======================================================
1. A python package *epimeter* with all implementation `here <https://github.com/intsystems/Epiplexity/tree/main/src>`_.
2. Runnable examples wired through the registries `here <https://github.com/intsystems/Epiplexity/tree/main/code>`_. The Colab demo notebook will be added to the same folder.

Repository layout
=================

.. code-block:: text

    src/epimeter/   the package: estimators, models, samplers, training, eval
    tests/          pytest
    code/           experiment code and demo
    doc/            Sphinx documentation
    paper/          arXiv-style manuscript
    figures/        figures for the paper
    slides/         presentations: project/ (project plan), talk/ (talk on the epiplexity paper)
    blogpost/       blog post and its preview script
    notes/          literature review (pdf + tex + refs)
    research_notes/ source notes behind the literature review
    plan/           plan, roles, protocol, calendar
    description.md  architecture and public API of the package

Development
===========

Requires Python 3.10+ and `uv <https://docs.astral.sh/uv/>`_. On a network that needs system TLS
certificates, add ``--system-certs`` to the uv commands.

.. code-block:: bash

    uv sync --extra dev
    .venv/bin/pytest tests/
    .venv/bin/ruff check src/ tests/ code/
