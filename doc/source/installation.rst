************
Installation
************

Requirements
============

- Python 3.10+
- PyTorch 2.0+

Installing from GitHub
======================

Install
-------
.. code-block:: bash

	python3 -m pip install git+https://github.com/intsystems/Epiplexity.git

Uninstall
---------
.. code-block:: bash

	python3 -m pip uninstall episynth

Development
===========

The development environment is managed by `uv <https://docs.astral.sh/uv/>`_. On a network that
needs system TLS certificates, add ``--system-certs`` to the uv commands.

.. code-block:: bash

	git clone https://github.com/intsystems/Epiplexity.git
	cd Epiplexity
	uv sync --extra dev
	.venv/bin/pytest tests/
	.venv/bin/ruff check src/ tests/ experiments/demos/

Building the documentation locally
==================================

The documentation uses Sphinx with the Read-the-Docs theme and ``sphinx.ext.autodoc``: API pages
are generated automatically from docstrings. To build locally:

.. code-block:: bash

	uv sync --extra dev          # installs sphinx and sphinx-rtd-theme
	sphinx-build -b html doc/source doc/_build

Then open ``doc/_build/index.html`` in a browser.

Deploying online (GitHub Pages)
===============================

The repository ships a GitHub Actions workflow (``.github/workflows/docs.yaml``) that builds the
Sphinx documentation and deploys it to the ``gh-pages`` branch. After pushing to ``main``, the
documentation is served from ``https://<user>.github.io/<repo>/`` (configure the ``base_url`` in
the workflow for a project site). See :ref:`architecture` for the structure of the docs.