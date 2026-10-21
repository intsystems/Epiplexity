Installation and deployment
===========================

Requirements
------------

Python 3.10+ and PyTorch 2.0+ are required. Small tests and the demonstration run
on CPU. Full experiment series use a CUDA GPU and much larger storage and training
budgets. CI validates the project on Python 3.11.

Install the library
-------------------

.. code-block:: bash

   python -m pip install git+https://github.com/intsystems/Epiplexity.git
   python -c "import episynth; print(episynth.__version__)"
   # To uninstall: python -m pip uninstall episynth

Development
-----------

Use `uv <https://docs.astral.sh/uv/>`_ and the committed lockfile:

.. code-block:: bash

   git clone https://github.com/intsystems/Epiplexity.git
   cd Epiplexity
   uv sync --locked --extra dev
   uv run --no-sync pytest tests/
   uv run --no-sync ruff check src/ tests/ experiments/demos/ doc/source/conf.py

On a network requiring system TLS certificates, add ``--system-certs`` to
``uv sync``. For slow downloads set ``UV_HTTP_TIMEOUT=300`` and
``UV_CONCURRENT_DOWNLOADS=4``. The development extra includes testing, notebooks,
plotting and documentation tools. See :doc:`stack` for the library choices.

Build and view documentation locally
------------------------------------

From the repository root, after installing the development dependencies:

.. code-block:: bash

   uv run --no-sync sphinx-build -b html -W --keep-going doc/source doc/_build

Open ``doc/_build/index.html`` in a browser. This directory is a standalone static
website and can be copied to any static web server. Optionally serve it locally:

.. code-block:: bash

   python -m http.server 8000 --directory doc/_build

The :doc:`api/estimators` page is generated automatically from the Python classes
and methods. Sphinx imports the real implementation; missing imports and build
warnings fail the CI documentation build. No GPU, pretrained checkpoint or
GIFT-Eval download is needed to build the documentation.

Publish with GitHub Pages
-------------------------

A repository administrator must select **Settings → Pages → Build and deployment
→ Source: GitHub Actions** once. The ``github-pages`` environment may require a
reviewer if the repository has deployment protection rules.

The workflow ``.github/workflows/docs.yaml`` then:

#. Builds and checks Sphinx documentation on pull requests to ``main``.
#. Builds and uploads a Pages artifact on pushes to ``main`` or a manual run
   (**Actions → docs → Run workflow**).
#. Deploys that artifact with ``actions/deploy-pages`` after a successful build.

Pull requests never publish. Deployment uses Pages artifacts, without a
``gh-pages`` branch. The expected project URL is
``https://intsystems.github.io/Epiplexity/``. After a deployment, check the URL
reported by the deployment job and open its API page. A passing local build
alone does not establish that the online site was published.
