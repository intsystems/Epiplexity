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

	python3 -m pip uninstall epimeter

Development
===========

The development environment is managed by `uv <https://docs.astral.sh/uv/>`_.

.. code-block:: bash

	git clone https://github.com/intsystems/Epiplexity.git
	cd Epiplexity
	uv sync --extra dev
	.venv/bin/pytest tests/
	.venv/bin/ruff check src/ tests/ code/
