Development stack
=================

The package name, distribution name and import name are all ``episynth``.
The repository name ``Epiplexity`` describes the research project.

.. list-table:: Libraries and responsibilities
   :header-rows: 1
   :widths: 28 72

   * - Library / tool
     - Use in this project
   * - Python 3.10+, PyTorch
     - Typed interfaces, Moirai model, tensor operations, autograd, Adam and EMA;
       CPU development and CUDA experiments.
   * - NumPy, SciPy, scikit-learn
     - Seeded synthetic generators, numerical routines, statistics and kernel generation.
   * - Pydantic
     - Validation of model, window and run configurations and JSON overrides.
   * - safetensors
     - Loading published Moirai weights without requiring the original uni2ts runtime.
   * - PyArrow
     - Reading GIFT-Eval Arrow files.
   * - pytest, pytest-cov
     - Unit and CPU training/resume tests, estimator contracts, coverage reporting.
       Integration tests requiring external data or weights are explicitly skipped
       unless their paths are supplied.
   * - Ruff
     - Import order, unused names, Python modernization and style checks configured
       in ``pyproject.toml``; notebooks are included in linting.
   * - Sphinx, sphinx-rtd-theme
     - Static HTML documentation. Autodoc reads signatures and docstrings;
       Napoleon renders Google/NumPy docstrings; viewcode links to implementation;
       MathJax displays mathematical formulas.
   * - Jupyter, nbconvert, Matplotlib
     - Executable CPU demonstration and plots.
   * - uv, setuptools, GitHub Actions
     - Locked development dependencies, package builds, automated lint/tests/docs
       and GitHub Pages deployment.

External integrations
---------------------

* Moirai 1.x / uni2ts: the port retains checkpoint parameter names. Compatibility
  checks use a separate interpreter with the original uni2ts installation.
* GIFT-Eval: evaluation reads a local dataset snapshot; downloading the large
  snapshot is separate from installing the library.
* Mix, Don't Pick: generators reproduce its processes. Optional ``checks``
  dependencies include ``fbm`` and TimeSynth. The original-generator comparison
  uses a separate reference environment with the versions documented in
  ``src/episynth/README.md``; it is not required for the base test suite.

Planned algorithms
------------------

The working estimator is prequential. Requential is registered and documented,
but its ``run`` method raises ``NotImplementedError``. Bayesian codes and other
estimators discussed in the research plans are future work, not installed or
implemented capabilities of this release.
