CPU demonstration and generated API
===================================

The notebook uses a one-layer model with width 64, a waveform generator and
64 training windows. It selects the prequential estimator through the same
registry used by the large runs, computes the decomposition in bits and plots
the losses. It also demonstrates registering a custom estimator.

:download:`Download the notebook <../../experiments/demos/demo_estimators.ipynb>`
or open ``experiments/demos/demo_estimators.ipynb`` from the repository root:

.. code-block:: bash

   uv sync --locked --extra dev
   uv run --no-sync jupyter lab experiments/demos/demo_estimators.ipynb

To execute all cells without a UI, starting from the repository root:

.. code-block:: bash

   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
     uv run --no-sync jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=300 \
     --output demo_estimators.executed.ipynb --output-dir /tmp/episynth-demo \
     experiments/demos/demo_estimators.ipynb

The executed copy is saved outside the source tree. Run artifacts are written
under the notebook's ``runs/`` directory (ignored by Git). Remove that demo run
directory if you want to retrain from scratch; an existing run resumes.

Automatically generated documentation
--------------------------------------

Open :doc:`api/estimators` to show the automatically generated documentation for
:class:`~episynth.estimators.prequential.PrequentialEstimator`, including
:meth:`~episynth.estimators.prequential.PrequentialEstimator.run` and
:meth:`~episynth.estimators.prequential.PrequentialEstimator.from_run_dir`.
The parameter and return descriptions come from docstrings, and type information
comes from annotations in the source. The class has a source-code link.

For example, ``build("prequential")`` returns that estimator;
``available()`` lists the registered names. ``requential`` appears in that list
as a planned implementation and raises ``NotImplementedError`` when run.
