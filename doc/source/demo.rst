CPU demonstration and generated API
===================================

The notebook runs three implemented estimators on a one-layer waveform model with 64 training windows.
Prequential coding decomposes the training loss, predictive V-information measures the context gain in bits per window, and requential coding transmits a student’s update choices as indices.
The embedded figures compare the archived measurements against downstream CRPS.

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

Use ``available()`` to list the registered names and ``build("prequential")`` to construct the estimator used in this notebook.
For separate CPU runs and the output units of each estimator, see :doc:`measurements`.
