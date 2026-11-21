.. _architecture:

Architecture
============

The ``episynth`` package estimates epiplexity in synthetic time series. An observer model
trains on a stream of windows from a synthetic generator. The estimator reports a
decomposition of the code length in bits or the predictive value of the context in bits
per window (V-information).

Layers
------

The library has shared layers and an estimator plugin layer. The shared layers are
independent of the estimation algorithm and serve all estimators in the same way.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Purpose
   * - ``episynth.generators``
     - Eleven univariate series generators from Mix, Don't Pick (including KernelSynth, ETS,
       Waveform and SDE), plus an OU reference generator. Each series depends only on its
       seed and length. ``generate(seeds)`` returns an array of shape ``(n, L)``.
   * - ``episynth.windows``
     - Forecasting windows. ``WindowSpec`` defines the task shape, ``WindowBatch`` holds
       a batch of windows, and ``WindowStream`` streams batches in ascending seed order
       on a background thread.
   * - ``episynth.model.moirai``
     - Moirai 1.x port with ``MoiraiModule``, ``MoiraiConfig``, ``Mixture`` (a mixture
       distribution with ``log_prob`` and ``sample``), and support for loading published
       checkpoints.
   * - ``episynth.training``
     - Shared ``Trainer`` for window batches, EMA weights, checkpointing, resuming runs,
       and held-out evaluation. Override ``compute_loss`` for a protocol-specific step.
   * - ``episynth.evaluation``
     - Forecasters, metrics, and the ``run`` entry point for GIFT-Eval evaluation.
   * - ``episynth.estimators``
     - **Plugin layer** for selecting epiplexity estimators and adding custom algorithms.

Estimator plugin interface
--------------------------

Every estimation algorithm implements the
:class:`episynth.estimators.base.EpiplexityEstimator` protocol.

.. code-block:: python

    class EpiplexityEstimator(Protocol):
        name: str
        def run(
            self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path,
        ) -> Mapping[str, Any]: ...

The ``run`` method carries out training and returns a JSON-compatible dictionary.
The prequential estimator returns ``model_part_bits`` and ``data_part_bits``.
The V-information estimator returns predictive entropies and their difference in bits
per window.

Register algorithms by name in :mod:`episynth.estimators.registry` and instantiate them
with the :func:`episynth.estimators.registry.build` factory.

.. code-block:: python

    from pathlib import Path
    from episynth.estimators import EpiplexityEstimator, register, build
    from episynth.generators.base import Generator
    from episynth.model.moirai import MoiraiModule
    from episynth.training import TrainConfig
    from episynth.windows import WindowSpec

    @register("my_estimator")
    class MyEstimator(EpiplexityEstimator):
        def run(
            self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path,
        ) -> dict[str, float]:
            ...  # Train and compute the code length.
            return {"model_part_bits": ..., "data_part_bits": ...}

    est = build("my_estimator")          # Select by name.
    result = est.run(model, generator, spec, config, "runs/my")

Users can select an algorithm by name or add a custom algorithm in a single module.

Built-in estimators
-------------------

* :class:`episynth.estimators.prequential.PrequentialEstimator` uses the shared ``Trainer``
  to sum window losses before their training step (the area under the loss curve).
* :class:`episynth.estimators.requential.RequentialEstimator` concerns the code for a student
  reconstructed from indices of samples drawn from the teacher's distribution. The
  registered stub raises :class:`NotImplementedError` on execution. See
  ``publications/research_notes/.../requential_coding.md``.

* :class:`episynth.estimators.v_information.VInformationEstimator` trains two Moirai
  models with shared training-set normalization, selects checkpoints on the validation
  set, and compares paired losses on the test set. The conditional predictor can
  ignore the context.

How estimators use the trainer
------------------------------

The shared ``Trainer`` in :mod:`episynth.training` runs the training loop and delegates
protocol-specific window losses to ``Trainer.compute_loss``. The prequential estimator
uses the standard implementation. The planned requential estimator subclasses
``Trainer`` and overrides ``compute_loss`` to estimate the teacher-student KL divergence.

CLI
---

The :mod:`episynth.train` CLI selects an estimator and launches large experiment
series. Pass the configuration file with ``--config``. It contains the ``data``,
``spec``, ``model``, and ``train`` sections and the ``estimator`` field. Override
any field with ``--set``.

.. code-block:: bash

    python -m episynth.train --config configs/kernelsynth.json --out runs/epi \
        --set train.lr=3e-4 --set estimator=prequential

The ``estimator`` field selects an algorithm from the :mod:`episynth.estimators`
registry and defaults to ``prequential``. The run writes standard ``Trainer`` artifacts
and saves the estimate as ``estimator_result.json`` in the run directory. The equivalent
Python call is ``build("prequential").run(model, generator, spec, config, out_dir)``.
Pass algorithm parameters through ``estimator_config``. The V-information estimator
writes two training branches and a shared manifest. See
``experiments/demos/v_information.json`` for an example.
