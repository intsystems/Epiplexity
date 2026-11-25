.. _architecture:

Architecture
============

The ``episynth`` package helps compare synthetic generators for forecast
pretraining. A generator produces a seeded time series, and the window
builder splits a segment into observed history and a forecast target. An
estimator uses these windows to measure a training run. Downstream
evaluation then scores the resulting forecaster on real series.
See :doc:`generators` for the data sources and :doc:`measurements` for what
each estimator measures.

Layers
------

The data, model, training, and evaluation layers provide reusable
components. Each estimator assembles the components its protocol needs.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Purpose
   * - ``episynth.generators``
     - Eleven univariate generator families, including Waveform, KernelSynth,
       and :ref:`Chaotic <chaotic-generator>`, plus the OU reference source.
       For a fixed generator configuration, ``generate(seeds)`` returns
       one series of length ``L`` per seed in an array of shape ``(n, L)``.
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

* :class:`episynth.estimators.prequential.PrequentialEstimator` sums the
  pre-update window losses and subtracts the cost of scoring the same
  number of windows at the final model's held-out loss rate.
* :class:`episynth.estimators.requential.RequentialEstimator` trains a
  student on forecasts sampled from the teacher and estimates the
  cumulative teacher–student KL before the student updates. Its finite
  ORC mode transmits indices into the student's seeded proposal stream.
* :class:`episynth.estimators.v_information.VInformationEstimator` trains two Moirai
  models with shared training-set normalization, selects checkpoints on the validation
  set, and compares paired losses on the test set. The conditional predictor can
  ignore the context.

How estimators use the trainer
------------------------------

The prequential estimator uses the shared ``Trainer`` and its ``compute_loss`` method.
The requential estimator owns separate teacher and student optimizers so that the teacher can train on real windows during projection recovery while the student and its message stream pause.

CLI
---

The :mod:`episynth.train` CLI reads the generator, window specification,
model, and training settings from a JSON configuration. In the following
CPU example, ``--set`` selects Chaotic as the data source and prequential
coding as the estimator.

.. code-block:: bash

    python -m episynth.train \
        --config publications/technical_report/examples/prequential.json \
        --out build/report_demo/chaotic --set data.generator=chaotic \
        --set estimator=prequential

The ``estimator`` field selects an algorithm from the :mod:`episynth.estimators`
registry and defaults to ``prequential``. The run writes standard ``Trainer`` artifacts
and saves the estimate as ``estimator_result.json`` in the run directory. The equivalent
Python call is ``build("prequential").run(model, generator, spec, config, out_dir)``.
Pass algorithm parameters through ``estimator_config``. The V-information estimator
writes two training branches and a shared manifest. See
``experiments/demos/v_information.json`` for an example.
