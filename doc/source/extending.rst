Extending: adding your own estimator
====================================

The protocol returns method-specific JSON-compatible results. See
:doc:`measurements` for coding decompositions, the implemented predictive-information
result and the proposed Bayesian reference wrapper. Registering a class supplies a name; its algorithm still
needs an implementation.

Add an epiplexity estimation algorithm in **one module** by implementing and
registering the :class:`episynth.estimators.base.EpiplexityEstimator` protocol.

Step 1. Implement the protocol
------------------------------

Each estimator must provide a ``name`` attribute and a ``run`` method.

.. code-block:: python

    def run(
        self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
        config: TrainConfig, out_dir: str | Path,
    ) -> Mapping[str, Any]:
        """Run training and return the estimate in bits."""

Arguments

* ``model`` (:class:`~episynth.model.moirai.MoiraiModule`) is the observer model.
* ``generator`` (:class:`~episynth.generators.base.Generator`) supplies time series.
* ``spec`` (:class:`~episynth.windows.WindowSpec`) defines the forecasting task shape.
* ``config`` (:class:`~episynth.training.TrainConfig`) supplies run settings.
* ``out_dir`` is the directory for run artifacts.

The method returns a JSON-compatible dictionary with algorithm-specific fields.
The prequential estimator returns ``model_part_bits`` and ``data_part_bits``.
The V-information estimator returns entropies and their difference in bits per window.
The protocol returns ``Mapping[str, Any]`` to support typed result dictionaries.

Step 2. Create the algorithm module
-----------------------------------

Create a file such as ``my_package/my_estimator.py``.

.. code-block:: python

    from pathlib import Path

    from episynth.estimators import EpiplexityEstimator, build, register
    from episynth.generators.base import Generator
    from episynth.model.moirai import MoiraiModule
    from episynth.training import TrainConfig
    from episynth.windows import WindowSpec

    @register("my_estimator")
    class MyEstimator(EpiplexityEstimator):
        name = "my_estimator"

        def __init__(self, variant: str = "ema") -> None:
            self.variant = variant

        def run(
            self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path,
        ) -> dict[str, float]:
            # Wrap the built-in algorithm in this example.
            # Replace the method body with your own algorithm.
            return build("prequential", variant=self.variant).run(
                model, generator, spec, config, out_dir
            )

Step 3. Use the estimator
-------------------------

After registration, the algorithm is available by name through the factory and CLI.

.. code-block:: python

    import my_package.my_estimator  # Register the external plugin.
    from episynth.estimators import build
    est = build("my_estimator", variant="ema")
    result = est.run(model, generator, spec, config, "runs/my")

Select a built-in estimator with the ``estimator`` field in the CLI configuration.
Pass constructor arguments through ``estimator_config``, for example
``{"variant": "raw", "test_windows": 1000}`` for V-information.
Import an external plugin **in the same process** before calling the CLI.

.. code-block:: python

    import my_package.my_estimator
    from episynth.train import main

    # episynth.train.main reads sys.argv, as in the shell CLI.
    import sys
    sys.argv = ["train", "--config", "configs/kernelsynth.json", "--out", "runs/epi",
                "--set", "estimator=my_estimator"]
    main()

To add an algorithm to ``episynth`` itself, place it in
``src/episynth/estimators/`` and arrange for it to be imported. Built-in estimators
register automatically in :mod:`episynth.estimators.registry` when imported.

A minimal estimator using the shared ``Trainer``
------------------------------------------------

Check registration as shown below.

.. code-block:: python

    from episynth.estimators import EpiplexityEstimator, available, build
    assert "prequential" in available()
    assert "requential" in available()
    est = build("prequential")
    isinstance(est, EpiplexityEstimator)  # True
