Choosing a synthetic generator
==============================

A generator specifies which patterns a forecaster sees during pretraining.
For example, Waveform gives the model recurring shapes, while GARCH gives it
periods of high and low volatility. ``data.generator`` chooses the source of
the series. ``estimator`` chooses how to measure what the forecaster can
learn from those series. The same generator can supply windows to prequential
coding, requential coding, and predictive V-information.

Available generators
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Configuration name
     - What it generates
   * - ``waveform``
     - Sums of sawtooth, square, or triangular waves.
   * - ``timesynth``
     - Combinations of periodic and autoregressive signals with noise.
   * - ``tsi``
     - Additive or multiplicative combinations of trend, seasonality, and noise.
   * - ``ets``
     - Series with exponential-smoothing dynamics for level, trend, and seasonality.
   * - ``stepfunction``
     - Flat segments joined by jumps, ramps, or sigmoids.
   * - ``arima``
     - Autoregressive and moving-average series with optional differencing.
   * - ``garch``
     - Returns with changing volatility, or cumulative sums of those returns.
   * - ``fbm``
     - Fractional Brownian motion or its increments.
   * - ``kernelsynth``
     - Gaussian-process draws from combinations of kernels, with optional trend.
   * - ``sde``
     - Mean-reverting trajectories with randomly switching regimes.
   * - ``chaotic``
     - A coordinate of a Lorenz trajectory or a scalar Mackey–Glass trajectory.
   * - ``ou``
     - A discrete Ornstein–Uhlenbeck process with a known prior on its
       autoregressive coefficient, used for Bayesian likelihood checks.

Each seed yields one fixed-length series. ``WindowSpec`` determines how much
of its history the model receives and how far ahead it forecasts. The same
interface can read a saved ``Corpus`` or sample from a ``Mixture`` of sources.
See :doc:`api/generators` for constructor options.

.. _chaotic-generator:

Chaotic as a forecasting test case
----------------------------------

For each seed, Chaotic chooses either Lorenz or Mackey–Glass and draws the
parameters, integration step, and starting state or history. Lorenz uses
fourth-order Runge–Kutta and exposes one of its three coordinates.
Mackey–Glass uses forward Euler and depends on a delayed value.
Both discard the first 2,000 steps. The trajectories have no added observation
noise. If the returned series contains nonfinite values, the generator
replaces it with Gaussian noise.

The forecaster sees only the sampled values. It has to infer the hidden
state from a short history, and small errors in that state can grow over
time. This makes Chaotic a test of how much of the short-term dynamics the
model can use. To compare it with another source, keep the forecaster,
window specification, and training budget fixed.

From the repository root, run the prequential CPU example on Chaotic:

.. code-block:: console

   python -m episynth.train --config publications/technical_report/examples/prequential.json --set data.generator=chaotic --out build/report_demo/chaotic

The same override works with the requential and V-information examples
in :doc:`measurements`. In the archived 40M-window runs, Chaotic had a
context gain of 402.41 bits per window and a real-data CRPS ratio of 1.285
relative to the seasonal naïve baseline (lower is better). ETS had a
similar gain of 403.02 bits per window and a lower CRPS ratio of 0.717.
The source-level gain measures how much the past helps on each generator.
The CRPS ratio measures how well the original pretrained forecaster
transfers to real series.

Numerical checks for Chaotic
----------------------------

The tests use fixed parameters and short trajectories. For Lorenz, they
compare fourth-order Runge–Kutta against a high-accuracy SciPy DOP853
solution. Halving the step reduced the maximum absolute error by factors
of 14.6 to 19.6 across the test trajectories, close to the fourth-order
factor of 16.

For Mackey–Glass, constant initial history gives an analytical solution
over the first delay interval. Halving the forward Euler step roughly
halved the error. A separate scalar calculation checks that after the
delay the implementation uses previously computed values at the right
step, including in a batch with different delay lengths.

Run the numerical checks with:

.. code-block:: console

   python -m pytest tests/episynth/test_chaotic.py -q
