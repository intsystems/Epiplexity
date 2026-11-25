Measurement methods and library contracts
=========================================

The practical goal is to select synthetic-data generators that improve forecasting
on real datasets under a fixed training-data budget. The measurement
methods below support the investigation of generator-selection signals.

The library implements synthetic generators, including :ref:`Chaotic
<chaotic-generator>`. It implements three estimation algorithms, **prequential
coding**, **requential coding**, and **predictive V-information**. The archived
comparisons pair these measurements with the original forecasters' CRPS at
matched training budgets. The library also provides a Bayesian predictive
log-loss reference for the OU generator.

Method status
-------------

.. list-table:: Implemented estimators and reference
   :header-rows: 1
   :widths: 22 27 28 23

   * - Method
     - Question answered
     - Current entry point
     - Readiness
   * - Prequential coding
     - How expensive was predicting the training sequence while learning it?
     - ``build("prequential").run(...)``;
       ``PrequentialEstimator.from_run_dir(...)``
     - Implemented estimator
   * - Requential coding
     - How expensive is reconstructing a student's learned predictive behavior?
     - ``build("requential").run(...)``
     - Direct estimate and finite ORC codec
   * - Predictive V-information
     - How much does context improve prediction for a specified model family?
     - ``build("v_information").run(...)``;
       ``VInformationEstimator.from_run_dir(...)``
     - Implemented estimator
   * - Bayesian predictive log-loss
     - What loss is achievable under the known OU prior and forecast factorization?
     - ``checks.ou_bayes.exact_window_nll`` and
       ``pointwise_window_nll``
     - Implemented reference; common public wrapper proposed

``estimators.available()`` lists the three registered estimators.
The OU reference is available in ``episynth.checks.ou_bayes``.

1. Prequential coding and its structural proxy
----------------------------------------------

**Inputs.** An initial model, an ordered seeded task stream, ``WindowSpec``,
``TrainConfig``, and an output directory. The recomputation interface takes a
completed run directory instead.

**Computation.** The model assigns a density to each future target before the
update on its batch. ``Trainer`` saves these losses in ``window_loss.f32``.
The final raw or EMA model is evaluated on separate held-out seeds. Summing
pre-update losses produces the prequential log-loss; subtracting the endpoint
cost produces the signed structural proxy used in this project.

For per-window losses in nats, the returned fields are

.. math::

   L = \frac{\sum_i\ell_i}{\ln 2},\qquad
   H = \frac{N\bar\ell_{\mathrm{heldout}}}{\ln 2},\qquad
   S = L-H.

**Outputs and units.** ``total_bits=L``, ``data_part_bits=H``,
``model_part_bits=S``, and ``n_windows=N``. The current return type is
``dict[str, float]``. ``model_part_bits`` is a signed proxy depending on the
observer, optimizer, data order, budget and floor variant. Continuous density
losses can be negative. An encoded file additionally needs a defined
quantization and codec; this interface returns measurements and run artifacts.

**Entry points and effects.** ``run`` trains or resumes the model and writes
checkpoints, configuration, and losses. ``from_run_dir`` only reads the
saved artifacts and computes a dictionary:

.. code-block:: python

   from episynth.estimators.prequential import PrequentialEstimator

   result = PrequentialEstimator(variant="ema").from_run_dir("runs/example")
   print(result["model_part_bits"], result["data_part_bits"])

The directory must contain ``window_loss.f32`` and ``heldout_ema_1.f32`` (or
``heldout_raw_1.f32`` for ``variant="raw"``). A ``steps.csv`` export alone
does not satisfy this entry point. The research article's ``analyze_runs.py``
reconstructs area from batch means as a separate analysis path.
Loss files must be nonempty, finite float32 arrays, with at least two held-out
windows. For Trainer outputs, the reader also checks the configured counts and
the final ``heldout.jsonl`` record. Historical directories containing only the
two loss files remain readable, but their expected counts and run completion
cannot be verified without metadata.
``analysis.run_area`` also reports floor-only standard error when individual
held-out losses are available. Tests: ``test_estimators.py`` and
``test_analysis.py``.

2. Requential coding
--------------------

The method compresses a student's training trajectory through teacher-selected
proposals regenerated from shared randomness. The reference is
`Qiu et al. (2026) <https://arxiv.org/abs/2607.11883>`_.

**CPU example.** The example uses a one-layer Moirai with a Student-t head, four-point horizons, and a budget of 32 real waveform windows.
Set ``RUN_ROOT`` to your experiment storage directory.

.. code-block:: bash

   python -m episynth.train --config experiments/demos/requential.json --out "$RUN_ROOT/req"
   python -m episynth.train --config experiments/demos/requential.json --out "$RUN_ROOT/orc" \
       --set estimator_config.mode=orc --set estimator_config.candidates=5 \
       --set 'estimator_config.projection_steps=[2]' --set estimator_config.sampling_seed=1

**Training and splits.** The teacher trains on real windows. The student trains on EMA-teacher samples conditional on public contexts, with the source horizon erased.
Use continuous heads. A pure log-normal mixture requires a shared lower endpoint and observed targets above it.
The example reserves seeds [0, 32) for real training, [32, 64) for public contexts, [64, 72) for projection validation, and [72, 80) for final evaluation.
Set ``public_first_seed``, ``validation_first_seed``, and ``validation_windows`` in the estimator configuration to place these splits in an existing corpus.
For a prequential comparison, share the exact initial weights, training seeds, batch size, and final held-out split.
Teacher-only recovery consumes the same real budget and sends zero messages.
The result records the validation loss before projection, the final recovery loss, the number of recovery windows, and whether the teacher recovered within budget.

**Estimated length.** Each observed horizon is one message.
``estimated_kl_bits`` sums signed Monte Carlo log likelihood ratios over the coded windows and can be negative on a short run.
Independent teacher draws (``mc_repeats`` per message) estimate each message KL for the nonlinear overhead.
The plug-in uses the nonnegative part of that mean as :math:`D` and sums :math:`D+2\log_2(1+D)+5.21` in ``plugin_expected_length_bound_bits`.
``estimated_overhead_bits`` reports the sum of the last two terms alone.
The reported Monte Carlo standard errors use within-message repeat variances for the leading term and a delete-one jackknife for the overhead, conditional on the visited model states.
Variation across training runs requires independent seeds.
Finite Monte Carlo sampling gives an estimate of the expected-length bound, without a certified upper bound.

**Replay.** Finite ordered random coding uses a fixed pool of student proposals for each horizon.
The encoder chooses a proposal by its ordered exponential arrival time and likelihood ratio, then writes its one-based Elias-delta index to ``indices.bin``.
The selected horizons approximate the teacher law, so the encoder uses separate direct teacher draws for the KL estimate.
The finite race is Algorithm 4 of `Theis and Yosri (2022) <https://proceedings.mlr.press/v162/theis22a.html>`_.
Large message KL can require exponentially many proposals.

.. code-block:: python

   from pathlib import Path
   from episynth.estimators import build

   codec = build("requential", **result["estimator_config"])
   payload = Path(result["artifacts"]["stream"]).read_bytes()
   raw, ema = codec.decode(initial_model, generator, spec, train, payload,
                           result["realized_message_bits"], result["message_count"])

The decoder needs the exact initial state, public source and seeds, and shared student update settings on the same deterministic runtime.
Corpus-backed contexts require the corresponding corpus.
``realized_message_bits`` counts only the meaningful payload bits. The byte padding, message count, bit-length framing, and shared setup are excluded.
``estimator_result.json`` records the split identities, generator settings, KL trajectory, projection events, and final EMA held-out NLLs in nats per window.
Each run stores ``student.pt``, its report, and any index stream in a separate subdirectory.
The evaluation CLI accepts this checkpoint with either ``--weights raw`` or ``--weights ema``.
The top-level report points to these artifacts so later runs can reuse the output directory while earlier streams remain replayable.

3. Predictive V-information
---------------------------

The quantity measures the reduction in optimal predictive log-loss when context
is available to a constrained predictive family; see
`Xu et al. (ICLR 2020) <https://arxiv.org/abs/2002.10689>`_. The report's
measurement section gives the paired empirical estimate.

**Entry points.** ``build("v_information", variant="ema", test_first_seed=2_000_000_000,
test_windows=100_000).run(model, generator, spec, config, out_dir)`` and
``VInformationEstimator.from_run_dir(out_dir)``. Parameters are also accepted via
``RunConfig.estimator_config`` or ``episynth.estimators.run --estimator-config``
(a JSON object). ``variant`` is ``ema`` or ``raw``; test size must be at least two.

**Task and family.** X is the observed context and Y is the forecast horizon from
``WindowSpec``. Two independent copies of the supplied initial Moirai are fitted
on the same seeded windows, each with the full training budget. The family uses
continuous density components and the existing product of pointwise forecast
densities. It explicitly includes ignoring context: the conditional selection
may use the fitted context-free predictor. Arbitrary supervised tasks and joint
autoregressive horizon densities are outside this implementation.

**Normalization and null input.** An additional streaming pass computes mean and
population variance over train horizon points in float64. Both models use the
same fixed shift and ``sqrt(variance + 1e-5)`` scale. The fixed constants are saved
in ``MoiraiConfig.fixed_scaling``; log densities retain their original units and
Jacobian correction. ``ignore_context=True`` removes context values, masks,
padding structure, lengths and time IDs from the horizon's inputs inside Moirai.
Forecast tokens receive canonical IDs and fixed mask representations. No target
values enter the prediction. Other Moirai runs retain their existing defaults.

**Splits and selection.** ``TrainConfig.first_seed/n_windows`` defines train;
``heldout_first_seed/heldout_windows`` defines validation; the estimator settings
define test. Reject empty or overlapping ranges and missing corpus coverage,
including overlaps after projecting nested mixture seeds to their source rows.
Checkpoints must include the final fraction 1.0. Choose the lowest validation NLL
for the requested weight variant at ``eval_precision``. Select baseline
independently, then choose the conditional predictor from conditional checkpoints
and that baseline. Ties prefer baseline, then the earlier step. Only after this
selection is persisted, evaluate both roles on the identical untouched test
windows, including identical dequantization.

For paired test losses in nats, reduce in float64:

.. math::

   d_i = (\ell_{\varnothing,i}-\ell_{X,i})/\ln 2,\qquad
   \widehat I_{\mathcal V}=\bar d,\qquad
   \mathrm{SE}=s_d/\sqrt M.

**Outputs and interpretation.** ``method="v_information"``,
``units="bits_per_window"``, ``unconditional_entropy_bits_per_window``,
``conditional_entropy_bits_per_window``, ``v_information_bits_per_window``,
``standard_error_bits_per_window``, ``n_test_windows`` and metadata for the family,
initialization, budgets, splits, normalization and selected checkpoints. Negative
empirical information is retained. The standard error conditions on fitted
models; it does not include variation across training runs. This is a
budget-limited approximation, not a claim of exact family optimization or of
satisfying the paper's PAC assumptions. No model/data coding-cost fields are used.

**Artifacts and resume.** ``unconditional/`` and ``conditional/`` hold normal
Trainer runs. ``v_information.json`` stores shared settings and selection;
``test_unconditional.f32`` and ``test_conditional.f32`` store ordered NLL in nats;
``pvi_bits.f64`` stores paired differences in bits. Independent branch resume
reuses normalization and rejects changed settings or initialization. Recomputing
with ``from_run_dir`` is read-only and rejects incomplete, nonfinite or
inconsistent artifacts. The CLI and estimator write ``estimator_result.json``.

**Executable CPU example.** From the repository root:

.. code-block:: bash

   python -m episynth.train --config experiments/demos/v_information.json \
       --out runs/v_information_cpu

Tests in ``test_v_information.py`` cover Gaussian analytical information,
independence, null-input invariance, train-only normalization, validation-only
selection, split coverage/isolation, interruption recovery and both CLIs.

Bayesian predictive log-loss by OU quadrature
---------------------------------------------

**Inputs.** A CPU ``WindowBatch`` from the OU prior, its ``WindowSpec``, and
quadrature nodes from ``phi_nodes(n_nodes)``. The node count controls numerical
accuracy. This implementation assumes the prior in ``generators.ou.OU``.

**Computation.** Marginalize the latent AR(1) coefficient under the known prior.
The joint score uses the horizon's full conditional density given context.
The pointwise score sums marginal future-point losses given context, matching
the factorization used by Moirai. The difference of their expected losses
quantifies the cost of ignoring dependencies between future targets.

**Outputs and units.** ``exact_window_nll`` and ``pointwise_window_nll`` return
arrays of shape ``(B,)`` in nats per window. Divide by ``horizon`` for
nats/point or by ``ln(2)`` for bits/window. The name ``exact_window_nll``
refers to the Bayesian formula; its implementation evaluates a numerical
quadrature. Compare node counts to establish convergence for a given task.

.. code-block:: python

   from episynth.checks.ou_bayes import (
       exact_window_nll, phi_nodes, pointwise_window_nll,
   )

   nodes = phi_nodes(2000)
   joint = exact_window_nll(batch, spec, nodes)
   pointwise = pointwise_window_nll(batch, spec, nodes)

``batch`` and ``spec`` in this example must describe the same OU windows.
Use ``dequantize_width=0`` to match the clean generative likelihood exactly;
the existing check otherwise evaluates that likelihood at dequantized values,
without convolving the density with the added noise. A model/reference gap
must use the same windows and matching pointwise or joint factorization.

**Current packaging.** The functions are implemented and tested in
``test_ou_bayes.py``. The CLI ``python -m episynth.checks.ou_bayes --run ...``
additionally loads raw/EMA checkpoints and prints losses and gaps; it requires
an OU training run. A wrapper in a common measurement API remains proposed.
There are no archived OU transfer runs in the report's 23-run comparison.

Shared infrastructure and release contract
------------------------------------------

Task generation converts a prior and seeds into masked forecasting examples.
``Trainer`` fits density models and writes the pre-update trajectory.
GIFT-Eval scoring measures forecast quality; Spearman/bootstrap analysis
compares generator-level rankings. These procedures support the measurement
methods above and remain useful independent library operations.

For a future common measurement API, each result should declare its method,
quantity definition, units and normalization, predictive family/prior, data
split and seed identifiers, compute/optimization budget, and artifact paths.
The estimator protocol accepts method-specific JSON-compatible results. Existing
prequential fields remain unchanged; V-information has the result contract above.
A unified schema for the estimators and a wrapper for the OU reference
remain proposals.

A released method should have an executable example and a check against its
mathematical definition. For prequential this includes the sum/decomposition
identity; for requential, seeded regeneration and separately labeled code
costs; for V-information, paired held-out scoring; for the OU reference,
the single-node Gaussian case and quadrature convergence.

The `technical report <https://github.com/intsystems/Epiplexity/blob/main/publications/technical_report/main.pdf>`_
contains a comparison of the methods and executable examples. The
`research article <https://github.com/intsystems/Epiplexity/blob/main/publications/article/main.pdf>`_
develops the coding interpretation and analyzes the transfer experiments.
See :doc:`api/estimators` for the current
estimator API and :doc:`extending` for registration mechanics.
