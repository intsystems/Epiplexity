Measurement methods and library contracts
=========================================

The practical goal is to select synthetic-data generators that improve forecasting
on real datasets under a fixed training-data budget. The measurement
methods below support the investigation of generator-selection signals.

The proposed four-method library scope is **prequential coding**, **requential
coding**, **predictive V-information**, and **Bayesian predictive log-loss**.
Prequential coding is the method used in the saved transfer experiments. The
Bayesian method uses the existing OU reference and has narrower applicability.
V-information is implemented; requential remains a stub. This is a concrete
proposal for the library. Generation, fitting, and benchmark evaluation provide the shared
execution layer.

Method status
-------------

.. list-table:: Current code and proposed public methods
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
     - Registered stub; raises ``NotImplementedError``
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

``estimators.available()`` lists registered names, including stubs. Currently
it returns ``prequential``, ``requential`` and ``v_information``; the first and
last run complete estimation procedures. A general Bayesian estimator is not
registered. The OU functions remain in ``episynth.checks.ou_bayes``.

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

**Required inputs.** Teacher/student models and update rules; real-data stream;
shared contexts; seeded proposal sampling; smoothing/projection settings;
training budget; and the code's shared initialization conventions.

**Required computation.** Implement teacher updates and student training on
selected synthetic targets, account for whole-message teacher/student KL,
and define reproducible selection and decoding. The KL estimate, expected-code
bound including overhead, and realized encoded length need distinct result
labels. Contexts must either be available to the decoder as declared side
information or have their transmission cost included.

**Required outputs.** Student checkpoint, cumulative KL estimate in bits,
overhead convention/bound, message count, and the settings needed to reproduce
the measurement. A real codec must additionally provide its messages and
decoder. These are proposed requirements, not fields returned by current code.

**Current behavior.** Construction succeeds; calling
``RequentialEstimator.run`` raises ``NotImplementedError``. The constructor's
parameters describe the intended implementation. They do not currently control
a functioning teacher/student algorithm. ``test_estimators.py`` verifies this
unsupported status.

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

4. Bayesian predictive log-loss by OU quadrature
------------------------------------------------

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
A unified schema for all four methods and wrappers for the remaining methods
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
