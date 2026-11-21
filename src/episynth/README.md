# episynth

Tools for studying epiplexity in synthetic time series. The package provides
generators, forecasting windows, a Moirai port, and training with per-window loss logs.

## Layers

- `episynth.generators` generates series from seeds.
- `episynth.windows` builds forecasting windows with a random context length, a
  horizon, patches, and dequantization. A background thread streams batches in
  ascending seed order.
- `episynth.model` ports Moirai 1.x from uni2ts and loads published checkpoints.
  Its mixture distribution reports log densities in nats.
- `episynth.corpus` uses worker processes to write seeded series to a file. A
  saved corpus can supply series through the generator interface to reduce CPU
  load during training.
- `episynth.training` runs training with pre-update per-window losses, a constant
  learning rate after warmup, EMA weights, checkpoints at selected fractions of
  the run, held-out evaluation, and resume support.
- `episynth.train` launches a run from a configuration file. It accepts a generator,
  corpus, or mixture, a context-length table, and command-line overrides.
- `episynth.analysis` computes excess area above final loss, its held-out
  uncertainty, the area curve over prefixes, comparison metrics by task (ranks and
  ratios to the naive forecast), and Spearman correlations with generator-level
  bootstrap intervals.
- `episynth.evaluation` evaluates on GIFT-Eval. It loads a `Salesforce/GiftEval`
  snapshot without GluonTS and builds the benchmark's test windows. It provides
  seasonal-naive and Moirai forecasters (using either the local protocol or the
  leaderboard notebook's protocol), per-window metrics and their exact aggregation
  under the leaderboard and scaled definitions, and the log density of observed targets.
- `episynth.estimators` provides the epiplexity estimator plugin interface. The
  `estimator` field of an `episynth.train` configuration selects a registered
  algorithm by name. Built-ins include `prequential` (excess loss area),
  `v_information` (predictive value of context), and the `requential` stub. Add
  another estimator in one module using `@register`.

```python
from episynth.generators import GENERATORS
from episynth.model.moirai import small
from episynth.training import TrainConfig, Trainer, build_model
from episynth.windows import WindowSpec

spec = WindowSpec(patch_size=32, context_lengths=(128, 256, 512))
model = build_model(small(patch_sizes=(32,), components=("student_t", "normal_fixed", "log_normal")), init_seed=0)
cfg = TrainConfig(n_windows=4_800_000, batch_size=256, device="cuda")
Trainer(model, GENERATORS["kernelsynth"](device="cuda"), spec, cfg, "runs/kernelsynth").run()
```

The `episynth.training` docstring lists the files in a run directory and their
units. Training and held-out seed ranges must be disjoint, including within
mixture components. When resuming, the trainer checks the data source, window
shape, model, and training settings. For prequential runs, `device`, `compile`,
`eval_batch_size`, `stream_workers`, `log_every`, and `fused_adam` can change.
Legacy checkpoints with missing source metadata require a new directory for
training. Their weights can still be evaluated. Every run produces a final
checkpoint and a held-out evaluation. A resumed run completes any interrupted
final evaluation.

The same workflow can use configuration files and a saved corpus. The corpus is
a float32 array of shape `(n, 1024)` with a manifest. Row `i` holds the series
for seed `first_seed + i`, so a corpus-backed run produces the same windows as
on-the-fly generation with the same seeds.

```bash
python -m episynth.corpus --generator kernelsynth --first-seed 0 --n 2000000 --out corpora/kernelsynth/train
python -m episynth.corpus --generator kernelsynth --first-seed 1000000000 --n 100000 --out corpora/kernelsynth/heldout
python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth --set train.lr=3e-4
```

The configuration defines `data`, `spec`, `model`, and `train` sections, a
`context_table` path, and the `estimator` name. See the `episynth.train` docstring
for the format and rules. An equally weighted mixture of generators or corpora
(`Mixture`, configured under `data.mixture`) helps choose a common learning rate.

The `estimator` field defaults to `prequential`. A run writes standard `Trainer`
artifacts and stores the estimate in `estimator_result.json`.

```bash
python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth \
    --set estimator=prequential --set train.lr=3e-4
```

The `episynth.estimators` docstrings describe the plugin interface. The prequential
estimator measures the excess area and reads the final held-out loss from
`heldout_ema_1.f32` through `episynth.analysis.run_area`. To add an estimator,
implement `EpiplexityEstimator` in a module and decorate the class with
`@register("name")`. Select it with `--set estimator=name`.

## V-information

`estimator=v_information` trains two Moirai models, with and without context,
using the same initial weights and budget. The `estimator_config` constructor
settings are `variant` (`ema` by default, or `raw`), `test_first_seed`
(2,000,000,000), and `test_windows` (100,000 by default, with a minimum of two).
The existing `train.heldout_*` fields define validation. Training, validation,
and test ranges must be disjoint and covered by the source, including the
underlying mixture component seeds. The estimator supports continuous density
components.

```bash
python -m episynth.train --config experiments/demos/v_information.json --out runs/v_information_cpu
python -m episynth.train --config experiments/demos/v_information.json --out runs/v_information_raw \
    --set estimator_config.variant=raw
python -m episynth.estimators.run --estimator v_information --generator waveform \
    --config-json experiments/demos/v_information.json --out runs/v_information_cli \
    --estimator-config '{"test_first_seed":2000,"test_windows":32}'
```

An additional streaming pass computes the mean and scale in float64 using only
the training targets. Both models use this fixed normalization. `log_prob` and
NLL remain in the original series units. `MoiraiConfig.fixed_scaling=(loc, scale)`
requires `scaling=True`, finite constants, and a positive scale.
`ignore_context=True` enables context-free prediction. Both settings are saved
with the model configuration.

Validation compares checkpoints for the chosen `variant` at
`train.eval_precision`. The conditional predictor is selected from conditional
checkpoints and the best baseline. Ties favor the baseline, then the earlier
step. The estimator reads test windows only after model selection. It scores
both predictors on identical targets, including dequantization. Horizon NLL
uses Moirai's current factorization as a product of pointwise densities.

The result includes `unconditional_entropy_bits_per_window`,
`conditional_entropy_bits_per_window`, their difference
`v_information_bits_per_window`, the paired `standard_error_bits_per_window`,
`n_test_windows`, and metadata. Negative estimates are retained. The standard
error reflects test-sample uncertainty conditional on the fitted models.
The estimate uses a finite optimization budget and supplies neither an infimum
over the predictor family nor a PAC guarantee. It reports predictive losses
without decomposing code length.

The `unconditional/` and `conditional/` directories contain standard Trainer
artifacts. `v_information.json` stores normalization, configurations, splits,
and selected checkpoints. `test_unconditional.f32` and `test_conditional.f32`
hold NLL in nats in test-seed order, and `pvi_bits.f64` holds the paired
differences in bits per window. Resuming checks initialization and settings
before continuing incomplete branches. A completed estimate can be recomputed
from the saved files after checking that they are complete and consistent.

```python
from episynth.estimators.v_information import VInformationEstimator
result = VInformationEstimator().from_run_dir("runs/v_information_cpu")
print(result["v_information_bits_per_window"])
```

To evaluate a checkpoint on the short-horizon GIFT-Eval tasks, download a dataset
snapshot with `huggingface_hub.snapshot_download("Salesforce/GiftEval", repo_type="dataset")`.

```bash
python -m episynth.evaluation.run --data <snapshot> --out eval/kernelsynth --model moirai \
    --checkpoint runs/kernelsynth/ckpt_1.pt --patch-size 32 --device cuda
```

The native `.pt` checkpoint supplies the model configuration. A supplied
`--config-json` must match it. `evaluation.json` records the model, checkpoint,
and evaluation settings, and existing `summary.csv` rows are reused only when
these settings match. Use a new output directory for a different experiment
or if the existing summary predates `evaluation.json`.

The summary always has the same columns. A metric with a zero denominator or no
eligible observations is recorded as `NaN`. Examples include MASE with a constant
history and metrics whose targets are all missing. Window and point counts make
these cases distinguishable, and evaluation can continue.

For a training check, run the `ou` generator (an AR(1) model with a coefficient
prior) and compare the learned loss with the exact Bayesian loss on the same
windows, obtained by quadrature over the prior
(`python -m episynth.checks.ou_bayes --run runs/ou`). The trained model is expected
to approach the reference from above.

Pipeline checks include comparing the port with uni2ts on identical inputs
(`python -m episynth.checks.moirai --checkpoint <directory> --uni2ts-python <interpreter-with-uni2ts-2.0.0>`),
checking data, windows, and metric formulas against the published seasonal-naive
baseline (`python -m episynth.checks.gift_eval_naive --data <snapshot> --published <results/Seasonal_Naive/all_results.csv>`),
and comparing a run with the model's published table
(`python -m episynth.checks.gift_eval_published`).

## Generators

`episynth.generators` includes eleven univariate time-series generators from
Mix, Don't Pick (Nagpal et al., 2026, arXiv:2606.09912). They are KernelSynth,
ETS, Waveform, SDE, StepFunction, TSI, ARIMA, fBm, GARCH, Chaotic, and TimeSynth.
The package follows the original generative processes with batch implementations
that work across series without the original `fbm` and `timesynth` dependencies.

ARIMA defaults to `coefficient_mode="legacy"`, which reproduces the published
series and the inherited polynomial-coefficient ordering bug. All 100 candidates
are rejected and a fallback vector is drawn from `U(-0.3, 0.3)`. To select
coefficients using the roots of the intended lag polynomial, explicitly set
`ARIMA(length=1024, coefficient_mode="corrected")` or pass
`--set data.generator_args.coefficient_mode='"corrected"'` to the training CLI.
This changes the data distribution. The mode is written to metadata, and
corpora with different modes cannot be combined. Default and `exact` generation
both use the selected coefficient rule. Reproducing the original generator
requires the `legacy` rule.

```python
import numpy as np
from episynth.generators import GENERATORS

gen = GENERATORS["sde"](length=1024)
x = gen.generate(np.arange(1000))          # (1000, 1024) float32, one series per seed
y = gen.generate([5, 6], exact=True)       # Original series for these seeds
corpus = gen.reproduce(random_seed=42, n=100)  # Original corpus with seed 42
```

A series depends only on its seed and length, regardless of seed order or
batch membership. The `exact` mode follows the original down to the order of
random number calls and serves as a validation reference. For most generators,
the default mode matches it bit for bit. See each class docstring for its
generative process.

## Checks

To compare against the original code, use a separate interpreter with the original
dependency versions (NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.7.1, fbm, and
timesynth). With NumPy 2, the original TimeSynth fails in its per-step loops and
silently substitutes fallback series.

```bash
uv venv .venv-ref && uv pip install --python .venv-ref/bin/python \
    numpy==1.26.4 scipy==1.11.4 scikit-learn==1.7.1 fbm==0.3.0 tqdm git+https://github.com/TimeSynth/TimeSynth.git
python -m episynth.checks.generators --original <mix-dont-pick-checkout> \
    --original-python .venv-ref/bin/python --out checks_out
```

Each generator produces a JSON report with the fraction of bit-for-bit matches,
Kolmogorov–Smirnov p-values comparing per-series statistics between the original
and default-mode corpora, and the speed of both implementations. KernelSynth
matches bit for bit only with the original NumPy build, because reproduction
mode constructs samples using an eigendecomposition. Run `pytest` to check
invariants without the original code.
