# episynth: architecture and development stack

The practical goal is to train a time-series foundation model on high-quality
synthetic data and select productive generators for that data. Productivity means
better forecasting on real datasets after pretraining under a fixed training-data
budget. `episynth` supports this decision by generating data, training forecasting
models, measuring learned structure, and evaluating transfer on GIFT-Eval. We test
epiplexity-based measurements as possible generator-selection signals. The Python
distribution and import name are `episynth`; the repository is `intsystems/Epiplexity`.

## Architecture and interaction

```text
Generator / Corpus / Mixture
           ↓ seeded series
WindowSpec → WindowBatch / WindowStream
           ↓ context and forecast windows
MoiraiConfig → MoiraiModule
           ↓
EpiplexityEstimator.run(model, generator, spec, config, out_dir)
           ↓
Trainer → per-window losses, held-out raw/EMA losses, checkpoints
           ↓
PrequentialEstimator.from_run_dir → model/data/total bits
           ↓
analysis + GIFT-Eval evaluation → tables, plots, generator comparisons
```

- `episynth.generators.base.Generator`: common contract (`generate`, `reproduce`)
  for seeded series. `GENERATORS` maps names to eleven Mix, Don't Pick processes
  plus the OU reference. `Corpus` reads generated series; `Mixture` combines sources.
- `episynth.windows`: `WindowSpec` validates the task, `WindowBatch` carries the
  tensors, `WindowStream` supplies deterministic batches.
- `episynth.model.moirai`: `MoiraiConfig`, `MoiraiModule`, distribution `Mixture`
  and published-checkpoint loading.
- `episynth.training`: `TrainConfig`, `Trainer.run`, `compute_loss`, `evaluate`,
  checkpoint save/resume and `build_model`.
- `episynth.estimators`: `EpiplexityEstimator` is a typed protocol; `register`,
  `build`, `available` provide the extension registry. `PrequentialEstimator`
  delegates training to `Trainer` and computes bits using `analysis.run_area`.
- `episynth.train`: validated JSON configuration, `--set` overrides, seed-range
  checks and execution of the selected estimator. Saves `estimator_result.json`.
- `episynth.analysis` and `episynth.evaluation`: area uncertainty, prefix curves,
  quality comparisons, forecast distributions and GIFT-Eval metrics.

## Measurement-method scope

The [measurement contracts](../doc/source/measurements.rst) propose four methods:
prequential coding (implemented), requential coding (registered stub), predictive
V-information (requires implementation), and Bayesian predictive log-loss using
the implemented OU reference (public wrapper proposed). They specify the quantity,
inputs, outputs, units and readiness of each method. Generation, training and
forecast scoring are shared operations; the saved experiments use prequential
coding. The common result contract must identify the measured quantity:
V-information is a predictive-loss reduction, while the current estimator
protocol returns a model/data code decomposition.

## Public API and annotations

The complete reference is generated from the actual annotated Python definitions,
including constructor and public method docstrings:

- [Architecture](../doc/source/architecture.rst)
- [API reference](../doc/source/api.rst)
- [Adding an estimator](../doc/source/extending.rst)

The shared contract is:

```python
class EpiplexityEstimator(Protocol):
    name: str

    def run(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        out_dir: str | Path,
    ) -> dict[str, float]: ...
```

Results contain `model_part_bits` and `data_part_bits`; prequential also returns
`total_bits` and `n_windows`. Per-window training losses are saved in nats and
converted to bits by dividing by `ln(2)`; the model part is the area above the
held-out final loss. See the [experiment protocol](../doc/source/experiments.rst).

Prequential is implemented. Requential is a registered skeleton whose `run`
raises `NotImplementedError`. Bayesian codes and proxies in the older research
plans are future directions; no old model/sampler registries are part of this API.

## Stack

- Python 3.10+, PyTorch, NumPy, SciPy, scikit-learn: models, generators and statistics.
- Pydantic: validated configurations. safetensors: checkpoint loading. PyArrow: GIFT-Eval data.
- Ruff: style and imports. pytest/pytest-cov: tests and coverage.
- Sphinx/autodoc/Napoleon, Read-the-Docs theme: API and narrative documentation.
- Jupyter/nbconvert/Matplotlib: the small CPU demo and plots.
- uv/setuptools/GitHub Actions: locked dependencies, packaging and CI.

External comparisons use separate original-generator or uni2ts environments;
large datasets and pretrained weights are not required for base tests or docs.

## Build, demo and deployment

From the repository root:

```bash
uv sync --locked --extra dev
uv run --no-sync ruff check src/ tests/ experiments/demos/ doc/source/conf.py
uv run --no-sync pytest tests/
uv run --no-sync sphinx-build -b html -W --keep-going doc/source doc/_build
uv run --no-sync jupyter lab experiments/demos/demo_estimators.ipynb
```

Open `doc/_build/index.html` locally and `doc/_build/api/estimators.html` for an
automatically generated API page. The docs workflow builds pushes and pull requests
to main. Automatic publication is disabled as required by the course; to publish,
use **Actions → docs → Run workflow** with branch `main`. A successful manual run
publishes through GitHub Pages artifacts. Select
**Settings → Pages → Source: GitHub Actions** once. Full local/online deployment
instructions are in [Installation and deployment](../doc/source/installation.rst).

Recorded large series and their artifact locations are described in
[experiments/episynth/README.md](../experiments/episynth/README.md).
