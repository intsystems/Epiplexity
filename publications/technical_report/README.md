# Technical report

[Read the report](main.pdf): *episynth: A Library for Studying Information in
Synthetic Time Series* (5 pages including references). Edit [main.tex](main.tex). References are in
[references.bib](references.bib).

The report introduces our library, explains its components and execution, and
walks through small CPU examples. The [research article](../article/README.md)
contains the coding analysis and the completed large-scale transfer experiments.

## Current methods

| Method | Output | Implementation |
| --- | --- | --- |
| Prequential | Total predictive cost and signed excess area, bits | Implemented |
| Predictive V-information | Paired predictive gain and standard error, bits/window | Implemented |
| OU Bayesian reference | Joint and pointwise predictive NLL, nats/window | Reference functions in `checks.ou_bayes` |
| Requential | Planned cumulative KL and coding-bound measurements | Registered stub |

See the [measurement guide](../../doc/source/measurements.rst) for input files,
units, entry points, and result fields. Method descriptions in this report
reflect the current code. The registry lists both runnable estimators and the
requential stub.

## Build

With pdfLaTeX and BibTeX installed, run from this directory:

```console
python build_report.py
```

The included figures and check counts suffice for compilation. The build checks
the five-page budget, unresolved references, and overflowing boxes. Evidence is
saved in `data/build_verification.json`.

## Run the examples and checks

Use Python 3.10 or newer. From the repository root, install the package and
publication dependencies in your environment:

```console
python -m pip install -e ".[dev]"
python -m pip install -r publications/technical_report/requirements.txt
python publications/technical_report/reproduce.py
```

`run_examples.py` executes the prequential configuration in
`examples/prequential.json` and the V-information configuration in
`experiments/demos/v_information.json`. It verifies the outputs against the
saved losses and checks the OU reference at two quadrature resolutions.
Training artifacts stay in `build/report_demo/`. Subsequent runs use the
library's resume behavior. The existing library test suite is then run and its
counts refreshed. `make_plots.py` produces the component diagram, execution
diagram, and measured example learning curve.

`python publications/technical_report/reproduce.py --skip-checks` reruns the
examples and figures while retaining the saved test-suite results.

| Evidence | Contents |
| --- | --- |
| `data/example_results.json` | Example measurements, verification outcomes, configuration hashes, and environment |
| `data/test_results.json`, `data/pytest_output.txt` | Existing test-suite outcomes and skip reasons |
| `data/check_summary.tex` | Generated test counts included in the report |
| `figures/architecture.pdf` | Component diagram |
| `figures/execution.pdf` | Prequential execution sequence |
| `figures/demo_loss.pdf` | Measured CPU waveform learning curve |

Experiment-analysis scripts and their outputs now live in
[`publications/article/`](../article/README.md). The public
`PrequentialEstimator.from_run_dir` reads per-window binary losses. The article
analysis can additionally reconstruct linear areas from archived batch-mean CSVs.
