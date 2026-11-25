# episynth technical report

[Read the report](main.pdf) · [LaTeX source](main.tex) · [Research article](../article/README.md)

Can a learning curve help choose a synthetic generator for forecast pretraining?
Sections 1–2 introduce epiplexity and work through the prequential estimate
on 256 Waveform windows. Section 3 shows how predictive V-information
measures the gain from seeing the past and how requential coding measures
the cost of teaching a student. Section 4 compares all three measurements
with real forecasting performance across eleven generators.

Section 5 introduces the implemented generators and shows how to run
the example on Chaotic (Lorenz and Mackey–Glass trajectories).
Sections 6–8 cover matched training runs, the Ornstein–Uhlenbeck likelihood
reference, validation, and how to evaluate a generator of your own.
The [meeting notebook] runs all three estimators on CPU and includes the archived CRPS comparisons.

[meeting notebook]: ../../experiments/demos/demo_estimators.ipynb

## Build the PDF

With pdfLaTeX and BibTeX installed, run from this directory:

```console
python build_report.py
```

The build reads the committed figures and test counts. It checks the
eight-page budget, cross-references, and text overflow, and records source
and PDF hashes in [data/build_verification.json](data/build_verification.json).
Source hashes use LF line endings so that LF and CRLF checkouts agree.
The report shares the prequential transfer figure and the cross-method
correlation table with the research article.

## Reproduce the measurements and validation

Use Python 3.10 or newer. From the repository root:

```console
python -m pip install -e ".[dev]"
python -m pip install -r publications/technical_report/requirements.txt
python publications/technical_report/reproduce.py
```

The [example runner](run_examples.py) uses the [prequential](examples/prequential.json),
[requential](../../experiments/demos/requential.json), and
[V-information](../../experiments/demos/v_information.json) configurations.
It checks the prequential decomposition and paired test loss difference from
saved vectors, requential KL and overhead totals from per-step records, and
the OU reference at two quadrature resolutions. The CPU run directories are
under `build/report_demo`.

The reproduction script runs the [library tests](../../tests/episynth/) and
regenerates the three report figures. Add `--skip-checks` to retain the
recorded test outcomes while refreshing the examples and figures. Tests for
POSIX file locks and staged fits run on Linux and are skipped on Windows.
To run the two external integration checks, supply `GIFT_EVAL_DIR` or
`MOIRAI_SMALL_DIR` as appropriate and invoke `pytest tests/episynth` directly.

| Artifact | Evidence |
| --- | --- |
| [data/example_results.json](data/example_results.json) | Numerical example results, configuration hashes, and environment |
| [data/test_results.json](data/test_results.json), [data/pytest_output.txt](data/pytest_output.txt) | Per-test outcomes and skip reasons |
| [data/check_summary.tex](data/check_summary.tex) | Component counts used in the validation table |
| [figures/architecture.pdf](figures/architecture.pdf) | CPU and archived Waveform series with their selected forecast windows |
| [figures/execution.pdf](figures/execution.pdf) | Pre-update training losses and final checkpoint evaluation |
| [figures/demo_loss.pdf](figures/demo_loss.pdf) | Learning curves and cumulative excess loss for the CPU and archived runs |
| [Shared transfer figure](../article/figures/transfer_results.pdf) | Geometric-mean CRPS ratios for eleven generators at both budgets |
| [Shared coding table](../article/data/coding_correlation_table.tex) | Full-set rank associations with archived CRPS for all three measurements |

The archived Waveform panels use the committed configuration, batch means,
and held-out EMA score in the [run directory](../../experiments/episynth/series2_2026-10-19/runs/s2_waveform/).
The [run archive](../../experiments/episynth/README.md) describes the
separately distributed loss vectors and checkpoints. The
[article analysis](../article/analyze_runs.py) reconstructs linear areas
from the committed batch means.

The [measurement guide](../../doc/source/measurements.rst) documents the
entry points, result fields, and units. The
[extension guide](../../doc/source/extending.rst) describes how to load an
external estimator in the same process as the training CLI.
