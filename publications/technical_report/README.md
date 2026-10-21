# Technical report

Open **[main.pdf](main.pdf)**: **7 pages total, including references and appendix**.
Edit [main.tex](main.tex); BibTeX references are in [references.bib](references.bib).

The report is written as a research article by **Danila Chernousov, Mark Ikonnikov,
Daniil Kazachkov, and Alexey Kravatskiy**, in English alphabetical order by surname.
It studies completed synthetic time-series experiments: eleven priors at approximately 2 million and 40 million
training windows, plus a Waveform initialization repeat, evaluated on 55 short-term
GIFT-Eval configurations. It uses the actual outputs uploaded in repository commit
`7d2b001d972169efdb19e3c8e256a182ba4c09f8`, informed by the supplied
`lit_talk_v2.pdf` and `proposal_3.4.pdf`.

The practical problem is training a time-series foundation model on high-quality
synthetic data and selecting productive generators for that data. Productivity
means improved forecasting on real datasets under a fixed training-data budget.
The report tests excess prequential loss as a possible signal for generator
selection, using the saved small-Moirai experiments.

The article moves from the research question and contributions through problem
formulation, methods, implementation, experimental evaluation, and discussion.
The supplied [Bensemble report](https://github.com/intsystems/bensemble/blob/master/paper/bensemble_tech_report.pdf)
was read as a structural example. Training-run attribution and reproduction
instructions are retained in the appendix and this README.

KernelSynth transfers best at both budgets. Across all eleven priors, excess area
has weak association with aggregate CRPS; stronger subset associations are
exploratory. The report explains sensitivity to the EMA endpoint and Waveform's
forecast tails, and distinguishes completed experiments from proposed studies.

## Four measurement methods and library contracts

Appendix B now proposes four measurement methods, with scientific purpose,
inputs, formulas, outputs and implementation status. The proposed library scope
is separate from the generation/training/evaluation infrastructure:

| Method | What it computes | Current status |
| --- | --- | --- |
| Prequential coding with held-out-floor decomposition | Total pre-update log-loss and its residual-data / signed model-proxy decomposition, in bits | Implemented: `PrequentialEstimator.run` and `from_run_dir` |
| Requential coding | Cost of teacher-selected synthetic training samples used to reconstruct a student's predictions | Registered stub: `run` raises `NotImplementedError` |
| Predictive V-information | Improvement in held-out prediction when context is available, for a specified predictive family | Proposed; requires two fitted predictors and a new result contract |
| Bayesian predictive log-loss by quadrature | Joint and pointwise predictive-loss references under the known OU/AR(1) prior | Implemented in `checks.ou_bayes`; a common public wrapper is proposed |

Prequential coding is the implemented information-estimation algorithm in the
saved transfer experiments. The OU reference is implemented and tested, but
has no archived training/transfer runs in that comparison. The other two
methods are described as future library implementations, with no claimed
experimental results. The OU method is proposed as the fourth method because
its computation already exists in the repository.

The [library measurement guide](../../doc/source/measurements.rst) gives current
entry points, file requirements, result fields, units, side effects, and proposed
contracts for the missing implementations. Generation supplies reproducible
context/target windows; training fits density models and logs their trajectory;
GIFT-Eval scoring and ranking analysis assess downstream utility.

The earlier research roadmap in [`plan/`](../plan/README.md) remains outdated.

## Rebuild the document

From this folder, with pdfLaTeX and BibTeX installed:

```console
python build_report.py
```

The included figures, generated results table, and test-count file are sufficient
to compile the report.
The build runs LaTeX, BibTeX, and two further LaTeX passes, checks the seven-page
limit and unresolved references/overfull boxes, and writes
`data/build_verification.json`. The document has fourteen bibliography entries.

## Recompute results and plots

With Python 3.10 or newer and NumPy, SciPy, and Matplotlib installed:

```console
python analyze_runs.py
python make_plots.py
```

The analysis reads committed files under
`../../experiments/episynth/series1_2026-10-19/` and
`../../experiments/episynth/series2_2026-10-19/`.
It recomputes signed excess areas from every exported training-step mean and the
final held-out EMA floor, geometric mean score ratios to seasonal naive, and
Spearman correlations with 10,000 paired bootstrap resamples of generator families.
It also extracts downstream `nll_per_point` values and averages them arithmetically
across the same 55 configurations, retaining nats per target point. NLL is neither
divided by the seasonal-naive baseline nor averaged geometrically. The saved
evaluator excludes windows with zero seasonal error for this field.
It also generates `data/results_table.tex` directly from the recomputed run records
so that the report's results table uses the same values and rounding as the CSVs.
It checks contiguous step/window counts and matching evaluation configurations,
compares all 22 primary areas with the author's archived per-window analysis,
and verifies the seasonal-naive point metrics against the committed leaderboard.
Large per-window vectors, model checkpoints, and the original benchmark data are
not required for this analysis.

The three experiment figures are listed below. Training and downstream-loss
figures appear in the article; the prefix figure is supplementary so the main
comparison can show both losses at both budgets within the seven-page limit.

- `figures/training_results.pdf`: training curves for both budgets.
- `figures/transfer_results.pdf`: epiplexity proxy versus downstream CRPS and NLL
  at both budgets. All eleven generators remain visible; filled circles and
  descriptive Theil–Sen trends show the exploratory subset excluding Chaotic and
  Waveform. The 40M subset CRPS correlation is −0.82, matching the anticorrelation
  in Chernousov's original regression figure. The corresponding 2M correlation
  is −0.47; mean-NLL correlations are −0.55 and −0.45. Full-set and subset
  correlations are labeled separately; the Waveform seed-1 repeat enters neither.
- `figures/prefix_results.pdf`: supplementary checkpoint statistics versus final transfer.

Each also has a PNG version. The optional `generator_gallery` and
`forecast_windows` figures contain newly generated descriptive examples and
are not historical experiment results or figures in the report.

## Reproduce the existing checks

Install [requirements.txt](requirements.txt) in an isolated Python environment.
A CPU build of PyTorch is sufficient. Then run:

```console
python reproduce.py
```

This records source/environment provenance, regenerates supplementary examples,
runs the author's existing `tests/episynth` suite, recomputes the saved-run
analysis, and redraws the figures. The current report rerun has **72 passed,
2 skipped, 0 failed**. The two skipped comparisons require external GIFT-Eval data
and the original Moirai checkpoint. Temporary training/checkpoint files stay in
the repository's ignored `build/` directory.

`python reproduce.py --skip-checks` runs the same reproduction pipeline without
repeating the tests and retains the existing test-result files.

## Evidence files

| File in `data/` | Contents |
| --- | --- |
| `run_statistics.csv` | Areas, endpoint losses, metric aggregates, and sensitivity statistics for 23 runs |
| `results_table.tex` | Generated main results table |
| `task_ratios.csv` | CRPS, MASE, and quantile-loss ratios, plus raw-unit NLL for every run/configuration |
| `correlations.csv` | Full-set and exploratory subset associations with bootstrap intervals |
| `checkpoint_statistics.csv`, `prefix_correlations.csv` | Prefix areas/losses and final-transfer associations |
| `analysis_verification.json` | Input hashes, rounding bounds, archived-area and point-baseline comparisons |
| `provenance.json` | Repository commit, supplied-document hashes, and environment versions |
| `test_results.json`, `pytest_output.txt` | Fresh outcomes of the existing tests |
| `check_summary.tex` | Generated test counts included in the report |
| `build_verification.json` | Compiled page count, bibliography count, and document hashes |
| `bibliography_metadata.json` | Primary-source metadata for the original research references |
| `figure_protocol.json`, `generator_examples.npz`, `window_examples.npz` | Supplementary descriptive-example provenance and arrays |

`collect_sources.py` optionally retrieves the papers' arXiv metadata into
`../../build/technical_report_sources/`; it is not required to compile or analyze.
No experiment training is rerun by the report analysis. Existing project source
and experiment files are preserved.
