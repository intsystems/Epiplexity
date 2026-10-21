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

The article moves from the research question and contributions through problem
formulation, methods, implementation, experimental evaluation, and discussion.
The supplied [Bensemble report](https://github.com/intsystems/bensemble/blob/master/paper/bensemble_tech_report.pdf)
was read as a structural example. Training-run attribution and reproduction
instructions are retained in the appendix and this README.

KernelSynth transfers best at both budgets. Across all eleven priors, excess area
has weak association with aggregate CRPS; stronger subset associations are
exploratory. The report explains sensitivity to the EMA endpoint and Waveform's
forecast tails, and distinguishes completed experiments from proposed studies.

## Four required algorithms

The report's implementation section states the four algorithms used by the current
`episynth` experiment. Appendix B gives their steps, outputs, code locations,
and existing checks.

| Algorithm | Implementation | Output |
| --- | --- | --- |
| Seeded synthetic task generation | [`make_windows`, `WindowStream`](../../src/episynth/windows.py), [`generators/`](../../src/episynth/generators/) | Reproducible, patched and masked context/target windows |
| Probabilistic training with AdamW and EMA | [`Trainer`](../../src/episynth/training.py) | Raw/EMA checkpoints, pre-update NLL logs, held-out losses |
| Excess prequential-loss estimation | [`run_area`](../../src/episynth/analysis.py), [`analyze_runs.py`](analyze_runs.py) | Signed excess area, bits, endpoint sensitivity; floor error when per-window losses are available |
| Transfer evaluation and ranking uncertainty | [`evaluation/`](../../src/episynth/evaluation/), [`quality_table`, `spearman_ci`](../../src/episynth/analysis.py) | GIFT-Eval metrics, baseline ratios, ranks and paired bootstrap intervals |

The earlier estimator roadmap in [`plan/`](../plan/README.md) is outdated.
The current package includes an implemented prequential estimator and a
requential stub; the report describes the completed `episynth` experiments.

## Rebuild the document

From this folder, with pdfLaTeX and BibTeX installed:

```console
python build_report.py
```

The included figures, generated results table, and test-count file are sufficient
to compile the report.
The build runs LaTeX, BibTeX, and two further LaTeX passes, checks the seven-page
limit and unresolved references/overfull boxes, and writes
`data/build_verification.json`. The document has twelve bibliography entries.

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
It also generates `data/results_table.tex` directly from the recomputed run records
so that the report's results table uses the same values and rounding as the CSVs.
It checks contiguous step/window counts and matching evaluation configurations,
compares all 22 primary areas with the author's archived per-window analysis,
and verifies the seasonal-naive point metrics against the committed leaderboard.
Large per-window vectors, model checkpoints, and the original benchmark data are
not required for this analysis.

The three experiment figures are:

- `figures/training_results.pdf`: training curves for both budgets.
- `figures/transfer_results.pdf`: excess area versus aggregate CRPS.
- `figures/prefix_results.pdf`: checkpoint statistics versus final transfer.

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
analysis, and redraws the figures. The current report rerun has **66 passed,
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
| `task_ratios.csv` | CRPS, MASE, and quantile-loss ratios for every run/configuration |
| `correlations.csv` | Full-set and exploratory subset associations with bootstrap intervals |
| `checkpoint_statistics.csv`, `prefix_correlations.csv` | Prefix areas/losses and final-transfer associations |
| `analysis_verification.json` | Input hashes, rounding bounds, archived-area and point-baseline comparisons |
| `provenance.json` | Repository commit, supplied-document hashes, and environment versions |
| `test_results.json`, `pytest_output.txt` | Fresh outcomes of the existing tests |
| `check_summary.tex` | Generated test counts included in the report |
| `build_verification.json` | Compiled page count, bibliography count, and document hashes |
| `bibliography_metadata.json` | Primary-source metadata for the eight research papers |
| `figure_protocol.json`, `generator_examples.npz`, `window_examples.npz` | Supplementary descriptive-example provenance and arrays |

`collect_sources.py` optionally retrieves the papers' arXiv metadata into
`../../build/technical_report_sources/`; it is not required to compile or analyze.
No experiment training is rerun by the report analysis. Existing project source
and experiment files are preserved.
