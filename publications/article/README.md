# Research article

[Read the article](main.pdf): *Epiplexity and Transfer in Synthetic Time-Series Pretraining*
(8 pages including the protocol appendix and references).
The manuscript is [main.tex](main.tex), with references in [references.bib](references.bib).

The article explains prequential and requential measurements, their use with
non-NLL training, and the interpretation of learning-curve areas under proper
scores. Section 3 gives practical recipes for each coding measurement
under non-NLL training and guidelines for generator, student, and evaluation
sample sizes. The precision rules explain how to choose held-out and teacher-audit
sample counts from pilot variances. It analyzes eleven synthetic generators at two training budgets using
the archived Moirai and GIFT-Eval experiments. The main empirical finding is a
weak association between excess area and transfer across the full comparison.

The [technical report](../technical_report/README.md) describes our `episynth`
library, architecture, interfaces, and executable examples. The
[journal](../journal/journal.tex) retains the broader research notes. This
directory is the current research manuscript and supersedes the earlier
`publications/paper/` outline.

## Build and reproduce

Use Python 3.10 or newer. From this directory:

```console
python -m pip install -r requirements.txt
python reproduce.py
python build_report.py
```

Compilation requires pdfLaTeX and BibTeX. Committed tables and figures also
allow `build_report.py` to run on its own. It checks the eight-page budget,
unresolved references, and overflowing boxes, and records PDF/source hashes.

`reproduce.py` runs the following scripts in order:

- `analyze_runs.py` reads committed configurations, batch-mean training curves,
  held-out scores, and GIFT-Eval summaries under `experiments/episynth/`.
- `prepare_tables.py` generates the correlation table and verifies the Bernoulli
  CRPS/KL ranking reversal from expected scores.
- `make_plots.py` generates the full-set comparison and supplementary figures.

The original training was conducted by Danila Chernousov and archived at commit
`7d2b001d972169efdb19e3c8e256a182ba4c09f8`. Reproduction here recomputes its analysis.
Large checkpoints, per-window vectors, and benchmark datasets have separate
artifact locations documented in [the experiment README](../../experiments/episynth/README.md).

## Evidence and figures

| Artifact | Contents |
| --- | --- |
| `data/run_statistics.csv` | Areas, endpoints, transfer scores, and sensitivities for 23 runs |
| `data/correlations.csv` | Full-set and exploratory correlations with bootstrap intervals |
| `data/task_ratios.csv` | Per-configuration forecast scores |
| `data/checkpoint_statistics.csv`, `data/prefix_correlations.csv` | Prefix measurements and final-transfer associations |
| `data/analysis_verification.json` | Input hashes, rounding bounds, baseline checks, and compression comparisons |
| `data/migration_verification.json` | Equality of the five migrated statistical tables with the previous analysis |
| `data/results_table.tex`, `data/correlation_table.tex` | Generated manuscript tables |
| `data/counterexample.json`, `data/counterexample.tex` | Direct calculation of the score-ranking reversal |
| `figures/transfer_results.pdf` | Main comparison: all eleven sources and the Waveform repeat |
| `figures/transfer_nll.pdf` | Same comparison using downstream NLL |
| `figures/training_results.pdf` | Training curves by generator and budget |
| `figures/prefix_results.pdf` | Point correlations across training prefixes |

CRPS and MASE are geometric mean ratios to the seasonal-naive baseline.
Downstream NLL is the arithmetic mean of original-unit nats per target point.
Area uses every saved batch mean and an independent held-out endpoint.
Bootstrap intervals resample generator families. The additional Waveform seed
is shown separately from primary correlations. Exploratory subsets are
identified in the table and text.

The coding discussion follows the cited constructions. The article reports
completed prequential experiments. A requential comparison and measurements
for quantile or diffusion models are proposed follow-up experiments.
