# Research article

[Read the article](main.pdf): *Epiplexity and Transfer in Synthetic Time-Series Pretraining*
(11 pages including references and appendices).
The manuscript is [main.tex](main.tex), with references in [references.bib](references.bib).

Section 2 connects synthetic forecasting, description length, and prior
evidence on transfer. Section 3 defines the zero-shot generator-selection
objective and its ranking hypothesis. Section 4 develops the prequential
estimate and describes requential coding, predictive V-information, and the
OU Bayes reference. Sections 5 and 6 define and report CRPS and MASE ratios
for eleven generators at both training budgets. The full comparison shows a
weak association between excess loss and transfer. Appendix B derives the
dependence of learning-curve areas on the scoring rule.

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
allow `build_report.py` to run on its own. It checks the eleven-page budget,
unresolved references, and overflowing boxes, and records PDF and source hashes.

`reproduce.py` runs the following scripts in order:

- `analyze_runs.py` reads committed configurations, batch-mean training curves,
  held-out scores, and GIFT-Eval summaries under `experiments/episynth/` and
  writes the pretraining and transfer table.
- `prepare_tables.py` generates the correlation table and verifies the Bernoulli
  CRPS/KL ranking reversal from expected scores.
- `make_plots.py` generates the comparison figures, seeded generator samples,
  and GIFT-Eval gallery. The GIFT-Eval figure uses the stored histories below.

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
| `figures/generator_examples.pdf` | Eleven seeded 192-point series in generator-table order, one vector trace per page |
| `figures/generator_gallery.pdf` | Five additional 192-point draws from each of the eleven generators |
| `data/gift_eval_examples.json` | Observations, task IDs, history offsets, and provenance for the 15 GIFT-Eval samples |
| `figures/gift_eval_gallery.pdf` | 192-point history excerpts from 15 evaluated GIFT-Eval configurations |
| `figures/transfer_results.pdf` | Labeled CRPS comparison for all eleven sources and the Waveform repeat |
| `figures/transfer_nll.pdf` | Labeled downstream NLL comparison, with a shared label for the near-overlapping Waveform runs at 40M |
| `figures/training_results.pdf` | Training curves by generator and budget |
| `figures/prefix_results.pdf` | Point correlations across training prefixes |

CRPS and MASE are geometric mean ratios to the seasonal-naive baseline.
Downstream NLL is the arithmetic mean of original-unit nats per target point.
Area uses every saved batch mean and an independent held-out endpoint.
Bootstrap intervals resample generator families. The additional Waveform seed
is shown separately from primary correlations. Exploratory subsets are
identified in the table and text.

The benchmark gallery takes the final 192 positions of the first test-window
history in each selected configuration of the [Salesforce GIFT-Eval Parquet release](https://huggingface.co/datasets/Salesforce/GiftEvalParquet/tree/faad405a631f3082bd3d4e4d47fa8ff976357205).
The stored excerpts preserve missing observations and enable offline plotting.

The coding discussion follows the cited constructions. The article reports
completed prequential experiments. A requential comparison and measurements
for quantile or diffusion models are proposed follow-up experiments.
