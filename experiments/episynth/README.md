# episynth run results (October 19–20, 2026)

This archive contains the two training series for the time-series track. They
compare the prequential epiplexity of 11 synthetic generators with Moirai-Small
performance on short-horizon GIFT-Eval tasks. The archive includes run configs,
per-step training curves, held-out losses by checkpoint, GIFT-Eval task summaries,
analysis tables, and figures. Per-window loss vectors (up to 160 MB per run) are
attached to the GitHub release `results-2026-10-19`. Checkpoints and per-window
GIFT-Eval metrics are available on request from object storage.

Both series follow the same protocol. Moirai-Small trains from scratch on each
of the eleven Mix, Don't Pick generators. It uses three mixture components,
patch size 32, a context of up to 512 points from the GIFT-Eval length table,
horizon 64, and dequantization of 10⁻³. The trainer logs each window's loss before
updating on its batch. We compare excess loss above the final loss (prequential
epiplexity) with performance on 55 short-horizon GIFT-Eval tasks. Training uses
AdamW with uni2ts parameter groups, learning rate 10⁻³ after 200 warmup steps,
an EMA time constant of 2% of the steps, bf16 with `torch.compile`, 100,000
held-out windows starting at seed 10⁹, and a batch size of 1,024.

| Series | Directory | Windows per generator | Steps | Series source | GPUs |
|---|---|---|---|---|---|
| 1 | `series1_2026-10-19/` | 1,999,872 | 1,953 | Precomputed corpora of 2 million series | RTX 4090 |
| 2 | `series2_2026-10-19/` | 39,999,488 | 39,062 | Process pool on demand, except a precomputed KernelSynth corpus on A100 (float64) | RTX 4090, A100 |

Both series use the same window seeds. Series 1 matches the first 2 million
windows of series 2.

## Archive contents

```text
series<k>_<date>/
  runs/<prefix><generator>/
    run.json          Full run config (data, windows, model, training, context table)
    config.json       Configuration received by Trainer
    steps.csv         Every 50th step: step, windows_seen, nats_per_point, lr, grad_norm, seconds
    train_curve.csv   Each step: step, windows_seen, nats_per_point. The loss is the
                      pre-update mean over 1,024 windows, in nats per point (window loss / 64).
    heldout.jsonl     Checkpoint fractions 0.01…1: held-out loss for raw and EMA weights
                      at training precision and fp32, in nats_per_window and nats_per_point.
  eval/<prefix><generator>/summary.csv   GIFT-Eval short, 55 tasks, final checkpoint's EMA weights
  eval/seasonal_naive/summary.csv        Seasonal-naive forecast on the same windows
  analysis/
    epiplexity_variants.md          Spearman correlations for 15 estimators (all 11 generators
                                   and the subset without Chaotic and StepFunction)
    epiplexity_variants_subsets.md  Same analysis without Chaotic and Waveform
    figures/                      Figures for the series
    timesfm3_reference.json       Series 2: leaderboard TimesFM-3 relative to our seasonal-naive
                                 baseline, with task-by-task alignment checks
  gzip_baseline.json  Series 1: gzip compressibility per generator, in bytes per point
scripts/              Analysis scripts documented below
reference/gift_eval_leaderboard/   Published all_results.csv files for TimesFM-3 and Seasonal_Naive
                                   from SalesforceAIResearch/gift-eval (commit 4d5ab3f)
```

Run prefixes are `series_` in series 1 and `s2_` in series 2.
`s2_waveform_seed1` uses a second initialization seed to measure seed
sensitivity. `episynth.evaluation.metrics.aggregate` documents the columns
of `summary.csv`, including `scaled_crps` (sample-path CRPS scaled by the
historical seasonal-naive error), `MASE[0.5]`,
`mean_weighted_sum_quantile_loss` (leaderboard CRPS), interval coverage and
width, and `nll_per_point`.

Loss is in nats in the original series units with dequantization of 10⁻³.
Dequantization explains negative values for nearly deterministic generators
(Chaotic, TSI, and fBm). `episynth.analysis.run_area` computes
$A = \sum_i \ell_i - N L_{\mathrm{fin}}$ from `window_loss.f32` and
`heldout_ema_1.f32`.

## Large files

- GitHub release `results-2026-10-19` contains `series1_2026-10-19_window_losses.tar`
  (all series 1 runs) and `series2_2026-10-19_<generator>_window_losses.tar`
  (one run per archive). Each contains `window_loss.f32` (headerless float32
  losses in window order) and `heldout_<variant>_<fraction>.f32` (held-out
  window losses at checkpoints, final checkpoint only in series 1). Extract
  the files over the series directory, for example
  `tar -xf series2_2026-10-19_ets_window_losses.tar -C series2_2026-10-19/`.
- Checkpoints (`ckpt_<fraction>.pt` and `last.pt`), per-window GIFT-Eval
  metrics, and the 40-million-series KernelSynth corpus (164 GB) are
  available on request from Danila Chernousov's object storage.

## Analysis scripts

Install the package or set `PYTHONPATH=src`, and install `matplotlib`,
`scipy`, and `pandas`. The series directory contains `runs/` and `eval/`.
Area calculations also require the extracted `window_loss.f32` and
`heldout_*.f32` files. The `SERIES_PREFIX` environment variable sets the
run prefix and defaults to `series_`.

- `series_analysis.py <series> [gzip_baseline.json]` summarizes areas, final
  losses, forecast accuracy, and correlations.
- `epiplexity_variants.py <series> <out.md>` compares 15 estimators against
  five performance measures. `variants_subset.py` performs the same analysis
  with exclusions from `EXCLUDE=chaotic,waveform`.
- `corr_vs_windows.py <series> <figures>` correlates checkpoint estimates
  with final forecast performance.
- `series_figures2.py <series> <figures>` plots ratios to the naive baseline
  versus area and shades excess area on the training curves.
  `series_figures3.py` compares three x-axis definitions.
  `series_figures4.py <series> <figures> <reference/gift_eval_leaderboard>`
  adds regression lines and the TimesFM-3 reference.
- `timesfm_reference.py <series> <reference/gift_eval_leaderboard> [prefix]`
  checks agreement with the leaderboard's seasonal-naive results by task and
  computes TimesFM-3 ratios to that baseline.
- `gzip_baseline.py <corpora> [n]` and `verify_corpora.py <corpora> [n]`
  operate on corpora on the training machine. `bundle_results.py <series>
  <prefix> <destination>` assembles this archive.

## Protocol deviations recorded in the data

- Series 2, KernelSynth. The run failed at the first checkpoint due to an error
  collecting the held-out set. It resumed from `last.pt` at step 390, preserving
  the training trajectory by construction. The 0.01 entry in `heldout.jsonl`
  was recovered from `ckpt_0.01.pt`.
- Series 2, TimeSynth. We rewrote the generator to batch across series after
  the per-series version ran at one step per second. A test verifies bit-for-bit
  agreement with the per-series implementation.
- Series 2, KernelSynth. The corpus was built on GPU using float64 Cholesky
  decomposition. The CPU path differs only in the decomposition's rounding.
- Series 2, Waveform. The geometric-mean CRPS ratio is 1.373 because one
  `bitbrains_rnd` forecast diverged. For `s2_waveform_seed1` the ratio is
  0.972. The median and rank are robust.

## V-information versus archived CRPS

Each of the 23 measurements pairs two new Moirai fits with the published CRPS ratio for the same generator, budget, and initialization.
The conditional and context-free predictors share training windows and target normalization.
The CSV reports predictive V-information in bits per window and the paired test standard error conditional on the selected fits.

From the repository root, run the sweep on the allocated GPUs with the remaining wall-time allowance:

```bash
export PYTHONPATH=src
export ROOT=/home/jovyan/shares/SR008.fs2/kazachkovdi/side-projects/epiplexity-v-information
python experiments/episynth/scripts/run_v_information_transfer.py --root "$ROOT" --gpus 0,1,2,3,4,5,6,7 --hours 12
```

The launcher prepares configs, builds the shared KernelSynth corpus, fits both branches per run, and writes the PDF, PNG, and source CSV to `$ROOT/results/`.
When the CPU quota permits, up to two low-priority CPU jobs prepare train-only normalization for upcoming fits.
During a fit’s CPU normalization pass, its GPU can generate a short KernelSynth shard.
A shard that overlaps the start of training runs to completion.
Once the queue drains, an idle GPU can fit the conditional 40M fBm branch alongside the context-free branch, including during another run’s CPU normalization pass.
The runner publishes the completed branch under the run directory with an exclusive symlink and records its actual CPU worker count in the branch config.
Run the same command with the remaining allocation to resume from completed shards and checkpoints.
To regenerate the figure from the paired losses, run `python experiments/episynth/scripts/v_information_transfer.py plot --root "$ROOT"`.

## Requential model length versus archived CRPS

The horizontal coordinate sums the pre-update EMA teacher/student log likelihood ratios over coded forecast windows, in bits per trained student.
The vertical coordinate is the archived geometric mean of the CRPS ratios on the same 55 transfer tasks as the V-information figure.
Both panels use the archive's initialization, generator, window distribution, and real training budget.
Projection recovery counts toward that budget and sends zero messages.
The eight repeated teacher draws per message supply the Monte Carlo standard error and the separate plug-in overhead in the CSV.

Set `REQ_ROOT` to the requential experiment directory and `VINFO_ROOT` to the archive containing the original KernelSynth corpus.
Public contexts start at seed 100,000,000.
For KernelSynth, place contiguous corpus shards covering 39,999,488 public seeds under `$REQ_ROOT/data/kernelsynth/<first_seed>` using the archived settings (length 1024, five kernels, chunk 64, CUDA factorization).
Projection validation and final evaluation each use 100,000 fixed windows starting at 1,000,000,000 and 2,000,000,000, respectively.

```bash
export PYTHONPATH=src
python experiments/episynth/scripts/requential_transfer.py --root "$REQ_ROOT" --source-root "$VINFO_ROOT" --run s2_waveform
python experiments/episynth/scripts/requential_transfer.py --root "$REQ_ROOT" --plot
```

Run one fit per generator with its archived `s2_` name, including the separate `s2_waveform_seed1` fit.
Use `series_kernelsynth --through 1999872` for the small KernelSynth point and `s2_kernelsynth` for the large point.
Each other seed-zero trajectory supplies both budget milestones.
The worker can resume from `runs/<run>/resume.pt`, including during recovery.
The plot command joins the 23 completed measurements to the archive and writes the PNG, PDF, and source CSV under `results/`.

## Gotchas

- **Triton loads compiled kernels by file path.**
  Concurrent writers on the shared FUSE mount produced missing-file errors during compilation.
  The launcher gives each GPU a compiler cache under `$XDG_CACHE_HOME/episynth-compile`, with `~/.cache` as the fallback.
  Corpus shards and model checkpoints live under `$ROOT/data` and `$ROOT/runs`.
