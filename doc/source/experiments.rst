Experiment workflow
===================

Research question
-----------------

We measure the structure a fixed forecasting learner can extract from each
synthetic generator, and study its relationship with forecasting quality.
The generator determines the data distribution; the estimator measures the
learning run. A small CPU demonstration illustrates the interface, while the
large experiment series supply the research measurements.

Pipeline
--------

#. Generate series online from deterministic seeds, or create a reusable corpus
   with ``python -m episynth.corpus``.
#. Draw forecasting windows using ``WindowSpec``; training and held-out seed
   ranges are disjoint.
#. Train Moirai with ``python -m episynth.train --config ... --out ...``.
   The JSON configuration specifies data, windows, model, training budget and
   estimator. CLI ``--set section.field=value`` overrides configuration fields.
#. Record each window's loss before its training step, checkpoint weights and
   evaluate raw and EMA models on held-out windows. Runs can resume.
#. Compute the prequential decomposition with ``episynth.analysis.run_area``.
   For losses in nats, ``total_bits = sum_loss / ln(2)``,
   ``model_part_bits = (sum_loss - N * heldout_mean_loss) / ln(2)``, and
   ``data_part_bits = N * heldout_mean_loss / ln(2)``. The reported uncertainty
   comes from the held-out losses. This finite-budget estimate is specific to
   the observer and training protocol.
#. Evaluate checkpoints on the short-horizon GIFT-Eval tasks and compare
   generator-level estimates with forecasting metrics using the analysis API.

Saved series
------------

``experiments/episynth`` contains the two recorded series: a corpus-based series
with 2 million windows and an online series with 40 million windows, together
with configurations, learning curves, held-out summaries, GIFT-Eval summaries,
analysis tables, figures and scripts. See
`the experiment README <https://github.com/intsystems/Epiplexity/blob/main/experiments/episynth/README.md>`_
for exact artifacts and the release holding large per-window losses.

Reproducibility and scope
-------------------------

Keep generator seeds, model initialization, window specification, data budgets,
optimizer settings, precision and the held-out protocol fixed when comparing
runs. Report the units and uncertainty rather than only a generator ranking.
The normal CI suite checks small CPU runs and resume behavior; it does not rerun
the expensive series or establish the scientific conclusions automatically.
