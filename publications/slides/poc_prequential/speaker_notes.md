# episynth proof of concept: speaker notes

Slides 1–12 take about 12–15 minutes, including a short demonstration of the
notebook and API. Slides 13–15 are optional appendix slides.

## 1. Title slide (20 seconds)

“We’ll show the working episynth proof of concept. We’ll follow the prequential
calculation from the formula to an estimate in bits and see where the
estimator fits in the library.”

## 2. Research question (60 seconds)

“We vary the synthetic time-series generator and train a separate instance
of the same model on each one. Training gives an estimate of learned
structure, and GIFT-Eval gives forecast performance on real data. We then
measure their correlation across generators.”

Q denotes forecast error. If more learned structure is associated with
better forecasts, the expected correlation is negative. The experiment
tests whether this relationship exists while fixing the architecture,
budget, and protocol within each series.

## 3. Architecture (50 seconds)

Follow the arrows from the generator through windows and Trainer. Training
produces two inputs to the correlation analysis. Losses give an estimate of
structure, and the weights allow us to measure transfer performance. Generators,
windows, and training are shared components. The estimator registry is the
extension point. A saved corpus can replace costly online generation with
disk reads.

## 4. Prequential coding (80 seconds)

“Prequential coding means predict, then learn. We score each window’s future
using the current model and then train on the batch.”

The procedure is **blockwise**. The same model scores all four windows in a
demo batch. The weights update once per batch, so the formula uses the
batch index b(i). Compute losses before `optimizer.step` and write these
pre-update values to disk after the step.

The optimizer uses mean NLL per horizon point. The file records the full
NLL per window, so dividing by the horizon for optimization leaves the
stored units unchanged.

## 5. From prequential loss to the model part (100 seconds)

“The sum of online errors reflects both learning structure and the
remaining difficulty of making predictions. We subtract the final held-out
loss scaled to the same budget.”

The held-out windows come from **the same synthetic generator**, with seeds
separate from training. The final predictor uses EMA weights. Subtracting
N times the mean held-out loss from total training loss gives signed area.
Dividing by ln(2) converts nats to bits.

Describe this as excess prequential loss for a fixed observer, budget, and
protocol. It is a proxy for the model part of a compute-bounded two-part code.
The final held-out loss includes approximation and optimization error.
The shared algorithm and initialization are fixed and setup cost is outside
this estimate. A code-length interpretation of continuous densities requires
a fixed precision or dequantization. The presentation uses recorded NLLs
in the original series units. This estimate alone cannot establish exact
theoretical epiplexity, weights-file size, or pure noise entropy.

## 6. Geometric interpretation (45 seconds)

“The estimate is a signed area, including contributions from regions below
the baseline. The slide uses a schematic to show how that works.”

The code sums signed excesses without clipping at zero. An actual estimate
can therefore be negative.

## 7. Implementation (70 seconds)

Show `PrequentialEstimator.run` → `Trainer.run` → `from_run_dir` → `run_area`.
The calculation sums the loss array and subtracts its length times the
held-out mean. `from_run_dir` computes the estimate from a saved run without
retraining. The files are headerless float32 arrays containing full window
losses in nats. The PDF links to the source.

## 8. Proof-of-concept configuration (50 seconds)

“We use one layer, 60,994 parameters, 64 training windows, and a CPU. The
interfaces and calculation are the same as for a larger run.”

The demo model has one Student-t component. The larger series use three
components. Training and held-out seeds are separate.

## 9. Demo results (70 seconds)

Open the [executed notebook](https://github.com/intsystems/Epiplexity/blob/main/experiments/demos/demo_estimators.ipynb).

Show the configuration, the call that runs the estimator, and the saved
plots. The model part is 788.284 bits, the data part is 5499.059 bits, and
the total is 6287.343 bits. All six code cells ran. The slide uses the plot
from those saved outputs.

The shading highlights regions above the baseline. The model part uses
the signed area, including negative contributions. This short run shows
that the interface works. Stability and correlation across eleven
generators are questions for the larger study. Start a fresh kernel for a
live rerun because the registry rejects duplicate `demo_constant` registration.

## 10. Extending the library (60 seconds)

The common `run` signature lets each algorithm use the same pipeline.
Register a new estimator and select it through `build` or the configuration.
Any algorithm-specific operations belong in its `run` implementation,
which can use the shared components.

The current requential estimator is a registered stub. The notebook's
`ConstantEstimator` demonstrates class registration.

## 11. Generated API documentation and checks (60 seconds)

Open the [run_area API reference](https://intsystems.github.io/Epiplexity/api/analysis.html#episynth.analysis.run_area).

Show the parameters, return value `RunArea`, and linked source. If helpful,
open `doc/source/api/analysis.rst` and point out that `automodule` obtains
this information from Python. This demonstrates an automatically documented
function.

The baseline verification reported 72 passed and 2 skipped. The skipped
tests require external weights and GIFT-Eval data. The docs build automatically
and site publication is manual.

## 12. Conclusion (40 seconds)

“The proof of concept confirms that the base estimator is integrated. The
shared interface covers generation, training, recording pre-update losses,
held-out evaluation, and decomposition. The research question uses the
long runs and GIFT-Eval results described in the technical report.”

## Appendix: possible questions

- **Why does area uncertainty grow with N?** With the recorded training
  losses fixed, N multiplies the uncertainty of the mean baseline. The
  formula conditions on the fitted predictor and assumes independent
  held-out windows. Training-seed variability needs a separate estimate.
- **Why is one number insufficient to establish generator utility?**
  The protocol depends on architecture, compute budget, EMA, and the target
  distribution. The largest area offers no guarantee of the best transfer.
- **What did the larger runs show?** Spearman correlation was about −0.18
  and 0.00 across all eleven families at the two budgets, with intervals
  including zero. The slide uses saved CSVs and excludes the extra Waveform seed.
- **Why plot nats?** At a fixed budget, converting the area to bits by
  multiplying by 1/ln(2) leaves rankings and Spearman correlation unchanged.
