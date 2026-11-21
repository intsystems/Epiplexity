# episynth: Proof of Concept / Prequential

The TM2 presentation has **12 main slides and 3 appendix slides** and is
designed for a 12–15 minute talk.

- [Compiled PDF](episynth_poc_prequential.pdf)
- [LaTeX source](main.tex)
- [Speaker notes](speaker_notes.md)
- [Sources for the numbers and figures](data/provenance.json)

The main slides cover the research question, architecture, blockwise prequential
coding, excess area above the final held-out EMA loss, implementation, CPU demo
configuration and results, the plugin interface, and generated API documentation.
The appendix covers uncertainty, the larger runs, and references.

## Build

Use XeLaTeX (TeX Live or MiKTeX) with Beamer, fontspec, polyglossia, TikZ,
and listings. The slides use DejaVu Sans and DejaVu Sans Mono when installed,
with the standard Beamer fonts as a fallback. Run from this directory.

```bash
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf episynth_poc_prequential.pdf
```

In Overleaf, select **XeLaTeX** and set `main.tex` as the main document.
The source archive includes every figure under `figures/` and can compile
without Python, PyTorch, datasets, model weights, or internet access.

## Figures and numerical results

`figures/demo_notebook.png` comes from the saved output of
`experiments/demos/demo_estimators.ipynb`. Building the slides uses the
existing run. The left plot shades positive excess loss, while the estimator
sums signed differences, as the slide explains.

`figures/area_schematic.*` is a labeled schematic. `large_series_context.*`
uses the committed CSVs in `publications/article/data/` for the eleven main
generators in each series, excluding the additional Waveform seed.

To regenerate the figures from a full checkout, install NumPy and Matplotlib
and run the following command.

```bash
python make_figures.py
```

The script reads the notebook and CSVs from the repository. The standalone
source archive already includes the figures needed to build the PDF.

Finzi et al. (2026) provide the theoretical motivation. The slides use
excess prequential loss as an operational estimate for a fixed observer and
protocol. The estimate alone cannot establish exact theoretical epiplexity,
the size of a weights file, or the utility of a generator for transfer.
