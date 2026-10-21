# Epiplexity and the information bottleneck: project proposal

Prepared on 9 October 2026 from the collaborator correspondence, the repository's existing literature review, and primary-source checks.

- [PDF](proposal.pdf)
- [LaTeX source](proposal.tex)
- [Bibliography](references.bib)

The proposal distinguishes loss of information from limited decoding accessibility and estimator artifacts. It recommends a gated study comparing V-information, MDL probing, a qualified prequential structural proxy, and a small separate model-coding validation. Direct modeling of activation distributions is included, with explicit quantization and density-model controls. It does not assume that the information bottleneck hypothesis must hold or that epiplexity is a new independent signal beyond existing MDL components.

Build from this directory with a standard LaTeX installation:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
bibtex proposal
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error proposal.tex
```

The document records source-verification limits and separates published findings from proposed experiments and original reasoning. No research experiments have been run for this proposal.
