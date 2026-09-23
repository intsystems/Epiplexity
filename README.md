# Epiplexity

Estimating epiplexity ([Finzi, Qiu et al., 2026](https://arxiv.org/abs/2601.03220)) with small models
(MLPs) on small datasets on a single GPU.

- [`notes/epiplexity.pdf`](notes/epiplexity.pdf): a literature review covering the definitions,
  theorems and estimators, the follow-ups (Requential Coding, EpiSelect/EpiGen), the papers citing it,
  alternative measures, and what all of this implies for small-scale estimation.
  Build it with `pdflatex epiplexity && bibtex epiplexity && pdflatex epiplexity && pdflatex epiplexity`.
- [`research_notes/`](research_notes/): the source notes behind the review, with a link on every claim.
