# Epiplexity

Estimating epiplexity ([Finzi, Qiu et al., 2026](https://arxiv.org/abs/2601.03220)) with small models
(MLPs) on small datasets on a single GPU.

This repository is the shared skeleton for a four-member project that closes two courses (Bayesian
methods and the R&D course) and targets CPAL 2027. The code lives in the `epimeter` package
(`src/`), launched through `experiments/` and `notebooks/`.

## Repository layout

```
plan/                  planning, roles, protocol, calendar
research_notes/        source notes behind the literature review
notes/                 literature review (pdf + tex + refs)
slides/                presentation
src/epimeter/          the package: estimators, models, samplers, training, eval
experiments/           runnable scripts wired through the registries (analogous to relaxit demo/)
notebooks/             RQ1-RQ4 notebooks (later stages)
tests/                 pytest
docs/                  Sphinx (skeleton now, filled at stage 7)
paper/                 arXiv-style manuscript (stage 7)
Dockerfile             single-GPU runtime (stage 7)
```

## Current stage (1-3): interface skeleton

`src/epimeter/` currently exposes only interfaces and skeleton classes. Importing the package is
side-effect free; concrete estimators (`Prequential`, `Requential`, `Bayesian`, `Proxies`) and
`train`/`evaluate` raise `NotImplementedError` until later stages fill them in. Two registries
(`models`, `samplers`) are already functional: register any network or sampler with
`@epimeter.register("name")` and build it with `epimeter.build("name", **config)` — see
`experiments/example_mlp.py` and `experiments/example_sampler.py`.

## Development

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). On a network that needs system TLS
certificates, add `--system-certs` to the uv commands.

```bash
uv sync --extra dev
uv pip install -e .
.venv/bin/pytest tests/          # 8 structural + pluggability tests
.venv/bin/ruff check src/ tests/
```

## Literature

- [`notes/epiplexity.pdf`](notes/epiplexity.pdf): literature review (definitions, theorems,
  estimators, follow-ups, alternatives, implications for small-scale estimation).
- [`plan/README.md`](plan/README.md): the full project plan, roles, protocol and calendar.

License: MIT.
