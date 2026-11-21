# episynth requirements audit (October 21, 2026)

> Historical audit dated October 21. Current materials are in the
> [technical report](technical_report/README.md) and the
> [research article](article/README.md).

This audit checked the full Technical Meeting 2 (TM2) requirements from the
project correspondence against `main` after the technical report was added
(`876f611`). It also recorded the switch to manual documentation deployment
and the correction of blog image paths. At the user's request, the audit
left the presentations as they were. It covered engineering readiness for
TM2. The talk, scientific results, and full final submission required
separate verification.

| Requirement | Result and evidence |
|---|---|
| Repository in `intsystems`, tests, and publication of documentation | GitHub Actions in `intsystems/Epiplexity` runs pytest, Ruff, and the Sphinx build. |
| Disable automatic GitHub Pages deployment | Pushes and pull requests to `main` still build the docs. Uploading the Pages artifact and running the `deploy` job require `github.event_name == 'workflow_dispatch'`. Publish from **Actions → docs → Run workflow** on `main`. |
| Project name | The distribution, import, and main documentation use `episynth`. `Epiplexity` is the repository name. The separate proposal `research_notes/information_bottleneck_epiplexity/proposal.tex` still used the old name at the time of this audit. |
| Code style and linter | CI runs Ruff with the configuration in `pyproject.toml`. It checks the library, tests, demo notebook, and Sphinx configuration. |
| Documentation engine | Sphinx uses autodoc, Napoleon, viewcode, MathJax, and the Read the Docs theme. |
| Test library | GitHub Actions runs pytest and pytest-cov. |
| Architecture and class interactions | [architecture.rst](../doc/source/architecture.rst) and [description.md](description.md) trace generators → windows → model → training → estimator → analysis and GIFT-Eval. |
| Annotated public functions and methods | All 149 audited public definitions had docstrings and generated API entries. The audit verified parameter and return annotations on functions and methods, including `*args` and `**kwargs` where present. It covered public definitions in public modules. |
| Dependencies and integrations | `pyproject.toml` declares dependencies, `uv.lock` pins versions, and [stack.rst](../doc/source/stack.rst) describes their roles and the external integrations. |
| Documentation structure | `doc/source/index.rst` links installation, the demo, architecture, the development stack, extension instructions, experiments, and the module API. |
| Build and view documentation locally | Build with `uv run --no-sync sphinx-build -b html -W --keep-going doc/source doc/_build`. Open the HTML directly or serve it with a static server. See [installation.rst](../doc/source/installation.rst). |
| Online documentation | [GitHub Pages](https://intsystems.github.io/Epiplexity/) hosts the docs. [GitHub Actions](https://github.com/intsystems/Epiplexity/actions/runs/39625074099/job/119314439960) confirmed deployment of `5148f88`. The cloud environment's proxy restricted direct HTTP access. |
| Automatically generated documentation for the demo | The [estimator API](https://intsystems.github.io/Epiplexity/api/estimators.html) documents signatures, parameters, results, and source links for `PrequentialEstimator.run` and `from_run_dir`. |
| Executable demo with saved plots | All six code cells in [demo_estimators.ipynb](../experiments/demos/demo_estimators.ipynb) ran in a new empty working directory. The embedded PNG contains both plots. The audit ran the existing code cells. |
| Proof of concept integrated into the planned architecture | The prequential estimator uses `EpiplexityEstimator`, the `build`/`register` registry, the shared `Trainer`, and `analysis.run_area`. [extending.rst](../doc/source/extending.rst) explains adding an estimator. TM2 requires at least one base algorithm. |
| Blog structure and planned figures | The [draft](blogpost/epiplexity.md) has sections and twelve illustrations. Relative links work in the repository. The [README](blogpost/README.md) explains preview and export for the separate site. |
| Technical report structure and planned figures | At the time of the audit, the [technical report](technical_report/main.pdf) had seven pages covering methodology, implementation, results, figures, discussion, and appendices. The repository also contains [source and reproduction instructions](technical_report/README.md). `paper/main.tex` is an older manuscript. |
| TM2 presentation of up to 15 minutes with the proof of concept and autodoc | Slides are under `publications/slides/`. An earlier review found that some slides diverged from the current time-series proof of concept. This audit left them as they were and could not verify talk readiness or duration. The notebook, generated API, and technical report were ready to show. |

## Demo verification

The CPU demo uses Waveform, a model with 60,994 parameters, 64 training
windows, 32 held-out windows, and fp32. The audit verified finite losses
and estimates, the sizes of the loss files, checkpoint creation, and
`model_part_bits + data_part_bits = total_bits`. The model part was about
788.284 bits, the data part 5499.059 bits, and the total 6287.343 bits.
The saved figures show per-window loss with a held-out baseline and the
model and data parts. The run demonstrates the interface. Estimator
convergence and large-run results require separate experiments.

## Remaining work for final submission

- **Future algorithms.** Prequential is implemented. `RequentialEstimator.run`
  raises `NotImplementedError`. Bayesian estimators and proxies from the
  original plan remain research directions. The planned API can be documented
  before those algorithms are implemented, and TM2 can use the working estimator.
- **Coverage.** The last complete local check of `5148f88` reported 72 passed,
  2 skipped, and 76% coverage for `episynth`. The original team plan set a
  goal above 90%. The audit had yet to meet this historical goal, which had
  no CI threshold and was absent from the TM2 requirements provided.
- **External checks.** Two tests were skipped without `MOIRAI_SMALL_DIR`
  and `GIFT_EVAL_DIR` for published Moirai weights and GIFT-Eval data.
  The audit used existing GPU results and original-implementation comparisons.
- **Colab.** The audit confirmed local notebook execution. The original plan
  separately called for a run in a fresh Colab session, which had yet to happen.
- **Plans and manuscript.** The repository retains the earlier plans and
  `paper/main.tex` for historical reference. At the time of the audit, the
  time-series manuscript was available in `technical_report/main.pdf`.
  Its README showed how to rerun the analysis and build it.

The engineering work, documentation, blog post, and technical report were
ready for TM2. Verification of the presentation, the live talk, and the
complete final submission remained open.
