# Member 4: testbeds, fast proxies, variance and the repository

**Role.** Owner of the data generators, the classical and fast complexity measures, the study of
seed variance, and the repository (package, CI, documentation, Docker).

## Research questions (RQ4 and open questions 3 and 6)

1. **Seed variance.** How large is the spread of $S_T$ over seeds and over the hull? How many
   seeds does a ranking of two datasets need at a given confidence? No published estimate reports
   this.
2. **Fast proxies.** Do they rank datasets the same way as $S_T$? The candidates are AIC, BIC,
   HQIC, WAIC, WBIC, EDL, and the reservoir score of Zhang and Levin measured against a
   shuffled-target baseline at matched sample size.

- **Hypothesis.** AIC, BIC and HQIC depend only on the number of parameters and the fitted loss.
  They will track total information and rank random labels high. EDL will agree with $S_T$
  in ranking. The reservoir score will agree only after the baseline is subtracted
  (review, Table 3).

## Library (`epimeter`)

- `data/`: the testbeds, all generated on the GPU where possible.
  - ECA $Y=F^t(X)$ with periodic boundary;
  - symmetric label noise and fixed pixel permutation for MNIST, MNIST-1D and CIFAR-10;
  - Markov sources with a known entropy rate;
  - teacher–student MLP (with Member 3);
  - sparse parity;
  - random hierarchy model.
- `criteria/`: AIC, BIC and HQIC; WAIC and WBIC with SGLD chains; EDL (Eq. 5 of the review);
  the reservoir score (Eq. 6) with a shuffled-target baseline.
- BMM algorithm activity: **basic methods** (AIC, BIC, HQIC, WAIC, WBIC) plus the fast proxies.

## Theory item (R&D checkpoint 2)

1. **Permutation invariance.** For an MLP with i.i.d. initialisation of the first layer and a
   coordinate-wise optimiser (SGD, Adam), a fixed permutation of input coordinates leaves the
   distribution of every training curve unchanged. Every estimator of $S_T$ built on these curves
   is therefore invariant. Member 2 tests this.
2. **Seeds.** The number of seeds needed to separate two values of $S_T$ at confidence
   $1-\alpha$, given the measured spread. Use it to fix the number of seeds in the protocol.

## Other BMM roles

- **Project wrapping.**
  - Package layout, `pip install` from GitHub, the MIT licence;
  - GitHub Actions CI (tests, coverage, documentation build);
  - one code style (ruff) across the team.
  - Evaluated at the checkpoint (3.11 or 10.11, short) and at TM3 (24.11, full).
- **Documentation.** Sphinx, with an API page per module and one tutorial page. Intermediate
  structure at TM2 (20.10), final at TM3 (24.11). Sphinx is also required by the R&D course.

## R&D course requirements (owner)

- `python3 train.py` trains every model of the paper.
- `python3 test.py` computes every table and figure from the saved runs.
- A `Dockerfile` in which both commands run.
- A script that downloads MNIST, MNIST-1D and CIFAR-10. Synthetic data needs none.
- Runs are stored as one JSON line per run, with the fields fixed in [`README.md`](README.md).
  Figures are computed only from these files.

## Tasks and dates

| Date | Deliverable |
|---|---|
| 29.09 | Repository skeleton: `src/epimeter`, `tests/`, `docs/`, `paper/`, CI, licence |
| 06.10 | ECA generator on the GPU; label-noise and permutation wrappers; AIC and BIC |
| 13–19.10 (R&D CP2) | Permutation invariance and the seed rule, written |
| 20.10 (BMM TM2) | Documentation structure; WAIC and WBIC with SGLD |
| 03.11 or 10.11 (BMM checkpoint) | Short presentation of the wrapped library; EDL and the reservoir score |
| 3–9.11 (R&D CP3) | `train.py`, `test.py` and Docker running end to end on one testbed |
| 10.11 | Variance study complete (20 seeds on 3 testbeds); proxy agreement on all testbeds |
| 17.11 | Figures frozen; paper sections written |
| 24.11 (BMM TM3) | Full presentation: install, CI, documentation |
| 01.12 | Cross-review of one other BMM team |

## Paper sections

Testbeds and protocol (Setup), the variance and proxy experiments (Results), and reproducibility
(Appendix).

## Compute (estimate)

The variance study (20 seeds × 3 testbeds × 4 sizes) should need 10–20 T4-hours. SGLD chains for
WBIC need several training runs each. Limit WBIC to the smaller observers if Colab time runs short.
