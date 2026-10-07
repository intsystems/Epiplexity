# Plan: epiplexity on one GPU (fall 2026)

The team has four members and uses one repository and one paper for two courses:

- the Bayesian course (BMM, [schedule](https://github.com/intsystems/BMM/blob/main-26-27/schedule.md),
  [projects](https://github.com/intsystems/BMM/blob/main-26-27/projects.md));
- "Intelligent systems creation" (R&D course,
  [page](https://intsystems.github.io/ru/course/rnd_in_ai/index.html)).

The primary venue is CPAL 2027: abstract on 23.11, paper on 05.12 ([`venues.md`](venues.md)). Background, notation and
equation numbers come from the literature review [`notes/epiplexity.pdf`](../notes/epiplexity.pdf).
Each member has a plan of their own:

| | Member 1 | Member 2 | Member 3 | Member 4 |
|---|---|---|---|---|
| Plan | [`member1.md`](member1.md) | [`member2.md`](member2.md) | [`member3.md`](member3.md) | [`member4.md`](member4.md) |
| Research | prequential code, sweep, ECA | requential code, label noise, permutation | Bayesian codes, theory, manuscript | testbeds, fast proxies, seed variance, repository |
| BMM algorithm | online coding | requential and two-part codes | variational coding, KFAC-Laplace evidence | AIC, BIC, HQIC, WAIC, WBIC, EDL, reservoir score |
| BMM roles | planning, PoC | tests, demo | blog post, tech report | wrapping, documentation |
| Theory item (R&D CP2) | floor error; expected area | requential overhead; noise | evidence identity; noise | permutation invariance; number of seeds |
| Paper | prequential method, ECA results | requential method, noise results | theory, Bayesian results, introduction, editing | setup, variance results, reproducibility |

Every member also has one BMM cross-review (1.12) and gives the weekly 5-minute R&D presentation
in turn.

## Goal

**Working title:** *Epiplexity on one GPU: estimators for small networks and when they agree.*
Planned contributions:

1. `epimeter`, a PyTorch library (the name is free on PyPI; `epiplexity` is taken).
   - One interface returns (model part, data part) in bits for the prequential, requential,
     variational and Laplace codes, the classical criteria, EDL and the reservoir score.
   - A sweep recovers $S_T$ at a compute bound $T$.
2. The first estimates of epiplexity with MLP and convolutional observers. The testbeds are
   ECA, label noise, permuted pixels and a planted teacher–student task.
3. The first measurements of seed variance, with a protocol that fixes the floor, the number of
   seeds, the units and the batch size.
4. A bridge from Bayes to MDL: the conditions under which Bayesian model parts agree with $S_T$.

## Research questions

| | Question | Owner | Open question in the review (Section 9) |
|---|---|---|---|
| RQ1 | Does an MLP reproduce the ECA ordering? From which $T$ on? | Member 1 | 1, 2 |
| RQ2 | Do the requential code and the held-out-floor area ignore label noise, while the raw code grows? | Member 2 | 4, 5 |
| RQ3 | Do Bayesian model parts rank datasets the same way as $S_T$? | Member 3 | 6 |
| RQ4 | How large is the seed spread? Do fast proxies agree with $S_T$? | Member 4 | 3, 6 |

## One project for two courses

**BMM.** Request the listed project "Theoretically informed deep learning model complexity
estimation" (2–4 people) at TM1. Epiplexity becomes the quantity that unifies it. Its four
algorithm groups map one to one onto members: basic criteria (Member 4), Bayesian codes
(Member 3), online coding (Member 1), and two-part MDL codes, extended by requential coding
(Member 2).

| Requirement | Artifact | Owner | Due |
|---|---|---|---|
| Project planning | this folder, presented | Member 1 | TM1, 29.09 |
| PoC | prequential code on MNIST through the library interface | Member 1 | TM2, 20.10 |
| Documentation | Sphinx | Member 4 | 20.10 (structure), 24.11 |
| Blog post | Habr or Medium | Member 3 | 20.10 (draft), 24.11 |
| Tech report, 3–5 pages | short version of the paper | Member 3 | 20.10 (draft), 24.11 |
| Wrapping | package, install, CI | Member 4 | 3.11 or 10.11 (short), 24.11 |
| Tests, coverage above 90% | pytest in CI | Member 2 | TM3, 24.11 |
| Demo | Colab notebook | Member 2 | 24.11, final 8.12 |
| One algorithm each | see the table above | all | TM1 (plan), TM3 (code) |
| Cross-review | one per member | all | 1.12 |

**R&D course.** Checkpoint weeks are from the course README. The dates assume the semester began
on 1 September 2026; confirm them with the instructor.

| Checkpoint | Content | Artifact | Owner |
|---|---|---|---|
| CP1, week 4 (≈22–28.09) | problem analysis | the literature review and the research questions above | all |
| CP2, week 7 (≈13–19.10) | theoretical results | one written statement with proof per member | Member 3 assembles |
| CP3, week 10 (≈3–9.11) | experiment prepared | `train.py`, `test.py`, Docker, pilot runs | Member 4 |
| CP4, week 13 (≈24–30.11) | paper and repository | arXiv-style paper, MIT licence, Sphinx, Colab notebook | Member 3, all |

## Calendar

| Week (Tue–Mon) | Milestones |
|---|---|
| 29.09–05.10 | **BMM TM1 (29.09)**: plan presented. Repository skeleton. Shared protocol and result format agreed |
| 06.10–12.10 | Every estimator runs once on MNIST or ECA. ECA generator on the GPU. Paper skeleton |
| 13.10–19.10 | **R&D CP2**: theory items written. Route chosen: archival CPAL, or the non-archival route that keeps ICML open |
| 20.10–26.10 | **BMM TM2 (20.10)**: PoC, blog and tech-report drafts, documentation structure. Sweep and hull |
| 27.10–02.11 | Pilot runs of every experiment on one seed. Tests above 60% |
| 03.11–09.11 | **BMM checkpoint (3.11 or 10.11). R&D CP3.** Main runs start, 5 seeds |
| 10.11–16.11 | Main runs complete. Variance study complete |
| 17.11–23.11 | **Experiments frozen on 17.11.** Paper draft v1 on 17.11. Internal review. **CPAL abstract registered by 23.11** |
| 24.11–30.11 | **BMM TM3 (24.11)**: all materials ready. **R&D CP4.** Paper v2 |
| 01.12–07.12 | **BMM cross-review (1.12).** **CPAL paper deadline 05.12** (9 pages, anonymised code link) |
| 08.12 | **BMM TM4**: reviews discussed, grades |

## Shared protocol

The protocol is fixed in the first week and changed only by agreement of all four members. It
follows Section 9 of the review.

- **Units.** Bits. Report $S$, $H$, $S$ per label, the dataset size $\mathcal D$, and $T$ in FLOPs
  from Eq. (1) of the review.
- **Target.** Code lengths cover labels only, which makes every value a conditional epiplexity
  $S_T(Y\mid X)$.
- **Floor.** The held-out loss of the EMA model on a held-out set whose size comes from the theory
  item of Member 1. A training-loss floor is never reported as a result.
- **Data.** One pass over fresh data whenever the generator allows. For fixed datasets, use the
  first-epoch online loss with a held-out floor (EDL, Eq. 5 of the review).
- **Seeds.** At least 5; Member 4 fixes the final number by 10.11. Report mean and standard error.
- **Controls in every experiment.** Random labels, constant labels, and shuffled targets at
  matched size.
- **Observers.**
  - Architectures: MLPs of widths 32–1024 and depths 1–3; a 1-D CNN for ECA.
  - Training: Adam with a learning rate scaled by 1/fan-in, a constant rate with an EMA of the
    weights.
- **Records.** One JSON line per run with the fields `estimator`, `testbed`, `config`, `seed`,
  `n_params`, `n_examples`, `flops`, `S_bits`, `H_bits`, `curve_path`, `git_sha`. All figures are
  computed from these files by `test.py`.
- **Compute.** One Colab GPU per member; the time estimates in the member plans assume a T4.
  Every run saves a checkpoint and can resume, because Colab sessions end without warning.

## Repository layout

```
notes/                 literature review (exists)
plan/                  this plan
src/epimeter/          codes/ criteria/ data/ observers/ sweep.py
tests/  docs/          pytest; Sphinx
paper/                 arXiv-style manuscript; the tech report is derived from it
notebooks/demo.ipynb   Colab demo
train.py  test.py  Dockerfile  LICENSE (MIT)
```

## Decisions needed this week

1. **Team size in the R&D course.** The course admits teams of 2–3.
   - Ask the instructor (A. V. Grabovoy) to admit one team of four with one paper.
   - Fallback: two teams on this repository, Members 1 and 2 (training-based estimators) and
     Members 3 and 4 (Bayesian codes and fast proxies). Each team presents its half; the paper
     merges both halves.
2. **BMM project.** Confirm at TM1 (29.09) that the team may take the listed project with the
   epiplexity focus and requential coding added.
3. **Names.** Assign people to Members 1–4. Member 3 needs the strongest Bayesian background;
   Member 4 needs the strongest engineering background.
4. **Compute.** A free Colab T4 limits the length of sessions. Plan Colab Pro for at least
   Members 1 and 2, whose sweeps are the longest.
5. **Venue.** CPAL 2027 is recommended: 9 pages, archival, one author in person in Tokyo
   (23–26.03.2027). Decide by 19.10 between CPAL and the non-archival route that keeps ICML 2027
   open ([`venues.md`](venues.md)). Decide who travels and how the trip is funded.

## Risks

| Risk | Response |
|---|---|
| MLPs do not learn rules 54 and 110 at large $t$, which leaves $S_T\approx0$ for every rule | Use small $t$ and the CNN observer. Report $S_T$ as a function of $t$; that curve is a result in itself |
| Signals of a few kilobits on MNIST are below the error of the floor | Size the held-out set by the theory item of Member 1. Keep the large synthetic testbeds primary |
| The per-message overhead exceeds the requential code at small batch | Choose the batch by the theory item of Member 2. Report the full bound next to $\sum_i\mathrm{KL}_i$ |
| The late-November deadlines of the two courses coincide | Freeze experiments on 17.11. After that, the paper depends on no new runs |
