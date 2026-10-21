# Member 2: requential code, label noise and the demo

**Role.** Owner of the requential estimator for classifiers (Eq. 4 and Section 3.2 of
[`notes/epiplexity.pdf`](../notes/epiplexity.pdf)). Owner of the experiments that separate
structure from noise.

## Research questions (RQ2 and open question 5)

1. **Label noise.** Symmetric label noise at rate $\rho$ over $C$ classes sets
   $H(Y\mid X)=h(\rho)+\rho\log_2(C-1)$ per example, where $h$ is the binary entropy.
   - **Hypothesis.** The raw prequential code grows linearly in this term. The held-out-floor
     area and the requential code stay close to their clean values, then fall at $\rho=1$.
2. **Teacher recipe.** Which of the two recipes gives the shorter and more stable code: a
   divergence threshold with a paused teacher (founding paper), or iso-loss projection
   (requential-coding paper)? Neither paper compares them.

## Library (`epimeter`)

- `codes/requential.py`:
  - an EMA teacher, a student trained on hard labels sampled from the teacher, and the exact
    categorical KL per batch;
  - factorised Bernoulli outputs for ECA;
  - both teacher recipes;
  - reports $\sum_i\mathrm{KL}_i$ and the full bound
    $\sum_i[\mathrm{KL}_i+2\log(1+\mathrm{KL}_i)+\kappa]$ with $\kappa=5.21$.
- `codes/two_part.py`: a naive two-part code with quantised weights, as a lower baseline.
- BMM algorithm activity: **requential coding and the two-part code** (from the "2-part MDL codes"
  item of the project list).

## Theory item (R&D checkpoint 2)

1. **Overhead.** At batch size $B$ over $M$ examples the per-message terms add at least
   $(M/B)\,\kappa$ bits. For MNIST at $B=128$ that is 2.4 kbits, the size of the whole MNIST
   code. Derive the smallest $B$ at which the overhead is below 10% of $\sum_i\mathrm{KL}_i$, and
   use it in the protocol.
2. **Noise.** Give the expected behaviour of the three codes (raw prequential, held-out-floor
   area, requential) under symmetric label noise. State the assumptions: one pass, a calibrated
   teacher.

## Experiments

- **Label noise.** $\rho\in\{0,0.1,0.2,0.4,0.6,0.8,1\}$ on MNIST, MNIST-1D and CIFAR-10. MLP
  observers of widths 64–1024. 5 seeds.
- **Pixel permutation.** One fixed permutation of pixels. The MLP should give an identical value
  (Member 4 proves the invariance); a CNN should not.
- **ECA.** The requential code on the same streams as Member 1, the small-scale analogue of Fig. 2c
  of the founding paper.

## Other BMM roles

- **Tests** (coverage above 90%, TM3 on 24.11). Set up pytest and coverage in week 1. Every member
  adds tests for their own code; Member 2 enforces the threshold in CI together with Member 4.
- **Demo** (TM3 on 24.11, final version at TM4 on 8.12). One Colab notebook that installs
  `epimeter`, computes the prequential and requential codes on MNIST with label noise, and plots
  the main figure of the paper. The same notebook satisfies the Colab requirement of the R&D course.

## Tasks and dates

| Date | Deliverable |
|---|---|
| 06.10 | Requential loop for an MLP on MNIST, one seed |
| 13–19.10 (R&D CP2) | Overhead bound and noise predictions, written |
| 20.10 (BMM TM2) | pytest and coverage running in CI; requential prototype |
| 03.11 | Both teacher recipes; pilot of the noise sweep on MNIST |
| 10.11 | Noise and permutation runs complete, 5 seeds |
| 17.11 | Figures frozen; paper sections written |
| 24.11 (BMM TM3) | Tests above 90%; demo notebook runs on Colab from a clean session |
| 01.12 | Cross-review of one other BMM team |
| 08.12 (BMM TM4) | Demo final |

## Paper sections

The requential estimator (Method), the noise and permutation experiments (Results), and the
comparison of teacher recipes (Appendix).

## Compute (estimate)

The requential code needs about 7/3 of a training run. The noise sweep (3 datasets × 7 rates × 5
widths × 5 seeds × 2 codes) should need 15–30 T4-hours with the datasets held on the GPU.
