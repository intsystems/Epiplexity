# Member 1: prequential code and the compute sweep

**Role.** Owner of the central estimator: the prequential area of Eq. (2) in
[`notes/epiplexity.pdf`](../notes/epiplexity.pdf) (Section 3.1). The same member recovers $S_T$
from a sweep over widths and depths (Section 3.3). Every other estimator is compared with this one.

## Research question (RQ1)

Does an MLP observer reproduce the ordering of Finzi et al. on elementary cellular automata (ECA)?
That ordering is rule 54 above rule 15 above rule 30, with rule 30 near zero. From which compute
bound $T$ does the order settle?

- **Hypothesis.** The order holds for MLPs at small numbers of automaton steps $t$. As $t$
  grows, rules 54 and 110 exceed what an MLP of width at most 1024 can learn. Their $S_T$ then
  falls towards zero.
- **Second observer.** A 1-D convolutional network with circular padding, which matches the
  locality of the automaton.
- **Stretch goal** (open question 2 of the review): does $S_T$ depend on direction at small
  scale? Compare $S_T(Y\mid X)$ with $S_T(X\mid Y)$ for $Y=F^t(X)$ under a bijective rule
  (rule 15, a shift with negation) and a non-bijective one (rule 30).

## Library (`epimeter`)

- `codes/prequential.py`:
  - the online code and the area above the floor;
  - the floor in three variants: held-out loss (default), first-pass loss (EDL, Eq. 5 of the
    review) and training loss (kept only to demonstrate pitfall 1);
  - the block-wise variants of Bornschein et al. (2022): incremental training with replay, and
    forward calibration.
- `sweep.py`:
  - a width × depth grid with a constant learning rate and an EMA of the weights;
  - FLOPs from Eq. (1) of the review;
  - the lower convex hull of (compute, code length), and $S_T$ as a function of $T$.
- BMM algorithm activity: **online coding** (from the project list of "Theoretically informed deep
  learning model complexity estimation").

## Theory item (R&D checkpoint 2)

1. With a held-out floor and one pass over i.i.d. data, the expected area is
   $\sum_i[\mathrm{KL}(p\Vert P_i)-\mathrm{KL}(p\Vert P_M)]$ (Eq. 3 of the review). The data
   entropy cancels.
2. A bias $\delta$ in the floor shifts the estimate by $M\delta$. Derive the size of the held-out
   set that makes the standard error of the floor small against $S/M$. Turn this into a rule for
   the protocol.

## Tasks and dates

| Date | Deliverable |
|---|---|
| 29.09 (BMM TM1) | **Project planning role:** present the plan (name, scope, stack, scheme), using [`README.md`](README.md) |
| 06.10 | Prequential area for an MLP on ECA and MNIST, one seed, through the shared interface |
| 13–19.10 (R&D CP2) | Written statements and proofs of the two theory items |
| 20.10 (BMM TM2) | **PoC role:** prequential code on MNIST through the library interface, with a notebook |
| 03.11 | Sweep and hull working; first curve of $S_T$ against $T$ for rules 15, 30 and 54 |
| 10.11 | ECA runs complete: rules 15, 30, 54, 110, identity and noise controls; $t\in\{1,2,4,8,16\}$; MLP and CNN; 5 seeds |
| 17.11 | Figures frozen; paper sections written |
| 24.11 (BMM TM3) | Algorithm presented; its tests pass |
| 01.12 | Cross-review of one other BMM team |

## Paper sections

The prequential estimator and the sweep (Method), and the ECA experiment (Results).

## Interfaces

- Receives the ECA stream from Member 4 (`data.ECA`, generated on the GPU).
- Provides the reference $S_T$ values that Members 2–4 compare against.
- Agree the result object with everyone in week 1 (fields `S_bits`, `H_bits`, `flops`, `curve`).

## Compute (estimate)

An MLP with $10^5$ parameters on $10^7$ examples needs about $6\times10^{12}$ FLOPs, which is
minutes on a T4. The full ECA grid (5 rules or controls × 5 values of $t$ × 2 observers × 8
sizes × 5 seeds) should need 20–40 T4-hours. Generating data on the GPU is required. The released
generator runs on the CPU and dominated the run time in our test (review, Section 9).
