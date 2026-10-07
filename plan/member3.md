# Member 3: Bayesian codes, theory and the manuscript

**Role.** Owner of the Bayesian estimators, the bridge from Bayes to MDL, and the manuscript
(editor). This is the part that ties the project to the Bayesian course.

## Research question (RQ3)

Do Bayesian model parts rank datasets the same way as the training-based $S_T$? The candidates are
the variational KL term and the Laplace Occam factor. Without a compute bound, when do they
coincide with $S_T$?

- **Hypothesis.** At the optimum of the variational two-part code, random labels go to the data
  part. Memorising them in the weights would need at least $M\log_2 C$ bits of KL. The model part
  therefore vanishes on noise, as $S_T$ does. The Laplace Occam factor grows with the number of
  parameters regardless of the data. It will rank datasets correctly only after subtracting its
  value on shuffled labels.
- **Testbed with a known answer.** A teacher–student MLP with planted structure. The labels come
  from a random teacher MLP of width $k$. Structure therefore grows with $k$ by construction.

## Library (`epimeter`)

- `codes/variational.py`: the variational code of Graves (2011) and Blier and Ollivier (2018).
  - Model part $\mathrm{KL}(q\Vert p)$ with a Gaussian mean-field posterior.
  - Data part $\mathbb E_q[-\log p(D\mid w)]$.
  - Both in bits, with the bits-back argument written in the docstring.
- `codes/laplace.py`: the log-evidence under a KFAC Laplace approximation (Ritter et al., 2018),
  via `laplace-torch`. The Occam factor is reported as the model part.
- The bridge to MDL that the BMM project description asks for: one interface returning
  (model part, data part) in bits for every code (Grünwald, MDL tutorial, Section 2.6.3).
- BMM algorithm activity: **variational coding and KFAC-Laplace evidence** (the "Bayesian-based
  codes" item of the project list).

## Theory item (R&D checkpoint 2)

1. **Evidence identity.** For an exact Bayesian predictive, the prequential code equals
   $-\log p(D)$ by the chain rule. Also
   $-\log p(D)=\mathbb E_q[-\log p(D\mid w)]+\mathrm{KL}(q\Vert p)-\mathrm{KL}(q\Vert p(w\mid D))$.
   State which term plays the role of $S_T$. State what the time bound of Definition 2 in the
   review removes from this identity.
2. **Noise.** Prove the statement of the hypothesis: the model part of the optimal variational
   code on labels independent of inputs is zero.
3. **WBIC.** Relate the WBIC term to the finite-time route of Ohzeki (2026), with Member 4.

## Tasks and dates

| Date | Deliverable |
|---|---|
| 06.10 | Variational code for an MLP on MNIST; paper skeleton in the arXiv style (`paper/`) |
| 13–19.10 (R&D CP2) | Theory items 1 and 2 written with proofs; this is the main theoretical result for the checkpoint |
| 20.10 (BMM TM2) | **Blog post and tech report**, draft versions; Laplace evidence working |
| 03.11 | Teacher–student generator (with Member 4); pilot of agreement with $S_T$ on two testbeds |
| 10.11 | Bayesian codes on all testbeds, 5 seeds; rank correlation with the $S_T$ of Members 1 and 2 |
| 17.11 | Figures frozen; **full paper draft v1** assembled from all sections |
| 24.11 (BMM TM3) | Blog post published (Habr or Medium); tech report final (3–5 pages, a short version of the paper) |
| 24–30.11 (R&D CP4) | Paper v2 after internal review |
| 01.12 | Cross-review of one other BMM team |
| 23.11, 05.12 | CPAL abstract registration; CPAL paper submission ([`venues.md`](venues.md)) |

## Paper sections

The theory section, the Bayesian experiment, the introduction and related work (condensed from the
literature review), and the final editing of all sections.

## Compute (estimate)

The variational code needs about 2–3 times one training run. Laplace needs one training run plus
one KFAC pass. The whole agreement study should need 10–20 T4-hours.
