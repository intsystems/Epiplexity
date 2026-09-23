# Follow-ups on epiplexity by eysu35 (Ellen Su): EpiSelect, EpiGen, and arXiv 2608.11746

Sources were fetched on 2026-09-23. I read the arXiv abstract page, the full HTML (v1), and the PDF text (21 pages). I cloned both GitHub repos and read every source and config file. I also queried the GitHub API, the Semantic Scholar API, the OpenReview search API and the author's homepage. Both repos have a single commit, so every file link below points at `main`. "Inference" marks my own derivation or reading of the code; "unverified" marks what I could not check.

Short URLs used below:
- Paper (abs): https://arxiv.org/abs/2608.11746 · HTML: https://arxiv.org/html/2608.11746v1 · PDF: https://arxiv.org/pdf/2608.11746v1
- EpiSelect: https://github.com/eysu35/EpiSelect · EpiGen: https://github.com/eysu35/EpiGen
- Founding paper: https://arxiv.org/html/2601.03220v1

---

## Q1. Which paper goes with which repo? Bibliographic facts

### Takeaway
One paper covers both repos. arXiv 2608.11746, "Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization" (Su\*, Potapczynski\*, Qiu, Hughes, Wilson; v1 12 Aug 2026), presents EpiSelect (Sec. 3) and EpiGen (Sec. 4) as two methods in the same paper. Neither repo has a separate paper. I found no venue, journal reference or OpenReview submission, and the paper had 0 citations as of 2026-09-23.

### Cited Findings
- **Title:** "Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization". **Authors:** Ellen Su, Andres Potapczynski, Shikai Qiu, Edward Hughes, Andrew Gordon Wilson. **Categories:** cs.LG (primary), cs.CL. Only v1 exists: Wed, 12 Aug 2026 07:38:58 UTC, 854 KB. The submitter is Ellen Su. — [arXiv abs](https://arxiv.org/abs/2608.11746)
- The abs page has no "Comments" or "Journal-ref" field. The only metadata fields are Subjects and submission history. I checked this in the page HTML. An LLM summary of the page claimed the code links were in a comments field; that is wrong. The code links are in the paper body, at Sec. 3.4 and Sec. 4.1. — [arXiv abs](https://arxiv.org/abs/2608.11746); [HTML](https://arxiv.org/html/2608.11746v1)
- **Affiliations:** Su, Potapczynski, Qiu and Wilson are at New York University; Hughes is at "Inherent". The PDF marks "∗Equal contribution" on Ellen Su and Andres Potapczynski. The paper licence is CC BY 4.0. The PDF is 21 pages. — [PDF](https://arxiv.org/pdf/2608.11746v1); [HTML](https://arxiv.org/html/2608.11746v1)
- The paper links each repo from its own section. Sec. 3.4 says "Code available at https://github.com/eysu35/EpiSelect" (data selection). Sec. 4.1 says "Code available at https://github.com/eysu35/EpiGen" (synthetic data generation). — [HTML](https://arxiv.org/html/2608.11746v1)
- Both READMEs carry the same arXiv badge and the same BibTeX (`@article{su2026epiplexity, ... journal = {arXiv preprint arXiv:2608.11746}, year = {2026}}`), and each calls itself "Code for" this paper. — [EpiSelect README](https://github.com/eysu35/EpiSelect/blob/main/README.md); [EpiGen README](https://github.com/eysu35/EpiGen/blob/main/README.md)
- Semantic Scholar has the paper (CorpusId 291033701), with an empty venue, publicationDate 2026-08-12, citationCount 0 and no citing papers. — [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2608.11746?fields=title,authors,venue,publicationDate,citationCount,externalIds,citations.title)
- An OpenReview full-text search for "epiplexity" returned `{"notes":[],"count":0}`. — [OpenReview API](https://api.openreview.net/notes/search?term=epiplexity)
- **Who eysu35 is:** the GitHub profile name is "Ellen Su" (account created 2021). Her homepage (dated Sep 12, 2026) says she is "a third year PhD student in the NYU Center for Data Science", advised by Todd Gureckis (Computation and Cognition Lab), and "a Visiting Researcher at Meta FAIR" mentored by Florian Bordes and Adina Williams under Meta's AI Mentorship (AIM) program. She has a BS in CS from Princeton. — [GitHub API user](https://api.github.com/users/eysu35); [homepage](https://eysu35.github.io/)
- The Acknowledgements list funding from DARPA AIQ HR00112590066, NSF CAREER IIS-2145492, NSF CDS&E-MSS 2134216, Google's TPU Research Cloud (TRC) and Lambda's Research Grant Program. Su is supported by the Meta AIM PhD Fellowship and Qiu by the Two Sigma PhD Fellowship. — [HTML](https://arxiv.org/html/2608.11746v1)
- eysu35 has 15 public repos. EpiSelect and EpiGen are the only epiplexity-related ones; both were created 2026-08-11. — [GitHub API repos](https://api.github.com/users/eysu35/repos)
- Press coverage: an AI Weekly item (8 Sep 2026) calls EpiSelect a "companion method" and repeats 0.394 vs 0.379 and r=0.88. — [AI Weekly](https://aiweekly.co/alerts/nyus-wilson-introduces-epiplexity-to-guide-ai-data-selection)

### Inferences
- "EpiSelect" and "EpiGen" are method names inside one paper. Anyone citing either should cite arXiv 2608.11746. Shikai Qiu and Andrew Gordon Wilson are also authors of the founding paper, so this is effectively a follow-up by the same lab.

### Gaps
- Google Scholar was not queried directly because it needs JavaScript or blocks scraping. The venue may therefore be missing if the paper was accepted somewhere after 12 Aug 2026 (for example a NeurIPS 2026 workshop). Unverified.
- The author's Publications page (eysu35.github.io/Publications) returned 404, so I could not check a self-reported venue.

---

## Q2. EpiSelect: what is scored, the estimator, scaling laws, baselines, data, models, results and cost

### Takeaway
EpiSelect scores whole data domains, not individual examples. During training it fits a cross-domain power-law scaling law to per-domain training-loss curves. It then differentiates a prequential-style epiplexity proxy, in which the floor is the current loss rather than the final loss, with respect to each domain's token count. Softmaxing that gradient (τ=1, with momentum) gives the domain sampling weights. On a token-capped Common Pile (30 or 31 domains, 15.7B training tokens), the average zero-shot score over 10 LM-Eval tasks is 0.394 for EpiSelect vs 0.379 for ADO vs 0.377 for Natural at 124M, and 0.431 vs 0.425 vs 0.422 at 1.3B. Each result is a single run. The scaling-law refits cost <5% (124M) and <1% (1.3B) of training time. No proxy model is needed.

### Cited Findings
**Motivating correlation (Sec. 3.1, App. B, Fig. 2 left, Fig. 8)**
- The authors trained separate 124M LLaMA-2-family models for 15B tokens on each of five Pile domains: arXiv, GitHub, PileCC, PubMed and Wikipedia. They then computed post-hoc prequential epiplexity (Eq. 1) and evaluated on 7 zero-shot LM-Eval tasks: ARC, HellaSwag, LogiQA, WinoGrande, LAMBADA, SciQ and PIQA. — [HTML Sec. 3.1, Fig. 2](https://arxiv.org/html/2608.11746v1)
- Epiplexity vs mean OOD accuracy: Pearson r=0.88 (two-tailed p≈0.05) and Spearman ρ=0.90. Weight norm vs accuracy: r=0.01 (p≈0.99), ρ=0.20. PileCC had the highest epiplexity and the best accuracy; ArXiv and GitHub were lowest on both. — [HTML App. B](https://arxiv.org/html/2608.11746v1)
- Only the 5 largest domains were used. The stated reason is that "smaller domains lack sufficient in-domain tokens to train without multiple epochs, which violates the prequential estimator's assumption in Equation 1 that credits code-length reduction to unseen tokens." — [HTML Sec. 3.1](https://arxiv.org/html/2608.11746v1)
- DoReMi and ADO "implicitly upsample data domains with higher epiplexity" on the Pile (Fig. 3). — [HTML Sec. 3.1, Fig. 3](https://arxiv.org/html/2608.11746v1)

**Estimator (Sec. 3.2, Eqs. 2–5, App. C–D, Alg. 1)**
- Eq. 2 gives the per-domain online epiplexity proxy: Ŝ(t) = Σ_{m=1}^{K} Ŝ_m(t) = Σ_m Σ_{s=1}^{t} (L_m(s) − L_m(t)). Here L_m(s) is domain m's loss at iteration s, and n_k^{(s)} is the cumulative count of domain-k tokens. — [HTML Sec. 3.2](https://arxiv.org/html/2608.11746v1)
- Eq. 3 is the cross-domain scaling law: L̂_m(n_1..n_K) = ε_m + β_m (Σ_k γ_{m,k} n_k)^{−α_m}. Here α_m, β_m > 0, ε_m is the irreducible loss, and γ_{m,k} > 0 with Σ_k γ_{m,k} = 1 is "the normalized influence of all the domains on domain m". Setting γ_{m,k} = 0 for k≠m recovers the single-domain (ADO-style) law. — [HTML Sec. 3.2](https://arxiv.org/html/2608.11746v1)
- Eq. 4 is the marginal epiplexity gain: ∂Ŝ/∂n_k = Σ_m [n_m α_m γ_{m,k} / Σ_i γ_{m,i} n_i] · (L̂_m(n) − ε_m). — [HTML Sec. 3.2](https://arxiv.org/html/2608.11746v1)
- App. D derives this as ∂Ŝ_k/∂n_k = Ŝ_k(n_k+1) − Ŝ_k(n_k) = −n_k ∂L̂_k/∂n_k. It treats S as a function of cumulative token counts rather than steps. — [HTML App. D](https://arxiv.org/html/2608.11746v1)
- Eq. 5 is the sampling rule: π_k ∝ exp((1/τ) ∂Ŝ(t)/∂n_k). τ→∞ gives uniform sampling and τ→0 gives greedy selection. "In our experiments, we fixed τ=1." — [HTML Sec. 3.2](https://arxiv.org/html/2608.11746v1)
- Algorithm 1: every ν steps, run FitScalingLaw on the (L^{(s)}, n^{(s)}) history. Then set π_k ∝ exp(∂Ŝ/∂n_k / τ) and apply momentum with clipping, π_k ← clip(ω π_k + (1−ω) π̄_k, δ_min). Sample the batch from π. The global mean is π̄_k ← (t/(t+1)) π̄_k + (1/(t+1)) π_k. Per-domain losses L_k^{(t)} are computed on the current batch before the optimizer update. — [HTML Alg. 1](https://arxiv.org/html/2608.11746v1)
- **Fitting (App. C):** minimize the Huber loss between observed and predicted per-domain losses, refitting every 1000 iterations. The parameters are α, log β, log ε and log γ; γ rows are mapped onto the simplex with a softmax. The init grid is α∈{0.0,…,0.8}, log β∈{−2,…,6} and log ε∈{−2.0,−1.5,…,2.0}, which gives 729 points. γ is initialized from Dir(p) with p_m=10 on the diagonal and p_k=1 elsewhere. Each fit adds about 12 s, compared with 3 s for ADO, which fits no γ. That is "less than 5% of the training time for our 124M model and less than 1% for the 1.3B model." — [HTML App. C](https://arxiv.org/html/2608.11746v1)
- **Fit quality (App. C.1, Fig. 9):** R² reaches 0.9 within the first 4k steps. At the final step the median R² across the 30 domains is 0.88, with median log-RMSE 0.02 nats (predictions "within roughly 2%"). R² is lower for DM Mathematics, Enron Emails and Ubuntu IRC, which have "near-flat, noisy loss curves". — [HTML App. C.1](https://arxiv.org/html/2608.11746v1)
- **ADO for comparison (App. C.2, Eq. 8):** π_k ∝ μ_k (α_k/n_k)(L̂_k(n_k) − ε_k) λ_k, where μ_k is the prior and λ_k is a "heuristic credit score". — [HTML App. C.2](https://arxiv.org/html/2608.11746v1)

**Code-level implementation (repo)**
- The README summarizes: "fits a cross-domain scaling law `L_m = beta_m (sum_k gamma_mk n_k)^-alpha_m + eps_m` and reweights batches toward the domains with the largest marginal contribution to epiplexity, refitting every 1,000 steps. No proxy model required. Adapted from the ADO reference implementation (github.com/yidingjiang/ado)." — [EpiSelect README](https://github.com/eysu35/EpiSelect/blob/main/README.md)
- `epiplexity_grads(tokens, params)` implements Eq. 4 exactly. It computes `dS = alpha*(loss-eps)`, `num = gamma*tokens` (γ_{m,k}·n_m) and `denom = Σ_k γ_{m,k} n_k`, then sums over m. `tests/test_scaling_law.py` checks it against a loop-based reference implementation to within 1e-7. — [scaling_law.py](https://github.com/eysu35/EpiSelect/blob/main/src/scaling_law.py); [test_scaling_law.py](https://github.com/eysu35/EpiSelect/blob/main/tests/test_scaling_law.py)
- `fit_scaling_law` fits all K domains jointly (K(K+3) parameters). It runs jaxopt `LBFGS(tol=1e-5, maxiter=200)` from 729 inits in parallel via `vmap`, sharded over a ("replica","data") mesh, and keeps the lowest objective (`nanargmin`). The objective is Huber (δ=1e-3) on **log** loss plus barrier penalties: α ≤ 0.8, α ≥ 0, log β ≤ 6.5, and `log_eps_min = -min(log_eps - 0.5, 0)`. The gamma grid uses a single Dirichlet draw (`gamma_grid_n = 1`, tilt 10). — [scaling_law.py](https://github.com/eysu35/EpiSelect/blob/main/src/scaling_law.py)
- `EpiSelect.curr_dist` computes weights = `exp(epiplexity_grads(D, params)/tau)`, then normalizes and optionally multiplies by the prior and the credit term (both off for EpiSelect). It then sets `p = clip_min_probability(ema*p_avg + (1-ema)*dist, 1e-8)` and updates the global average `p_avg`. Before training reaches the refit start step, the loss curves are smoothed with a Savitzky–Golay filter and subsampled every `fit_freq` steps. — [episelect.py](https://github.com/eysu35/EpiSelect/blob/main/src/data_selectors/episelect.py)
- EpiSelect loader config: `ignore_steps: 1000`, `start_step: 1000`, `update_interval: 1000`, `use_prior: false`, `use_credit_assignment: false`, `do_global_avg: true`, `ema: 0.9`, `tau: 1.0`, `fit_freq: 20`, `window_length: 101`, `polyorder: 3`. The fit target defaults to `y_axis="train_loss"`, with `val_loss` as an option. — [commonpile_episelect.yaml](https://github.com/eysu35/EpiSelect/blob/main/configs/loader/commonpile_episelect.yaml); [adaptive_loader.py](https://github.com/eysu35/EpiSelect/blob/main/src/dataloader/adaptive_loader.py)
- The ADO baseline in the repo computes slope = max(α, 0.05)·(L̂ − ε)/n_k (`divide_by_n=True`), multiplies by `p_history**0.5` (credit term) and by the empirical prior, clips the minimum probability to 1e-2, and applies the same 0.9 EMA. — [ado.py](https://github.com/eysu35/EpiSelect/blob/main/src/data_selectors/ado.py)
- The code also logs a running prequential estimate per domain, `train_epi_K = Σ_s c_k(s)·L_k(s) − L_k(t)·Σ_s c_k(s)`, where c_k(s) is the number of domain-k sequences in step s's batch and L is the per-sequence mean-token training loss measured before the update. A held-out version, `val_epi_K`, uses validation loss weighted by the counts seen between eval points. These are metrics only; selection uses the scaling-law gradient. — [ado.py `compute_epiplexity`, `update_val_loss`](https://github.com/eysu35/EpiSelect/blob/main/src/data_selectors/ado.py)

**Pile saturation (Sec. 3.3, Fig. 4)**
- On uncopyrighted Pile (~210B tokens, 17 domains; Books3, BookCorpus2, OpenSubtitles, YTSubtitles and OWT2 excluded), PileCC is 41% of tokens. "Training only on the PileCC domain leads to better performance than the state-of-the-art (SOTA) data selection method" (ADO). The authors conclude that Pile is "uninformative as a benchmark for distinguishing between selection strategies" and move to Common Pile (8 TB public-domain text, 30 sources, largest domain Stack V2 ≈ 14%). — [HTML Sec. 1, 3.3](https://arxiv.org/html/2608.11746v1)

**Main experiment setup (Sec. 3.4, App. A, E.1)**
- **Models:** decoder-only LLaMA-2-family transformers, implemented in JAX with Equinox and Optax. 124M = 12 layers, 12 heads, d=768. 1.3B = 24 layers, 16 heads, d=2048. Both use SwiGLU, RMSNorm and RoPE, the GPT-NeoX-20B tokenizer (50,304 vocab) and context 1024. Parameters are stored in fp32 and forward passes run in bf16. — [HTML App. E.1](https://arxiv.org/html/2608.11746v1)
- **Training:** 60,000 steps at a global batch of 256 sequences (262,144 tokens/step, ≈15.7B tokens). AdamW with peak LR 1e-3, β2=0.95, weight decay 1e-4, 500-step warmup and cosine decay to 1e-5. Selection weights are updated every 1,000 steps, after an initial 1,000 steps on the natural distribution. — [HTML App. E.1](https://arxiv.org/html/2608.11746v1); [configs/main.yaml](https://github.com/eysu35/EpiSelect/blob/main/configs/main.yaml)
- **Data (paper, App. A):** up to 2B tokens were streamed for each of the 6 largest of 30 domains, and all tokens for the 24 smaller ones, for a total of 29.7B tokens. The "Natural" baseline samples in proportion to domain size under this construction. — [HTML App. A](https://arxiv.org/html/2608.11746v1)
- **Data (repo):** "Streams the 31 `common-pile/*_filtered` domains and tokenizes with GPT-NeoX-20B — 43.3B tokens, ~95 GB, capped at 2B tokens per domain". `TOKEN_CAP = 2_000_000_000` and `VAL_FRACTION = 0.01`, and the `TASKS` list has 31 entries. — [README](https://github.com/eysu35/EpiSelect/blob/main/README.md); [common_pile_grouped.py](https://github.com/eysu35/EpiSelect/blob/main/src/tfds/common_pile_grouped.py)
- **Baselines:** Natural (no selection) and ADO. DoReMi is omitted because it requires "trained smaller proxy models, incur[s] larger costs, and underperform[s] ADO". There is no perplexity filtering, RHO-loss, DSIR or dedup baseline. — [HTML Sec. 3.4](https://arxiv.org/html/2608.11746v1)

**Results (Table 1): zero-shot accuracy, 10 LM-Eval tasks**

| Model | Method | ARC | BBQ | BoolQ | CSQA | HSwag | LAM | OBQA | PIQA | SciQ | WinoG | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 124M | Natural | .199 | .352 | .529 | .194 | .279 | .263 | .134 | .597 | .708 | .519 | .377 |
| 124M | ADO | .206 | .365 | .493 | .197 | .277 | .252 | .146 | .596 | .734 | .524 | .379 |
| 124M | EpiSelect | .212 | .409 | .587 | .210 | .273 | .260 | .156 | .595 | .707 | .529 | .394 |
| 1.3B | Natural | .338 | .332 | .590 | .196 | .306 | .404 | .196 | .604 | .745 | .514 | .422 |
| 1.3B | ADO | .344 | .327 | .586 | .192 | .307 | .351 | .200 | .623 | .790 | .526 | .425 |
| 1.3B | EpiSelect | .352 | .356 | .599 | .209 | .315 | .404 | .184 | .613 | .772 | .511 | .431 |

— [HTML Table 1](https://arxiv.org/html/2608.11746v1)
- EpiSelect attains "the top score on 6 of 10 tasks for both model sizes". The paper also says: "Our improvement over ADO exceeds previously reported improvements over baselines [ADO paper]". — [HTML Sec. 3.4, Table 1 caption](https://arxiv.org/html/2608.11746v1)
- Fig. 2 (middle) states that EpiSelect "achieves higher epiplexity than alternatives" (Uniform/Baseline and ADO) on Common Pile. — [HTML Fig. 2](https://arxiv.org/html/2608.11746v1)
- **Cross-domain γ (Fig. 5):** the fitted γ matrix at the end of training is "diagonally dominant", with "a sparse set of strong cross-domain effects" that are asymmetric. — [HTML Sec. 3.4](https://arxiv.org/html/2608.11746v1)
- **Compute:** "All of our training was done on a single Cloud TPU v4-32 slice (4 host VMs, each with 8 TPU v4 chips)", with data parallelism of 8 sequences per chip. Wall-clock time is ≈8 h per 60k-step run for 124M and ≈29 h for 1.3B. — [HTML App. E.1](https://arxiv.org/html/2608.11746v1)
- Seeds in the configs: 124M runs use `seed: 21` and 1.3B runs use `seed: 0`, with one config per method. — [configs/experiment](https://github.com/eysu35/EpiSelect/tree/main/configs/experiment)

### Inferences
- **EpiSelect vs ADO.** With γ = I, Eq. 4 reduces to ∂Ŝ/∂n_k = α_k(L̂_k − ε_k). That is exactly ADO's per-domain slope α_k(L̂_k − ε_k)/n_k multiplied by n_k. The "epiplexity gain" is n times the marginal rate of loss decrease, which is App. D's −n ∂L/∂n. The operational differences from ADO are therefore: (i) no 1/n_k, so large, already-seen domains are not penalised; (ii) no prior μ_k and no credit term λ_k; (iii) learned cross-domain γ; (iv) exponential (softmax) rather than linear normalisation; (v) a minimum-probability clip of 1e-8 instead of 1e-2. This is derived from Eq. 4, Eq. 8 and the two selector files.
- **Softness of the reweighting.** The fit bounds α ≤ 0.8, and L̂ − ε is a few nats at most. When γ is diagonally dominant, the pre-softmax scores are therefore O(0.1–1), so exp(·) at τ=1 changes the relative weights across domains by at most a factor of about e. After the 0.9 EMA toward the global mean, the resulting reweighting is quite soft. This is my inference from the code; the paper reports no π trajectories numerically.
- **Where the gains come from.** Most of the 124M improvement over ADO (+1.5 pts average) comes from BBQ (+4.4) and BoolQ (+9.4). Excluding those two tasks, the averages are 0.3678 (EpiSelect) vs 0.3665 (ADO) vs 0.3616 (Natural). At 1.3B, excluding them gives 0.420 vs 0.417 vs 0.413. BoolQ scores of 0.49–0.59 are near or below a majority-class baseline (unverified here, but BoolQ is known to be label-imbalanced). With single seeds and no error bars, the average gains of 0.6–1.7 points should be treated as preliminary.
- **Estimator mismatch.** Eq. 2 sums unweighted over steps. The logged code metric weights each step by the domain's sample count, which matches the per-token prequential sum in Eq. 1. The selection signal uses neither directly: it uses the analytic derivative of the fitted law.
- **Internal inconsistency in App. C.1.** It reports "median R² across the 30 domains" (Common Pile) but names DM Mathematics and Enron Emails. Those are Pile domains and do not appear in the repo's 31-domain Common Pile list. The fit-quality analysis may therefore come from a Pile run. Unverified.
- **Barrier on ε.** The code comment reads "eps >= 0.5", but the barrier acts on log ε ≥ 0.5, which means ε ≥ e^0.5 ≈ 1.65 nats. This is my reading of `_fit_objective`. It is harmless for LMs but matters if the fitter is reused on low-loss tasks such as MLP classification, where the irreducible loss may be far below 1.65 nats.

### Gaps
- The numeric values behind Fig. 2 (epiplexity magnitudes), Fig. 3, Fig. 4 (PileCC-only vs ADO accuracy) and Fig. 5 are only in images and could not be extracted from the text.
- There is a domain-count and data-size discrepancy: the paper says 30 domains and 29.7B tokens, while the repo says 31 domains and 43.3B tokens. Which build was used for Table 1 is unresolved.
- "ARC" in Table 1 is ambiguous. The eval config lists both `arc_challenge` and `arc_easy`, making 11 harness tasks for 10 table columns ([eval.yaml](https://github.com/eysu35/EpiSelect/blob/main/configs/eval.yaml)).
- The repo does not contain the Pile experiments (single-domain runs, DoReMi, the PileCC-only run); it only builds Common Pile.
- No multi-seed variance, total TPU-hours or per-example scoring is reported. Baselines such as RHO-loss, DSIR and perplexity filtering are not compared.

---

## Q3. EpiGen: method, reward/estimator, models, results and compute

### Takeaway
EpiGen fine-tunes a pretrained GPT-2 **generator** with REINFORCE. The reward for each generated batch is the drop in a GPT-2 **learner's** mean loss over a buffer of all past generated batches (16 sampled batches plus the current one) after K=10 learner steps on the new batch. The paper frames this as the step-wise change in prequential epiplexity. The learner trained on EpiGen data scores 0.770 average on 7 GLUE tasks, vs 0.743 for pretrained GPT-2 and 0.759 for a frozen generator. The paper's compute is 4× L40S for ≈10 h per run. The gain depends on pretrained initialization: from random init the change is −0.003.

### Cited Findings
**Method (Sec. 4, Eqs. 6–7, Alg. 2)**
- Eq. 6 is the reward: r_t = S(t) − S(t−1) = Σ_{x∈𝓑} (log 1/P_{θ_{t−1}}(x) − log 1/P_{θ_t}(x)). Here 𝓑 is the generation buffer, which includes the current batch 𝒳_t. The buffer "is not used to train either model; instead, it serves as a reference distribution for reward computation and provides a tractable estimate of epiplexity." — [HTML Sec. 4](https://arxiv.org/html/2608.11746v1)
- Eq. 7 is the gradient: ∇_{θ^g} 𝒥 = E_{𝒳_t∼P_{θ^g}}[(r_t − b)·Σ_{x∈𝒳_t} ∇ log P_{θ^g}(x)], with an EMA baseline b ← βb + (1−β)r_t. — [HTML Sec. 4](https://arxiv.org/html/2608.11746v1)
- Algorithm 2: θ^g and θ both start from θ_0. Each iteration: sample B sequences i.i.d. from P_{θ^g}(·|τ); subsample M items from the buffer and add 𝒳_t; compute L^pre as the mean learner loss on that set; take K learner SGD steps on 𝒳_t; compute L^post; set r_t = L^pre − L^post; update the generator; update b; append 𝒳_t to 𝓑. — [HTML Alg. 2](https://arxiv.org/html/2608.11746v1)
- The stated rationale: "Memorizing a particular batch 𝒳_t of noise data would not reduce learner loss on the remainder of 𝓑", so the buffer "may partially regularize the generator model against mode collapse". — [HTML Sec. 4.1](https://arxiv.org/html/2608.11746v1)

**Setup (Sec. 4.1, App. E.2)**
- **Models:** GPT-2 small (117M parameters, 12 layers, 12 heads, d=768) via HuggingFace, for both generator and learner. A frozen copy serves as the reference for perplexity. GPT-2 tokenizer (50,257). PyTorch with gradient checkpointing. — [HTML App. E.2](https://arxiv.org/html/2608.11746v1)
- **Hyperparameters:** T=6,000 generator steps and K=10 learner steps per generator step, giving 60,000 learner updates. Batch of 32 sequences × 512 tokens at τ=1.0. 16 buffer batches are sampled uniformly and combined with the current batch. Gradient-norm clipping 1.0; EMA β=0.99. Learner: AdamW, lr 1e-5, weight decay 1e-2. Generator: AdamW, lr 1e-6, weight decay 1e-2. Validation loss is computed every 1000 learner steps on 500K OWT tokens. — [HTML App. E.2](https://arxiv.org/html/2608.11746v1); [epigen.yaml](https://github.com/eysu35/EpiGen/blob/main/configs/glue_table/epigen.yaml)
- **Baselines:** Pretrained (GPT-2); FrozenGen (frozen generator, matched token count); PPL (generator rewarded by "negative log perplexity" under frozen pretrained GPT-2); NoBuffer (learning-based reward on the current batch only). — [HTML Sec. 4.1](https://arxiv.org/html/2608.11746v1)
- **Evaluation:** full fine-tuning plus a linear head on 7 GLUE tasks (CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE), 3 epochs, lr 2e-5, batch 32, scored on the dev set. CoLA uses MCC; the others use accuracy. — [HTML Sec. 4.1](https://arxiv.org/html/2608.11746v1). The code uses `GPT2ForSequenceClassification`, `max_seq_len` 128 and 10% linear warmup then linear decay. — [datagen/eval.py](https://github.com/eysu35/EpiGen/blob/main/datagen/eval.py); [configs/eval.yaml](https://github.com/eysu35/EpiGen/blob/main/configs/eval.yaml)

**Results**
- Table 2 (GLUE):

| Method | CoLA | SST-2 | MRPC | QQP | MNLI | QNLI | RTE | Avg |
|---|---|---|---|---|---|---|---|---|
| Pretrained | .264 | .930 | .779 | .891 | .814 | .884 | .639 | .743 |
| FrozenGen | .388 | .924 | .774 | .893 | .817 | .882 | .632 | .759 |
| PPL | .380 | .911 | .770 | .892 | .815 | .882 | .643 | .756 |
| NoBuffer | .367 | .919 | .767 | .892 | .815 | .881 | .661 | .757 |
| EpiGen | .422 | .920 | .777 | .892 | .817 | .886 | .675 | .770 |

"Epiplexity-guided synthetic data training yields a 2.7-point average improvement … over the Pretrained baseline." — [HTML Table 2](https://arxiv.org/html/2608.11746v1)
- Table 3 (perplexity on 5M-token slices; lower is better): Pretrained 24.64 (OWT) / 29.80 (SlimPajama) / 35.12 (FineWeb); FrozenGen 26.49 / 31.14 / 37.57; PPL 104.20 / 106.81 / 171.55; NoBuffer 33.10 / 38.60 / 48.43; EpiGen 27.58 / 33.09 / 39.74. "No method improves over Pretrained perplexity." — [HTML App. G, Table 3](https://arxiv.org/html/2608.11746v1)
- Initialization ablation (Fig. 6): with pretrained weights, EpiGen raises the GLUE average from 0.743 to 0.770. From random initialization the change is "essentially no change (−0.003)". "Having a pre-trained generator proves necessary for RL to discover useful synthetic data." — [HTML Sec. 4.1](https://arxiv.org/html/2608.11746v1)
- Mixture (Fig. 7): with the trained generator frozen, learners trained on "50% and 25% of synthetic data (and, subsequently, 50% and 75% of OWT data) achieve higher average performance on the OOD GLUE benchmark". — [HTML Sec. 4.1](https://arxiv.org/html/2608.11746v1)
- Buffer grounding (App. G.1, Fig. 11): the authors mixed 0/25/50/100% OWT into the **evaluation buffer**. OWT perplexity falls monotonically, and the average over 15 LM-Eval zero-shot tasks improves beyond pretrained GPT-2. Compared with learners trained on the same number of real OWT tokens, EpiGen learners perform "comparably", which the authors read as showing "the synthetic data from the generator is as informative as real natural web text." — [HTML App. G.1](https://arxiv.org/html/2608.11746v1)
- Training dynamics (App. F, Fig. 10): the reward "rises steadily over training and remains positive". Learner loss on generated data falls for EpiGen and stays flat for FrozenGen. The paper also says "the variance of our REINFORCE gradients is not high". — [HTML Sec. 4.1, App. F](https://arxiv.org/html/2608.11746v1)
- Samples (App. H): the text stays web-like (news, Q&A, blog) from step 1,000 to step 60,000, and the authors describe it as "stylistically plausible" but possibly "factually hallucinated". — [HTML App. H](https://arxiv.org/html/2608.11746v1)
- **Compute (paper):** "4 NVIDIA L40S GPUs with PyTorch DDP. Wall-clock training time is approximately 10 hours per 60,000-step run." — [HTML App. E.2](https://arxiv.org/html/2608.11746v1)
- **Compute (repo):** "Roughly 24 h per run on 2 GPUs". `train.slurm` requests `--gres=gpu:2`, `--time=24:00:00` and 64 GB RAM, and the eval job requests 1 GPU. There are 8 runs in total. — [EpiGen README](https://github.com/eysu35/EpiGen/blob/main/README.md); [train.slurm](https://github.com/eysu35/EpiGen/blob/main/scripts/train.slurm)

**Code-level implementation (repo)**
- **Reward is computed on a lookahead copy.** `_loss_drop` deep-copies the learner and creates a **fresh** AdamW(lr_learner, wd 0.01). It measures the mean loss over the eval batches, takes S steps on the generated tokens, measures again and returns `loss_before - loss_after`. The copy is then discarded. The real learner is trained separately afterwards, with S steps using its own persistent optimizer. — [reward.py](https://github.com/eysu35/EpiGen/blob/main/datagen/reward.py); [train.py](https://github.com/eysu35/EpiGen/blob/main/datagen/train.py)
- Losses are `F.cross_entropy(..., reduction="mean")`, averaged over batches, so r_t is in nats per token rather than the sum over x∈𝓑 in Eq. 6. — [reward.py](https://github.com/eysu35/EpiGen/blob/main/datagen/reward.py)
- REINFORCE step: `(-advantage * token_log_probs.mean()).backward()`. One scalar advantage is shared by all tokens and sequences in the batch, followed by gradient clipping at 1.0. Generation samples 512 tokens autoregressively with a KV cache, starting from the EOS token (empty prompt). — [train.py](https://github.com/eysu35/EpiGen/blob/main/datagen/train.py)
- Only the generator is wrapped in DDP. The learner is an unwrapped model on each rank that trains on that rank's local batch of `batch_size/world_size` sequences. Each rank keeps its own buffer (`sample_from_buffer // world_size` batches). The reward is local, while the baseline is all-reduced. Rank 0's learner is the one saved. — [train.py](https://github.com/eysu35/EpiGen/blob/main/datagen/train.py)
- The buffer is an unbounded `deque` of detached token tensors, kept on the device. — [buffer.py](https://github.com/eysu35/EpiGen/blob/main/datagen/buffer.py)
- Logged `learner_epi` = Σ_t B·L_t − L_t·Σ_t B, the same current-floor prequential form as in EpiSelect. It uses the loss from the last of the S learner steps. — [train.py](https://github.com/eysu35/EpiGen/blob/main/datagen/train.py)
- Reward modes are `epiplexity`, `no_buffer` and `pretrained_ppl`. The code also has `real_sample_n` and `eval_real_only` options for adding OWT to the evaluation buffer (the App. G.1 ablation), but no config file for them is provided. — [train.py](https://github.com/eysu35/EpiGen/blob/main/datagen/train.py); [reward.py](https://github.com/eysu35/EpiGen/blob/main/datagen/reward.py)
- The mixture configs set `real_frac` to 0.25, 0.5 and 1.0 (the fraction of the learner batch replaced by OWT). They use `real_train_tokens: 100_000_000` and `eval_offset: 200_000_000`, and the generator is frozen at `out/epigen/generator`. — [mixture_25p.yaml](https://github.com/eysu35/EpiGen/blob/main/configs/mixture/mixture_25p.yaml)

### Inferences
- **The GLUE gain is concentrated in two small tasks.** CoLA (+15.8 MCC points) alone accounts for about 2.26 of the 2.7-point average gain over Pretrained, and RTE adds +3.6 points. Without CoLA and RTE, EpiGen averages 0.858 vs Pretrained 0.860. Against FrozenGen (+1.1 points average), the whole gap is CoLA (+3.4) and RTE (+4.3). Both are small, high-variance GLUE tasks, and the paper uses a single seed (`seed: 42`). The EpiGen-specific effect beyond "more in-distribution tokens" therefore rests on a small margin. Computed from Table 2.
- **Code differs from the algorithm as written.** Algorithm 2 implies the reward comes from the actual learner update. In the code it comes from a lookahead copy with a fresh Adam state, and the first Adam steps with fresh moments behave roughly like sign-SGD. The reward therefore measures a slightly different update from the one the learner actually receives. Per-rank learners also mean the learner's effective batch is 32/world_size, not 32. Both points come from reading the code, not from running it.
- **Cost per iteration.** Each iteration costs about 2× the learner training (lookahead copy plus real steps), 2×(M+1) forward evaluations on the buffer and 512-step autoregressive generation. That is roughly 3–4× the cost of training the learner on 60k steps of real data. This is an order-of-magnitude estimate from the code structure.
- **Mixture discrepancy.** The paper's Fig. 7 text mentions 25% synthetic / 75% OWT, but no config gives 75% real. The configs give 25%, 50% and 100% real, which is 75%, 50% and 0% synthetic. Either the paper's wording or the config names are inverted, or a 75%-real config is missing.
- **Credit assignment is per batch.** The reward is a per-batch scalar. EpiGen's "epiplexity" signal is therefore the aggregate loss drop from a batch, not a per-sample epiplexity.

### Gaps
- The numeric values of Fig. 6 (beyond 0.743, 0.770 and −0.003), Fig. 7, Fig. 10 and Fig. 11 are image-only.
- The compute figures conflict: the paper says 4× L40S for ~10 h, the README says ~24 h on 2 GPUs, and the README's GPU model is unspecified.
- The eval config lists 8 harness tasks, while App. G.1 reports 15. The 15-task config and the OWT-in-buffer configs are not in the repo.
- The README labels the GLUE experiment "Table 1", but in the paper it is Table 2. This is minor.

---

## Q4. Code structure, frameworks, dependencies, entry points, compute, licence and dates

### Takeaway
EpiSelect is a JAX/Equinox/Optax + Hydra + TFDS codebase. It is built for multi-host TPU pods, with a CPU install path, and is adapted from ADO. EpiGen is a small PyTorch + HuggingFace + Hydra codebase for SLURM/DDP. Both are MIT-licensed, have a single "init" commit, contain no notebooks or toy-scale examples, and have no issues.

### Cited Findings
- **EpiSelect metadata:** created 2026-08-11T15:33:24Z, last push 2026-08-25T05:53:28Z, 1 star, language Python, MIT licence ("Copyright (c) 2026 The EpiSelect Authors"). There is one commit, `c45889c` "init", dated 2026-08-25 01:53 −0400. Issues: none. — [GitHub API](https://api.github.com/repos/eysu35/EpiSelect); [commits](https://github.com/eysu35/EpiSelect/commits/main); [LICENSE](https://github.com/eysu35/EpiSelect/blob/main/LICENSE)
- **EpiGen metadata:** created 2026-08-11T17:34:42Z, last push 2026-08-13T22:41:57Z, 1 star, MIT licence ("Copyright (c) 2026 Ellen Su"). There is one commit, `7555ccb` "init", dated 2026-08-13 18:40 −0400. Issues: none. — [GitHub API](https://api.github.com/repos/eysu35/EpiGen); [commits](https://github.com/eysu35/EpiGen/commits/main); [LICENSE](https://github.com/eysu35/EpiGen/blob/main/LICENSE)
- **EpiSelect layout:** `launch.py` is the Hydra entry point (`python launch.py +experiment=124M/EpiSelect loader.data_dir=... rundir=... multihost=false`). `src/train.py` holds the training loop (Orbax checkpoints, W&B logging). `src/scaling_law.py` holds the cross-domain law fit and Eq. 4. `src/data_selectors/{episelect,ado}.py` hold the selectors. `src/dataloader/{adaptive_loader,commonpile_loader}.py` and `src/tfds/common_pile_grouped.py` handle data. `src/model.py` and `src/layers.py` define the model (Equinox GPT with SwiGLU, RoPE and RMSNorm). `src/eval.py` and `src/harness/` run lm-eval, C4, SlimPajama and FineWeb evaluation. `scripts/tpu_{setup,launch,check,stop}.sh` manage TPU pods, `scripts/build_commonpile_tfds.py` builds the data, and `tests/test_scaling_law.py` holds the tests. — [EpiSelect repo](https://github.com/eysu35/EpiSelect); [README](https://github.com/eysu35/EpiSelect/blob/main/README.md)
- **EpiSelect dependencies (pyproject):** datasets==2.19.2, equinox==0.13.6, gcsfs, hydra-core>=1.3.2, jaxopt==0.8.3, lm-eval==0.4.11, optax==0.1.7, orbax-checkpoint==0.5.20, scipy, tensorflow>=2.21.0, tensorflow-datasets>=4.9.9, transformers>=5.7.0 and wandb. The optional TPU extra is jax[tpu]>=0.6.2. It requires Python >=3.10, and `.python-version` pins 3.12.3. `requirements.sh` installs "CPU jax (pulled in by equinox/optax)" by default. — [pyproject.toml](https://github.com/eysu35/EpiSelect/blob/main/pyproject.toml); [requirements.sh](https://github.com/eysu35/EpiSelect/blob/main/requirements.sh)
- **EpiSelect experiments:** six experiments, `{124M, 1.3B} × {Natural, ADO, EpiSelect}`. Data must be pre-built to TFDS (~95 GB, 43.3B tokens), and the scripts assume GCS buckets. — [README](https://github.com/eysu35/EpiSelect/blob/main/README.md)
- **EpiGen layout:** `datagen/train.py` (REINFORCE loop, generation, real-data mixing), `datagen/reward.py` (epiplexity, no-buffer and pretrained-ppl rewards), `datagen/buffer.py` and `datagen/eval.py` (GLUE fine-tuning, perplexity, harness). Configs are `configs/{glue_table,random_init,mixture}/*.yaml` and `configs/eval.yaml`, with scripts `scripts/{train.slurm,eval.slurm,launch_all.sh}`. The entry point is `torchrun --nproc_per_node=2 -m datagen.train --config-path ../configs/glue_table --config-name epigen`. There are eight runs. OWT is streamed from HuggingFace on first use and cached. Any key can be overridden, for example `H=100 no_wandb=true`. — [EpiGen README](https://github.com/eysu35/EpiGen/blob/main/README.md)
- **EpiGen dependencies:** torch>=2.4, transformers>=4.44, datasets>=2.19, hydra-core>=1.3, omegaconf>=2.3, numpy>=1.26, wandb and lm-eval>=0.4. Setup uses Python 3.10 via conda. — [requirements.txt](https://github.com/eysu35/EpiGen/blob/main/requirements.txt); [README](https://github.com/eysu35/EpiGen/blob/main/README.md)
- **Compute summary:** EpiSelect ran on a TPU v4-32 slice, ≈8 h per run (124M) and ≈29 h per run (1.3B). EpiGen ran on 4× L40S for ≈10 h per run according to the paper, or ~24 h on 2 GPUs according to the README. — [HTML App. E](https://arxiv.org/html/2608.11746v1); [EpiGen README](https://github.com/eysu35/EpiGen/blob/main/README.md)

### Inferences
- Adding up Table 1, EpiSelect alone used about 3×8 + 3×29 ≈ 111 v4-32 slice-hours, before the Pile single-domain and baseline runs. EpiGen's 8 runs come to roughly 320–390 GPU-hours (8 × 40–48 GPU-h). Neither is feasible on a single Colab GPU at the published scale. These are my estimates.
- EpiSelect's single commit postdates arXiv v1 by 13 days, and the repo was created on 11 Aug. The public code is therefore a post-submission cleanup, which may explain the 30-vs-31-domain mismatch. Unverified.
- `jaxopt` is pinned at 0.8.3. My understanding is that jaxopt is in maintenance mode and its optimizers are moving into optax (unverified here), so pin carefully when reusing it.

### Gaps
- Neither repo includes notebooks, small or toy configs, pretrained checkpoints or logged results.
- There is no CI, and nothing shows the tests were run.

---

## Q5. Claims that refine, contradict or extend the founding paper (arXiv 2601.03220)

### Takeaway
The follow-up keeps the founding paper's prequential view, "area under the loss curve above the final loss", but changes the floor. It replaces the final-model loss with the **current** training loss so the quantity can be optimised online. It also drops the compute-optimal search over model size and tokens and works at a fixed model and fixed token budget. It adds two tractable proxies: an analytic per-domain derivative from a fitted cross-domain scaling law, and a lookahead loss drop over a buffer. It also offers the first direct test of the founding paper's OOD-generalisation hypothesis (r=0.88 over 5 Pile domains). No per-example epiplexity is defined.

### Cited Findings
- **Founding paper, prequential estimator:** "area under the loss curve above the final loss"; |P_preq| ≈ Σ_{i=0}^{M−1} (log 1/P_i(Z_i) − log 1/P_M(Z_i)) (Sec. 4.1, Eq. 8). — [2601.03220 HTML](https://arxiv.org/html/2601.03220v1)
- **Founding paper, requential estimator:** |P_req| ≈ Σ_i KL(P_i^t ∥ P_i^s) (Sec. 4.2, Eq. 9). — [2601.03220 HTML](https://arxiv.org/html/2601.03220v1)
- **Founding paper, compute bound:** "We optimize the training hyperparameters (e.g., learning rate) and the trade-off between N and D subject to the time bound 6ND+2N𝒟≤T to find the optimal P⋆ that minimizes the two-part code" (Sec. 4.1). — [2601.03220 HTML](https://arxiv.org/html/2601.03220v1)
- **Founding paper, on OOD generalisation:** it motivates "selecting high epiplexity data that induces more structural information in the model, since these structures can then be reused for unseen out-of-distribution (OOD) tasks". It also cautions that "epiplexity is a measure of information, *not* a guarantee of OOD generalization to specific tasks" (Sec. 1). The fetch summary found no per-sample epiplexity notion. — [2601.03220 HTML](https://arxiv.org/html/2601.03220v1)
- **Follow-up's framing:** Finzi et al. "contends that data with higher epiplexity should lead to better OOD generalization … this hypothesis is largely untested". Finzi et al. showed ADO "can *inadvertently* select for higher epiplexity data (relative to random batches), but neither develops a methodology that directly uses epiplexity for selection". — [2608.11746 HTML Sec. 1](https://arxiv.org/html/2608.11746v1)
- **The modification:** "we introduce two novel estimation approaches, both of which replace the final loss with the current training loss, toward greedily maximizing epiplexity gain throughout training." The follow-up also calls these "the first practically tractable estimators for epiplexity" built on the prequential approximation. — [2608.11746 HTML Sec. 2](https://arxiv.org/html/2608.11746v1)
- **Assumption made explicit:** the prequential estimator "credits code-length reduction to unseen tokens", so multi-epoch training violates it. — [2608.11746 HTML Sec. 3.1](https://arxiv.org/html/2608.11746v1)
- **Stated limitation:** "we relied on a tractable proxy for epiplexity rather than the quantity itself, and we leave to future work the theoretical guarantees that maximizing for this proxy during training maximizes the true epiplexity." The experiments are also limited to small models. — [2608.11746 HTML Sec. 5](https://arxiv.org/html/2608.11746v1)
- **New empirical claims:**
  - Epiplexity correlates with OOD accuracy across Pile domains (r=0.88, ρ=0.90, n=5), while weight norm does not.
  - DoReMi and ADO implicitly upsample high-epiplexity domains.
  - A PileCC-only model beats ADO on the Pile.
  - EpiSelect raises epiplexity and accuracy over ADO on Common Pile.
  - Epiplexity-maximising synthetic data improves GLUE fine-tuning only from pretrained initialization.
  - Synthetic data is "as informative as real natural web text" at matched tokens (App. G.1).
  
  — [2608.11746 HTML](https://arxiv.org/html/2608.11746v1)
- **Discussion framing:** "while we want the model for a given dataset to be as compressible as possible, perhaps we want to expose the model to data that makes the model as incompressible as possible." — [2608.11746 HTML Sec. 5](https://arxiv.org/html/2608.11746v1)

### Inferences
- **Refinement: the current-loss floor.** Replacing the final-loss floor with the current loss L(t) turns epiplexity into a running quantity. Ŝ(t) = Σ_{s≤t}(L(s) − L(t)) always lower-bounds the eventual final-floor estimate for the same trajectory while loss is still falling. Its increments satisfy ΔŜ ≈ −n·ΔL, so maximising it greedily favours data with large remaining loss decrease scaled by the amount already seen. This is my derivation from Eq. 2 and App. D.
- **Extension, not contradiction.** Neither method searches over (N, D) at a fixed compute T, which the founding paper's definition requires. The quantities measured are therefore "prequential area at one fixed model and token budget", not compute-optimal epiplexity S_T. The follow-up does not claim otherwise.
- **Granularity.** EpiSelect is per-domain (K≈30 groups) and EpiGen is per-batch. Neither provides per-example epiplexity, so no follow-up addresses per-sample scoring.
- **Implicit failure cases:**
  - The random-init EpiGen run gives no gain.
  - The PPL and NoBuffer reward variants collapse perplexity (104 and 33 on OWT vs 24.6 pretrained).
  - The authors suggest the buffer-based reward is what prevents reward hacking.
  - All EpiGen learners lose in-distribution perplexity.
  - The correlation evidence rests on n=5 points (p≈0.05).

### Gaps
- The follow-up does not compare its online proxies with the founding paper's compute-optimal prequential or requential estimates on the same data, so how faithful the proxies are is unquantified.
- No theoretical link between the "current-floor" proxy and S_T is given.

---

## Q6. What is directly usable for estimating epiplexity with MLPs on small datasets on one Colab GPU?

### Takeaway
Neither repo can run end-to-end at small scale. Both are hard-wired to LMs (Common Pile/TFDS/TPU pods; GPT-2/OWT/HF generation). Three pieces are small, framework-level and reusable:
- EpiSelect's `src/scaling_law.py`: joint cross-domain power-law fit plus the analytic ∂Ŝ/∂n_k (Eq. 4), in pure JAX + jaxopt.
- The few-line online per-group prequential accumulator (`compute_epiplexity` / `update_val_loss` in `ado.py`).
- EpiGen's `_loss_drop` lookahead-reward function in PyTorch.

For small datasets trained over multiple epochs, the paper's own caveat applies: training-loss prequential sums are invalid once data repeats. Use the held-out-loss variant (`val_epi` / `y_axis="val_loss"`) or single-pass streaming.

### Cited Findings
- `fit_scaling_law(D, L, mesh, K)` takes D and L as [K, T] arrays of cumulative per-group counts and losses and returns θ of size K(K+3). `epiplexity_grads(tokens[K,1], θ)` returns ∂Ŝ/∂n_k. The test file shows single-host use with `Mesh(devices.reshape(1, len(devices)), ("replica","data"))`, a synthetic fit with K=17 and T=111, and a 729-point init grid. — [scaling_law.py](https://github.com/eysu35/EpiSelect/blob/main/src/scaling_law.py); [test_scaling_law.py](https://github.com/eysu35/EpiSelect/blob/main/tests/test_scaling_law.py)
- The online accumulator keeps `train_epi_ceiling += counts_K * curr_train_loss` and `train_total_nk += counts_K`, then computes `train_epi_K = ceiling - curr_train_loss * total_nk`. The held-out analogue is `val_epi_ceiling += interval_cnt_K * val_loss_K` with `val_epi_K = val_epi_ceiling - val_loss_now * val_total_nk`. — [ado.py](https://github.com/eysu35/EpiSelect/blob/main/src/data_selectors/ado.py)
- Fit hyperparameters to reuse or adapt: Savitzky–Golay smoothing (window 101, order 3) of per-step losses; fitting from step 1000 onward, subsampled every 20 steps; refit every 1000 steps; bounds α ∈ [0, 0.8], log β ≤ 6.5, log ε ≥ 0.5. — [commonpile_episelect.yaml](https://github.com/eysu35/EpiSelect/blob/main/configs/loader/commonpile_episelect.yaml); [scaling_law.py](https://github.com/eysu35/EpiSelect/blob/main/src/scaling_law.py)
- `_loss_drop(learner, tokens, eval_batches, S, lr)` deep-copies the model, measures mean CE on `eval_batches`, takes S AdamW steps on `tokens`, re-measures and returns the drop. It assumes an HF-style `model(x).logits` and next-token targets. — [reward.py](https://github.com/eysu35/EpiGen/blob/main/datagen/reward.py)
- The paper's caveat: multi-epoch training "violates the prequential estimator's assumption … that credits code-length reduction to unseen tokens". — [HTML Sec. 3.1](https://arxiv.org/html/2608.11746v1)
- Neither repo ships notebooks, small configs or an MLP or non-LM example. The smallest published EpiSelect run is 124M parameters on 15.7B tokens, and EpiGen uses GPT-2 small for 60k learner steps. — [EpiSelect README](https://github.com/eysu35/EpiSelect/blob/main/README.md); [EpiGen README](https://github.com/eysu35/EpiGen/blob/main/README.md)

### Inferences
- **Recipe for MLPs.** For per-subset (per-class, per-source or per-augmentation) epiplexity on a small dataset:
  1. Log each minibatch's per-group loss before the gradient step. For true prequential validity, do this only on first exposure, i.e. a single epoch or a stream; otherwise evaluate on a fixed held-out split at intervals.
  2. Accumulate Ŝ_k(t) = Σ_s c_k(s)L_k(s) − L_k(t)Σ_s c_k(s).
  3. Optionally fit the Eq. 3 law with `scaling_law.py` to get forecasts and ∂Ŝ/∂n_k for adaptive sampling.
  
  For small K (≤ 10) the 729-start vmapped LBFGS with K(K+3) ≤ 130 parameters should be cheap on one Colab GPU. This is an estimate; I did not run it.
- **Loss-floor barrier.** If the ε barrier is reused as is, it forces ε ≥ e^0.5 ≈ 1.65 nats. That is inappropriate for MLP classification losses that fall well below 1.65 nats, so relax `log_eps_min` or re-parameterise. Likewise, α ≤ 0.8 may be too tight for fast-converging small models. This is my reading of the code.
- **Floor choice.** The "current-loss floor" gives an online, monotone-in-time running estimate that needs no final model. To match the founding paper's definition instead, compute the area above the final loss at the end, or repeat the run over (model size, steps) at a fixed compute budget. The follow-up's code does not do the latter.
- **Adapting EpiGen's reward.** For a classifier, replace `model(x).logits` next-token CE with CE against labels. The generator side (REINFORCE over sequences) has no direct MLP analogue unless you have a parametric generator. With pretrained-init dependence (random init gave −0.003), a toy-scale reproduction of EpiGen is unlikely to show gains.

### Gaps
- There are no published epiplexity values, units or magnitudes for small models in either work, because the figure data were not extractable. So there is no reference scale to check a Colab reproduction against.
- The JAX version compatibility of the pinned `jaxopt==0.8.3` / `equinox==0.13.6` / `optax==0.1.7` combination on current Colab images is untested. The TPU extra requires jax>=0.6.2, and optax 0.1.7 is old relative to that.
