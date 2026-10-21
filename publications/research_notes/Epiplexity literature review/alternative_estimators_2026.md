# Alternative and cheaper estimators of epiplexity (2026): Excess Description Length (Donoway et al.), reservoir "learnable novelty" (Zhang & Levin), and the small-data prequential estimator (Liu et al.)

Scope and method. I read all three arXiv PDFs in full (converted with pdftotext) and pulled the arXiv HTML versions to recover exact LaTeX. I also read the ISIT "full" PDF hosted on the first author's site, checked the ISIT DOI through the Crossref API, and cloned both code repositories and read the estimator code line by line. Scratch files are in `C:/TMP/claude/d--GitHubD-Epiplexity/8843dc96-d6bc-4e7a-985b-ba9a35de477c/scratchpad/edl_levin/`. I also ran a small replication of the Zhang & Levin estimator on CPU using their own code. Those numbers are marked **[own replication]**; they are not from the papers. Scripts: `.../edl_levin/tests/eca_sanity.py` and `.../edl_levin/tests/eca_N_scaling.py`.

Notation: S_T is epiplexity (Finzi, Qiu et al., arXiv 2601.03220). "Finzi's prequential estimate" means |P_preq| ≈ Σ_i [log 1/P_i(Z_i) − log 1/P_M(Z_i)], the area under the single-pass online loss curve above the final loss, taken at the MDL-optimal point of a compute sweep.

---

## Q1. Excess Description Length (EDL): definitions, proved properties and their assumptions, toy models, experiments, multi-epoch and memorization handling, usability for MLPs on MNIST/CIFAR, and the relation to S_T

### Takeaway
EDL is the first-exposure (single-pass) prequential codelength minus n times the final model's held-out (population) loss. Unlike Finzi's estimate, it subtracts a test loss, not a training loss, and it allows any amount of later multi-epoch compute, so memorization cannot inflate it. The ISIT 2026 version proves six properties: non-negativity, additivity, compute monotonicity, a processing bound, an n·I(X;Y) saturation bound, and a generalization-gap penalty. The arXiv preprint (v1) contains only theory and analytic toy models: no neural-network experiments, no dataset numbers, no code. Its empirical claims are deferred to a "companion empirical paper" that I could not find in public. Computationally, EDL is one training run plus one held-out evaluation, so it applies directly to MLPs on MNIST/CIFAR. But it measures one fixed algorithm, with no minimization over observers, so it is not S_T.

### Cited Findings

**Versions**
- arXiv 2601.04728 has only v1 (submitted Thu 8 Jan 2026, 747 KB). Subjects: cs.LG, cs.AI. Authors: Elizabeth Donoway (Dept. of Physics, UC Berkeley), Hailey Joren, Fabien Roger and Jan Leike (Anthropic, San Francisco). The PDF has 26 pages and is marked "Preprint". — [arXiv abs 2601.04728v1](https://arxiv.org/abs/2601.04728v1); [PDF](https://arxiv.org/pdf/2601.04728v1)
- The ISIT paper has a different title and a single author: "Excess Description Length: An Information Measure for Generalizable Structure Learned from Finite Data", Elizabeth Donoway (UC Berkeley, Dept. of Physics). Venue: 2026 IEEE International Symposium on Information Theory (ISIT), Guangzhou, China, 28 Jun – 3 Jul 2026, pp. 1–6, IEEE, DOI 10.1109/ISIT62367.2026.11654044, type proceedings-article. — [Crossref record](https://api.crossref.org/works/10.1109/ISIT62367.2026.11654044); [DOI](https://doi.org/10.1109/ISIT62367.2026.11654044) resolves to [IEEE Xplore 11654044](https://ieeexplore.ieee.org/document/11654044/). The Xplore page is JavaScript-rendered and returned no content to my fetch.
- The author's publications page lists the ISIT paper as "Submitted to ISIT 2026" and links a "full" PDF. That PDF has 7 pages: the 6-page paper plus an Appendix A on transfer learning. Its PDF metadata gives a creation date of 16 Jan 2026. — [elizabethdonoway.com/publications](https://elizabethdonoway.com/publications); [EDL_ISIT_full.pdf](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- The ISIT paper cites arXiv 2601.04728 as "[10]" and calls it "An extended companion paper [10] [that] applies the EDL framework to analyze capability emergence in language models" (§I). — [EDL_ISIT_full.pdf](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- The "population loss" framing, additivity, compute monotonicity, the processing bound and the saturation bound appear only in the ISIT version. arXiv v1 uses the empirical test loss L_test(θ*) and proves non-negativity in expectation, the regret decomposition, convergence to SDL and a generalization-improvement identity. — compare [arXiv v1 §3–4](https://arxiv.org/abs/2601.04728v1) with [ISIT full PDF §II–V](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)

**Definitions in arXiv v1 (§2–3)**
- Loss: ℓ(θ; x, y) = −log p_θ(y|x) (Eq. 1). Population loss: L(θ) = E_{(x,y)∼D}[ℓ(θ; x, y)] (Eq. 2). The paper works in nats and reports bits (÷ ln 2). A training algorithm A maps (θ_{i−1}, (x_i, y_i)) ↦ θ_i and "encompasses the choice of optimizer, learning rate schedule, and all other hyperparameters". θ* denotes the final parameters "considerate of any early stopping condition". — [arXiv v1 §2.1–2.2](https://arxiv.org/abs/2601.04728v1)
- Def. 3.1 (prequential MDL): MDL(D; θ0, A) = Σ_{i=1}^n ℓ(θ_{i−1}; x_i, y_i) (Eq. 6). For batches, "we accumulate Σ_{(x,y)∈B_j} ℓ(θ_{j−1}; x, y) before the update". The paper notes that MDL depends on example order and that the same ordering, or an average over orderings, should be used when comparing. — [arXiv v1 §3.1](https://arxiv.org/abs/2601.04728v1)
- SDL (Whitney et al. 2021) as restated: SDL(D; A) = Σ_i ℓ(θ_{i−1}; x_i, y_i) − n·L*, with L* = inf_θ L(θ) (Eq. 5). — [arXiv v1 §2.5](https://arxiv.org/abs/2601.04728v1)
- **Def. 3.2 (EDL): EDL(D; θ0, A) = MDL(D; θ0, A) − n·L_test(θ*)** (Eq. 7). L_test(θ*) is "estimated in practice by averaging over a held-out test set drawn from the same distribution as the training data". Equivalently, EDL = Σ_{i=1}^n [ℓ(θ_{i−1}; x_i, y_i) − L_test(θ*)] (Eq. 9). θ* is the final model "whether by convergence, early stopping, or a fixed compute budget". — [arXiv v1 §3.2](https://arxiv.org/abs/2601.04728v1)
- "EDL requires only quantities that are directly measurable: the accumulated training loss during the first epoch and the test loss of the final model." — [arXiv v1 §3.5](https://arxiv.org/abs/2601.04728v1)
- Normalizations: EDL_ex = EDL/n. EDL_tok = EDL/D, where D is "the total number of tokens in the dataset scored in computing MDL (typically, the label tokens across all examples)". EDL_par = EDL/P, where P is the number of trainable parameters. — [arXiv v1 §3.4](https://arxiv.org/abs/2601.04728v1)
- Relation to SDL: EDL = SDL + n·(L* − L_test(θ*)) (Eq. 10), so EDL ≤ SDL, with equality when the learner reaches the optimum. — [arXiv v1 §3.5](https://arxiv.org/abs/2601.04728v1)

**Definitions in the ISIT version (§II)**
- Log-loss in bits: ℓ(θ; x, y) ≜ −log2 p_θ(y|x). Population loss: L(θ) ≜ E_{(x,y)∼D}[ℓ] (Eq. 1). The trained parameters are θ_T = A(θ0, S; T) "after compute budget T (measured in optimization steps, epochs, or FLOPs)". — [ISIT full PDF §II](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- Def. 1 (first-exposure prequential codelength): MDL_1(S; A) ≜ E_π[Σ_{i=1}^n ℓ(θ^π_{i−1}; x_{π(i)}, y_{π(i)})] (Eq. 2). π is a uniform random permutation, and θ^π_i are the parameters "after processing the first i examples once". Remark 1: for SGD, θ_i is the state after i minibatches in the first epoch. — [ISIT §II-A](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- **Def. 2 (compute-indexed, "population" EDL): EDL_T(A, n) ≜ E_{S∼D^n}[MDL_1(S; A) − n·L(θ_T)]** (Eq. 3). Quote: "This definition formalizes the operational estimator 'area of the first-epoch prequential loss above the final population loss,' while allowing θ_T to come from multi-epoch training (T ≫ n)." — [ISIT §II-B](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- Remark 3 (operational estimation): "(1) computing cumulative loss during the first epoch of training, (2) estimating L(θ_T) on held-out data after training completes, and (3) taking the difference." — [ISIT §II-B](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- The abstract describes EDL as "the excess codelength incurred on first exposure to data relative to a predictor whose expected log-loss equals the final model's population loss, thereby excluding memorization effects on the training set." — [ISIT full PDF abstract](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)

**Properties proved, with their assumptions**
- *Population-monotonic algorithm* (arXiv v1 Def. 4.1): E[L(θ_j) | θ_{j−1}] ≤ L(θ_{j−1}) for all steps j. The paper says this is satisfied by "Gradient descent on convex losses with appropriate learning rate", "SGD with sufficiently small learning rate on smooth losses" and "Any algorithm where early stopping prevents overfitting". It is violated by "Large learning rates that cause oscillation", "Training well past the optimal early stopping point on finite data" and "Algorithms that memorize training data at the expense of generalization". The ISIT Def. 3 is the unconditional version: E[L(θ_{t+1})] ≤ E[L(θ_t)]. — [arXiv v1 §4.1](https://arxiv.org/abs/2601.04728v1); [ISIT §III-B](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Non-negativity* (arXiv v1 Thm 4.2 / ISIT Thm 2). Assumptions: i.i.d. training data, an independent test sample, a population-monotone A, and (ISIT) T at least the first-exposure length. Conclusion: E[EDL] ≥ 0. The proof uses E[EDL] = Σ_i [L(θ_{i−1}) − L(θ*)] (Eq. 14), which is the ISIT "expected area form", Thm 1: EDL_T(A, n) = Σ_{i=1}^n E[L(θ_{i−1}) − L(θ_T)] (Eq. 4). Remark 4.3: "For algorithms that are not population-monotonic, EDL may be negative in expectation… In practice, we observe E[EDL] ≥ 0 for well-tuned training procedures (see companion empirical paper)." — [arXiv v1 §4.1, App. A.1](https://arxiv.org/abs/2601.04728v1); [ISIT §III-A–B](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Regret decomposition* (arXiv v1 Thm 4.4): MDL = Σ_i ℓ(θ; x_i, y_i) + R_n(θ) for any fixed θ (Eq. 16), so MDL ≈ n·L_train(θ*) + R_n(θ*) (Eq. 17). Cor. 4.5: with sublinear regret, EDL/n → L_train(θ*) − L_test(θ*) (Eq. 18). The ISIT version (§VI) states the exact identity EDL_T = E_{S,π}[R_T(S_π)] + n·(E[L_train(θ_T)] − L(θ_T)), where R_T(S_π) = Σ_i [ℓ(θ^π_{i−1}; z_π(i)) − ℓ(θ_T; z_π(i))]. — [arXiv v1 §4.2](https://arxiv.org/abs/2601.04728v1); [ISIT §VI](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Convergence to SDL* (arXiv v1 Thm 4.6). Assumes consistency, L(θ*) → L* almost surely as n → ∞. Then EDL/n − SDL/n = L* − L_test(θ*) → 0 (Eqs. 19–20). Remark 4.7 gives rates of O(√(log|Θ|/n)) for finite classes and O(1/√n) for bounded-complexity parametric models; "We do not claim rates for neural networks." — [arXiv v1 §4.3](https://arxiv.org/abs/2601.04728v1)
- *Generalization-improvement identity* (arXiv v1 Thm 4.8, population-monotone): L(θ0) − E[L(θ*)] = (L(θ0) − L̄) + E[EDL]/n (Eq. 21), where L̄ = (1/n) Σ_i E[L(θ_{i−1})]. So E[EDL] = n·(L̄ − E[L(θ*)]) (Eq. 22). — [arXiv v1 §4.4](https://arxiv.org/abs/2601.04728v1)
- *Algorithm dependence* (arXiv v1 Prop. 4.9): different A give different EDL on the same D and θ0. The paper calls this "not a defect but a feature". ISIT Remark 4 adds that sup_A EDL_T(A, n) is the algorithm-independent object and that "An algorithm that learns slowly incurs high regret and thus high EDL". — [arXiv v1 §4.5](https://arxiv.org/abs/2601.04728v1); [ISIT §III-A](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Compute monotonicity* (ISIT Thm 3). Assumes T2 ≥ T1 ≥ n and E[L(θ_T2)] ≤ E[L(θ_T1)]. Then EDL_T2 ≥ EDL_T1, with equality iff the population losses are equal. Proof: EDL_T2 − EDL_T1 = n·(E[L(θ_T1)] − E[L(θ_T2)]). Cor. 4: overfitting decreases EDL, so EDL_T is maximized at the early-stopping point T* = argmin_T E[L(θ_T)]. Remark 5: EDL becomes negative when n·L(θ_T) > MDL_1. — [ISIT §III-C](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *KL decomposition* (ISIT Thm 5): EDL_T = Σ_i E[ΔKL(θ_{i−1}, θ_T)] (Eq. 5), where ΔKL(θ, θ′) = D_KL(p_D‖p_θ) − D_KL(p_D‖p_θ′). — [ISIT §III-D](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Additivity* (ISIT Thm 6). Assumes D = D1 × D2 with independent samples S1, S2, a factorized model p(y1, y2|x1, x2) = p^(1)(y1|x1)·p^(2)(y2|x2), and an A that respects the factorization. Then EDL_T(A, S1∪S2) = EDL_T1(A1, n1) + EDL_T2(A2, n2) (Eq. 6). — [ISIT §III-D](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Sequential decomposition, warm start* (ISIT Thm 7). S1 and S2 are independent samples from the same D. Then EDL_T(A, S1∪S2) = EDL_T1(A, n1) + EDL_T2|T1(A, n2) + n1·E[L(θ_T1) − L(θ_T)] (Eq. 7). The last term is the "transfer benefit". Appendix A of the full PDF (Cor. 15) decomposes EDL_cold − EDL_warm into a trajectory advantage Δ_traj plus an endpoint correction (Eq. 13). — [ISIT §III-E, App. A](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Mutual-information bound* (ISIT Lemma 8). Assumes a population-monotone A and "marginal-calibrated initialization (i.e., p_θ0(y|x) = p(y) so that L(θ0) = H_D(Y))". Then EDL_T(A, n) ≤ n·I_D(X;Y) (Eq. 8), tight when the final predictor is Bayes-optimal. Without calibrated initialization the bound is EDL_T ≤ n·(L(θ0) − L*) (Remark 7). — [ISIT §III-F](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Processing bound* (ISIT Thm 9). For any deterministic elementwise f with pushforward D′ = f_#(D): sup_A EDL_T(A, f(S)) ≤ sup_A EDL_T(A, S) (Eq. 9). Cor. 10: invariance under sufficient statistics. — [ISIT §III-G](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Finite-data saturation* (ISIT Thm 11). Same assumptions as Lemma 8: sup_T EDL_T(A, n) ≤ n·I_D(X;Y) (Eq. 10), "Immediate from Lemma 8". Cor. 12 ("Decoupling data from compute"): with T1 = n (end of the first epoch) and T2 > T1, EDL_T2 = EDL_T1 + n·E[L(θ_T1) − L(θ_T2)] (Eq. 11). "No resampled example is redundantly encoded as additional information." — [ISIT §IV](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Generalization-gap penalty* (ISIT Thm 13). Compares A_gen, which reaches L(θ_T) ≈ L*, with A_mem, which reaches zero training loss but L(θ_T) = L0 > L*, "Assuming similar first-exposure trajectories". Then EDL_T(A_gen) − EDL_T(A_mem) = n·(L0 − L*) > 0 (Eq. 12). Cor. 14: "For data with random labels Y ⊥ X, EDL_T(A, n) ≲ 0 for any algorithm, regardless of training loss achieved." — [ISIT §V](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- *Variance and ordering* (arXiv v1 App. C). Prop. C.1: Var[EDL] = O(n), with test-term variance O(n²/n_test); relative uncertainty in EDL/n is O(1/√n). Prop. C.2: for i.i.d. data, E[MDL] is independent of ordering. Def. C.3 gives a continuous-time MDL_cont(T) = ∫_0^T L(θ(t)) dt. — [arXiv v1 App. C](https://arxiv.org/abs/2601.04728v1)

**Toy models (arXiv v1 §5, App. B)**
- *Random labels* (§5.1, Prop. 5.1, App. B.1). y ∼ Uniform(Y), |Y| = k. The optimal predictor is uniform, with loss log k. For a calibrated model, MDL ≈ n·log k and L_test = log k, so E[EDL] = 0. If the model memorizes, "for the first exposure to each example (which determines MDL), the model cannot predict better than chance… the memorized labels do not help predict independent test labels, so L_test(θ*) ≥ log k". With MDL = n·log k + ε, E[EDL] ≈ ε, and "For well-behaved models, ε ≈ 0." Quote: "Random labels yield EDL near zero regardless of the training effort or computational resources expended." — [arXiv v1 §5.1, App. B.1](https://arxiv.org/abs/2601.04728v1)
- *Hypothesis collapse* (§5.2, Prop. 5.2). m hypotheses with a uniform prior; one diagnostic example; hypotheses maximally distinguishing on x (m/|Y| per label). "a single example contributes exactly log|Y| bits to EDL, and the generalization improvement is log m bits". Without the maximal-distinguishing assumption it contributes at most min(log|Y|, log m). App. B.2 (Prop. B.3, Eq. 69) instead states "A single diagnostic example can contribute up to (but not necessarily) log m bits to EDL" and writes "EDL = n·(L(θ0) − L(θ*)) = 1·log m". — [arXiv v1 §5.2, App. B.2](https://arxiv.org/abs/2601.04728v1)
- *Disjoint subdistributions* (§5.3, Prop. 5.3). Learning a rule that lowers loss by δ on a subdistribution of weight π contributes about π·δ bits per example to expected generalization; EDL/n = Σ_j π_j (L̄_j − L*_j) (Eq. 79). — [arXiv v1 §5.3, App. B.3](https://arxiv.org/abs/2601.04728v1)
- *Coupon collector* (§5.4, Prop. 5.4, App. B.4). K concepts, per-concept improvement Δ = L_high − L_low, u = n/K. E[EDL] ≈ KΔ[1 − (1+u)e^{−u}] (Eq. 90). For small n it is ≈ Δn²/(2K); for large n it saturates at KΔ. E[EDL]/n peaks at n ≈ 1.79K with value ≈ 0.298Δ (Eq. 97, Fig. 2/6). — [arXiv v1 §5.4, App. B.4](https://arxiv.org/abs/2601.04728v1)
- *Format vs capability* (§5.5, Prop. 5.5, App. B.5). With linear learning curves, EDL/n ≈ n(L_F0/n_F + L_C0/n_C) while both components are being learned (increasing). For n > n_C, EDL ≈ (n_F·L_F0 + n_C·L_C0)/2 is constant, so EDL/n decays as 1/n. The paper recommends scoring only answer tokens. — [arXiv v1 §5.5, App. B.5](https://arxiv.org/abs/2601.04728v1)

**Experiments**
- arXiv v1 has no empirical experiments on real models or datasets; §5 and App. B are analytic toy models with schematic figures (Figs. 1–8). §7.1.1, "Empirical validation preview", says the framework's predictions are "validate[d] empirically in a companion paper". Claims listed there: (1) elicitation shows monotonically decreasing EDL/token with dataset size, while teaching shows an initial increasing phase, "robust across models (Llama 3 1B–8B, TinyStories-1B, Qwen2.5 1.5B–14B) and tasks (arithmetic, reasoning)"; (2) "Pre-teaching a skill (e.g., multiplication via operator notation) converts a teaching task to an elicitation task, reducing information thresholds by 10–100×"; (3) "EDL collapses to near zero when labels are randomly permuted"; (4) performance "degrades significantly when EDL per parameter exceeds regime-dependent thresholds, with elicitation thresholds (≈0.01–0.1 bits/parameter) roughly 100× lower than teaching thresholds (>1 bit/parameter)". No numbers, figures or code accompany these claims in v1. — [arXiv v1 §6.4, §7.1.1](https://arxiv.org/abs/2601.04728v1)
- The ISIT paper has no experiments. Fig. 1 is a schematic of single-epoch and multi-epoch EDL areas. — [ISIT full PDF](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- A related paper by the same group, "Quantifying Elicitation of Latent Capabilities in Language Models" (NeurIPS 2025; Donoway, Joren, Somani, Sleight, Michael, DeWeese, Schulman, Perez, Roger, Leike), reports that "training as few as 10–100 randomly chosen parameters can recover up to 50% of the performance gap between pretrained-only and full fine-tuned models, while 1,000s to 10,000s of parameters can recover 95%". — [NeurIPS 2025 proceedings](https://proceedings.neurips.cc//paper_files/paper/2025/hash/dd16e008534dc52fab72fdedad069d5c-Abstract-Conference.html) (via search snippet; the OpenReview PDF was behind a browser check)

**Multi-epoch training and memorization**
- arXiv v1 §3.2: "Training typically continues beyond a single pass through the data… Let θ* denote the final parameters after training completes". MDL is taken from the first pass only. App. D.2: "increases in EDL in later epochs reflect additional predictive information extracted from the train data", and "For a finite dataset, maximum EDL can be obtained by training to best achievable performance on the validation set (with early stopping prior to overfitting) and evaluating test loss on the best model checkpoint." — [arXiv v1 §3.2, App. D.2](https://arxiv.org/abs/2601.04728v1)
- App. D.2 of arXiv v1 calls EDL "compute-bound-agnostic: EDL quantifies the information cost of generalization for any training procedure under any computational bound or stopping condition". — [arXiv v1 App. D.2](https://arxiv.org/abs/2601.04728v1)
- In the ISIT version, memorization is handled by referencing population loss (Thm 13, Cor. 14), and multi-epoch compute adds EDL only through lower population loss (Cor. 12). The first innovation listed in §VII is "Decoupling data from compute: EDL counts information only for first exposure, while allowing arbitrary subsequent computation to extract structure." — [ISIT §IV, §V, §VII](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)

**Relation to epiplexity, as stated by the author (ISIT §VI, Table I)**
- "Concurrent independent work [9] introduces 'epiplexity' S_T as a compute-bounded notion of structural content in data… EDL differs in three respects: (i) it evaluates a *specified algorithm* rather than optimizing over all feasible programs to evaluate the *data*, (ii) it uses an explicit population-loss reference to exclude memorization, and (iii) it remains meaningful when compute exceeds dataset size (T ≫ n) by fixing a first-exposure codelength." — [ISIT §VI](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- "In the one-pass i.i.d. regime, the prequential *proxy* for the epiplexity estimator takes an 'area above final loss' form that is geometrically similar to single-epoch instantiations of EDL_T when the empirical risk coincides with the population risk." Also: "EDL_T,converged > EDL_T,single-epoch when additional epochs improve generalization." — [ISIT §VI](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)
- Table I, "Comparison of information measures for learning". Rows for S_T: Non-negative ✓; Additive ?; Processing bound ?; Finite data ✓*; Multi-epoch ×; Compute-indexed ✓; Operationally computable "Bounded". EDL gets ✓ in every row. The footnote on the asterisk reads: "Epiplexity assumes data are sampled from an effectively infinite distribution with small generalization gap; sufficiently large compute bounds (exceeding the domain of the distribution) can violate this assumption." — [ISIT Table I (p. 5)](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf) (read from a rendered page image because the symbols did not survive text extraction)
- The ISIT Discussion says sup_A lim_{T→∞} EDL_T(A, n) "defines an intrinsic property of (D, n) analogous to channel capacity; characterizing this quantity remains open." — [ISIT §VII](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf)

### Inferences
- **Precise relation to S_T.** Take a single epoch on i.i.d. data (T = n) with a fixed algorithm A. In expectation, EDL_n(A, n) is exactly the "area above final loss" Σ_i E[L(θ_{i−1}) − L(θ_n)], which is Finzi's prequential estimate |P_preq| for the same run with the final training loss replaced by its expected population loss. The two coincide when A is the run that Finzi's compute sweep selects as MDL-optimal and when one pass over fresh data makes the final training loss an unbiased test-loss estimate. Beyond that case there are three differences. (a) S_T minimizes the two-part code over observers (model size, data budget and hyperparameters) at compute T; EDL takes A as given. (b) EDL allows T ≫ n and uses held-out loss; Finzi's estimate needs a single pass and uses the final training loss. (c) Finzi codes a dataset of 𝒟 tokens that can exceed the D* the model trains on; EDL's n is the whole training set.
- **The "intrinsic" sup_A EDL is not a structural-information quantity.** By Thm 1, EDL_T = Σ_i [L(θ_{i−1}) − L(θ_T)]. An algorithm that does not update during the first pass and then trains for many epochs gets EDL_T = n·(L(θ0) − L(θ_T)). With a calibrated start and a reachable Bayes-optimal predictor this attains the Lemma 8 ceiling n·I(X;Y). So sup_A EDL_T(A, n) = n × (best achievable population-loss reduction from n samples), which grows linearly in n even for a one-line deterministic rule. S_T, by contrast, stays small for simple rules. Remark 4 ("An algorithm that learns slowly incurs high regret and thus high EDL") already points in this direction. EDL is therefore meaningful for a fixed, sensible algorithm, and its "maximize over algorithms" version has the opposite sign convention to the minimization in epiplexity. This is my derivation; the papers do not state it.
- **Compute monotonicity is nearly definitional.** ISIT Thm 3 assumes the population loss does not increase, and the conclusion restates that assumption multiplied by n. The substantive content is Cor. 12: later compute enters only through L(θ_T).
- **Internal inconsistencies in arXiv v1:**
  - §5.2 (Prop. 5.2) says one diagnostic example contributes "exactly log|Y| bits to EDL". App. B.2 (Prop. B.3, Eq. 69) says it contributes "up to… log m" and uses EDL = n(L(θ0) − L(θ*)), which is the generalization improvement, not Def. 3.2.
  - The random-label memorization argument (App. B.1) shows E[EDL] ≈ ε with an unsigned ε. The ISIT Cor. 14 states EDL ≲ 0.
  - Remark 4.3 and §7.1.1 cite a "companion empirical paper", but the ISIT version cites arXiv 2601.04728 itself as the "extended companion paper" that "applies the EDL framework to analyze capability emergence in language models", and v1 contains no such analysis.
- **Usability for MLPs on MNIST/CIFAR.** EDL is directly usable and cheap. It needs one training run with the loss of every first-epoch minibatch logged before its update (the Def. 3.1 batch rule), then held-out loss at the final or early-stopped checkpoint. There is no model-size sweep and no teacher–student distillation. For MNIST with a calibrated start, the saturation bound is n·I(X;Y) ≤ n·log2 10 ≈ 3.32n bits (about 2.0e5 bits for n = 60k). Test-set noise contributes a standard deviation of about n·σ_ℓ/√n_test. For example, σ_ℓ ≈ 0.3 bits with n = 50k and n_test = 10k gives about 150 bits [ESTIMATE, arithmetic]; this is small relative to EDL values in the thousands of bits. Caveats:
  - The value depends on learning rate, batch size and initialization (Prop. 4.9).
  - It is a per-algorithm quantity, not S_T. To mimic S_T one would still have to sweep architectures or compute and choose by total code length, which the papers do not prescribe.
  - For a from-scratch MLP the initialization is not marginal-calibrated (L(θ0) ≈ log k, not H(Y)), so the Lemma 8 bound becomes n·(L(θ0) − L*) (Remark 7).

### Gaps
- I could not access the IEEE Xplore page content (JavaScript-rendered), so I have not verified that the published 6-page ISIT paper is identical to the 16-Jan-2026 "full" PDF on the author's site. The theorem numbering in these notes follows that PDF.
- The "companion empirical paper" (EDL scaling on Llama 3 1B–8B, TinyStories-1B, Qwen2.5 1.5B–14B; the bits-per-parameter thresholds; the random-label control) could not be located. The NeurIPS 2025 elicitation paper is related, but I could not open its PDF to check whether it computes EDL.
- Neither EDL paper reports code, runtime, datasets, model sizes or compute. No official implementation was found.
- Neither paper evaluates EDL on MLP/MNIST/CIFAR, or on any data used in the epiplexity paper (ECA, induction tasks, OpenWebText, etc.), so there is no numerical EDL–S_T comparison.

---

## Q2. Zhang & Levin, "Intelligence from Learnable Novelty": the closed-form spectral description length on a fixed random reservoir. Exact formulas, the ridge-as-MDL argument, the choice of α, η, λ and reservoir size, the ECA ranking against Finzi et al., the NCA/MNIST/RL experiments, runtime, and validation

### Takeaway
S^φ(Y|X) = ½ Σ_i log2(1 + η·s_i(W_λ)²), where W_λ is the ridge readout, with column-standardized features, from a frozen random reservoir φ (a fixed feature map) to the targets. It is a model-cost-only, two-part-code score for a linear readout, not a prequential/online code. The paper motivates ridge as the small-‖W‖ Taylor limit of the log-det penalty and shows an exact MDL minimizer ranks systems almost identically (Spearman 0.997). The paper never compares it against Finzi's training-based S_T. Among the 88 ECA rules it ranks rule 110 first, then 25, with 54 fourth and 30 twelfth, so rule 54 beats rule 30 as in Finzi et al. The paper's claim that noise contributes zero to |M| holds only as N → ∞. At the paper's ECA settings, pure i.i.d. noise targets score about 18 bits, close to rule 15 (about 21) and 70% of rule 30 (about 25) [own replication]. That puts rule 30 above rule 15, the opposite of Finzi's 15 > 30. Scoring one system takes under a second on CPU.

### Cited Findings

**Bibliographic and code facts**
- arXiv 2607.18433v1, submitted Mon 20 Jul 2026. Comments: "24 pages, 10 figures, 4 tables". Subjects: cs.LG, cs.AI, nlin.AO. Authors: Yanbo Zhang (Allen Discovery Center at Tufts University) and Michael Levin (Allen Discovery Center at Tufts; Wyss Institute at Harvard); corresponding author Michael.Levin@tufts.edu. The PDF is dated July 22, 2026. — [arXiv abs 2607.18433](https://arxiv.org/abs/2607.18433); [PDF](https://arxiv.org/pdf/2607.18433v1)
- Code: [github.com/Zhangyanbo/learnable-novelty](https://github.com/Zhangyanbo/learnable-novelty). PyTorch (torch ≥ 2.9, < 2.11), Python ≥ 3.11, managed with `uv`; dependencies include stable-baselines3 ≥ 2.8, gymnasium[box2d], gymnasium-robotics and mujoco ≥ 3.9. License: "Apache 2.0 with the Tufts Open Source License Rider v.1 (academic, non-commercial research use only)". The estimator is in `src/rc_epiplexity/` (`core.py`, `reservoirs.py`, `online.py`). Entry points: `src/eca.py` (Fig. 2), `src/inverse_nca.py` (Fig. 3), `src/inverse_mnist.py` (Fig. 4), `src/rl_classic_control_epiplexity.py --seeds 0..9` (Table 1), `src/appendix/surrogate_vs_mdl.py --task eca` (App. D), `src/robustness/eca_hparam_scan.py` (App. B), `src/dynamical_systems_experiment.py` (App. F), `src/reservoir_criticality.py` (App. A.3). — [README](https://github.com/Zhangyanbo/learnable-novelty/blob/main/README.md); [pyproject.toml](https://github.com/Zhangyanbo/learnable-novelty/blob/main/pyproject.toml)
- The README lists a "Steps-to-threshold column of Table 1" (`src/rl_steps_to_threshold.py`). Table 1 in arXiv v1 has no such column. — [README](https://github.com/Zhangyanbo/learnable-novelty/blob/main/README.md) vs [arXiv v1 Table 1](https://arxiv.org/html/2607.18433v1)

**Definitions and formulas (§2–3)**
- Prequential cumulative surprise: L = Σ_i ℓ_i = −Σ_i log2 p(y_i | y_<i, X) = −log2 p(Y|X) (Eq. 1). Two-part split: L ≈ min_M [|M| − log2 p(Y|X, M)] (Eq. 2). Bounded version: L_φ(Y|X) = min_{M∈M_φ} [|M| − log2 p(Y|X, M)], with M*_φ its argmin (Eq. 3). "**S^φ(Y|X) = |M*_φ(Y|X)|**" (Eq. 4). The paper states that |M*_φ| "is precisely the epiplexity of Finzi et al. (2026)". — [arXiv v1 §2](https://arxiv.org/html/2607.18433v1)
- Observer: "let φ be an untrained, fixed random nonlinear map… H = φ(X) ∈ ℝ^{N×m}… targets Y ∈ ℝ^{N×D}… All learnable capacity resides in the linear readout matrix W ∈ ℝ^{m×D}, so M_φ can be defined as φ followed by every possible linear readout." — [arXiv v1 §3](https://arxiv.org/html/2607.18433v1)
- Total description length with a Gaussian residual: L(W) = ‖Y − HW‖²_F / (2σ² ln 2) + C(W, φ) (Eq. 5). — [arXiv v1 §3](https://arxiv.org/html/2607.18433v1)
- **Spectral description length: C_spec(W) = α·log2 det(I_m + η·WWᵀ) = α·Σ_i log2(1 + η·s_i(W)²)** (Eq. 6), where "η is a resolution parameter and α is an overall scale". Quote: "Since α changes neither the ranking between systems nor the direction of the gradient, all experiments fix α = 1/2 and set η = 1, except the MNIST encoder, which uses η = 30 (Table 3)." — [arXiv v1 §3](https://arxiv.org/html/2607.18433v1)
- Normalization: H̃_{·c} = (H_{·c} − μ_c)/(σ̂_c·√m) and Ỹ = (Y − μ_Y)/u_Y (Eq. 7). u_Y is "a fixed scale factor… posited in advance", used "because the target's magnitude itself carries information". The √m "keeps the output scale of a random readout with i.i.d. standard normal entries invariant across feature widths". — [arXiv v1 §3](https://arxiv.org/html/2607.18433v1)
- Ridge readout: W_λ = argmin_W ‖Ỹ − H̃W‖²_F + λ‖W‖²_F = (H̃ᵀH̃ + λI_m)^{-1} H̃ᵀỸ (Eq. 8). **Full estimator: S^φ(Y|X) = ½·log2 det(I_m + η·W_λW_λᵀ) = ½·Σ_i log2(1 + η·s_i(W_λ)²)** (Eq. 9). "The optimum is unique, and S^φ is differentiable in (X, Y)." — [arXiv v1 §3](https://arxiv.org/html/2607.18433v1)

**Why ridge is used as an approximation to MDL**
- "With equation 6 as the weight part… the total description length equation 5 has no closed-form minimizer… We therefore approximate the minimization by a ridge regression." Justification: "for small W, log2 det(I_m + η·WWᵀ) ≈ η‖W‖²_F/ln 2, so the spectral description length itself reduces to the quadratic penalty of the ridge, and the residual scale σ² merges with α and η into a single ridge parameter λ; the ridge's own shrinkage in turn keeps the readout in the small-norm regime where the expansion is accurate." The paper also cites Hinton & van Camp (1993) and Musat (2026, arXiv 2605.10878, "Neural weight norm = Kolmogorov complexity"). — [arXiv v1 §3](https://arxiv.org/html/2607.18433v1)
- App. C derives the log-det from a matrix-Gaussian prior on W with a Wishart-type hyperprior on the row precision Λ. The result is −log2 p(W) = C + ((a+D)/2)·log2 det(I_m + η·WWᵀ) (Eq. 18), "which is the spectral description length equation 6 with α = (a + D)/2… the estimator treats α as a free overall scale and fixes α = 1/2 throughout, rather than the D-dependent value the hierarchical derivation would assign." Holding the precision fixed at Λ = λI_m instead gives the ridge penalty −log2 p(W | λI_m) = C + (λ/(2 ln 2))‖W‖²_F (Eq. 15). The Wishart hyperprior is p(Λ) ∝ |Λ|^{(a−m−1)/2} exp(−tr Λ/(2η)), with a > m − 1. — [arXiv v1 App. C](https://arxiv.org/html/2607.18433v1)
- App. D minimizes J_MDL(W) = ‖Ỹ − H̃W‖²_F/(2σ² ln 2) + C_spec(W) exactly, with σ² = λ/(2αη) (Eq. 19). It uses a majorize–minimize weighted-ridge iteration W_{k+1} = (H̃ᵀH̃ + λM_k)^{-1}H̃ᵀỸ with M_k = (I_m + η·W_kW_kᵀ)^{-1} (Eq. 20). Results:
  - S^φ_MDL exceeds S^φ "by a factor growing from about 1 to about 2.3 with the score itself".
  - Under an isotropic Gram, stationary points satisfy w(1 + λ/(1 + ηw²)) = σ, versus the ridge σ/(1+λ), so S^φ ≤ S^φ_MDL. The ordering "holds in every paired solve we ran (880 for the ECA, 99 for the NCA)".
  - Ranking agreement: "Spearman ρ = 0.997 over the rules, the same top-fourteen set, rule 110 maximal under both with its margin over the runner-up widening from 24.7 to 38.7 bits". "the largest rank change anywhere is rule 41, second under the ridge readout and thirteenth under the exact one".
  - On the NCA trajectories: "Pearson r = 0.996, with a nearly constant 35-bit offset on the converged plateau".
  
  — [arXiv v1 App. D, Fig. 8](https://arxiv.org/html/2607.18433v1)
- App. E: the ridge solve uses the augmented least-squares system [H̃; √λ·I_m] w_j = [ỹ_j; 0_m] with a reduced QR and a triangular solve (Eqs. 21–22). It is differentiable through QR, the triangular solve and the SVD, "without ever materializing H̃ᵀH̃". — [arXiv v1 App. E](https://arxiv.org/html/2607.18433v1)

**Code implementation of the estimator (as read)**
- `core.py`, `svd_log_volume(w, lambda_code)` returns `0.5 * torch.sum(torch.log1p(lambda_code * s.square())) / ln2`, where `s = torch.linalg.svdvals(w)`. So α = 0.5 is hard-coded and η is named `lambda_code`. The docstring notes this replaced "the earlier `0.5 * lambda * ||w*||^2`". — [core.py](https://github.com/Zhangyanbo/learnable-novelty/blob/main/src/rc_epiplexity/core.py)
- `multioutput_epiplexity` centers the target and divides by a fixed `sigma_y` (= u_Y, default 1.0). It standardizes features with `(features − mean)/(std·√d + eps)` and solves one ridge readout per output column via `torch.vmap` of `ridge_least_squares` (augmented QR). It returns the epiplexity plus the residual; the residual is not used in the score. — [core.py](https://github.com/Zhangyanbo/learnable-novelty/blob/main/src/rc_epiplexity/core.py)
- Reservoirs are built from `Conv1d` or `Linear` layers with a parameter-free `PreActNorm` (normalize over channels) before every ELU, "which holds a random reservoir at the edge of chaos". The 1D reservoir flattens (batch × space) sites into rows, so one readout is shared across sites. — [reservoirs.py](https://github.com/Zhangyanbo/learnable-novelty/blob/main/src/rc_epiplexity/reservoirs.py)
- `online.py` implements covariance-form recursive least squares with Sherman–Morrison rank-1 updates, "O(d²) per step, so the full S_t trajectory costs O(T d²)". It "matches the batch estimator bit-for-bit under identical normalization (verified to ~1e-12)" but "requires the feature/target normalization to be frozen in advance". — [online.py](https://github.com/Zhangyanbo/learnable-novelty/blob/main/src/rc_epiplexity/online.py); [arXiv v1 App. F.2](https://arxiv.org/html/2607.18433v1)

**Hyperparameters and reservoir sizes (Tables 2–3)**
- ECA ranking: circular 1D conv, 256 channels, kernel 3 (1 in the final layer), depth 3, ELU, λ = 0.03. Data: lattice width 64, Bernoulli(1/2) initial state, 1000 burn-in steps, stacked target window τ = 32, N = 512 samples per rule, 10 independent draws of reservoir and data.
- Inverse NCA: the same conv reservoir at depth 4, λ = 0.3.
- Continuous flows: random MLP, depth 4, width 64, λ = 0.1.
- MNIST encoder: random MLP, depth 4, width 2048, λ = 3, η = 30.
- RL: random MLP, depth 4, width 32, λ = 0.3.
- All reservoirs use pre-activation normalization; all runs use α = ½ and η = 1 except MNIST.

— [arXiv v1 Tables 2–3, App. A](https://arxiv.org/html/2607.18433v1)
- App. A.3: "the plain reservoir sits in the ordered phase for every architecture we test, at χ ≈ 0.55 for depths up to sixteen and rising to χ ≈ 0.75 at depth thirty-two". Normalizing pre-activations pins χ ≈ 1 regardless of depth, input scale and architecture (Fig. 5). — [arXiv v1 App. A.3](https://arxiv.org/html/2607.18433v1)
- The ECA code states that the data generation "follows the original epiplexity reference implementation: https://github.com/shikaiqiu/epiplexity (reference commit … 3aa12a1…)" (width 64, burn-in 1000). The in-code comment on `TAU_MAX = 32` reads: "Paired with a deliberately *local* reservoir (depth 3, kernel 3 → radius 2), this long window surfaces rule 110 as the epiplexity maximum among ECA rules." — [src/systems/eca.py](https://github.com/Zhangyanbo/learnable-novelty/blob/main/src/systems/eca.py); [src/eca.py](https://github.com/Zhangyanbo/learnable-novelty/blob/main/src/eca.py)

**ECA ranking (§4.1, Fig. 2, App. B) and comparison with Finzi et al.**
- "Rule 110 ranks highest in the entire rule space, the lowest scores, exactly zero, go to the rules whose sampled attractor dynamics are constant, the near-trivial periodic rules 1 and 2 score low, and the chaotic rule 30 lies in between, below the complex rule 54 (Figure 2)." Fig. 2 shows the top fourteen in order 110, 25, 73, 54, 22, 9, 41, 105, 150, 106, 45, 30, 142, 154, plus the reference rules 3, 1 and 2. — [arXiv v1 §4.1, Fig. 2](https://arxiv.org/html/2607.18433v1)
- App. B robustness:
  - At the reference configuration, rule 110's margin over the runner-up (rule 25) is 20.7 bits over three draws, and "stays between 9.6 and 28.4 bits at every other configuration where it is first".
  - The Spearman correlation with the reference ranking stays above 0.90 except at depth ≤ 2.
  - Exceptions: at τ = 4, rule 110 falls 0.5 bits behind rule 73 (18.2 ± 0.4 vs 18.7 ± 0.2). At depth 2, rule 25 overtakes it (13.5 ± 0.4 vs 10.4 ± 0.5). At depth 1, rule 110 falls to rank 40.
  - Rules 54 vs 30: "At the reference configuration the complex rule 54 ranks fourth and the chaotic rule 30 twelfth, with a score gap S54 − S30 = 10.9 ± 1.0 bits". The order "reverses exactly where the receptive field grows beyond radius two: one more layer (depth 4, radius three) puts rule 30 fifth and rule 54 thirteenth (gap −14.6 ± 1.0 bits), and a wider kernel (kernel 5, radius four) does the same (−11.9 ± 0.5 bits)."
  - Scan ranges: η ∈ [0.03, 30], λ ∈ [0.001, 1], τ ∈ {4, …, 64}, depth ∈ {1, …, 5}, kernel ∈ {3, 5, 7}, channels ∈ {64, …, 512}, N ∈ {128, …, 1024}.
  
  — [arXiv v1 App. B, Fig. 7](https://arxiv.org/html/2607.18433v1)
- The paper does not mention rule 15 and does not compare its ranking with the numbers in Finzi et al.; Finzi et al. appears only as the definitional source and as the expensive baseline ("the original work trains a neural network for every system scored"). — [arXiv v1 §1, §3](https://arxiv.org/html/2607.18433v1)
- Finzi et al.'s ECA results, for comparison (from the main-paper notes in this folder, citing Finzi v2 Fig. 3): S_T(54) ≈ 5.4e6 bits at ≈1e17 FLOPs, S_T(15) ≈ 0.2e6 after a low-compute peak of ≈1.1e6, and S_T(30) ≈ 0. "The ordering 54 > 15 > 30 only settles from about 1e13 FLOPs." Finzi's task is Y = F^48(X) on 64 cells after a 1000-step burn-in, with up to 𝒟 = 100M test tokens. — [Finzi et al. v2 §5.1, Fig. 3](https://arxiv.org/html/2601.03220v2#S5.SS1) (via `main_paper.md` and `small_scale_estimation.md` in this folder)
- **[own replication]** Repo code, reference ECA configuration (depth 3, 256 ch, kernel 3, λ = 0.03, η = 1, N = 512, τ = 32), mean ± sd over 3 draws using the repo's seeding: S(110) = 63.92 ± 0.64, S(25) = 43.23 ± 1.50, S(54) = 37.10 ± 0.42, S(30) = 25.47 ± 0.96, S(15) = 20.91 ± 0.63, S(0) = 0.00 bits. The 110 − 25 margin (20.7 bits) and the 54 − 30 gap (11.6 bits) match App. B (20.7; 10.9 ± 1.0). — script `C:/TMP/claude/d--GitHubD-Epiplexity/8843dc96-d6bc-4e7a-985b-ba9a35de477c/scratchpad/edl_levin/tests/eca_sanity.py`
- **[own replication] Null controls, same reservoir and ridge:**
  - "Noisy TV": X ∼ Bernoulli(1/2) and Y an independent Bernoulli(1/2) 32 × 64 target. S = 17.95 ± 1.29 bits at N = 512 and 6.88 ± 0.54 at N = 4096.
  - Trivial map Y_k = X for all 32 horizons: S = 3.17 ± 0.10 bits.
  - N-scaling with one reservoir draw, rules 110 / 54 / 30 / 15 / noise:

    | N | 110 | 54 | 30 | 15 | noise |
    |---|---|---|---|---|---|
    | 128 | 62.4 | 38.7 | 34.4 | 29.9 | 28.4 |
    | 512 | 63.2 | 36.6 | 24.9 | 20.3 | 17.6 |
    | 2048 | 63.7 | 35.5 | 16.8 | 11.0 | 9.5 |

  — scripts `.../edl_levin/tests/eca_sanity.py` and `.../edl_levin/tests/eca_N_scaling.py`

**Other experiments**
- *Inverse NCA* (§4.1, Fig. 3, Table 4, App. G.1). A two-channel unit-norm NCA on a width-64 ring. g is a radius-2 (kernel 5) lift conv to 128 channels with BatchNorm and GELU, then a pointwise conv and a projection to 2 channels. Training: 32 no-grad burn-in steps, Gaussian state noise sd 0.1, τ = 8, batch 2048, AdamW with cosine annealing, LR 1e-4, "S scaled by 1/100 before the gradient step", gradient clipping 0.5, 2,000 steps. The objective is max_θ S^φ(Y(X, θ)|X) (Eq. 12). "every seed develops complex solitons". The score plateaus by ≈1,500 steps at S = 86–89 (three seeds); the nine-seed direct and residual variants end at S = 83.2–90.8 (Fig. 6). — [arXiv v1 §4.1, Table 4, Fig. 6, App. G.1](https://arxiv.org/html/2607.18433v1)
- *MNIST* (§4.2, Fig. 4, App. G.2). A trainable MLP encoder (hidden 64, 128, 256) produces a code D = 64, normalized to unit length. It maximizes S^φ(Z = E(X)|X) against a frozen random MLP reservoir (depth 4, width 2048, λ = 3, η = 30), with batch 128, AdamW with cosine annealing, 500 steps and no labels. Linear-probe accuracy rises "from 0.53 to 0.89 and the 5-nearest-neighbor from 0.66 to 0.89". Robustness (150-step runs, Fig. 10): λ ≤ 0.3 or η ≤ 0.1 drive accuracy below 0.5 (e.g. λ = 0.01 → 0.32; η = 0.03 → 0.39), while learning rate, batch size, code dimension and depth keep 0.80–0.90. — [arXiv v1 §4.2, App. G.2, Fig. 10](https://arxiv.org/html/2607.18433v1)
- *RL* (§4.3, Table 1, App. H). PPO (Stable-Baselines3) with 8 envs, rollout 1024, minibatch 256, γ = 0.999, GAE λ = 0.98, LR 3e-4, 600,000 steps, 10 seeds, evaluated on 100 episodes. The reward is r_t = r_t^task + β(S_t − S_{t−1}), maintained online by RLS. β is calibrated so that the whole-episode bonus equals 0.1 × the random-policy task-return scale. The reservoir is an MLP of depth 4, width 32, λ = 0.3, with per-task horizon τ in [8, 48]. Table 1 (task only / epiplexity only / task + magnitude / task + epiplexity):
  - Acrobot −167±166 / −500±0 / −89±4 / −83±2
  - MountainCarContinuous 28±43 / −97±4 / 93±1 / 93±1
  - Hopper 1879±325 / 1006±21 / 516±128 / 2192±270
  - BipedalWalker 125±74 / −59±42 / 146±51 / 151±49
  - HalfCheetah 362±201 / −181±244 / 623±212 / 393±260
  - Walker2d 296±45 / 327±45 / 294±32 / 285±41
  - Swimmer 181±76 / −12±26 / 194±76 / 206±86
  - PointMaze 229±77 / 6±3 / 242±79 / 256±22
  - LunarLander 169±74 / −438±393 / −171±35 / 208±25
  - Pendulum −1044±42 / −1092±56 / −1016±21 / −987±23
  
  The epiplexity bonus improves on the task reward in 9 of 10 tasks; Walker2d is 4% lower. — [arXiv v1 Table 1, App. H](https://arxiv.org/html/2607.18433v1)
- *Continuous flows* (App. F.1). MLP reservoir (depth 4, width 64, λ = 0.1), 512 samples, 8 reservoir draws, 10 lead times from 0.1 to 1.0. Scores: Lorenz 46, Rössler 25, Thomas 17; linear systems (pure rotation, damped spiral, stable node) 6.9–8.4. — [arXiv v1 App. F, Fig. 9](https://arxiv.org/html/2607.18433v1)

**Runtime and validation**
- The paper gives no wall-clock or FLOP figures. It says the estimator is "cheap to evaluate, deterministic, and fully differentiable". Stated complexity: from-scratch recomputation along a stream costs O(N²m²); the RLS form costs O(m²) per step, plus O(m²D) for the SVD when scored (App. F.2). — [arXiv v1 §3, App. F.2](https://arxiv.org/html/2607.18433v1)
- **[own replication]** CPU timing (torch 2.8.0+cpu, 8 threads), including numpy data generation with the 1000-step burn-in: ≈0.7 s per ECA rule score at N = 512 (32,768 site-rows, m = 256, D = 32); ≈0.45 s for noise targets with no burn-in; ≈4.3 s at N = 4096. — `.../edl_levin/tests/eca_sanity.py`
- The only quantitative validations are: (a) the ridge readout vs the exact MDL minimizer on the same reservoir (App. D); (b) agreement with Wolfram classes and continuous-flow orderings (App. B, App. F); (c) one-at-a-time robustness scans (App. B, App. G.2). "Limitations" says: "In its present form the estimator captures only the shallow structure that is linearly readable from random features", and "once an optimized system exhausts what its observer can read, learnable novelty saturates." — [arXiv v1 App. B, D, F, §5](https://arxiv.org/html/2607.18433v1)

### Inferences
- **Relation to S_T.** S^φ is the model-cost term |M| of a two-part code. The program class is "fixed random features φ + linear readout". The model is chosen approximately by ridge rather than by minimizing the stated two-part objective, and priced with an arbitrary scale (α fixed at ½ rather than (a+D)/2, η a free "resolution", and u_Y a posited unit). It is neither prequential nor time-bounded: there is no training trajectory, and the "boundedness" is the observer's capacity (m, depth, receptive field, λ, η), not a compute budget T. Its bit values are therefore not commensurable with S_T. Rule 110 scores ≈64 "bits" here, while Finzi's S_T for rule 54 is ≈5.4e6 bits. Its dependence on data size also differs: S^φ for structured rules is almost flat in N (≈62–64 for rule 110 from N = 128 to 2048 [own replication]), whereas S_T grows with dataset size and compute. The paper's claim that |M*_φ| "is precisely the epiplexity of Finzi et al." holds at the level of the definition (Eq. 4), not for the estimator actually computed (Eq. 9).
- **ECA ranking against Finzi.** For the three Finzi rules, Z&L's reference observer gives 54 > 30 > 15 (37.1 > 25.5 > 20.9 [own replication]; the paper reports 54 fourth and 30 twelfth). Finzi gives 54 > 15 > 30. Both put rule 54 on top. They disagree on 15 vs 30, and Z&L's own App. B shows 30 overtaking 54 when the receptive field exceeds radius two. The N-scaling [own replication] shows that most of rule 30's and rule 15's scores are the finite-sample noise floor: both fall roughly in parallel with the pure-noise score as N grows (30 − noise ≈ 6–7 bits; 15 − noise ≈ 1.5–2.7 bits), while rules 110 and 54 stay flat. A plausible mechanism is that with τ = 32 and a radius-2 observer, rule 15's pure shift moves information out of the receptive field after about two steps, so a trivially simple rule looks like noise to this observer. This is an interpretation, not tested beyond the numbers above.
- **Does it vanish on noise?** Only asymptotically. The ridge fits sampling noise (λ = 0.03 is negligible relative to H̃ᵀH̃ ≈ (N·64/m)·I), giving a positive floor of ≈18 bits at the paper's N = 512 that decays slowly with N (28.4 → 17.6 → 9.5 for N = 128 → 512 → 2048) [own replication]. At the paper's settings the noisy TV therefore scores about as much as rule 15 and ≈70% of rule 30. The paper's statement that "the noisy television contributes nothing to |M|" (§2) and the RL claim of "immunity to the noisy-television problem" (§4.3) are not supported at finite N without a noise-baseline subtraction or a larger λ. In the RL setting the paper instead limits this by keeping the reservoir small (width 32), so that it "fits a structured trajectory but not a chaotic one" (App. H.2); that is a deliberate choice of observer, not a property of the estimator.
- **Trivial data.** Constant targets give exactly 0, which follows directly from target centering: centered Y = 0 gives W = 0. A deterministic but simple map (identity repeated over 32 horizons) gives ≈3 bits [own replication], which is appropriate: the log-det prices the 32 duplicated columns as essentially one direction.
- **Observer dependence and tuning.** The score is explicitly observer-relative (§5). The ECA code comment shows the τ = 32 plus radius-2 pairing was chosen with the stated aim of surfacing rule 110 as the maximum, and App. B shows the 30/54 order is set by the observer's radius. The ECA headline is therefore partly a property of a hand-picked observer. The authors present this observer-relativity as a feature.
- **Internal inconsistency.** App. D says rule 41 is "second under the ridge readout", but Fig. 2 and App. B put rule 25 second and rule 41 seventh, and App. D's runner-up margin (24.7 bits) differs from App. B's (20.7). They may use different draws; unresolved.
- **Suitability as a cheap S_T proxy.** It is useful as a differentiable, ~1-second, ranking-level proxy for "linearly readable structure from random features". It needs (i) a noise-floor baseline (score shuffled or independent targets of the same marginals at the same N) and (ii) a reservoir matched to the structure's scale before rankings can be compared with S_T.

### Gaps
- No validation of S^φ against Finzi's prequential or requential S_T on any shared dataset (ECA 15/30/54, induction tasks, etc.) was found in the paper or the repo.
- I did not run the RL, MNIST or NCA experiments, and the paper reports no wall-clock times for them. GPU requirements are not stated; the code falls back to CPU when CUDA is absent.
- I only saw the latest repo state (shallow clone, HEAD "Fail fast on ECA rules missing a Wolfram class"), so I cannot say whether results were produced by exactly this code version. The README's "steps-to-threshold" column does not appear in arXiv v1.
- Finzi et al. App. C.7 scores 10 rules including 110. I did not check whether rule 110 is Finzi's maximum there, so the cross-paper comparison is limited to rules 15, 30 and 54.

---

## Q3. Liu, Qi, Du & He, "Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain": the exact small-data modifications of the epiplexity estimator (Appendix B), formulas, and `epiplexity_mdl.py`

### Takeaway
Liu et al. keep Finzi's prequential form but make three changes for small, multi-epoch LLM fine-tuning datasets:
1. The online NLL is taken from epoch 1 only and then locked.
2. The "final loss" is recomputed on every training example with the multi-epoch model (in-sample), instead of using the last running loss.
3. The checkpoint (epoch) is chosen by a per-token MDL, S/N_train + H_val/N_val, with the entropy term from a 10% validation split.

The resulting S equals online regret against the final in-sample model. It is EDL plus n times the generalization gap, so memorization inflates S and only the MDL-optimal epoch selection holds it back. The model size is fixed for each reported value (no minimum over observers).

### Cited Findings

**Bibliographic facts**
- arXiv 2603.02218: v1 Tue 10 Feb 2026, v2 Fri 15 May 2026. Comments: "10 pages, 6 figures, 7 formulas, accepted by ICML 2026 position paper track". Subjects: cs.LG, cs.AI, cs.CL, cs.IT. Authors: Wei Liu, Siya Qi, Yali Du, Yulan He; affiliations "¹King's College London ²The Alan Turing Institute" (Du and He carry both). The running head is "From Self-Play to Self-Evolution". The PDF (v2) has 22 pages including appendices. — [arXiv abs 2603.02218](https://arxiv.org/abs/2603.02218); [PDF v2](https://arxiv.org/pdf/2603.02218v2)
- The repo README titles it "Position: Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain". The repo is a single commit, "init", dated 2026-07-07. Files: `epiplexity_mdl.py` (core, 399 lines), `run_all_epiplexity.py` (batch runner), `position/self_play.sh`, `position/calc.py`. Requirements: torch, transformers, peft, numpy, tqdm (Python 3.10 conda env). — [GitHub thinkwee/SelfPlay_SelfEvo_Gap](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap)

**Formulas in the paper**
- Bounded MDL optimizer: P* = argmin_{P∈P_{C,T}} |P| + E[log 1/P(X)] (Eq. 1); S_{C,T}(X) := |P*| (Eq. 2); H_{C,T}(X) := E[log 1/P*(X)] (Eq. 3); MDL_{C,T}(X) := S_{C,T}(X) + H_{C,T}(X) (Eq. 4). P_{C,T} is "a family of LLM observers implementable within budgets (C, T)". — [arXiv v2 §2.2](https://arxiv.org/html/2603.02218v2)
- Prequential estimate: |P_preq| ≈ Σ_{i=0}^{M−1} (log 1/P_i(Z_i) − log 1/P_M(Z_i)) (Eq. 7). Z_i is "the i-th training token", P_i the predictive distribution "before observing Z_i", and P_M "the predictive distribution of the fully trained model". — [arXiv v2 §4](https://arxiv.org/html/2603.02218v2)
- Main text: "The epiplexity is defined as the difference between the prequential and final training losses, representing the model's cumulative online regret. The MDL score combines the normalised epiplexity (model cost) and validation loss per-token (data cost). The algorithm then returns the epiplexity corresponding to the epoch that minimises this MDL score." — [arXiv v2 §4](https://arxiv.org/html/2603.02218v2)
- **Algorithm 1 ("Epiplexity Estimation via Prequential MDL")**. Inputs: observer M_θ0, D = D_train ∪ D_val, max epochs K, learning rate η. Initialize θ ← θ0, L_online ← 0, N_train ← 0, MDL* ← ∞, E* ← 0. For k = 1..K: for each x ∈ D_train, compute ℓ ← −log P_θ(x); if k = 1, set L_online ← L_online + ℓ and N_train ← N_train + CountTokens(x); then θ ← θ − η∇_θℓ. At the end of each epoch: L_train ← Σ_{x∈D_train} −log P_θ(x); L_val ← Σ_{x∈D_val} −log P_θ(x); N_val ← CountTokens(D_val); **S ← (L_online − L_train)/ln 2**; **MDL ← S/N_train + (L_val/ln 2)/N_val**; if MDL < MDL*, set MDL* ← MDL and E* ← S. Return E*. — [arXiv v2 §4, Algorithm 1](https://arxiv.org/html/2603.02218v2)
- **Appendix B, verbatim:**
  - "Given a synthetic dataset generated by the PROPOSER, we train a SOLVER model on this data and use the prequential coded length at the checkpoint that achieves the optimal MDL as the epiplexity. As shown, although the entropy decreases monotonically and the coded length increases monotonically, all experimental runs attain the optimal MDL at an intermediate stage of training."
  - "Due to the relatively small data scale in our proof of concept experiments, we adopt an improved computation scheme compared with Finzi et al. (2026). All tokens within the same batch are assigned an identical training loss value, and per-token averages are used when computing the prequential coded length |P_preq| and entropy H. These averaged quantities are then summed to obtain a per-token MDL value, which is used to determine the MDL optimal point."
  - "In addition, the epiplexity computation in Finzi et al. (2026) assumes a sufficiently large dataset, such that training for a single epoch suffices and the final loss −log P_M(Z_last) can be used as an approximation of the loss −log P_M(Z_i) incurred by the trained observer on each data point. In our setting, the dataset is relatively small, so we adopt a more accurate procedure by re-computing the loss for each training example after the observer has been fully trained. Furthermore, for multi-epoch training, we define the prequential coded length as the difference between the training loss in the first epoch and the loss of the model trained for multiple epochs, evaluated on all data points, to satisfy the assumptions underlying prequential coding."
  
  — [arXiv v2 App. B](https://arxiv.org/html/2603.02218v2)
- Appendix A setup:
  - LoRA with rank 16, alpha 32, dropout 0.05, on "the Qwen2.5 family of models, spanning from 0.5B to 14B parameters".
  - Seed datasets are generated by Qwen2.5 7B, Qwen2.5 14B and Qwen3 4B; 10% is held out for validation "to calculate the bounded entropy".
  - Max sequence length 2048, LR 1e-4, ≤ 20 epochs, MDL and prequential length computed at the end of each epoch.
  - Early stopping after 5 epochs without MDL improvement; "in practice, all runs terminate before reaching the 20-epoch limit."
  - The self-play runs follow Absolute Zero's RL with Qwen2.5 3B, and the same model is used to compute epiplexity.
  
  — [arXiv v2 App. A](https://arxiv.org/html/2603.02218v2)

**Code (`epiplexity_mdl.py`)**
- Defaults: `Qwen/Qwen2.5-3B-Instruct`, `max_length 2048`, `batch_size 4`, `grad_accum 4`, `lr 1e-4`, AdamW `weight_decay 0.1`, `max_epochs 20`, `dtype bf16`, `val_ratio 0.1`, `patience 5`, LoRA `r 16`, `alpha 32`, `dropout 0.05`, `target_modules "q_proj,v_proj"`. — [epiplexity_mdl.py L27–58](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/epiplexity_mdl.py)
- Text format: `f"Question: {question}\nAnswer: {answer}"`. Labels are the input ids with only padding masked (−100), so the loss covers question and answer tokens. — [epiplexity_mdl.py L69–83, L228–232](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/epiplexity_mdl.py)
- Online NLL is recorded per micro-batch as `nll = outputs.loss.item() * num_tokens`, with `num_tokens = attention_mask.sum()`, in `model.train()` mode. The step is taken only every `grad_accum` micro-batches. NLLs are appended "Only for Epoch 1" and then locked: `fixed_online_nats = sum(all_batch_nlls)` and `fixed_train_tokens = sum(all_batch_tokens)`. — [epiplexity_mdl.py L224–255](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/epiplexity_mdl.py)
- `compute_mdl` at the end of each epoch:
  - `h_train_nats` is the current model's NLL over the full training loader, in `eval()` mode under `no_grad`.
  - `estimated_final_model_cost_nats = (h_train_nats/total_train_capacity_tokens) * fixed_train_tokens`, which equals `h_train_nats` because the same data is re-tokenized.
  - `s_preq_bits = (fixed_online_nats − estimated_final_model_cost_nats)/ln 2`.
  - `mdl_per_token = s_preq_bits/fixed_train_tokens + (h_val_nats/ln 2)/val_tokens`. The code comment says this "is equivalent to assuming we care about a test set of size equal to the training set size."
  - The reported epiplexity is `s_preq_bits` (total bits) at the minimum of `mdl_per_token`.
  
  — [epiplexity_mdl.py L311–388](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/epiplexity_mdl.py)
- The batch runner uses 5 solver models (Qwen2.5-0.5B/1.5B/3B/7B/14B-Instruct) × 6 data directories (14b, 14b_coder, 3b_coder, 7b, 7b_coder, qwen3_4b) × 3 task types (code_f induction, code_i abduction, code_o deduction). Jobs are spread over `NUM_GPUS = 4`. The 14B model uses batch 1, grad_accum 16 and gradient checkpointing. `position/calc.py` contains the same estimator logic (it differs only in comments). — [run_all_epiplexity.py](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/run_all_epiplexity.py); [position/calc.py](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/position/calc.py)
- README method notes: "Epiplexity `S = Σ(online_loss_i − final_loss_i) / ln 2`… The online NLL is accumulated during epoch 1 and then locked; later epochs only update the model and recompute the final-model NLL." No data files are shipped: input must be produced by running Absolute Zero Reasoner self-play. — [README](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap/blob/main/README.md)

**Results**
- Experiment 1 (Fig. 5). Stronger proposers (Qwen2.5 7B → 14B → Qwen3 4B) yield more epiplexity, and "As the SOLVER size increases, the learnable information first increases and then decreases", which the paper attributes to larger models opting "for direct memorisation". Induction is much higher than abduction or deduction. Read from the y-axes: induction ≈ 2.4e5–3.4e5 bits, abduction ≈ 0.9e5–1.3e5, deduction ≈ 0.8e5–1.2e5 [read from plot]. The x-axis labels run 0.5b…72b, although App. A says 0.5B–14B. — [arXiv v2 §4, Fig. 5](https://arxiv.org/html/2603.02218v2)
- Experiment 2 (Fig. 6, Table A1). Epiplexity (×10³ bits) over nine self-play iterations "fluctuates substantially" while rewards rise:
  - Induction: 75.7, 76.7, 458.9, 459.6, 76.5, 76.7, 462.1, 77.2, 474.3
  - Abduction: 48.8, 49.4, 49.2, 162.5, 50.2, 177.6, 181.3, 183.3, 49.2
  - Deduction: 146.2, 156.3, 148.9, 157.4, 45.4, 45.2, 151.7, 151.8, 160.0
  
  — [arXiv v2 App. C, Table A1](https://arxiv.org/html/2603.02218v2)

### Inferences
- **Precise relation to EDL and S_T.** Liu's S = Σ_i ℓ_online,i (epoch 1) − Σ_i ℓ(θ_T; z_i) over the same training tokens. This is exactly the regret R_T(S) against the final model. By the ISIT identity EDL_T = E[R_T] + n·(E[L_train(θ_T)] − L(θ_T)), we get **S_Liu = EDL_T + n·(L(θ_T) − L_train(θ_T))**: EDL plus n times the generalization gap. So S_Liu ≥ EDL whenever the model fits training data better than held-out data, and it grows with memorization; Appendix B itself notes the coded length "increases monotonically" over epochs. Finzi's single-pass estimate uses the final *running* training loss, which on fresh data is a test-loss estimate, so it sits closer to EDL than to S_Liu. Liu's claim that recomputing in-sample losses is "more accurate" holds only for the "model-only code" reading of Finzi's Eq. 8. It moves away from a generalization-based quantity.
- **What controls memorization.** Only the per-token MDL epoch selection. Because S/N_train rises with epochs and H_val/N_val eventually rises under overfitting, the argmin lands at an "intermediate stage". The per-token weighting sets the data-cost weight as if the test set had N_train tokens, per the code comment. That differs from Finzi's total-code criterion over a 𝒟-token dataset: in effect 𝒟 = N_train. It is also chosen over epochs for a fixed model, not over observers at fixed compute.
- **It is not S_T.** Each reported number is |P_preq| for one solver size at its best epoch. The "inverted-U vs solver size" in Fig. 5 is a curve of per-model prequential lengths. S_T at a budget would take the minimum-MDL observer across sizes. The quantity is also joint over question + answer tokens (S(X) of the text), not S(Y|X) on the answer, because labels include the question.
- **Probable estimator artifacts** (from reading the code; untested):
  1. The online NLL is computed in train mode with LoRA dropout 0.05, while the final NLL is computed in eval mode. This biases S upward by roughly N_train × (dropout-induced loss increase).
  2. `num_tokens = attention_mask.sum()` counts one unpredicted token per sequence, while HF's `outputs.loss` averages over shifted labels. The small overcount appears on both sides, so it mostly cancels in S but inflates both token counts.
  3. The optimizer steps only every 4 micro-batches, so "prequential" coding is at the 16-example granularity, which is legitimate.
  4. The near-bimodal Table A1 values (≈76k vs ≈460k bits for induction, ≈49k vs ≈160–183k for abduction, ≈45k vs ≈150k for deduction) look like jumps in the selected epoch (epoch 1 vs later) rather than smooth changes in the data. This is speculative without the `metrics_history.json` logs.
- **Noise and trivial data.** On pure-noise tokens, multi-epoch LoRA training can lower in-sample NLL (memorization) while validation NLL stays flat or rises. The per-token criterion should then select epoch 1, where S = online − in-sample-after-one-epoch, which is small but not zero for an overparameterized LLM. Vanishing on noise therefore depends on the epoch selection, not on the estimator. For trivial data the online loss is low from the start, so S is small, as intended.

### Gaps
- Dataset sizes (examples or tokens per proposer and task), GPU type, GPU-hours and wall-clock times are not reported in the paper or README. The runner assumes 4 GPUs.
- The paper includes no noise or random-label control and no comparison with Finzi's single-epoch estimate on the same data.
- Whether Fig. 5 contains 32B/72B points, as the axis labels suggest, is unclear: App. A and the runner stop at 14B.
- The ICML 2026 proceedings entry (PMLR volume and pages) was not verified.

---

## Q4. For each estimator, what it gets right and wrong as an epiplexity estimator: vanishing on noise, vanishing on trivial data, compute-boundedness, observer-dependence

### Takeaway
None of the three estimates S_T itself. Each fixes one observer or algorithm instead of minimizing the two-part code over a compute-bounded class. EDL is the most principled on noise and memorization, going to zero or below on random labels by theorem, but it measures a fixed algorithm's generalization gain, not a model description length. Zhang & Levin's score is cheap and differentiable but has a finite-sample noise floor and an arbitrary bit scale. Liu et al.'s version is closest to Finzi's operational recipe but counts memorized in-sample fit, relying on epoch selection to cap it.

### Cited Findings
- EDL on noise: "For data with random labels Y ⊥ X, EDL_T(A, n) ≲ 0 for any algorithm, regardless of training loss achieved." — [ISIT Cor. 14](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf). arXiv v1 Prop. 5.1: E[EDL] = 0 for i.i.d. random labels. — [arXiv v1 §5.1](https://arxiv.org/abs/2601.04728v1)
- EDL is compute-indexed but algorithm-specific: "EDL depends not only on the final model, but also on the learning trajectory… it is not an algorithm-independent notion of final information content." — [ISIT Remark 4](https://elizabethdonoway.com/papers/EDL_ISIT_full.pdf). "EDL depends on the training algorithm… this complicates comparison across settings." — [arXiv v1 §7.2](https://arxiv.org/abs/2601.04728v1)
- Z&L on noise and trivial data (claimed): "a noisy television is all residual, and a dark room contributes to neither part"; "that boundedness is why both the noisy television and the dark room contribute zero to |M*_φ|". — [arXiv 2607.18433 §1–2](https://arxiv.org/html/2607.18433v1). Measured [own replication]: noise targets score 17.95 ± 1.29 bits at N = 512 and 6.88 ± 0.54 at N = 4096; constant targets score exactly 0; the identity map scores 3.17 bits. — `.../edl_levin/tests/eca_sanity.py`
- Z&L observer dependence: the 30/54 order flips with receptive-field radius (App. B). A small reservoir width (32) is chosen in RL so that the observer "fits a structured trajectory but not a chaotic one". — [arXiv 2607.18433 App. B, App. H.2](https://arxiv.org/html/2607.18433v1)
- Liu et al. observer dependence: "the same object can appear structured to a stronger observer and random to a weaker one". The coded length "increases monotonically" with epochs. — [arXiv 2603.02218 §2.2, App. B](https://arxiv.org/html/2603.02218v2)

### Inferences
- **Comparison table** (my synthesis of the cited findings above):

| Property | Finzi prequential S_T (reference) | EDL (Donoway) | S^φ (Zhang & Levin) | S_Liu (Liu et al.) |
|---|---|---|---|---|
| What is subtracted | final (running) train loss at the end of one pass | n × held-out/population loss of the final model | nothing: model-cost term of a two-part code | in-sample NLL of the multi-epoch model on the same training tokens |
| Minimizes over observers at fixed compute | yes (sweep of N, D and hyperparameters; Pareto hull) | no (fixed A; the "intrinsic" sup_A is a maximization) | no (one fixed reservoir) | no (fixed solver; minimum over epochs only) |
| Multi-epoch | not allowed (assumes a single pass) | yes (Cor. 12) | n/a | yes, but in-sample |
| Random labels / noise | ≈0 in theory; the estimate is exposed to baseline noise at large D | ≲0 by theorem; test-noise std ≈ n·σ_ℓ/√n_test | positive finite-N floor (~18 bits at the paper's ECA N) that shrinks slowly with N | ≈ memorization at the chosen epoch; relies on the epoch criterion |
| Trivial data | small | 0 with calibrated init; small otherwise | 0 for constant targets; small for simple maps | small |
| Compute-bounded | yes (T = FLOPs of training + evaluation) | indexed by T but no bound on evaluation; "compute-bound-agnostic" (v1 App. D.2) | no time bound; capacity-bounded (m, depth, λ, η) | only via the fixed model and ≤ 20 epochs |
| Observer-dependent | yes (model class) | yes (A, θ0) | yes, strongly (reservoir radius flips rankings) | yes (solver size) |
| Units comparable to S_T | same | same order for one-pass runs | no (≈10²-bit scale, α = ½ arbitrary) | same kind (bits of prequential excess), but inflated by the generalization gap |
| Cost | many training runs | 1 run + held-out evaluation | ≈1 s CPU per system [own replication] | 1 LoRA run of ≤ 20 epochs per (solver, data) pair |

- **Best use for a small-scale epiplexity project.**
  - EDL is the most defensible replacement for "area above final loss" when data are finite and training is multi-epoch: log first-epoch losses, early-stop on validation, and subtract n·L_val. It inherits Finzi's single-pass form when T = n.
  - Liu's in-sample recomputation should be avoided, or reported next to EDL, because the difference between the two is n × generalization gap.
  - Z&L's S^φ can serve as a cheap pre-screen or differentiable objective. It should always be paired with an N-matched shuffled-target baseline, because otherwise its noise floor can reorder low-structure systems.

### Gaps
- None of the three papers applies its estimator to the same datasets as Finzi et al. (ECA 15/30/54, easy vs hard induction, CIFAR/ImageNet-style, OpenWebText), so all cross-estimator comparisons above rely on theory or on my small ECA replication.
- None reports a random-label or shuffled-target experiment with numbers, apart from EDL's un-quantified companion-paper claim.

---

## Q5. Bibliographic data for BibTeX

### Takeaway
The bibliographic details below come from arXiv abstract-page metadata, the Crossref API (for the ISIT DOI) and the NeurIPS proceedings page. The ISIT paper is single-author and has a different title from the arXiv preprint.

### Cited Findings
- arXiv 2601.04728: "Excess Description Length of Learning Generalizable Predictors"; Donoway, Elizabeth; Joren, Hailey; Roger, Fabien; Leike, Jan. Citation date 2026/01/08; cs.LG (cs.AI). — [arXiv abs](https://arxiv.org/abs/2601.04728)
- ISIT: "Excess Description Length: An Information Measure for Generalizable Structure Learned from Finite Data"; Elizabeth Donoway; 2026 IEEE ISIT, Guangzhou, China, 2026-06-28 to 2026-07-03; pp. 1–6; IEEE; DOI 10.1109/ISIT62367.2026.11654044. — [Crossref](https://api.crossref.org/works/10.1109/ISIT62367.2026.11654044)
- arXiv 2607.18433: "Intelligence from Learnable Novelty"; Zhang, Yanbo; Levin, Michael. Citation date 2026/07/20; cs.LG (cs.AI, nlin.AO). — [arXiv abs](https://arxiv.org/abs/2607.18433)
- arXiv 2603.02218: "Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain"; Liu, Wei; Qi, Siya; Du, Yali; He, Yulan. v1 2026-02-10, v2 2026-05-15; "accepted by ICML 2026 position paper track". — [arXiv abs](https://arxiv.org/abs/2603.02218)
- NeurIPS 2025 (related): "Quantifying Elicitation of Latent Capabilities in Language Models"; Donoway, Joren, Somani, Sleight, Michael, DeWeese, Schulman, Perez, Roger, Leike. — [NeurIPS proceedings](https://proceedings.neurips.cc//paper_files/paper/2025/hash/dd16e008534dc52fab72fdedad069d5c-Abstract-Conference.html)

### Inferences
Suggested BibTeX, assembled from the facts above. The arXiv DOIs follow the standard 10.48550/arXiv.<id> pattern.

```bibtex
@misc{donoway2026edl,
  title         = {Excess Description Length of Learning Generalizable Predictors},
  author        = {Donoway, Elizabeth and Joren, Hailey and Roger, Fabien and Leike, Jan},
  year          = {2026},
  month         = jan,
  eprint        = {2601.04728},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2601.04728},
  url           = {https://arxiv.org/abs/2601.04728}
}

@inproceedings{donoway2026edlisit,
  title     = {Excess Description Length: An Information Measure for Generalizable Structure Learned from Finite Data},
  author    = {Donoway, Elizabeth},
  booktitle = {2026 IEEE International Symposium on Information Theory (ISIT)},
  address   = {Guangzhou, China},
  pages     = {1--6},
  year      = {2026},
  month     = jun,
  publisher = {IEEE},
  doi       = {10.1109/ISIT62367.2026.11654044}
}

@misc{zhang2026learnablenovelty,
  title         = {Intelligence from Learnable Novelty},
  author        = {Zhang, Yanbo and Levin, Michael},
  year          = {2026},
  month         = jul,
  eprint        = {2607.18433},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.18433},
  url           = {https://arxiv.org/abs/2607.18433},
  note          = {Code: https://github.com/Zhangyanbo/learnable-novelty}
}

@misc{liu2026selfplay,
  title         = {Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain},
  author        = {Liu, Wei and Qi, Siya and Du, Yali and He, Yulan},
  year          = {2026},
  eprint        = {2603.02218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2603.02218},
  url           = {https://arxiv.org/abs/2603.02218},
  note          = {Position paper, ICML 2026 position paper track (per arXiv comments); v2, 15 May 2026. Code: https://github.com/thinkwee/SelfPlay_SelfEvo_Gap}
}

@inproceedings{donoway2025elicitation,
  title     = {Quantifying Elicitation of Latent Capabilities in Language Models},
  author    = {Donoway, Elizabeth and Joren, Hailey and Somani, Arushi and Sleight, Henry and Michael, Julian and DeWeese, Michael R. and Schulman, John and Perez, Ethan and Roger, Fabien and Leike, Jan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

### Gaps
- ICML 2026 proceedings details for Liu et al. (PMLR volume and pages) and NeurIPS 2025 volume and pages for the elicitation paper were not verified.
- The ISIT paper's abstract was not available through Crossref, and the IEEE Xplore page content did not load. The abstract quoted above is from the author-hosted PDF.
- The arXiv ID 2603.02218 carries a v1 date of 10 Feb 2026, as shown on the abstract page. I did not investigate the ID/date mismatch.
