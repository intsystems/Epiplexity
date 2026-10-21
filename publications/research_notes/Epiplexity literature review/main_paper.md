# The founding epiplexity paper (arXiv 2601.03220) and its official code (shikaiqiu/epiplexity)

Provenance and conventions used in these notes
- Primary sources read in full: the arXiv abstract page, the v2 HTML (converted to text with LaTeX taken from MathML alttext), the v1 HTML (word-level diffed against v2), the v1 and v2 PDFs (65 pages each; figures rendered and read visually), and a full clone of the GitHub repo (all scripts, configs and notebook code cells). Accessed 2026-09-23.
- Base links: ABS = https://arxiv.org/abs/2601.03220 ; V2 = https://arxiv.org/html/2601.03220v2 ; V1 = https://arxiv.org/html/2601.03220v1 ; REPO = https://github.com/shikaiqiu/epiplexity
- "[read from plot]" means the number was read by eye from a rendered figure and is approximate (about ±10%). "[inference]" means my own calculation or reasoning, not a claim made by the authors. "[COLAB]" flags details relevant to the goal of estimating epiplexity with MLPs or small transformers on one Colab GPU.
- Units: the paper's definitions use bits (log base 2). The repo logs nats (natural-log cross-entropy and KL) and converts to bits only in notebooks or at the logging step (divide by ln 2).

## Q1. Exact definitions: time-bounded two-part (MDL) code, epiplexity S_T(X), time-bounded entropy H_T(X), the compute bound T, the program class

### Takeaway
Epiplexity S_T(X) is the length in bits, |P*|, of the program P* that minimizes the time-bounded two-part code |P| + E[log 1/P(X)] over all prefix-free programs on a fixed universal Turing machine (UTM) that can both evaluate probabilities and sample within T(n) steps. Time-bounded entropy H_T(X) = E[log 1/P*(X)] is the expected code length of the data under that P*. In practice X is the entire dataset (the "test dataset", of 𝒟 tokens). T is measured in FLOPs as 6ND (replaying training of an N-parameter model on D tokens) plus 2N𝒟 (evaluating it on X). The program class is restricted to neural-network training runs, coded either prequentially or requentially.

### Cited Findings
**Paper identity and versions**
- The title is verified exactly: "From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence". The authors are Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter and Andrew Gordon Wilson (Carnegie Mellon University and New York University). The PDF marks Finzi, Qiu and Jiang as equal contributors (∗). — [arXiv abs](https://arxiv.org/abs/2601.03220); [v1 PDF](https://arxiv.org/pdf/2601.03220v1)
- v1 was submitted Tue 6 Jan 2026 (3,934 KB) and v2 on Mon 16 Mar 2026 (3,935 KB). No v3 existed as of 2026-09-23. Subjects are cs.LG and stat.ML, and the arXiv comment is "Code available at" the GitHub repo. — [arXiv abs](https://arxiv.org/abs/2601.03220)
- The v1-to-v2 changes (from my word-level diff of the two HTML versions) are mostly editorial:
  - "CSPRNG" was renamed "PRG" in Definition 3 and Theorems 9 and 12.
  - Theorem 9 now uses explicit H_Poly/S_Poly notation.
  - Theorem 12 was restated as H_Poly(G(U_k)) − H_Poly(U_k) > n − k − nε(k) − c.
  - §4.1 gained a sentence noting that prequential losses are effectively test-loss estimates.
  - The emergence discussion (§5.3.2) was softened.
  - Related work was added: excess entropy, surplus description length (Whitney et al. 2020), information transfer (Zhang et al. 2020), and Achille & Soatto 2025.
  - An ADO footnote and the code link were added, and an Appendix H paragraph calling regret "a generalization of epiplexity for all coding schemes" was removed.
  
  No experimental numbers changed in the text. — [v1 HTML](https://arxiv.org/html/2601.03220v1), [v2 HTML](https://arxiv.org/html/2601.03220v2)
- The paper frames epiplexity as the dual of MDL: "While MDL is a criterion for model selection given a fixed dataset, epiplexity ... can be viewed as its dual: a criterion for data selection given a fixed computation budget." — [v2 §2.3](https://arxiv.org/html/2601.03220v2#S2.SS3)

**Background definitions (paper §2)**
- **Def. 1 (prefix Kolmogorov complexity):** K(x) = min{|p| : 𝒰(p) = x}. K(x|y) is the length of the shortest program that outputs x given y. — [v2 §2.1](https://arxiv.org/html/2601.03220v2#S2.SS1)
- **Def. 3 (non-uniform PRG):** G stretches k bits to n bits. For every non-uniform probabilistic polynomial-time (PPT) distinguisher D_k with poly(k) advice, |Pr_{s∼U_k}[D_n(G(s))=1] − Pr_{u∼U_n}[D_n(u)=1]| = ε(k) < negl(k) (Eq. 1).
- **Def. 4 (non-uniform one-way function, OWF):** f is computable in poly(n) and, for every non-uniform PPT A_n, Pr_{x∼U_n}[A_n(f(x)) ∈ f⁻¹(f(x))] < negl(n). — [v2 §2.1](https://arxiv.org/html/2601.03220v2#S2.SS1)
- **Def. 5 (naive sophistication):** nsoph_c(x) = min_S {K(S) : K(x|S) > log|S| − c} (Eq. 2). GitHub issue #1 points out that the definition omits the condition x ∈ S. — [v2 §2.2](https://arxiv.org/html/2601.03220v2#S2.SS2); [issue #1](https://github.com/shikaiqiu/epiplexity/issues/1)
- **Def. 6 (two-part MDL):** for data x ∈ {0,1}^{n×d} and model set 𝓗, L(x) = min_{H∈𝓗} L(H) − log P(x|H). — [v2 §2.3](https://arxiv.org/html/2601.03220v2#S2.SS3)

**Core definitions (paper §3)**
- **Def. 7 (time-bounded probabilistic model).** Let T: ℕ→ℕ be non-decreasing and time-constructible, and 𝒰 a fixed prefix-free UTM. A prefix-free program P is a T-time probabilistic model over {0,1}^n if:
  - (Evaluation) on input (0,x), 𝒰(P,(0,x)) halts within T(n) steps and outputs Prob_P(x) ∈ [0,1] with a finite binary expansion;
  - (Sampling) on input (1,u), with u ∈ {0,1}^∞ an infinite random tape, it halts within T(n) steps and outputs Sample_P(u) ∈ {0,1}^n;
  - with Σ_x Prob_P(x) = 1 and Pr_{u∼U_∞}[Sample_P(u) = x] = Prob_P(x) for all x.
  
  𝒫_T is the set of all such programs. Italic P denotes the probability mass function and roman P the program. Footnote 3: other constraints can replace 𝒫_T with 𝒫_𝓕, "One such possibility is to constrain the function class to all models reachable by a given optimization procedure with a given neural network architecture." — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Def. 8 (epiplexity and time-bounded entropy).** For X on {0,1}^n, P* = argmin_{P∈𝒫_T} { |P| + E[log 1/P(X)] } (Eq. 3), with ties broken by the smallest program, logs base 2 and |P| in bits. Then **S_T(X) := |P*|** and **H_T(X) := E[log 1/P*(X)]** (Eq. 4), and MDL_T(X) := S_T(X) + H_T(X) is the "total time-bounded information content". — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Basic properties:**
  - (1) S_T(X) ≥ 0 and H_T(X) ≥ 0.
  - (2) H(X) ≤ S_T(X) + H_T(X) ≤ n + c₁.
  - (3) MDL_{T′}(X) ≤ MDL_T(X) whenever T′(n) ≥ T(n).
  - (4) MDL_{T′}(f⁻¹(X)) ≤ MDL_T(X) + |f| + c₂ with T′(n) = T(n) + Time(f), for a program f implementing a bijection in fixed time.
  
  The paper stresses that under a fixed compute budget "having a short program for f⁻¹ does not imply one for f, and vice versa". — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Worked examples:**
  - For uniform U_n with a constant time bound T(n) ≥ c₁, S_T(U_n) + H_T(U_n) ≤ n + c₂ and H_T ≥ H = n, so S_T(U_n) ≤ c₂.
  - For the mixture of 0101… and 1010… (probability ½ each) with T(n) = Θ(n), S_T = O(1) and H_T = O(1).
  
  — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Def. 11 (conditional versions).** 𝒫^X_{T(n)} is the set of models P such that for each fixed x the conditional P_{Y|x} ∈ 𝒫_{T(n)}. Then P*_{Y|X} = argmin {|P| + E_{(X,Y)}[−log P(Y|X)]} (Eq. 5), S_T(Y|X) := |P*_{Y|X}| and H_T(Y|X) := E_{(X,Y)}[−log P*_{Y|X}(y|x)] (Eq. 6).
  - "In general this definition is not equivalent to the difference of the joint and individual entropies, H_T(Y,X) − H_T(X) ≠ H_T(Y|X)."
  - Conditioning on a deterministic string d is also defined (Eq. 7), e.g. S_T(X|m) given a model m, for "what additional data is most useful ... on top of a pretrained LLM".
  
  — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Machine-learning instantiation.** "We take the random variable X to refer to the entire dataset of interest, i.e. typically a collection X = [X₁, X₂, …] of many iid samples"; E[log 1/P(X)] scales with dataset size, epiplexity "typically grows with the size of the dataset", and "the epiplexity of a typical dataset is orders of magnitudes smaller than the random information content." — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **What T measures in practice.** "We will measure time by the number of floating-point operations (FLOPs) and dataset size by number of tokens, so that training a model with N parameters on D tokens takes time approximately 6ND ..., while evaluating it on X takes time 2N𝒟 with 𝒟 = |X|." X is called the test dataset, as distinct from the training data one is free to choose. The constraint is 6ND + 2N𝒟 ≤ T. — [v2 §4, §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)
- **Program class in practice.** "We restrict attention to probabilistic models parameterized by neural networks". Directly storing architecture and weights "can significantly overestimate the information content in the weights, particularly for large models trained on relatively little data", so they instead encode the training process via prequential coding (Dawid 1984) or requential coding (Finzi et al. 2026). — [v2 §4](https://arxiv.org/html/2601.03220v2#S4)
- **Conditional estimation in practice.** For S_T(Y|X), "decoding time takes into account both the input (X) and label (Y) tokens, but code length only needs to be computed for the label tokens (tokens contributing to the training loss)." — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **Architecture.** The default is the GPT-2 transformer trained with Adam; learning rates are tuned on a small model and transferred via μP and CompleteP. "In μP, the per-layer learning rate is base learning rate divided by the input dimension, so our reported base learning rate is larger than typical learning rates used for Adam." — [v2 App. C](https://arxiv.org/html/2601.03220v2#A3)

### Inferences
- [inference] S_T is not monotone in T by definition; only MDL_T is (property 3). The emergence experiment (Fig. 6) shows S_T falling at high compute. Any Colab replication should therefore report the whole S_T(T) curve, not a single value.
- [inference][COLAB] S_T and H_T depend on the test-set size 𝒟, the tokenization, the compute bound T and the program class (architecture plus optimizer). Comparisons across datasets are meaningful only with all four held fixed. For MLP classifiers the natural object is the conditional S_T(Y|X) (Def. 11): code length counts only label predictions, while FLOPs count forward passes on inputs.
- [inference] Def. 7 requires both sampling and evaluation within T. Autoregressive transformers, and classifiers with softmax outputs, satisfy both trivially at the same cost, so no extra sampling cost is charged in the estimators.

### Gaps
- The paper gives no finer-grained FLOP model than 6ND + 2N𝒟. Attention FLOPs (quadratic in sequence length) and embedding FLOPs are not separately accounted for. The code counts all parameters, including embeddings: `P = sum(p.numel() ...)` in [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py).
- No formal treatment of how a finite-precision neural-network program maps to the UTM definition (constants for architecture, initialization and optimizer are "omitted"). — [v2 §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)

## Q2. Theorems and propositions: statements and assumptions

### Takeaway
All rigorous results are asymptotic and rely on cryptographic assumptions under a polynomial-time bound: existence of PRGs, one-way functions, pseudorandom functions (PRFs) or one-way permutations, secure against non-uniform PPT adversaries. They show:
- PRG outputs have near-maximal H_Poly and near-zero S_Poly (Thm 9).
- High-epiplexity random variables exist, but only S_Poly = Ω(log n) (Thm 10).
- Deterministic maps can raise H_Poly by about n − k (Thm 12, Paradox 1).
- One-way permutations create a factorization gap of ω(log n) in H_Poly (Thm 13, Paradox 2), and no polynomial-time model fitting the forward direction can satisfy Bayes' rule (Cor. 26).

Paradox 3 (induction and emergence) is supported only by a definition (Def. 14), a stated conjecture and experiments, not a theorem. Appendix B adds non-cryptographic results about how the neural-network estimators scale (Thm 30, B.3, B.4.2, B.4.3).

### Cited Findings
- **Lemma 15 (maximum expected description length).** For any X on {0,1}^n there are constants c₁, c₂, c₃ with S_T(X) + H_T(X) ≤ n + c₁ for T(n) ≥ c₂n + c₃. The proof uses the uniform model Q_unif, a constant-size program running in linear time. — [v2 App. A](https://arxiv.org/html/2601.03220v2#A1)
- **Lemma 16.** For X = U_n and T(n) ≥ c₂n + c₃, n ≤ H_T(X) ≤ n + c₁ (lower bound since E[−log Q] = H + KL ≥ H). — [v2 App. A](https://arxiv.org/html/2601.03220v2#A1)
- **Theorem 9 (PRGs: high random content, little structure; main text).** For any G ∈ PRG that stretches its input to n = poly(k) bits with advantage at most ε(k): n − 2 − nε(k) < H_Poly(G(U_k)) ≤ n + c, and S_Poly(G(U_k)) ≤ c + nε(k).
  - Contrast: Shannon entropy is H(G(U_k)) = k, and polynomial-time-bounded Kolmogorov complexity is at most k + c.
  - Appendix Theorem 17 gives the lower bound "for every polynomial time bound T(n)". The proof builds threshold distinguishers D_t(x) = 𝟙{P(x) ≥ 2^{−(n−t)}}, uses |P| ≤ n + c so that P serves as polynomial-size advice, and shows Pr[D_t(U_n)=1] ≤ 2^{−t}. Appendix Theorem 19 gives the S_T bound.
  
  — [v2 §3](https://arxiv.org/html/2601.03220v2#S3); [App. A.1, A.3](https://arxiv.org/html/2601.03220v2#A1.SS1)
- **Theorem 10 (existence of high-epiplexity random variables).** Assuming OWFs secure against non-uniform PPT adversaries, there is a sequence {X_n} over {0,1}^n with S_Poly(X_n) = Ω(log n).
  - The authors caution that "logarithmically growing information content only admits a very modest amount of structural information, still far from the power law scaling we see with some natural data", and that "the argument is nonconstructive".
  - Appendix Theorem 24 states it via a PRF family F_K: {0,1}^m → {0,1}^k secure with advantage ε(m); for n = m + k ≥ n₀ such X_n exist.
  - Construction: sample x ∼ U_m and output z = (x, F_K(x)), so H(P_K) = m. The keyed model Q_K costs ≤ 2m + c₁. A heavy-set argument (Def. 21, Lemmas 22–23) plus a union bound over short models gives the lower bound, with parameters s = Δ = log m, t = m + c₁ + log m, k = 4m + 4Δ + 2c₁.
  - The PRFs come from OWFs via standard constructions (Håstad et al. 1999).
  
  — [v2 §3](https://arxiv.org/html/2601.03220v2#S3); [App. A.4](https://arxiv.org/html/2601.03220v2#A1.SS4)
- **Theorem 12 (Paradox 1: deterministic transformations create time-bounded information).** Let G: {0,1}^k → {0,1}^n be a PRG with advantage ε(k). Then H_Poly(G(U_k)) − H_Poly(U_k) > n − k − nε(k) − c.
  - Take-away for synthetic data: "if we want to produce interesting information, we should make sure the functions we use do not have simple and efficiently computable inverses."
  - Contrast with the classical data processing inequality, H(f(X)) ≤ H(X), and K(f(x)) ≤ K(x) + K(f) + c.
  - Appendix Theorem 18 is the same result; its appendix version still says "CSPRNG" and "−O(1)", a leftover v1 wording.
  
  — [v2 §5.1](https://arxiv.org/html/2601.03220v2#S5.SS1); [App. A.2](https://arxiv.org/html/2601.03220v2#A1.SS2)
- **Theorem 13 (Paradox 2: factorization dependence).** For a one-way permutation f, X = U_n and Y = f(X): H_Poly(X|Y) + H_Poly(Y) > H_Poly(Y|X) + H_Poly(X) + ω(log n).
  - Appendix Theorem 25 states it as "for every constant c > 0 there exists N such that for all n ≥ N, … + c log n", assuming a polynomial-time-computable OWP secure against non-uniform PPT inverters with negligible success probability.
  - Proof ingredients: H_poly(Y|X) = O(1) (the forward map is deterministic and efficient); |H_poly(Y) − H_poly(X)| ≤ c₀; and a hard conditional term for H_poly(X|Y).
  - Contrast: Shannon satisfies H(Y|X) + H(X) = H(X|Y) + H(Y), and Kolmogorov satisfies K(y|x) + K(x) = K(x|y) + K(y) + O(1).
  
  — [v2 §5.2](https://arxiv.org/html/2601.03220v2#S5.SS2); [App. A.5](https://arxiv.org/html/2601.03220v2#A1.SS5)
- **Corollary 26 (no polynomial-time model fitting a one-way permutation's forward direction can satisfy Bayes' rule).** Consider a model family allowing both factorizations, P_{1→2}(X,Y) = P₁(X)P₂(Y;X) and P_{2→1}(X,Y) = P₂(Y)P₁(X;Y). If E[−log P₁(X)] ≤ n + ε and E[−log P₂(f(X)|X)] ≤ ε, then for any c and large n there is an x with P₁(x)P₂(f(x);x) > n^c 2^{−2ε} P₂(f(x))P₁(x;f(x)) (Eq. 39). — [v2 App. A.5](https://arxiv.org/html/2601.03220v2#A1.SS5)
- **Definition 14 (epiplexity-emergent).** For a computable family Φ_n: {0,1}^n → {0,1}^n and random variables X_n, (Φ, X) is emergent if there are time bounds T₁ = o(T₂) and a schedule k(n) with S_{T₁}(Φ(X)|X,n) − S_{T₂}(Φ(X)|X,n) = Θ(1) and S_{T₁}(Φ^k(X)|X,n,k) − S_{T₂}(Φ^k(X)|X,n,k) = ω(1) (Eq. 10). The authors state: "We have not proven that the Game of Life satisfies this definition". — [v2 §5.3.2](https://arxiv.org/html/2601.03220v2#S5.SS3.SSS2)
- **"Limited Epiplexity Increase Property" (Paradox 3, stated as likely false, not proven).** "It seems likely that there are no constants c₁ and c₂ for which" S_{T₂}(G(U_k)) ≤ |G| + c₁ holds for all programs G: {0,1}^k → {0,1}^n running in time T₁, with T₂(n) > T₁(k) + c₂. By contrast, K(F⁻¹) = K(F) + O(1). — [v2 §5.3.1](https://arxiv.org/html/2601.03220v2#S5.SS3.SSS1)
- **Paradox 3 statement.** argmin_P E_{X∼Q}[−log P(X)] = Q suggests models only match the generator. The paper counters with induction (models must learn inverse or inference circuits absent from the generator) and emergence. Appendix G argues this is due to maximum-likelihood estimation rather than autoregressive factorization (e.g. VAE encoders approximate P_{Z|X}). — [v2 §5.3](https://arxiv.org/html/2601.03220v2#S5.SS3)
- **Lemma 29 (naive time-bounded sophistication is O(1)).** Replacing K(x) with K^t(x) in Koppel-style sophistication (Def. 28) gives soph^t_c(x) ≤ C_t for every string, because a universal interpreter with a timeout is total. This motivates the distributional, two-part definition. — [v2 App. A.6](https://arxiv.org/html/2601.03220v2#A1.SS6)
- **Theorem 30 (monotone growth of compute-optimal N and D; non-cryptographic).** Let D̃ = 6D + 2𝒟 and T = N·D̃. Assume J(N, D̃) is C², there is a unique interior optimizer, and in log coordinates μ = log N, ν = log D̃ at the optimum:
  - complementarity ∂²J/∂μ∂ν ≤ 0 ("larger models are more sample-efficient");
  - ∂²J/∂μ² > 0;
  - ∂²J/∂ν² > 0.
  
  Then N*(T) and D̃*(T) are strictly increasing in T. With additional assumptions (N*→∞, D*→D_∞, and an infinite-model-size limit |P|_∞(D) motivated by μP/CompleteP limits), S_T grows asymptotically and H_T is nonincreasing in T. — [v2 App. B.4.1](https://arxiv.org/html/2601.03220v2#A2.SS4)
- **B.4.2 (monotonicity in dataset size).** In the infinite-compute limit, S_∞(X_𝒟) is nondecreasing and h_∞ = H_∞/𝒟 nonincreasing in 𝒟, for any coding scheme, assuming a minimizer exists (exchange argument, Eqs. 93–95). — [v2 App. B.4.2](https://arxiv.org/html/2601.03220v2#A2.SS4)
- **B.4.3 (prequential D* → 𝒟).** If ℒ(N,D) is nonincreasing in N with a limit ℒ_∞(D) that is C¹ and strictly decreasing, then J_∞(D) = ∫₀^D ℒ_∞(u)du + (𝒟 − D)ℒ_∞(D) has J′_∞ = (𝒟 − D)ℒ′_∞(D). So D*(T) → 𝒟, approached from below, and N*(T) ∼ T/(8𝒟). — [v2 App. B.4.3](https://arxiv.org/html/2601.03220v2#A2.SS4)
- **Scope of the proofs.** "While the results we prove in this paper are based on the polynomial vs nonpolynomial separation in cryptographic primitives, it seems likely that a much wider array of compute separations are relevant for information in the machine learning context", e.g. quadratic versus cubic time for attention, or fixed versus variable circuit depth with chain of thought. — [v2 §2.1](https://arxiv.org/html/2601.03220v2#S2.SS1)

### Inferences
- [inference] None of the cryptographic theorems apply to the finite, FLOP-bounded estimators used in experiments; they justify the concept, not the numbers. The estimator-relevant theory is Appendix B (Thm 30, B.2–B.4), which rests on empirical scaling assumptions.
- [inference] Theorem 12's gap is about H_T (random content), not S_T. The paper's claim that computation creates *structural* information (Paradox 1 for S_T) is supported only empirically (ECA rule 54, Fig. 3).

### Gaps
- There is no theorem lower-bounding S_T beyond Ω(log n) for any explicit distribution, and none connecting S_T to OOD performance; the authors explicitly disclaim such a guarantee (see Q6).
- I did not verify every proof step (e.g. the heavy-set union bound in A.4). Minor typos exist: Theorem 24 writes "{X_k}_{k=1}^n", and Cor. 26 has "lef".

## Q3. Practical estimators: prequential and requential formulas, required runs, setting T, number of runs, biases and variance, hyperparameter sensitivity

### Takeaway
- **Prequential estimate.** |P_preq| ≈ Σ_i [log 1/P_i(Z_i) − log 1/P_M(Z_i)], the area under the online training-loss curve above the final loss. It needs only standard single-epoch training and a held-out loss.
- **Requential estimate.** |P_req| ≈ Σ_i KL(P^t_i ‖ P^s_i): a student is trained on samples from an EMA teacher that is itself trained on real data, and the teacher-to-student KL is summed. It is rigorous (an explicit code with known runtime) but 2–10× slower.

In both cases S_T is read off the Pareto frontier of two-part code versus compute (6ND + 2N𝒟). That frontier comes from a sweep of model sizes (width × depth) under a constant learning rate with EMA iterates, so every run contributes a curve over D. The authors then take a lower convex hull and keep one median point per run. Prequential is typically several times larger than requential but ranks datasets similarly.

### Cited Findings
**Prequential coding (§4.1)**
- **Coding scheme.** Starting from P₀, at each step i, encode Z_i with log 1/P_i(Z_i) bits and then train on it. L(Z_{:M}, P_M) = Σ_{i=0}^{M−1} log 1/P_i(Z_i) codes both data and final weights, decodable in time 6ND. — [v2 §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)
- **Model-only code (Eq. 8, heuristic from Zhang et al. 2020 and Finzi et al. 2025).** Estimate L(Z_{:M}|P_M) = Σ log 1/P_M(Z_i) and appeal to symmetry of information: **|P_preq| ≈ Σ_{i=0}^{M−1} (log 1/P_i(Z_i) − log 1/P_M(Z_i))**. "If Z_i is sampled i.i.d. ... the code length for the model can be visualized as the area under the loss curve above the final loss". For random data the loss never decreases, and for simple data it drops quickly and stabilizes; both give small |P_preq|. — [v2 §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)
- **Two-part code.** |P_preq| + E[log 1/P_M(X)], with runtime 6ND + 2N𝒟. Training hyperparameters (e.g. learning rate) and the N-versus-D trade-off are optimized subject to 6ND + 2N𝒟 ≤ T. Then S_T(X) = |P*_preq| and H_T(X) = E[log 1/P*(X)], with E[log 1/P_M(X)] estimated by "the validation loss scaled by the size of X". μP keeps the optimal learning rate and initialization consistent across sizes. — [v2 §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)
- **Why the prequential estimate is not rigorous.**
  - (i) Both L(Z,P_M) and L(Z|P_M) only upper-bound the respective Kolmogorov complexities, so their difference is not an upper bound on K(P_M).
  - (ii) Symmetry of information does not extend to time-bounded Kolmogorov complexity, so the runtime is not guaranteed to be 6ND.
  - The paper also notes that the prequential losses "are effectively taken on estimates of the test loss ... In cases where train and test diverge, such as when there is overfitting, this difference could become important."
  
  — [v2 §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)
- **Cheap approximation used in all experiments except ADO.** "Since all of our experiments are in the one-epoch training regime without data repeat and training data Z_i are drawn i.i.d. ..., we ... estimate Σ log 1/P_M(Z_i) as M log 1/P_M(Z_M)", a rescaled loss on unseen data. For ADO (non-i.i.d. selection) they compute the sum exactly. — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **Test-loss scaling (Eqs. 45–48).** E[log 1/P(X)] ≈ (K/K̂) Σ_{i=1}^{K̂} log 1/P(X̂_i), using a held-out set of K̂ ≪ K examples. This "does not affect our time-bound calculation". — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)

**Requential coding (§4.2; Finzi et al. 2026, i.e. arXiv 2607.11883)**
- **Scheme.** At step i a student P^s_i is trained on a synthetic token Z̃_i ∼ P^t_i from teacher checkpoints P^t_0 … P^t_{M−1}, which are "typically ... the checkpoints from training on the original real training set". Using relative entropy coding (Theis & Ahmed 2022), each synthetic token costs KL(P^t_i‖P^s_i) + log(1 + KL) + 4 bits in expectation. **|P_req| = Σ_{i=0}^{M−1} [KL(P^t_i‖P^s_i) + log(1+KL(P^t_i‖P^s_i)) + 4] + O(1) ≈ Σ_i KL(P^t_i‖P^s_i)** (Eq. 9); the overheads are negligible "due to large sequence length and batch size". — [v2 §4.2](https://arxiv.org/html/2601.03220v2#S4.SS2)
- **Visualization.** Approximately the area between the student's and teacher's loss curves on real data, since KL ≈ log 1/P^s_i(Z_i) − log 1/P^t_i(Z_i) when P^t_i ≈ P_X. Two-part code: |P_req| + E[log 1/P^s_M(X)], with runtime 6ND (D = student training tokens) + 2N𝒟. Training hyperparameters, teacher choices and N-versus-D are optimized subject to T. — [v2 §4.2](https://arxiv.org/html/2601.03220v2#S4.SS2)
- **KL estimator per sequence (Eqs. 43–44).** KL(P^t‖P^s) = Σ_{j=1}^L E_{Z_{<j}∼P^t}[Σ_{Z_j∈𝒱} P^t(Z_j|Z_{<j}) log(P^t(Z_j|Z_{<j}) / P^s(Z_j|Z_{<j}))]. This is estimated on the single teacher sample used to train the student, using full-vocabulary teacher and student logits recorded along it. — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **Two interventions that shorten the requential code.** (1) Distilling from an EMA of teacher checkpoints, which reduces noise. (2) A maximum teacher–student KL threshold: when exceeded, "the teacher is frozen while the student catches up". The EMA time scale and max-KL threshold are extra hyperparameters. — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **B.2: prequential approximates requential with a static teacher.** With P^t_i = P^t ≈ P_{X₁}, three approximations turn Σ_i KL(P^t‖P^s_i) into Σ_i E[log 1/P^s_i − log 1/P^s_M] (Eq. 50), which is the prequential estimate. This "lends some justification to treating 6ND as the decoding time" for prequential. "We expect the prequential estimate to be an overestimate of the requential code length." — [v2 App. B.2](https://arxiv.org/html/2601.03220v2#A2.SS2)

**Comparison and recommendations (§4.3, Fig. 2c)**
- "The prequential estimate is typically several times larger than the requential estimate, the two estimates correlate well, particularly within each group"; "a good correlation between the two is not guaranteed". — [v2 §4.3](https://arxiv.org/html/2601.03220v2#S4.SS3)
- Fig. 2c ranges [read from plot; axes labelled "MB"]:

  | Group | S_req (MB) | S_preq (MB) |
  |---|---|---|
  | ECA | ≈0–0.9 | ≈0–4 |
  | Easy induction | ≈0.15–0.93 | ≈4.2–6.2 |
  | Hard induction | ≈0.45–1.0 | ≈3.9–6.2 |
  | Natural data | ≈11–45 | ≈15–70 |

  — [v2 Fig. 2](https://arxiv.org/html/2601.03220v2#S4); [v2 PDF p.13](https://arxiv.org/pdf/2601.03220v2)
- **Cost.** "Requential coding ... is typically 2× to 10× slower than prequential coding". The overhead is smaller for large batches and short sequences because of repeated teacher sampling. The authors "recommend using prequential coding for crudely estimating epiplexity and ranking the epiplexity of different datasets, particularly when one has access to the loss curve from an existing expensive training run ..., and requential coding for obtaining the most accurate estimates otherwise." — [v2 §4.3](https://arxiv.org/html/2601.03220v2#S4.SS3)

**Setting T and running the sweep (App. B.1, Fig. 10)**
- **Procedure:**
  1. Find a good learning rate on a small model and transfer it with μP and CompleteP. For requential, also tune the EMA time scale and max-KL threshold on the small model.
  2. "Train models of various depths and widths to simultaneously sweep over model size and width-depth ratios, for a total number of training tokens chosen to be larger than the test dataset size 𝒟", because optimal training tokens do not exceed 𝒟.
  3. Use "an EMA of the iterates ... under a constant learning rate schedule, rather than using a decaying learning rate schedule, following Hägele et al. (2024)", so each run traces a curve in the (|P| + E log 1/P(X)) versus T plane.
  4. The Pareto frontier over runs gives the optimal (N, D, width, depth) for each T.
  
  — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **Frontier smoothing.** The empirical frontier is "jagged". They use "the lower convex hull" and then "retain only the median point (ordered by compute) per training run (which has a fixed model size) on the lower convex hull". Fig. 10 demonstrates this on synthetic Chinchilla scaling-law curves with prequential coding. — [v2 App. B.1, Fig. 10](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **Sources of error acknowledged:**
  1. systematic error from the convex hull and median point;
  2. a fixed architecture and learning algorithm "rather than considering all possible programs";
  3. suboptimal hyperparameters (learning rate, Adam β₁ and β₂).
  
  "In most cases, we believe these sources of errors only contribute sub-leading corrections ... unlikely to alter the ordering between datasets if the estimated epiplexity gap is already significant". — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)

**Scaling-law (analytic) estimator (App. B.3, C.9)**
- **Setup.** With ℒ(N,D) = E + (N₀/N)^α + (D₀/D)^β, T = 6ND + 2N𝒟, and prequential |P_preq| = Σ_{i=1}^D [(i/D₀)^{−β} − (D/D₀)^{−β}] ≈ (β/(1−β)) D₀ d^{1−β} (Eq. 54, natural units d = D/D₀).
- **Optimality condition.** βd^{−β−1}(δ − d) = 3αδ t^{−α}(3d + δ)^{α−1} (Eq. 56), with n* = t/(3d* + δ), δ = 𝒟/D₀ and t = T/(2N₀D₀).
- **Large compute.** d* → δ, n* ∼ t/(4δ), **S_∞(X) = (β/(1−β)) D₀^β 𝒟^{1−β}** (Eq. 59) and H_∞ = 𝒟E + D₀^β 𝒟^{1−β} (Eq. 60).
- **Small compute.** d* = (β/3α)^{1/(β+1)} t^{α/(β+1)} δ^{(1−α)/(β+1)}, n* ≈ t/δ, S_T ∝ T^{α(1−β)/(β+1)} and H_T − 𝒟E ∝ T^{−αβ/(β+1)}. With Chinchilla α ≈ 0.34 and β ≈ 0.28, S_T ∝ T^0.19 and H_T − 𝒟E ∝ T^−0.07.

— [v2 App. B.3](https://arxiv.org/html/2601.03220v2#A2.SS3)

**Estimator as implemented in code**
- **PyTorch, synthetic experiments ([soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py)):**
  - `K_auc = np.trapz(Ls − L, Ts)`: trapezoidal integral over training tokens of the logged, interval-averaged prequential training loss minus the current loss, in nats. It is computed at every log point, so each checkpoint of a constant-LR run gets its own estimate.
  - `K_req` accumulates the per-token KL (teacher softmax versus student log-softmax, computed from the student's pre-update logits on the teacher-generated batch) × number of masked tokens, in nats.
  - The teacher used for generation is the EMA teacher if `ema_steps > 1`, with decay = exp(−1/ema_steps).
  - The teacher step is skipped ("blocked") while the last student KL > `max_kl`.
  - `compute = 6 * P * tokens_per_iter * step`, where P counts all parameters.
- **Notebook conversion to bits ([notebooks/eca_3rules.ipynb](https://github.com/shikaiqiu/epiplexity/blob/main/notebooks/eca_3rules.ipynb)):**
  - requential: K(M) = K_req/ln2, K(X|M) = student_loss × test_tokens / ln2, total compute = 6·P·student_tokens + 2·P·test_tokens;
  - prequential: K(M) = K_auc/ln2, K(X|M) = train_loss × test_tokens / ln2;
  - `compute_lower_convex_hull(..., reduced=True, tol=2e-2)` implements the hull plus median-point-per-run rule; test_tokens = 1e8.
- **JAX, natural data ([picodo/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/train.py)):**
  - K(X) accumulates teacher training loss × tokens / 1e6 / ln2 (Mbits); K(X|M) = (recent average training loss) × teacher tokens seen / 1e6 / ln2; the prequential K(M) = K(X) − K(X|M).
  - K(M)_req = cumulative mean distillation KL × distill tokens per step / 1e6 / ln2.
  - The student is trained with cross-entropy on hard teacher samples (unconditional generation from BOS, temperature 1.0, KV cache); the KL is only logged for the code length.
  - Model size is set from `model.P` (millions of parameters) and depth `model.N` via width D = round(√(P·1e6/N/12)/64)·64.

**Hyperparameters used (App. C)**

| Setting | Batch | Base LR | Warm-up | EMA (steps) | max-KL |
|---|---|---|---|---|---|
| ECA | 1536 sequences | 0.03 | 100 | 50 | none |
| Hard induction | 1536 | 0.03 | — | 100 | 0.03 nats/token |
| Easy induction | 384 | 0.03 | 15 | 50 | none stated in paper; code uses `max_kl=[0.005]` |
| Chess / OWT / CIFAR | 256 × 512 tokens | 2 | — | 50 | 0.1 nats/token |
| ECA emergence | 147,456 tokens | 0.06 | 100 | 50 | none |

— [v2 App. C](https://arxiv.org/html/2601.03220v2#A3); [induction_easy.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/induction_easy.py)

**Number of runs in the provided sweeps (code configs)**
- ECA 3 rules: widths {16,32,64,128,256,512} × depths {1,2,4,6,9} = 30 configurations per rule, × 3 rules = 90 runs, each T = 10,000 steps. — [eca_3rules.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_3rules.py)
- ECA 10 rules: 4 widths × 3 depths × 10 rules = 120 runs. — [eca_rules.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_rules.py)
- Natural data: depths {3,6,12,24} × sizes {1,2,5,10,20,40,60,80,100,120,140,160}M × 4 datasets, one seed, 5e9 tokens each. — [sweeps/requential.yaml](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/sweeps/requential.yaml)
- Induction: one model configuration per h, because the loss converges ("Further increasing model size or training data led to no improvement"). — [v2 App. C.2–C.3](https://arxiv.org/html/2601.03220v2#A3)

### Inferences
- [inference][COLAB] The minimal prequential estimate for one (dataset, T, 𝒟) needs:
  1. roughly 5–30 constant-LR runs over model sizes, each logging the online (pre-update) training loss at log-spaced points and held-out loss via weight EMA;
  2. converting each run to (compute = 6ND + 2N𝒟, total = K_auc + 𝒟·loss) curves in bits;
  3. the lower convex hull plus median-point-per-run rule.
  
  For ranking datasets, a single fixed-size run's K_auc already gives the "crude" estimate the authors endorse.
- [inference][COLAB] **Key pitfall for small datasets.** The cheap approximation Σ log 1/P_M(Z_i) ≈ M × held-out loss assumes a single epoch over i.i.d. data (App. B.1). With multi-epoch training on a small dataset the prequential "online" losses are no longer test-loss estimates after the first epoch. The code-length argument then needs either (a) only the first-epoch prefix counted, (b) the exact Σ log 1/P_M(Z_i) as in the ADO case, or (c) requential coding, where the student always sees fresh teacher samples. B.4.3 also implies the optimal training tokens D* approach 𝒟 from below, so a small 𝒟 caps S_T.
- [inference][COLAB] **Requential is cheap for classification.** For conditional S_T(Y|X) with an MLP classifier, a "teacher sample" is a single categorical label, so the KL over classes is exact and costs one extra forward pass. The 2–10× overhead the paper reports comes from autoregressive sequence sampling and should mostly disappear.
- [inference] The two estimators differ by several-fold in absolute value (Fig. 2c), so absolute S_T numbers are estimator-dependent. Only within-estimator rankings are claimed to be robust.
- [inference] The notebook uses `student_loss` (the student's cross-entropy on teacher-generated samples) as the per-token H_T proxy for requential. That is not the same as held-out real-data loss (App. B.1 says held-out test set); they coincide only when the teacher ≈ the data distribution. This subtlety matters when reimplementing.

### Gaps
- No variance or seed study is reported. The sweeps use a single seed (`seed: 0`; PyTorch seed fixed at 1337 + rank), and no error bars appear in any figure.
- No quantitative study of sensitivity to EMA time scale, max-KL or learning rate, beyond "we optimize ... for the small model".
- There is no rule for choosing 𝒟 (the test-set size in the time bound), which directly scales S_T and H_T. The paper uses 100M tokens for ECA, 250M for the ECA prequential/requential comparison, and 5B for natural data.

## Q4. Every experiment: data, models, tokens, compute, main results, and Colab feasibility

### Takeaway
Most synthetic experiments use tiny GPT-style transformers (about 3k to 28M parameters) and 10^13–10^17 FLOPs per run: ECA rules, the rule-30 one-way function, hard and easy induction, and ECA emergence. Their smaller configurations are feasible on one Colab GPU. The natural-data experiments (chess orderings, OpenWebText, CIFAR-5M) use models up to 160M parameters trained on 5B tokens (about 6×10^18 FLOPs) on Titan RTX GPUs and TPUv4, which is not Colab-scale. The scaling-law estimates and the ADO comparison involve no new training by the authors; ADO results are adapted from Jiang et al. 2025. Headline results:
- Rule 54 has high S_T, rule 30 maximal H_T with S_T ≈ 0, and rule 15 has little of either.
- The reverse chess ordering has higher H_T, higher S_T and better centipawn transfer.
- Induction increases S_T.
- OWT > chess > CIFAR-5M in S_T; language > VQ-image > video > pixel images at 10^25 FLOPs.
- ADO-selected data shows higher prequential epiplexity.

### Cited Findings
**Compute resources (global).** "A cluster of 6 2080Ti was used for many of the smaller scale experiments. A cluster of 6 Titan RTX and 32 TPUv4 provided by the Google TPU Research Cloud was used for the more computationally expensive natural data experiments." No GPU-hours are reported. — [v2 Appendix Outline](https://arxiv.org/html/2601.03220v2#Ax1)

**E1. ECA rules 15, 30, 54 (Fig. 3, §5.1, App. C.1)**
- **Task.** Predict Y = F(X), with F = ECA iterated 48 steps on 64 cells; X comes from a 1000-step burn-in from a uniform state. This is the conditional Y|X; the test set is 𝒟 = 100M tokens counting Y only.
- **Sweep.** Widths {16,32,64,128,256,512} × depths {1,2,4,6,9}; 1536 sequences per batch; base LR 0.03; 100 warm-up steps; EMA 50; no max-KL; requential.

— [v2 §5.1](https://arxiv.org/html/2601.03220v2#S5.SS1), [App. C.1](https://arxiv.org/html/2601.03220v2#A3)
- **Results:**
  - Rule 15 (class II) gives "little information (low H_T, low S_T)".
  - Rule 30 (class III) gives "lots of unpredictable random information (high H_T, low S_T)".
  - Rule 54 (class IV) gives "both random and structural information (medium H_T, high S_T)".
  
  [read from plot] Over roughly 10^12–10^17 FLOPs:
  - rule 30: MDL ≈ 1.0×10^8 bits (1 bit per token) and S_T ≈ 0;
  - rule 54: S_T rises to ≈5.4×10^6 bits by ≈10^17 FLOPs while H_T falls to ≈1.5×10^7 bits;
  - rule 15: S_T peaks at ≈1.1×10^6 bits at low compute, then settles at ≈0.2×10^6, and its MDL goes to ≈0.
  
  — [v2 Fig. 3](https://arxiv.org/html/2601.03220v2#S5.SS1); [v2 PDF p.17](https://arxiv.org/pdf/2601.03220v2)
- **Code.** [experiments/eca_3rules.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_3rules.py) sets B = 384, A = 4 gradient-accumulation steps, T = 10,000 steps and `requential: True`, with sequence length 128 and loss on the second half.

**E2. ECA prequential versus requential across 10 rules (Fig. 2c, App. C.7)**
- **Setup.** Rules {0, 32, 4, 15, 22, 30, 41, 54, 106, 110} covering all 4 Wolfram classes; widths {16,32,64,128} × depths {1,2,3}; up to 10,000 steps; LR 0.03; batch 384; 𝒟 = 250M; "For each rule, we report the maximum epiplexity over the resulting compute range." — [v2 App. C.7](https://arxiv.org/html/2601.03220v2#A3)
- **Code discrepancy.** [eca_rules.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_rules.py) uses `steps: [128]`, `d_head: 32` and `requential: False`, whereas the paper says the setup is otherwise identical to C.1 (48 steps).

**E3. Rule 30 as a conjectured one-way function: forward versus reverse (Fig. 4a, §5.2)**
- **Setup.** f = 8 steps of rule 30 with state size n and periodic boundaries. The forward pass is shown to be expressible by an explicit RASP-L program (App. D). "The model achieves the Shannon entropy (gray) in the forward direction, but has a consistent gap in the reverse direction." — [v2 §5.2](https://arxiv.org/html/2601.03220v2#S5.SS2)
- [read from plot] n ranges 16–64. At n = 64, H_T(A|B) + H_T(B) is ≈77 bits in reverse versus ≈65 bits forward, with the gap growing in n. — [v2 PDF p.19](https://arxiv.org/pdf/2601.03220v2)
- **Code.** [experiments/soi.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/soi.py): 8 layers, width 128 (n ≤ 32) or 192; batch 512; 50k or 100k steps; widths {16,24,32,48,64}; seeds {68,37}; prequential training via `soph.train_old.train_basic`.

**E4. Chess ordering (Fig. 4c, §5.2, App. C.4)**
- **Data.** Lichess games (HF `Lichess/standard-chess-games`), character-level, formatted "<moves>|<board>" (forward) or "<board>|<moves>" (reverse; FEN final board).
- **Models.** 1M–160M parameters, depth 3–24; base LR 2; batch 256; sequence length 512; EMA 50; max-KL 0.1 nats/token; teachers trained on 5B tokens; test set 5B tokens.
- **Result.** "The reverse order has both time-bounded higher entropy and epiplexity. This gap vanishes at small compute budgets". The authors note "there is no clear polynomial vs non-polynomial time separation in this setup."

— [v2 §5.2](https://arxiv.org/html/2601.03220v2#S5.SS2), [App. C.4](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot] Over ≈3×10^15–5×10^18 FLOPs, H_T falls from ≈7.8×10^9 to ≈3.4×10^9 bits (forward) and ≈3.95×10^9 (reverse). S_T rises to ≈2.4×10^8 (forward) and ≈2.8×10^8 bits (reverse). — [v2 PDF p.19](https://arxiv.org/pdf/2601.03220v2)

**E5. Chess OOD transfer (Fig. 7, §6.1, App. C.4)**
- **Setup.** Pre-trained models are fine-tuned on 50k examples (a 10M-parameter model with depth 24) and scored by greedy-decoding accuracy on:
  - (1) Lichess puzzles with rating > 2000 (HF `EleutherAI/lichess-puzzles`);
  - (2) Stockfish centipawn evaluation in 9 buckets (≤−800, −800..−400, −400..−200, −200..−50, −50..+50, +50..+200, +200..+400, +400..+800, ≥+800 cp; HF `Lichess/chess-position-evaluations`).
- **Result.** The reverse ordering yields "matching accuracy on chess puzzles but significantly higher accuracy on the centipawn task".

— [v2 §6.1](https://arxiv.org/html/2601.03220v2#S6.SS1), [App. C.4](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot] Epiplexity is ≈2.4×10^8 (forward) versus ≈2.8×10^8 (reverse). Puzzle accuracy is ≈0.26 versus ≈0.26; centipawn accuracy ≈0.21 versus ≈0.28. — [v2 PDF p.24](https://arxiv.org/pdf/2601.03220v2)
- **Code.** [picodo/sweeps/chess.yaml](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/sweeps/chess.yaml): depth `model.N=24`, `model.P=10`, T = 5e9, T_downstream = 25e6, B_ft = 32, one seed.

**E6. Hard induction: rule 30 with hidden bits (Fig. 5b, §5.3.1, App. C.3)**
- **Setup.** Z = U_32; f = 4 steps of rule 30; the first h ∈ {0,…,5} input bits are hidden; loss only on f(Z) given m(Z). One model (3 layers, width 256) for 20,000 steps; batch 1536; LR 0.03; EMA 100; max-KL 0.03.
- **Result.** The loss converges to h bits, but "the total compute required for this loss to converge grows exponentially with h"; "as the model is forced to induct on the missing bits, the epiplexity grows."

— [v2 §5.3.1](https://arxiv.org/html/2601.03220v2#S5.SS3.SSS1), [App. C.3](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot] Measured epiplexity (×10^6) for h = 0…5 is ≈2.5, 2.9, 3.3, 4.0, 4.3, 5.5, over 10^14–2×10^16 FLOPs. — [v2 PDF p.20](https://arxiv.org/pdf/2601.03220v2)

**E7. Easy induction: Markov chains with hidden transition rows (Fig. 5c, App. C.2)**
- **Setup.** Statistical induction heads setup (Edelman et al. 2024): V = 8 symbols, h hidden rows (columns in the text) of the transition matrix, sequence n = 512. Model: 3 layers, width 128, LR 0.03, batch 384, 3000 steps, 15 warm-up steps, EMA 50.
- **Result.** "Values 0 < h < 8 having higher epiplexity than h = 0 or h = 8". The model learns strategy 1 (use given rows) first, then in-context induction.

— [v2 §5.3.1](https://arxiv.org/html/2601.03220v2#S5.SS3.SSS1), [App. C.2](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot] Epiplexity (×10^6) for h = 0, 2, 4, 6, 8 is ≈1.0, 2.7, 3.95, 4.7, 3.2. Training loss is ≈1.26–1.51×10^3 bits per sequence; compute is up to ≈2×10^15 FLOPs. — [v2 PDF p.20](https://arxiv.org/pdf/2601.03220v2)

**E8. ECA emergence with looped versus non-looped prediction (Fig. 6, §5.3.2, App. C.8)**
- **Setup.** Rule 54, t = 64 steps; widths {16,32,64,128} × depths {1,2,4,8,16,32} × loops ℓ ∈ {1,2,4,8,16}. The ℓ-loop model predicts intermediate states, and its negative log-likelihood (NLL) upper-bounds the final-state NLL (Eq. 104). Only ℓ = 16 beats ℓ = 1. LR 0.06; batch 147,456 tokens; test set 𝒟 = 100M final-state tokens.
- **Result.** Non-looped S_T rises with compute until "a compute threshold beyond which the looped model suddenly becomes favorable, causing an abrupt drop in MDL and epiplexity".

— [v2 §5.3.2](https://arxiv.org/html/2601.03220v2#S5.SS3.SSS2), [App. C.8](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot] Non-looped S_T peaks at ≈1.45×10^7 bits near ≈10^15 FLOPs. After the switch, the looped S_T is ≈1.0×10^7, falling to ≈0.55×10^7 at ≈10^16.5 FLOPs, and the MDL drops from ≈0.2×10^9 to ≈0. — [v2 PDF p.23](https://arxiv.org/pdf/2601.03220v2)

**E9. Lorenz system (App. F, Fig. 11)**
- An LLM predicts the first B = 10 bits of Φ_t(X), with t = 30 Lyapunov times (λ₁ ≈ 0.9) and X ∼ U[−20,20]³ + 20[0,0,1] quantized to B bits. "The resulting model has a nearly identical loss and estimated epiplexity in the two settings", and the model learns the SRB invariant measure. No model size or compute is given. — [v2 App. F](https://arxiv.org/html/2601.03220v2#A6)

**E10. Natural data decomposition (Fig. 8a, §6.2, App. C.4–C.6)**
- **Data:**
  - OpenWebText (HF `Skylion007/openwebtext`), restricted to documents using only 96 common alphanumeric symbols, character-level;
  - Lichess chess, in both orders;
  - CIFAR-5M converted to greyscale and flattened to 1024 raster-order tokens with vocabulary 0–255.
- **Setup.** Same as chess: up to 160M parameters, at most 5B tokens, requential, time bound 6×10^18 FLOPs.
- **Result.** "Epiplexity accounts for only a tiny fraction of the total information, with the OpenWebText carrying the most epiplexity, followed by chess data. Despite having the most total information, CIFAR-5M data has the least epiplexity, as over 99% of its information is random".

— [v2 §6.2](https://arxiv.org/html/2601.03220v2#S6.SS2), [App. C.5–C.6](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot, log scale]

  | Dataset | S_T (bits) | Total S_T + H_T (bits) |
  |---|---|---|
  | OWT | ≈3×10^8 | ≈7×10^9 |
  | Chess (reverse) | ≈2.8×10^8 | ≈4×10^9 |
  | Chess (forward) | ≈2×10^8 | ≈3.5×10^9 |
  | CIFAR-5M | ≈9×10^7 | ≈2.3×10^10 |

  — [v2 PDF p.25](https://arxiv.org/pdf/2601.03220v2)
- **Internal inconsistency.** The Fig. 8a caption says "1B OpenWebText, Chess, and CIFAR-5M tokens", whereas the §6.2 text says "5B tokens" and App. C.4 sets the test set to 5B tokens. — [v2 §6.2 / Fig. 8](https://arxiv.org/html/2601.03220v2#S6.SS2)

**E11. Scaling-law estimates (Fig. 8b, Fig. 9, §6.3, App. C.9, Table 1)**
- **Setup.** 𝒟 = 10^12 tokens and T = 10^25 FLOPs ("equivalent to the training compute of Llama3 70B"). Language uses Chinchilla; images and video use Henighan et al. 2020, corrected for embedding parameters via N = N_{\E} + ωN_{\E}^{1/3}, ω = (V + L_ctx)(A/12)^{1/3}, aspect ratio A = 5 (Pearce & Song 2024). Parameters are converted with α = γ/δ, β = γ/(1−δ) and D₀ = (Ĉ/6) N₀^{α/β} (β/α)^{−1/β}.
- **Result.** "Language data has the highest epiplexity, while image data has the least"; VQ tokenization increases image epiplexity; video has less H_T and S_T than image at the same resolution.

— [v2 §6.3](https://arxiv.org/html/2601.03220v2#S6.SS3), [App. C.9](https://arxiv.org/html/2601.03220v2#A3)
- **Table 1 (α, β, N₀, D₀ in tokens, E in nats):**

  | Domain | α | β | N₀ | D₀ | E |
  |---|---|---|---|---|---|
  | Image 8×8 | 0.331 | 0.566 | 8.0e1 | 2.66e6 | 3.14 |
  | Image 16×16 | 0.307 | 0.820 | 2.8e2 | 8.94e7 | 2.68 |
  | Image 32×32 | 0.258 | 0.399 | 6.3e1 | 1.95e6 | 2.30 |
  | Image VQ 16×16 | 0.322 | 0.441 | 2.7e4 | 4.44e7 | 4.23 |
  | Image VQ 32×32 | 0.287 | 0.560 | 1.9e4 | 1.63e8 | 3.32 |
  | Video VQ 16³ | 0.428 | 0.718 | 3.7e4 | 1.79e8 | 1.15 |
  | Language (Chinchilla) | 0.339 | 0.285 | 4.91e7 | 1.49e9 | 1.69 |

  — [v2 Table 1](https://arxiv.org/html/2601.03220v2#A3)
- [read from plot] Language S_T ≈ 9×10^10 bits; VQ images ≈1–1.5×10^10; Video VQ ≈ 8×10^9; pixel images ≈1–5×10^9. [inference] This matches the closed form: for language, S_∞ = (0.285/0.715)·(1.49e9)^0.285·(1e12)^0.715 ≈ 6.2×10^10 nats ≈ 9.0×10^10 bits. — [v2 PDF p.25](https://arxiv.org/pdf/2601.03220v2)

**E12. ADO data selection (Fig. 8c, §6.4)**
- **Setup, adapted from Jiang et al. 2025.** 1.3B-parameter decoder-only models on 125B tokens of the Pile; 7 zero-shot tasks; out-of-distribution perplexity on SlimPajama and FineWeb.
- **Result.** "ADO indeed achieves higher epiplexity measured by prequential coding". The exact sum Σ log 1/P_M(Z_i) is used here because the data are not i.i.d.

— [v2 §6.4](https://arxiv.org/html/2601.03220v2#S6.SS4), [App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- [read from plot] Epiplexity ≈2.45×10^10 (ADO) versus ≈1.95×10^10 (natural); accuracy ≈0.590 versus ≈0.585; perplexity-1 ≈12.21 versus ≈12.25; perplexity-2 ≈15.0 versus ≈15.9. — [v2 PDF p.25](https://arxiv.org/pdf/2601.03220v2)

### Inferences
Colab feasibility estimates, all [inference] from code configs using 6·N·tokens with N ≈ 12·depth·width² (embeddings negligible), for the teacher only. Requential adds a student pass plus sampling, roughly 2–3× more. I assume a T4 or L4 sustaining roughly 5–20 TFLOP/s effective; tiny models run far below peak.

| Experiment | Tokens | Compute | Colab verdict |
|---|---|---|---|
| E1/E2 ECA | 384×4×128 ≈ 1.97×10^5 tokens/step × 10^4 steps ≈ 2×10^9 | ≈4×10^13 (width 16, depth 1) to ≈3.3×10^17 (width 512, depth 9) | [COLAB] configurations with width ≤ 128 and depth ≤ 4 (≤ ~0.8M params, ≤ ~10^16 FLOPs) take minutes to about an hour each. A reduced 3-rule sweep is feasible; the full 90-run grid including width 512 is not practical. |
| E7 easy induction | ≈384 × ~530 × 3000 ≈ 6×10^8 | ≈0.6M params → ≈2×10^15 | [COLAB] minutes; the best Colab target. |
| E6 hard induction | 1536 × 64 × 20,000 ≈ 2×10^9 | 2.4M params → ≈3×10^16 per h | [COLAB] ~1–3 h per h value with requential; 6 values total. |
| E3 SOI | — | ≈10^16 (small n) to ≈10^17 (n = 64) | [COLAB] feasible for n ≤ 32. |
| E8 emergence | 10^9 | depth 32 × width 128 ≈ 6.3M params → ≈4×10^16 per run | [COLAB] 120 runs is too many; a subset of widths and loops is feasible. |
| E4/E5/E10 natural data | 5×10^9 | 160M params → ≈4.8×10^18 (matches the 6×10^18 bound) | not Colab-feasible. The smallest 1M-param runs (≈3×10^16) are, so a scaled-down 𝒟 (e.g. 10^7–10^8 tokens) sweep of 1–10M-param models is plausible. |
| E11 scaling laws | — | CPU only (`notebooks/scaling_laws.ipynb`) | trivially reproducible. |

- [inference] Fig. 8a magnitudes support 5B as the correct token count: OWT total ≈7×10^9 bits / 5×10^9 characters ≈ 1.4 bits/char, whereas 1B would imply ≈7 bits/char, implausible for character-level English.

### Gaps
- No per-experiment wall-clock, GPU-hours or FLOP totals are reported. The Lorenz experiment (E9) gives no model or compute details.
- The exact numeric values behind Figs 3–8 are not tabulated in the paper; the numbers above are plot readings. Fig. 2c and Fig. 5 units ("MB", and unlabelled ×10^6) could not be reconciled with each other (megabits versus megabytes is ambiguous).
- ADO numbers (E12) are adapted from Jiang et al. 2025, not re-run; their compute is referred to that paper.

## Q5. The official repository (github.com/shikaiqiu/epiplexity)

### Takeaway
The repo is MIT-licensed and has a single commit ("add files", 2026-03-08 UTC). It pairs PyTorch code for all synthetic experiments (`soph/`, `experiments/`) with JAX/Flax (NNX) code for natural data (`picodo/`), plus notebooks that rebuild the figures from wandb logs. Every script is a grid sweep that logs to wandb, and there is no standalone small example beyond a `debug = True` single-run flag. Runtimes are undocumented. Only two notebooks can be re-plotted without re-running experiments: scaling laws (analytic) and chess ordering (ships `runs.pkl`). One open issue concerns a typo in the paper's Definition 5.

### Cited Findings
- **Metadata.** Created 2026-01-05; the last push and only commit was 2026-03-08 03:06 UTC ("add files", by shikai); MIT licence ("Copyright (c) 2024 Marc Finzi"); 84 stars and 14 forks on 2026-09-23; GitHub lists the language as Jupyter Notebook. — [GitHub API](https://api.github.com/repos/shikaiqiu/epiplexity); [LICENSE](https://github.com/shikaiqiu/epiplexity/blob/main/LICENSE)
- **Issues.** Exactly one: #1 "Missing term for sophistication" (open, 2026-05-25, 0 comments), noting that Definition 5 omits x ∈ S. No issues report code bugs. — [issue #1](https://github.com/shikaiqiu/epiplexity/issues/1)
- **Structure (51 files).**
  - `experiments/`: ca_shared.py, eca_3rules.py, eca_rules.py, eca_emergence.py, induction_easy.py, induction_hard.py, soi.py.
  - `soph/`: train.py (697 lines, single-GPU and DDP), model.py (GPT), train_old.py, datasets/CA.py, datasets/markov.py, utils/config_generator.py (grid_iter, dispatch_multigpu), executor.py, rng.py.
  - `picodo/`: main.py (Hydra), train.py (858 lines), model.py, data.py, configs/{default,chess}.yaml, sweeps/{chess,requential}.yaml, dataset/prepare_{chess,cifar5m,eval,fen2cp,open,puzzles}.py, reorder.py, launch.sh, run.sh.
  - `notebooks/`: chess_order, eca_3rules, eca_emergence, eca_rules, induction_easy, induction_hard, scaling_laws, soi (.ipynb), plus openai+chinchilla.csv and runs.pkl (≈33 MB).
  
  — [repo](https://github.com/shikaiqiu/epiplexity)
- **Dependencies (README).**
  - PyTorch environment: `conda create -n epi python=3.10; pip install torch numpy wandb tqdm fire pandas plum-dispatch`.
  - JAX environment: `pip install jax[cuda12] flax optax chex wandb hydra-core omegaconf tqdm numpy`.
  - Run synthetic experiments with `CUDA_VISIBLE_DEVICES=0 python experiments/<script>.py` (multi-GPU by listing more devices).
  - "Set `debug = True` at the top of each script for a quick single-point test run."
  
  — [README](https://github.com/shikaiqiu/epiplexity/blob/main/README.md)
- **Logged quantities.**
  - Synthetic: `train_loss`/`student_loss`, their EMA versions ("preferred"), `K_auc` (prequential AUC) and `K_req` (cumulative teacher-to-student KL).
  - Natural: `K(X)`, `K(M)`, `K(X|M)`, `K(M)_req` in Mbits, `distill_kl`, `down_acc`/`down_acc_ft`.
  - "Epiplexity is the model description length of the compute-limited MDL minimizer. In practice, this means sweeping over model sizes and training durations, then taking the Pareto frontier ... See `notebooks/eca_3rules.ipynb`."
  
  — [README](https://github.com/shikaiqiu/epiplexity/blob/main/README.md)
- **README mapping of scripts to figures.** eca_3rules → Fig. 3; eca_rules → Fig. 2c; soi → Fig. 4a; induction_easy/induction_hard → Fig. 5; eca_emergence → Fig. 6; scaling_laws → Figs 8b and 9 ("No training required"); picodo `sweeps/requential.yaml` → Figs 4c and 8a; `sweeps/chess.yaml` → Fig. 7. The README mislabels some section numbers relative to the paper (e.g. hard induction as "Section 5.3.2", emergence as "Section 5.4"). — [README](https://github.com/shikaiqiu/epiplexity/blob/main/README.md)
- **Model details (PyTorch).** [soph/model.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/model.py) implements μP/CompleteP-style choices:
  - init std 1/√fan_in;
  - some layers zero-initialized;
  - residual branches divided by L^block_exponent (default 1);
  - QK LayerNorm with a 1/d_head-type attention scale (`qk_scale = 8/√d_head`);
  - matrix learning rate = lr/fan_in;
  - embeddings, LayerNorm and biases on a fixed `vec_lr = 6e-4`;
  - dtype defaults to bf16 only on A100/H100, otherwise float16 with GradScaler.
- **Model details (JAX).** Model width is derived from `model.P` in millions of parameters; the gradient-accumulation parameter is called `A` in both codebases. — [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py); [picodo/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/train.py)
- **Sample JAX single-job command (README):** `python main.py -cn chess ... train_student=true train_teacher=true teacher_ema=50 student_ema=50 model.N=3 model.P=5 ds_path=chess opt.lr=2 B=256 model.L=512 max_kl=0.1 A=8 opt.schedule=const opt.warmup_tokens=16384000 T=5000000000 T_eval=1000000 num_evals=50`.
- **Hard-coded internal paths.** Checkpoint paths point to `gs://wandb_model_checkpoints_us_central2/...`, and launch.sh/run.sh `cd ~/sophistication/picodo`. — [README](https://github.com/shikaiqiu/epiplexity/blob/main/README.md); [picodo/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/train.py); [launch.sh](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/launch.sh)
- **Notebook data sources.**
  - eca_3rules.ipynb fetches runs with `wandb.Api().runs("shikai/requential", filters={'config.ema_steps': 50, 'config.tag': 'test', 'config.T': 10000})`, i.e. the author's wandb project. The filter tag 'test' differs from the script's tag 'arxiv_3rules'.
  - chess_order.ipynb and scaling_laws.ipynb read local `runs.pkl` and `openai+chinchilla.csv`.
  
  — [eca_3rules.ipynb](https://github.com/shikaiqiu/epiplexity/blob/main/notebooks/eca_3rules.ipynb)
- **Pairing of teacher and student in the synthetic code.** For conditional tasks (ECA), the teacher's synthetic sample regenerates only the second half (targets) given the real first half (inputs), via `generate_synthetic` in [experiments/ca_shared.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/ca_shared.py). ECA data are generated on the fly by `soph/datasets/CA.py` (numpy roll-based evolution, 1000-step burn-in, seeded by a SHA hash of (iteration, seed, rank)).

### Inferences
- [inference][COLAB] The PyTorch `soph/train.py` loop is the most reusable piece. It already implements, in one file:
  - online prequential AUC;
  - the requential teacher/student pair with EMA teacher, max-KL gating and per-token KL accounting;
  - log-spaced logging suitable for Pareto sweeps.
  
  It is transformer-specific: it assumes a `GPT` class with `generate`, `forward_features` and `configure_optimizers`. An MLP version would need to replace the model class and `generate_synthetic`; for classifiers, sampling a label from the softmax replaces autoregressive generation.
- [inference][COLAB] On a T4 (not A100/H100) the code falls back to float16 with GradScaler; `compile` is False in the experiment configs. wandb logging is on by default, so set `wandb_log=False`, `debug=True`, or `WANDB_MODE=disabled` on Colab.
- [inference] Because figures are rebuilt from the authors' private wandb runs, a replicator must re-run sweeps and repoint the notebooks. The two exceptions (scaling_laws and chess_order via runs.pkl) are quick wins for validating one's own Pareto and estimator code against the paper's figures.

### Gaps
- Runtimes per run or per sweep are not documented anywhere in the README or code.
- No small tutorial notebook or MLP example exists. The paper has no stated project page or blog post: the authors' homepage lists the paper only as an arXiv entry. — [Shikai Qiu homepage](https://shikaiqiu.github.io/)
- I did not execute any code, so I cannot confirm that scripts run as-is (e.g. the wandb tag mismatch or GCS paths).

## Q6. Limitations and open questions acknowledged by the authors, plus review and venue status

### Takeaway
The authors acknowledge several limits:
- Epiplexity measures the amount of structural information, not its relevance, so it does not guarantee OOD transfer to a given task.
- The prequential estimator is heuristic and the requential one is costly.
- Estimates are restricted to one architecture and optimizer, and depend on imperfect hyperparameter sweeps and Pareto-frontier artifacts.
- The rigorous theory covers only polynomial-time cryptographic separations and yields only Ω(log n) epiplexity; the emergence and induction claims are conjectural or empirical.
- Typical scaling trends can fail (grokking, emergence).

No peer-review record could be verified: the paper is listed as an arXiv preprint, and an OpenReview PDF surfaced by search is access-restricted.

### Cited Findings
- **Not a generalization guarantee.** "We emphasize, however, that epiplexity is a measure of information, not a guarantee of OOD generalization to specific tasks." Likewise, "higher epiplexity does not guarantee better generalization to any specific task ... these structures may or may not be relevant to the particular downstream task of interest." — [v2 §1](https://arxiv.org/html/2601.03220v2#S1), [§6.1](https://arxiv.org/html/2601.03220v2#S6.SS1)
- **Estimator rigor and cost.** The prequential estimate is "not rigorous for two reasons" (see Q3). Requential is "typically 2× to 10× slower", though "it is possible that the overhead can be reduced with more efficient algorithms". — [v2 §4.1, §4.3](https://arxiv.org/html/2601.03220v2#S4.SS3)
- **Program-class restriction and sweep artifacts.** Errors come from the convex hull and median point, from "using a fixed architecture ... and learning algorithm ... rather than considering all possible programs", and from suboptimal hyperparameters. — [v2 App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- **Scaling trends are "typical", not guaranteed.** They "should be understood only as typical trends, with a counterexample shown in Section 5.3.2". The B.4 assumptions "may fail to capture rare exceptions like grokking and sudden improvement in performance above certain compute thresholds". — [v2 §4.4](https://arxiv.org/html/2601.03220v2#S4.SS4), [App. B.4](https://arxiv.org/html/2601.03220v2#A2.SS4)
- **Weak explicit lower bound.** Theorem 10 gives only Ω(log n), "still far from the power law scaling we see with some natural data"; the construction is nonconstructive. — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Emergence is not proven.** "We have not proven that the Game of Life satisfies this definition, which is likely difficult as small changes to the evolution rule can destroy the emergent behavior". The looped-versus-non-looped result is "a more uncommon situation where the brute-force solution is accessible". — [v2 §5.3.2](https://arxiv.org/html/2601.03220v2#S5.SS3.SSS2)
- **Natural data might admit simpler programs.** "It is possible that with significantly more compute much simpler programs can model these natural datasets ... but ... we must treat natural data as having high epiplexity for all practical purposes." — [v2 §6.3](https://arxiv.org/html/2601.03220v2#S6.SS3)
- **Chess analogy is loose.** "There is no clear polynomial vs non-polynomial time separation in this setup". — [v2 §5.2](https://arxiv.org/html/2601.03220v2#S5.SS2)
- **Evidence on downstream value is limited.** "While these downstream evaluations do not capture everything about a pretrained model, they do offer evidence that epiplexity is a potentially useful concept". — [v2 §6.4](https://arxiv.org/html/2601.03220v2#S6.SS4)
- **Open directions named in §8:**
  - fine-grained theory of how structural information changes with compute budget, model class and data transformations ("new lower bounds and impossibility results for representation learning and transfer");
  - explaining why scaling-law exponents depend weakly on architecture and optimizer;
  - "a compute-aware analogue of classical notions such as sufficient statistics and information bottlenecks";
  - using epiplexity to guide synthetic data generation;
  - new hardness notions "capturing not sample complexity but the size of the structure that must be extracted".
  
  "Epiplexity in isolation is not a measure of generalization, or a complete theory of learning". — [v2 §8](https://arxiv.org/html/2601.03220v2#S8)
- **Other compute separations are unexplored.** Quadratic versus cubic time, attention, and chain of thought are "likely" relevant but are not covered by the theory. — [v2 §2.1](https://arxiv.org/html/2601.03220v2#S2.SS1)
- **Unused alternative constraints.** Footnote 3 suggests constraining to "all models reachable by a given optimization procedure with a given neural network architecture"; the paper does not develop this. — [v2 §3](https://arxiv.org/html/2601.03220v2#S3)
- **Venue.** Shikai Qiu's homepage lists the paper under venue "arXiv", with no conference. — [shikaiqiu.github.io](https://shikaiqiu.github.io/)
- **Third-party reading.** A blog summary notes that "Estimation is still approximation-heavy (especially prequential proxy)", that "The framework depends on the choice of observer/model class/time budget", and that results are "currently more explanatory/interpretive than directly actionable at large scale". This is a secondary source. — [Lixin Xu notes](https://davidlxu.github.io/posts/2026/02/epiplexity-paper-notes/)
- **Related follow-ups (pointers only; outside this note's scope):**
  - "Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data" (Qiu, Finzi, Zheng, Zhang, Wilson; arXiv 2607.11883, July 2026), the full treatment of the requential estimator — [arXiv 2607.11883](https://arxiv.org/abs/2607.11883);
  - "Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization" (Su, Potapczynski, Qiu, Hughes, Wilson; arXiv 2608.11746), which proposes EpiSelect and EpiGen — [arXiv 2608.11746](https://arxiv.org/html/2608.11746);
  - "Financial Epiplexity" (arXiv 2607.02695) — [arXiv 2607.02695](https://arxiv.org/pdf/2607.02695).

### Inferences
- [inference][COLAB] Main risks for democratization:
  - (a) the single-epoch, i.i.d. assumption behind the cheap prequential estimate breaks on small datasets;
  - (b) the estimate depends on the chosen test-set size 𝒟 and the compute bound T, and without a principled choice cross-paper comparisons are ill-defined;
  - (c) no seed variance is reported, so a small-scale replication should itself measure seed-to-seed spread of K_auc and K_req before interpreting differences;
  - (d) absolute values differ several-fold between prequential and requential, so only within-estimator rankings should be trusted.
- [inference] Missing evaluations that a small-scale study could add: MLPs and non-sequence models (the paper uses transformers throughout); multiple seeds; hyperparameter-sensitivity curves for EMA and max-KL; and a check that prequential and requential rankings agree on MLP classification data.

### Gaps
- **No verifiable peer reviews.** A search surfaced an OpenReview PDF (id e08dc9eb…), but fetching it returned "Error 403 ... Access to this page is restricted", so I could not confirm whether it is this paper or its reviews. The OpenReview API title search returned nothing relevant. ICML/NeurIPS 2026 status is unknown. — [OpenReview PDF (403)](https://openreview.net/pdf/e08dc9eb572c3d4c31dd91a17a142a2dfefcf85b.pdf)
- No author blog post or talk specific to this paper was found.
