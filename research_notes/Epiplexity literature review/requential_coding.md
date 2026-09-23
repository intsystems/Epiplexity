# Requential Coding (Qiu, Finzi, Zheng, Zhang, Wilson; arXiv 2607.11883, Jul 2026) as an estimator of epiplexity

Scope and provenance. I read the full text of arXiv 2607.11883 (only one version exists: v1, 13 Jul 2026), converted from the arXiv HTML with LaTeX kept, and I also read the PDF. I cloned the repo at commit `58c9796` and read all of its code. I read the relevant sections of 2601.03220 (the epiplexity paper, v2) and 2504.15208 (the compute-optimal generalization-bounds paper). Many figure values are not printed in the text. I recovered those from the PDF's vector graphics by calibrating against the tick labels. They are marked **[digitized]**, carry an estimated error of a few percent, and are saved in `C:/TMP/claude/d--GitHubD-Epiplexity/8843dc96-d6bc-4e7a-985b-ba9a35de477c/scratchpad/requential/digitized_figs.txt` and `fig9_values.txt`. Unless a note says otherwise, logs are base 2 and KL is in bits, following the paper's convention.

Short URLs used below:
- REQ = https://arxiv.org/html/2607.11883v1
- REQ-PDF = https://arxiv.org/pdf/2607.11883v1
- REPO = https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/
- EPI = https://arxiv.org/html/2601.03220v2
- CO = https://arxiv.org/html/2504.15208v1

---

## Q1. Exact definition of the requential code: protocol, sample selection (REC, shared randomness, KL threshold / teacher pausing), code-length formula, validity theorems, overhead terms

### Takeaway
The student P_t is a generative model trained only on batches it draws from its own distribution. The draws use a shared PRNG keyed by (seed, t, i). At each step the encoder uses relative entropy coding (REC: PFR or ORC) to pick the index of a proposal whose marginal law is the teacher Q_t, and writes that index with an Elias-delta code. The rigorous per-step expected length is KL(Q_t‖P_t) + 2 log(1+KL(Q_t‖P_t)) + κ with κ < 5.21 (Prop. A.1 / Thm. A.2). Theorem B.4 adds a Chebyshev bound on the realized length. In practice the code length is just the cumulative teacher-student KL. In the 2607.11883 paper there is no KL threshold. The epiplexity paper (2601.03220) did use one, freezing the teacher when KL exceeded the threshold. The 2607 paper replaces that with teacher EMA smoothing plus "iso-loss projection", which resets the teacher to the student and then retrains the teacher while the student is paused.

### Cited Findings
**Protocol (Section 3.1 and Figure 2)**
- The encoder and decoder agree on the student initialization P_0, the update rule G (for example gradient descent), the PRNG seed s, the number of steps and the batch size. At step t both sides derive the same proposal sequence Y_t^(0), Y_t^(1), … i.i.d. ~ P_t with a counter-based PRNG keyed by (s,t,i), so proposal i can be regenerated without generating the earlier ones. The encoder, which has Q_t, uses REC to choose an index i_t* such that Y_t^(i_t*) is marginally distributed as Q_t, then sends a prefix-free code m_t for that index. The decoder sets X_t = Y_t^(i_t*). Both sides then apply P_{t+1} = G(P_t, X_t). The teachers are "arbitrary, typically obtained by training on a stream of real data, and are needed only on the encoder side and never transmitted" — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1)
- Each "sample" is a whole batch of data. Training on X_t is hard distillation from Q_t: only the samples are used, never the logits — [REQ §3.1 and footnote 1](https://arxiv.org/html/2607.11883v1#S3.SS1)
- The pseudocode for Requential.Encode and Requential.Decode is in Figure 2 (top). Figure 2 (bottom) gives an illustrative REC by rejection sampling: draw Y_i ~ P and U_i ~ Unif[0,1) from S(i), and accept if U_i ≤ Q(Y_i)/(R·P(Y_i)) with R ≥ max_x Q(x)/P(x). The paper calls this "the simplest, inefficient implementation" — [REQ Fig. 2](https://arxiv.org/html/2607.11883v1#S3.F2)
- REC background: Li & El Gamal's Poisson functional representation (PFR) gives expected code length ≤ KL(Q‖P) + log(1+KL(Q‖P)) + 5 but needs an unbounded number of proposals. Ordered random coding (ORC; Theis & Ahmed 2022) meets the same bound approximately with about 2^{KL(Q‖P)} proposals. When P = Q only O(1) bits are sent, compared with H(Q) bits for entropy coding — [REQ §2 "Relative Entropy Coding"](https://arxiv.org/html/2607.11883v1#S2.SS0.SSS0.Px3)
- The PFR selection rule (Eq. 30) draws proposals X̃_i ~ P and unit-rate Poisson arrival times T_1 < T_2 < …, selects J = argmin_i T_i·P(X̃_i)/Q(X̃_i), and sets X = X̃_J. Then X ~ Q. Conditionally on X = x, J is geometric with mean 1 + a(x), where a(x) = Σ_y P(y)(R(x) − R(y))_+ and R = Q/P (Eqs. 34–38) — [REQ App. B.2](https://arxiv.org/html/2607.11883v1#A2.SS2)

**Code length (Eq. 2, Section 3.1 "Code Length")**
- Let F_{t−1} be the history before REC call t. Then L̄_req := Σ_t E[ℓ_t | F_{t−1}] ≤ Σ_{t=0}^{T−1} [KL(Q_t‖P_t) + 2 log(1+KL(Q_t‖P_t)) + κ] =: L̂_req, with κ < 5.21 — [REQ §3.1 Eq. 2](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px1)
- The extra log term, compared with the usual REC bound KL + log(1+KL) + O(1), comes from coding the index with a universal integer code. A Zipf code tuned to KL is not possible because "the decoder cannot access" KL — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px1)
- The authors report L̂_req as "the code length in all experiments". They say the log and constant terms are "negligible … for large batch sizes (typically ≳1M tokens for language models), so in practice L̂_req reduces to the cumulative teacher-student KL" — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px1). Appendix C.1 adds that the lower-order terms "are included explicitly in the block-size tradeoff calculation and omitted elsewhere" — [REQ App. C.1](https://arxiv.org/html/2607.11883v1#A3.SS1)

**Theorems (Appendices A and B)**
- Elias-delta length: ℓ_Δ(j) = ⌊log j⌋ + 2⌊log(⌊log j⌋+1)⌋ + 1 ≤ log j + 2 log(1+log j) + 1 (Eqs. 4–5). The constants are β = e^{−1} log e + 1 and κ = 1 + β + 2 log(1+β) < 5.21 (Eqs. 6–7). The PFR/ORC log-index bound is E[log J_t | F_{t−1}] ≤ KL(Q_t‖P_t) + β (Eq. 8) — [REQ App. A.1](https://arxiv.org/html/2607.11883v1#A1.SS1)
- **Proposition A.1** (conditional mean REC message length): E[ℓ_t | F_{t−1}] ≤ KL + 2 log(1+KL) + κ. **Theorem A.2** sums this over t. The paper notes that log-star codes (Rissanen 1983) could shrink the log term further — [REQ App. A.1](https://arxiv.org/html/2607.11883v1#A1.SS1)
- **Proposition B.1**: E[L − L̄] = 0 and E[(L − L̄)^2] = Σ_t E[Var(ℓ_t | F_{t−1})]. The proof is a martingale-difference argument — [REQ App. B.1](https://arxiv.org/html/2607.11883v1#A2.SS1)
- **Lemma B.2** (PFR log-index variance): with γ = 1 + 2/ln 2 < 4, E[(log J − I_+(X))^2] ≤ γ^2 and sqrt(Var log J) ≤ sqrt(Var_{X~Q} I(X)) + γ, where I(x) = log Q(x)/P(x). **Lemma B.3**: sqrt(Var ℓ) ≤ γ(sqrt(Var_Q I) + γ) + 3/2 — [REQ App. B.2–B.3](https://arxiv.org/html/2607.11883v1#A2.SS2)
- **Theorem B.4** (realized code length). Define μ_t = KL(Q_t‖P_t), s_t^2 = Var_{Q_t}(I_t), h_t = 2 log(1+μ_t) + κ and r_t = γ(s_t+γ) + 3/2, and let M = Σμ_t, H = Σh_t, V = Σr_t^2. Then L̄ ≤ M + H, E[(L − L̄)^2] ≤ E[V], and with probability ≥ 1−δ, L ≤ M + H + sqrt(E[V]/δ) (Eqs. 61–63) — [REQ App. B.4](https://arxiv.org/html/2607.11883v1#A2.SS4)
- Scale guide. With δ = 0.01, L ≤ M + H + 10 r sqrt(T). The relative fluctuation is ≈ 10γ (s′/μ′)/sqrt(BT), where μ′ and s′ are the per-sample mean and standard deviation of the information density and B is counted in sequences, not tokens. A sufficient condition for a fluctuation below 1% with probability 99% is **BT ≥ 10^6 γ^2 (s′/μ′)^2** (Eqs. 66–70). On a 100M FineWeb run the estimate falls below 1% within the training budget (Fig. 11) — [REQ App. B.4 and Fig. 11](https://arxiv.org/html/2607.11883v1#A2.F11)
- The variance result holds only for PFR (ORC in the limit of infinitely many candidates). The authors "establish that there exists a code with the claimed mean and variance … even though finding that code (encoding) can be computationally impractical" — [REQ App. B](https://arxiv.org/html/2607.11883v1#A2)

**Evaluating the code length vs. actually transmitting**
- To measure the code length it suffices to "replace REC encoding and decoding by sampling X_t directly from the teacher Q_t". Each step then costs one teacher forward pass for sampling, one student forward-backward pass and one teacher forward-backward pass on real data, so T_eval/T_train ≈ (3D+D+3D)/3D = 7/3 (about 2× memory). The overhead drops to 0.33× if teacher checkpoints already exist (Eq. 17) — [REQ §3.1 "Runtime"](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px2) and [App. A.2](https://arxiv.org/html/2607.11883v1#A1.SS2)
- Encoding with ORC costs T_enc/T_train ≈ 2 + (2B/3D) Σ_t 2^{KL_t+ρ} (Eq. 18). Decoding costs about 1× training (Eq. 19), or 4/3× if the sampling pass is not reused — [REQ App. A.2](https://arxiv.org/html/2607.11883v1#A1.SS2)
- Block-size tradeoff (Eq. 21). Split each batch of B tokens into blocks of B′, with per-sample KL ε_t = KL/B. Then L̂_req(B′) = Σ_t [Bε_t + (B/B′)(2 log(1 + B′ε_t) + κ)] — [REQ App. A.2](https://arxiv.org/html/2607.11883v1#A1.SS2). Figure 10 (104M-parameter FineWeb run, B = 524,288 tokens) shows the code length falling from about 10^10 bits at B′ = 1 token to its floor Σ KL ≈ 6×10^8 bits at B′ ≈ 10^4 tokens, while encoding time grows exponentially and passes 10^3× training within a few tens of tokens per block **[digitized, visual]** — [REQ Fig. 10](https://arxiv.org/html/2607.11883v1#A1.F10.fig1)
- Discussion: "requential coding is primarily a tool to evaluate the compressed model size rather than to transmit the model, as the REC encoder requires runtimes that scale exponentially in the KL divergence." Requential coding also gives a lossless code for the **student**, not the teacher — [REQ §5](https://arxiv.org/html/2607.11883v1#S5)

**Teacher choice, KL threshold and pausing**
- Default teacher: same architecture and hyperparameters as the student, trained on real batches. "Width, depth, learning rate, batch size, and initialization seed are shared between the student and teacher, so the per-step KL starts at zero" — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px3), [App. C](https://arxiv.org/html/2607.11883v1#A3)
- Two improvements. (1) Teacher smoothing: generate from an EMA of the teacher. (2) Iso-loss projection: at geometrically spaced steps S = {100, 150, 225, …} (×1.5), reset the teacher (weights and Adam state) to the student, then train the teacher on real data "with the student paused" until its pre-reset validation loss is recovered. The recovery "adds teacher compute and data but no student code length" — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px3) and [App. C.3, Algorithm 1](https://arxiv.org/html/2607.11883v1#A3.SS3)
- EMA formulas. Student: P^EMA_t = e^{−1/(αt)} P^EMA_{t−1} + (1 − e^{−1/(αt)}) P_t with α = 0.01. Teacher: e^{−1/max(50, αt)}. The floor of 50 steps avoids "very large initial KL". The EMA student is the REC reference and the EMA teacher is the REC target — [REQ App. C.3](https://arxiv.org/html/2607.11883v1#A3.SS3)
- The words "threshold", "MLP", "MNIST", "classif" and "ablat" do not appear anywhere in 2607.11883. This is from my full-text grep of the HTML at [REQ](https://arxiv.org/html/2607.11883v1).
- By contrast, the epiplexity paper (2601.03220, App. B.1) found "two interventions that reduce the model code length under requential coding: (1) distilling from an EMA of teacher checkpoints … and (2) imposing a maximum KL threshold between teacher and student—when exceeded, the teacher is frozen while the student catches up." Both are extra hyperparameters there — [EPI App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1). The settings used were: max KL = 0.03 nats/token with EMA timescale 100 steps for hard induction ([EPI App. C.3](https://arxiv.org/html/2601.03220v2#A3.SS3)), and max KL = 0.1 nats/token with EMA 50 steps for chess, where "the student models are trained for slightly more [tokens] due to hitting the max KL threshold" ([EPI App. C.4](https://arxiv.org/html/2601.03220v2#A3.SS4))
- The epiplexity paper quotes the REC cost as KL + log(1+KL) + 4 bits in expectation (citing Theis & Ahmed 2022), so |P_req| = Σ[KL + log(1+KL) + 4] + O(1) ≈ Σ KL (its Eq. 9) — [EPI §4.2](https://arxiv.org/html/2601.03220v2#S4.SS2). The later requential paper uses the looser but decoder-realizable KL + 2 log(1+KL) + κ with κ < 5.21 — [REQ Eq. 2](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px1)

**Implementation in the released code**
- In `train.py`, each loop iteration (1) takes a teacher step on a real batch, (2) samples a synthetic batch from the EMA teacher (autoregressive, temperature 1, per-example PRNG keys), (3) estimates the KL by Monte Carlo on that batch as log Q(x) − log P_EMA(x) at the sampled tokens only (never the full [B, L, V] logits), and (4) takes a student step on the batch. L_req = cumulative per-token KL × tokens per step / ln 2, in bits — [REPO train.py](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/train.py)
- Appendix C.1 states that "since X_t ~ Q_t, log Q_t(X_t) − log P_t(X_t) is an unbiased estimate of KL(Q_t‖P_t). Our batch size exceeds 10^5 tokens in all experiments" — [REQ App. C.1](https://arxiv.org/html/2607.11883v1#A3.SS1). The epiplexity paper instead used the lower-variance per-position full-vocabulary sum along the teacher's sampled prefix: KL ≈ Σ_j Σ_{z∈V} P^t(z|Z_<j) log P^t(z|Z_<j)/P^s(z|Z_<j) (Eqs. 43–44) — [EPI App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)

### Inferences
- The requential code is a code for P_T *given* the shared conventions (architecture, P_0 seed, G, PRNG seed, T, batch size), which cost O(1) bits. The paper, like the epiplexity paper, omits these constants.
- For small models the per-message term (κ + 2 log(1+KL_t)) ≈ 5.2–15 bits per step is **not** negligible when KL_t is small. That term, not Σ KL, dominates Eq. (2) whenever KL per REC call is ≲ 10 bits. See Q7.
- The mechanisms differ. The epiplexity paper paused the teacher once the KL passed a threshold. The requential paper pauses the student and moves the teacher back toward it (iso-loss projection). Both keep the per-step KL bounded. Only the second is described and implemented in 2607.11883 and the repo, so reimplementations should choose one explicitly.

### Gaps
- Neither the paper nor the repo actually runs REC encoding or decoding. No PFR/ORC implementation is released, and every reported number is the KL-based bound from sampling the teacher. Real transmission is shown only through the runtime model in Fig. 10.
- Paper vs. repo ambiguity on the teacher EMA timescale. Appendix C.3 writes max(50, αt) after stating α = 0.01 for the student. The repo default and every sweep use `teacher_ema: 0.1` (window = 0.1·t, minimum 50) with `student_ema: 0.01` — [REPO configs/default.yaml](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/configs/default.yaml). The teacher α is therefore probably 0.1 (unverified in the paper text).

---

## Q2. The decomposition showing prequential code length contains a data-entropy term, and how requential removes it

### Takeaway
In expectation over the data, the prequential code equals Σ_t H(X_t) (the data entropy) plus Σ_t E KL(P*‖P_t) (the approximation error); this is Eq. 1. The entropy term grows linearly even after learning stops. The requential code instead pays only Σ_t KL(Q_t‖P_t), the divergence between a moving teacher and the student. It has no entropy term because REC only has to deliver *some* sample from Q_t, and it pays only the incremental gap between teacher and student, not the full gap to the truth.

### Cited Findings
- Prequential coding: L_preq(X_{0:T−1}) = Σ_{t=0}^{T−1} log 1/P_t(X_t), the "area under the training loss curve". It also codes the model P_T because the decoder replays training — [REQ §2 "Prequential Coding"](https://arxiv.org/html/2607.11883v1#S2.SS0.SSS0.Px2)
- **Eq. (1)**: E[L_preq(X_{0:T−1})] = Σ_t H(X_t) [data entropy] + Σ_t E[KL(P*‖P_t)] [approximation error]. The first term "is paid even by a perfect predictor, accumulating at a linear rate even after the model stops learning". The second term is excessive because "actual learning is incremental … yet the prequential code pays the full remaining gap to the truth at every step" — [REQ §2](https://arxiv.org/html/2607.11883v1#S2.SS0.SSS0.Px2)
- How requential coding fixes both: "instead of coding a particular realization of a training batch, we can code a random batch from the training distribution. Moreover, we only need to specify how that distribution departs from what the model already knows" — [REQ §1](https://arxiv.org/html/2607.11883v1#S1). Each step costs about KL(Q_t‖P_t), so "the resulting code length is independent of parameter count and data entropy" — [REQ abstract](https://arxiv.org/abs/2607.11883)
- Visual link: the student's loss on real data tracks the teacher's, with a loss gap roughly equal to their KL, so the code length "is roughly equal to the integral of their loss gap" (Fig. 1 middle, FineWeb) — [REQ Fig. 1](https://arxiv.org/html/2607.11883v1#S1.F1). The epiplexity paper visualizes the same thing: KL(P^t_i‖P^s_i) ≈ log 1/P^s_i(Z_i) − log 1/P^t_i(Z_i), which "is accurate when P^t_i ≈ P_X" — [EPI §4.2](https://arxiv.org/html/2601.03220v2#S4.SS2)
- Empirical confirmation on synthetic controls (Fig. 9, D = 5B tokens, ~100M params). For uniformly random strings the prequential code is ≈ 3.27×10^10 bits against a requential code of ≈ 1.4×10^5 bits **[digitized]**. The paper describes random strings as having "little structure" while "the prequential code is inflated by the data entropy" — [REQ §4.4 and Fig. 9](https://arxiv.org/html/2607.11883v1#S4.SS4). The repo's `noise` docstring says: "the prequential code stays high while the requential code stays low (teacher and student both sit at ~uniform init, low KL)" — [REPO data.py](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/data.py)

### Inferences
- Eq. (1) follows because X_t ~ P* is independent of P_t, which depends only on X_{<t}. Then E[log 1/P_t(X_t)] = H(P*) + E KL(P*‖P_t) per batch.
- Sanity check on the digitized Fig. 9 value. 5×10^9 tokens × log2 96 = 6.585 bits gives 3.29×10^10 bits, which matches the prequential bar for "Random" (≈ 3.27×10^10) to within digitization error. The prequential code on noise is therefore essentially pure data entropy.
- The "Repeat" control has irreducible entropy of ln 96 nats per sequence ([REPO data.py](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/data.py)). That is ≈ 9.77×10^6 sequences × 6.585 bits ≈ 6.4×10^7 bits, a large share of its digitized prequential code of ≈ 1.7×10^8 bits, while its requential code is ≈ 1.0×10^6 bits.
- A limiting case, my derivation rather than the paper's: with a fixed perfect teacher Q_t = P*, the requential cost is Σ_t KL(P*‖P_t^s). That is Eq. (1)'s approximation-error term with the entropy term removed, although the student trajectory P^s, trained on teacher samples, can differ from a real-data trajectory. Moving teachers that stay close to the student reduce it further.

### Gaps
- The paper gives no closed-form expression for how much the approximation-error term shrinks under a moving teacher. The only evidence is empirical (Fig. 4).

---

## Q3. Comparison with the prequential estimate of 2601.03220 and the "prequential heuristic" of Finzi et al. 2504.15208: numbers, ratios, agreement

### Takeaway
The "prequential estimate" of 2601.03220 (its Eq. 8) and the heuristic of 2504.15208 are the same quantity: Σ_t [log 1/P_t(Z_t) − log 1/P_T(Z_t)], the area under the loss curve above the final loss. The requential paper plots it as the dashed "Prequential heuristic". For ~100M transformers at ~2B tokens, the fully tuned requential code (smoothing plus projection) is about 5.5–13× below the heuristic and about 40–300× below the true prequential code **[digitized]**. Plain requential coding, without smoothing or projection, is roughly equal to the heuristic: 0.97× on FineWeb, 1.14× on OWT, 2.2× on CIFAR-5M. The epiplexity paper reports that the prequential estimate is "typically several times larger" than the requential one, and that the two "correlate well, particularly within each group" of datasets. Rankings agree, but absolute values do not.

### Cited Findings
**Definitions**
- 2504.15208 §5.2: K(h) is estimated through the symmetry of information, K(h) = K(X,h) − K(X|h) − c. K(X,h) is upper-bounded by the prequential code −Σ_k log2 p_{h_{k−1}}(X_k|X_<k) and K(X|h) is estimated by −Σ_k log2 p_{h_D}(X_k|X_<k), which gives K(h)·log 2 ≤ Σ_k [R_{h_{k−1}} − R_{h_D}] "up to small constant factors". The heuristic is attributed to Zhang et al. (2020) — [CO §5.2](https://arxiv.org/html/2504.15208v1#S5.SS2)
- 2504.15208 numbers (Pythia models): a power-law fit to the prequential K(h) gives 6×10^5 · N^{0.5±0.1}. Parameter counting and quantization give tighter bounds over the Pythia range, but the sublinear prequential curve "overtakes it eventually, somewhere around 30B sized models" (the body text says ≈ 20B). Prequential-based generalization bounds are worse than quantization bounds but "improve substantially with scale" — [CO §5.2–5.3, Fig. 3](https://arxiv.org/html/2504.15208v1#S5.SS3)
- 2601.03220 Eq. (8): |P_preq| ≈ Σ_{i=0}^{M−1} (log 1/P_i(Z_i) − log 1/P_M(Z_i)), "we adopt the heuristic in Zhang et al. (2020) and Finzi et al. (2025)". It is described as "not rigorous": the difference of two upper bounds does not bound K(P_M), and symmetry of information does not extend to time-bounded complexity — [EPI §4.1](https://arxiv.org/html/2601.03220v2#S4.SS1)
- In practice 2601.03220 estimates Σ_i log 1/P_M(Z_i) ≈ M · log 1/P_M(Z_M) (final loss on unseen data × M), assuming one epoch and a small generalization gap. The exception is the ADO experiment, where it is computed exactly — [EPI App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- The requential paper describes this heuristic as "a commonly used heuristic … [that] only provides a non-rigorous estimate of the compressed model size but does not provide a valid compression and decompression scheme" — [REQ §2](https://arxiv.org/html/2607.11883v1#S2.SS0.SSS0.Px2)

**Theoretical link**
- 2601.03220 App. B.2: with a **static** teacher P^t ≈ P_X, requential ≈ Σ_i E_{P_X}[log 1/P^s_i(X) − log 1/P^s_M(X)] (Eq. 50), which "recovers precisely the prequential estimate" (Eq. 51). "Since a static teacher is generally suboptimal … we expect the prequential estimate to be an overestimate of the requential code length" — [EPI App. B.2](https://arxiv.org/html/2601.03220v2#A2.SS2)

**Empirical comparisons**
- 2601.03220 §4.3 and Fig. 2c cover four groups of datasets: ECA, easy and hard induction, and natural data. "While the prequential estimate is typically several times larger than the requential estimate, the two estimates correlate well, particularly within each group … a good correlation between the two is not guaranteed." Requential coding "is typically 2× to 10× slower than prequential coding". The authors recommend prequential coding for crude estimates and rankings and requential coding "for obtaining the most accurate estimates otherwise" — [EPI §4.3](https://arxiv.org/html/2601.03220v2#S4.SS3)
- Figure 4 of 2607.11883 (~100M params; curves truncated at D = 20N ≈ 2B student tokens) — [REQ Fig. 4](https://arxiv.org/html/2607.11883v1#S3.F4), values **[digitized]** from [REQ-PDF p.7](https://arxiv.org/pdf/2607.11883v1). Final code lengths in bits:

  | dataset | prequential | preq. heuristic | requential (vanilla) | + smoothing | + smoothing + projection | preq / best | heuristic / best |
  |---|---|---|---|---|---|---|---|
  | OpenWebText (char) | 3.8e9 | 9.0e8 | 1.0e9 | 1.4e8 | 7.0e7 | ≈54× | ≈13× |
  | CIFAR-5M (pixels) | 9.8e9 | 2.3e8 | 4.9e8 | 4.4e7 | 3.3e7 | ≈295× | ≈7× |
  | FineWeb (GPT-2 BPE) | 1.3e10 | 1.75e9 | 1.7e9 | 6.3e8 | 3.2e8 | ≈40× | ≈5.5× |

  The paper's own summary: the per-token requential cost "runs one to two orders of magnitude below the prequential per-token cost". The requential code "stops left of the 4-bit per parameter reference, with a significant gap on OpenWebText and CIFAR-5M, whereas the prequential code exceeds the FP32 parameter size". "Even the prequential heuristic … sits well above the requential code" — [REQ §3.2](https://arxiv.org/html/2607.11883v1#S3.SS2)
- Best requential per-token cost, from the totals above divided by 2B tokens: ≈ 0.035 bits/token on OWT, ≈ 0.017 on CIFAR-5M and ≈ 0.16 on FineWeb. The prequential cost is ≈ 1.9, 4.9 and 6.3 bits/token respectively **[digitized/derived]** — [REQ Fig. 4](https://arxiv.org/html/2607.11883v1#S3.F4)
- Generalization-bound context: 2504.15208 made the "vanishing gap" extrapolation "based on the non-rigorous prequential heuristic, whereas requential coding certifies the same trend with a rigorous code" — [REQ §4.2](https://arxiv.org/html/2607.11883v1#S4.SS2.SSS0.Px2)

### Inferences
- The two estimates agree when the teacher is static or plain. In Fig. 4, requential without smoothing or projection lands within about 1–2× of the heuristic, which fits the static-teacher argument of EPI B.2. They diverge when the teacher sequence is optimized: EMA smoothing buys about 3–8× and projection about another 1.3–2×.
- For a reimplementation, the ratio heuristic/requential depends strongly on how well the teacher is tuned. It is not an intrinsic property of the dataset, so compare datasets only under a fixed teacher recipe.
- The true prequential code (without subtracting the final loss) is dominated by entropy on every dataset here. It is a poor epiplexity estimator. The heuristic is the fairer baseline.

### Gaps
- I did not extract the numerical scatter behind Fig. 2c of 2601.03220 (the ratios per dataset group). That figure is an image with no values in the text, and 2601.03220 is only partially in scope.
- No head-to-head comparison of the two papers' requential settings exists: KL threshold versus iso-loss projection, and full-vocabulary versus sampled-token KL estimator.

---

## Q4. Experiments: datasets, model sizes, token counts, compute, main results (dataset ranking, Figure 9), generalization bounds

### Takeaway
Every experiment uses GPT-2-style transformers with 8 blocks and context 512, trained with Adam and μP (base LR 2), on OpenWebText (character level, V = 96), CIFAR-5M (greyscale pixels, V = 256), FineWeb (GPT-2 BPE, V = 50,257), and two synthetic controls. Model sizes range from 1.7M to about 1B parameters and data from 0.1B unique tokens to 20B tokens, on TPU v6e-8 hosts; no GPU or TPU hours are reported. The main results are these. Larger models and ensembles need fewer bits at a fixed loss. The requential PAC-Bayes bound beats an idealized lossless 4-bit PTQ bound in the compute-optimal regime, and bits per parameter decay as N^−0.46 (CIFAR-5M) and N^−0.15 (FineWeb). The bound becomes U-shaped under multi-epoch training. Datasets rank by learnable information as FineWeb (≈3.6e8 bits) > CIFAR-5M (≈3.6e7) > Repeat (≈1e6) > Random (≈1.4e5) **[digitized]**, while the prequential code ranks Random highest.

### Cited Findings
**Common setup**
- GPT-2 architecture, 8 blocks, context 512, size varied through width. Adam with β1 = 0.9, β2 = 0.95 and no weight decay. Constant LR after linear warmup, with EMA of iterates instead of LR decay. μP transfer with base LR 2 (per-layer LR = base / input dim) — [REQ App. C](https://arxiv.org/html/2607.11883v1#A3) and [App. C.3](https://arxiv.org/html/2607.11883v1#A3.SS3)
- Datasets. OpenWebText is filtered to documents using only 96 common alphanumeric symbols and tokenized by character (V = 96). CIFAR-5M is converted to greyscale and raster-flattened to 1024 tokens (V = 256). FineWeb uses GPT-2 BPE (V = 50,257). Sequence length is 512 throughout — [REQ App. C.2](https://arxiv.org/html/2607.11883v1#A3.SS2)
- Hardware: "The paper's experiments ran on TPU v6e-8 hosts" — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md). The paper acknowledges Google's TPU Research Cloud — [REQ Acknowledgements](https://arxiv.org/html/2607.11883v1#S5.SS0.SSS0.Px1)

**Per-figure configurations**
- **Fig. 4 (vs. prequential and PTQ)**: about 100M params; batch 1024 sequences (0.5M tokens/step); warmup 131M tokens. The teacher sees up to 5B real tokens (OWT, CIFAR-5M) or 20B (FineWeb). Curves are truncated at D = 20N student tokens. Vanilla requential keeps the student EMA — [REQ App. C.4](https://arxiv.org/html/2607.11883v1#A3.SS4). In the sweeps the widths are 1024 (OWT, CIFAR-5M) and 640 (FineWeb) — [REPO sweeps/hundredM_*.yaml](https://github.com/shikaiqiu/requential-coding/tree/58c97962f7d0c265a469a279da175370c2d87bfe/sweeps)
- **Fig. 5 (model scaling at fixed data)**: OWT and CIFAR-5M widths 128–1600 (1.7M–247M params), 5B real tokens, batch 256 sequences, warmup 16M tokens. FineWeb widths 128–2752 (14M–1B), 20B tokens, batch 1024. Smoothing and projection are on — [REQ App. C.5](https://arxiv.org/html/2607.11883v1#A3.SS5). Code length to reach the smallest plotted model's final loss **[digitized]** — [REQ Fig. 5](https://arxiv.org/html/2607.11883v1#S4.F5):
  - OWT: 4.1e7 bits (1.7M) → 2.5e7 (3.7M) → 2.0e7 (6.5M, 14.5M) → 1.7e7 (33M) → 1.6e7 (58M) → 1.55e7 (129M) → 1.45e7 (249M)
  - CIFAR-5M: 2.3e7 (1.7M) → 1.8e7 → 1.7e7 → 1.6e7 (15M) → 1.5e7 (33M) → 1.4e7 (58M–249M)
  - FineWeb: 4.0e8 (15M) → 1.9e8 (23M) → 1.5e8 (42M) → 1.2e8 (65M) → 1.1e8 (135M–267M) → 1.04e8 (522M) → 1.16e8 (1.0B)
  - The paper's own statement: "As model size increases, their code length drops significantly below the 1-bit per parameter floor achievable by quantization" — [REQ §4.1](https://arxiv.org/html/2607.11883v1#S4.SS1)
- **Fig. 6 (ensembles)**: E ∈ {1, 2, 4, 8} members of 76.9M params each (width 512), trained on FineWeb with a shared teacher and a shared synthetic stream, per-member Chinchilla budget of 1.54B tokens. Teacher smoothing is on and projection is off. The KL is taken against the *average* of the members' predictions — [REQ App. C.6](https://arxiv.org/html/2607.11883v1#A3.SS6). Code to reach the E = 1 final loss is ≈ 4.96e8 bits (E=1), 4.05e8 (E=2), 3.54e8 (E=4) and 3.37e8 (E=8) **[digitized]** — [REQ Fig. 6](https://arxiv.org/html/2607.11883v1#S4.F6). The authors note that the gain "owes in part to this extra compute" — [REQ §4.1](https://arxiv.org/html/2607.11883v1#S4.SS1)
- **Fig. 7 and Fig. 12 (PAC-Bayes bounds)**: Theorem D.1 (adapted from Finzi et al. 2025) with |K| = 1000 and δ = 0.01. D counts the teacher's real tokens. Empirical risk and Σ are measured on the teacher's training data. The PTQ baseline codes a normally trained model at L = 4N bits, assuming no loss from quantization. Fixed-data runs stop at D = 2B, and compute-optimal runs use D = 20N. The bits-per-parameter power law is fit to the 5 largest models — [REQ App. C.7](https://arxiv.org/html/2607.11883v1#A3.SS7)
  - The bound: R_s(h) ≤ R̂(h) + C ln V + (Σ + √2)√C, with C = [L(h) ln 2 + ln(|K|/δ)]/D — [REQ §4.2](https://arxiv.org/html/2607.11883v1#S4.SS2) and [App. D, Thm. D.1](https://arxiv.org/html/2607.11883v1#A4)
  - Compute-optimal FineWeb, from 7.0M to 1.02B params **[digitized]**: the requential bound falls from 7.47 to 4.16 nats; the lossless 4-bit PTQ bound from 8.18 to 5.09; test loss from 5.83 to 2.88. Compute-optimal CIFAR-5M, 1.7M → 251M: requential 4.39 → 3.36, PTQ 4.90 → 4.53, test 3.55 → 3.19 — [REQ Fig. 7](https://arxiv.org/html/2607.11883v1#S4.F7)
  - Bits per parameter at D = 20N **[digitized]**. CIFAR-5M: 0.42 (15M), 0.29 (33M), 0.21 (58M), 0.15 (130M), 0.115 (250M), fit N^−0.46. FineWeb: 1.92 (65M), 1.72 (135M), 1.62 (267M), 1.43 (523M), 1.25 (1.0B), fit N^−0.15 — [REQ Fig. 7 right](https://arxiv.org/html/2607.11883v1#S4.F7). The paper's summary is "a rate near 1 bit per parameter for compute-optimal LLMs" — [REQ §1](https://arxiv.org/html/2607.11883v1#S1)
  - Fixed data (2B tokens) **[digitized]**: the FineWeb requential bound goes 5.66 → about 5.20 nats (7M → 525M) and 5.23 at 1B. Idealized PTQ is tighter for N ≲ 43M (4.90–5.15) and then blows up. The CIFAR-5M requential bound is nearly flat (3.417 → 3.404), with PTQ better only at the two smallest sizes — [REQ Fig. 7 left](https://arxiv.org/html/2607.11883v1#S4.F7)
  - OWT (Fig. 12): the fixed-data bound falls "from 1.51 to 1.38 nats as N grows from 1.7M to 247M", and the "certified gap at the Chinchilla budget is down to 0.26 nats by 247M parameters" — [REQ Fig. 12](https://arxiv.org/html/2607.11883v1#A3.F12)
  - If the power law persists, "C → 0 and the certified generalization gap vanishes with scale" — [REQ §4.2](https://arxiv.org/html/2607.11883v1#S4.SS2.SSS0.Px2)
- **Fig. 8 (overfitting)**: FineWeb, width 512, 100M unique tokens trained for up to 10B tokens (`unique_tokens=1e8`), no projection. "The best bound [is] attained around one epoch of training", after which the complexity penalty rises faster than the training loss falls — [REQ §4.3](https://arxiv.org/html/2607.11883v1#S4.SS3) and [REPO sweeps/overfit_fineweb.yaml](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/sweeps/overfit_fineweb.yaml)
- **Fig. 9 (learnable information by dataset)**: "Each dataset has 5B tokens and the model is trained for one epoch." The ranking: "uniformly random strings and trivially repeating strings have little structure, images have substantial structure, and text has the most structure". The prequential code gives "similar and vacuous estimates" for random strings, images and text. Quantization at 1–4 bits/param "primarily reflects the model size and is insensitive to the data" — [REQ §4.4](https://arxiv.org/html/2607.11883v1#S4.SS4). The notebook compares Random and Repeat (synthetic, V = 96, width 1024) with CIFAR-5M (width 1024) and FineWeb (width 640, ≈104M), all read at D = 5B. Req is taken from the run with `isoloss_proj=true` and Preq from the vanilla run — [REPO notebooks/fig9_learnable_info.ipynb](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/notebooks/fig9_learnable_info.ipynb). Bar heights **[digitized exactly from vector rectangles; log axis calibrated on grid lines]** — [REQ-PDF p.10](https://arxiv.org/pdf/2607.11883v1):

  | dataset | requential (bits) | prequential (bits) | preq / req |
  |---|---|---|---|
  | Random (uniform, V=96) | ≈1.4e5 | ≈3.27e10 | ≈2.3e5 |
  | Repeat | ≈1.0e6 | ≈1.7e8 | ≈170 |
  | CIFAR-5M | ≈3.6e7 | ≈2.3e10 | ≈650 |
  | FineWeb | ≈3.6e8 | ≈2.8e10 | ≈78 |
  | PTQ band (1N–4N bits) | 1.0e8 – 4.1e8 | | |

- **Fig. 10 and Fig. 11**: see Q1 (104M and 100M FineWeb runs) — [REQ App. A.2, B.4](https://arxiv.org/html/2607.11883v1#A1.SS2)
- The epiplexity paper reaches a consistent ordering with requential coding (models ≤160M, ≤5B tokens, time bound 6×10^18 FLOPs): OpenWebText has the most epiplexity, then chess (Lichess), with CIFAR-5M lowest, "as over 99% of its information is random" — [EPI §6.2](https://arxiv.org/html/2601.03220v2#S6.SS2)

### Inferences
- Text holds about 10× more requential-learnable information than images at an equal 5B-token budget (FineWeb 3.6e8 vs. CIFAR-5M 3.6e7 bits). The prequential code separates them by only about 1.2× and puts random noise at the top.
- The FineWeb 100M model at 5B tokens uses ≈3.5 bits/param, just under the 4-bit line. CIFAR-5M uses ≈0.35 bits/param.
- Fig. 7 (FineWeb) shows 9 model sizes, the smallest ≈7M, but the released `scaling_fineweb.yaml` lists 8 widths (128–2752), and App. C.5 says the range is 14M–1B. The ≈7M point is probably a width-64 run that is not in the released sweep (unverified). Parameter counts assume an untied embedding and head: width 128 gives ≈14.4M, width 64 gives ≈6.8M.

### Gaps
- No GPU/TPU hours, wall-clock times or FLOP totals are reported anywhere in the paper or README. There is only the relative 7/3× evaluation overhead and the hardware type.
- All results come from single runs: every sweep uses `seed: 0`, and there are no error bars. Fig. 1 (middle) does not state its FineWeb model size. By eye it shows ≈8×10^8 bits after 10B tokens at loss ≈3.6 nats (unverified).

---

## Q5. Small-scale experiments (MLPs, toy data, MNIST, synthetic)? Minimal settings? Sensitivity to hyperparameters and seeds?

### Takeaway
2607.11883 contains no MLP, MNIST or toy-classification experiments. The smallest models are 1.7M-parameter transformers trained on 5B tokens, and the only synthetic data are the Random and Repeat controls at ~100M parameters. Small-scale settings that use requential coding appear in the earlier epiplexity paper (ECA and induction tasks with 1–3-layer models, batch 384–1536), with a KL threshold. The paper reports no seed variance and no formal hyperparameter ablations. The only sensitivity shown is to the teacher recipe (vanilla vs. smoothing vs. projection), which is worth up to about 15× in code length (Fig. 4).

### Cited Findings
- A full-text grep of 2607.11883 for "MNIST", "MLP", "classif", "ablat" and "error bar" returns zero hits (my search of the paper text).
- Smallest settings in the paper: OWT and CIFAR-5M widths from 128 (1.7M params), 5B real tokens, batch 256 × 512 = 131k tokens/step — [REQ App. C.5](https://arxiv.org/html/2607.11883v1#A3.SS5). "Our batch size exceeds 10^5 tokens in all experiments" — [REQ App. C.1](https://arxiv.org/html/2607.11883v1#A3.SS1)
- Synthetic controls: `noise` draws fresh uniform tokens with V = 96 and is "unlearnable". `repeat` repeats one random token for the whole sequence and is "trivially learnable", with a loss floor of ln(V)/(seq_len−1) nats/token. Neither needs data preparation — [REPO data.py](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/data.py) and [README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md). In the paper they run at width 1024 with 5B tokens — [REPO sweeps/synthetic_controls.yaml](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/sweeps/synthetic_controls.yaml)
- The smallest example in the README is `python main.py ds_path=open model.width=256 T=100_000_000 B=256 opt.warmup_tokens=16_384_000` — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md)
- Small settings in 2601.03220. ECA: widths {16, 32, 64, 128}, depths {1, 2, 3}, up to 10,000 steps, LR 0.03, batch 384 ([EPI App. C.7](https://arxiv.org/html/2601.03220v2#A3.SS7)). Hard induction: 3 layers, width 256, batch 1536 sequences, EMA 100 steps, max KL 0.03 nats/token ([EPI App. C.3](https://arxiv.org/html/2601.03220v2#A3.SS3))
- Sensitivity to the teacher recipe (Fig. 4, **[digitized]**). On OWT, vanilla ≈1.0e9 bits drops to ≈1.4e8 with smoothing and ≈7.0e7 with smoothing plus projection. CIFAR-5M: 4.9e8 → 4.4e7 → 3.3e7. FineWeb: 1.7e9 → 6.3e8 → 3.2e8 — [REQ Fig. 4](https://arxiv.org/html/2607.11883v1#S3.F4). The authors "expect substantial further gains from optimizing the teacher sequence" — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px3)
- The teacher EMA floor of 50 steps exists because under-smoothing the teacher early on "led to very large initial KL" — [REQ App. C.3](https://arxiv.org/html/2607.11883v1#A3.SS3)
- The requential code length depends on the training process, so "unlike parameter-based codes such as quantization that can be applied directly to the trained parameters", it requires "running the teacher-student training" — [REQ §5](https://arxiv.org/html/2607.11883v1#S5)
- The epiplexity paper treats the EMA timescale and max-KL threshold as hyperparameters tuned on a small model — [EPI App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)

### Inferences
- Plausible minimal Colab-scale setups using the repo unchanged: `ds_path=repeat` or `noise` (no data), or `open`, with `model.width=128`–`256`, T around 1e8 tokens and B = 256 on one GPU. Throughput is limited by autoregressive teacher sampling, as the README notes ("Sampling from the teacher dominates the runtime"). With B·L well below 10^5 tokens, the Monte Carlo KL noise and the per-message overhead of Eq. (2) both become visible.
- No seed-variance data means that stability across seeds has to be established in any small-scale reimplementation. Theorem B.4 covers only the REC coding fluctuation, not variation in the training trajectory. The paper states that controlling the variation of L̄ across runs "would require a separate stability or concentration result … not relevant to our investigation" — [REQ App. B.1](https://arxiv.org/html/2607.11883v1#A2.SS1.SSS0.Px1).

### Gaps
- Nothing in the paper or repo covers MLPs, classification, MNIST or small non-sequence toy problems. There is no sensitivity study on learning rate, EMA α, projection schedule or batch size, and no multi-seed error bars.

---

## Q6. Repo: framework, structure, entry points, configs, notebooks, runtime, licence, last commit, small example; plus bibliographic data

### Takeaway
The repo `shikaiqiu/requential-coding` is written in JAX, Flax NNX and Optax, with Hydra configs and wandb logging. It has one commit ("Initial release", 2026-07-14) under the MIT license, and implements a single training loop (`train.py`) for a GPT-2-style transformer. It includes one wandb sweep per paper experiment and five figure notebooks (Figs. 5–9) that pull from wandb and ship with no stored outputs. It has no MLP, classification or tiny example and no REC encoder. It measures code length by sampling from the teacher.

### Cited Findings
- Structure: `train.py` (770 lines; requential loop, per-step MC KL, iso-loss projection, bound logging), `model.py` (322; GPT-2-style transformer in Flax NNX with KV-cache `generate` returning per-token log-probs), `bounds.py` (144; PAC-Bayes quantities), `data.py` (255; loaders for fineweb, open, cifar5m, noise and repeat), `ckpt.py`, `main.py` (Hydra entry point), `configs/default.yaml`, `data_prep/` (3 scripts), `sweeps/` (9 yamls) and `notebooks/` (fig5–fig9) — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md) (line counts from my clone)
- Install: `pip install -U "jax[cuda12]" flax optax hydra-core omegaconf wandb tqdm numpy` (Python 3.10), with `requirements.txt` asking for jax>=0.5 and flax>=0.10. "Everything runs on GPUs and, with jax[tpu], on TPUs … both single- and multi-GPU setups work." For out-of-memory errors, raise the gradient-accumulation factor `A` or set `model.gradient_checkpointing=true` — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md)
- Key flags: `T`/`tpp` (real-token budget D), `model.width`, `isoloss_proj`, `teacher_ema`/`student_ema`, `ensemble_size`, `unique_tokens` (multi-epoch), `log_bound`, `train_student=false` (plain LM baseline only), `ckpt_dir`. Logged metrics include `kl` (nats/token), `L_req` and `L_preq` (bits), `ema_{student,teacher}_eval_loss`, and `req_bound/*` and `ptq_bound/*` — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md)
- Defaults: seed 0, B = 1024 sequences, A = 1, T_eval = 1,048,576, teacher_ema 0.1 with minimum 50, student_ema 0.01, isoloss_proj false, proj_first_step 100, proj_step_mult 1.5, width 768, depth 8, head_dim 64, seq_len 512, embed_init_std 0.1, float32, lr 2.0 (μP), embed_lr_mult 0.025, b1 0.9, b2 0.95, warmup 131,072,000 tokens — [REPO configs/default.yaml](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/configs/default.yaml). Adam eps is 1e-20 in `build_tx` — [REPO train.py](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/train.py)
- Implementation details in `train.py`. The teacher is created with `cfg.seed` and student member m with `cfg.seed + m`, so the teacher and student 0 share their initialization. `L_preq` = cumulative teacher train loss (measured before the update) × tokens/step. Iso-loss projection deep-copies the student's weights *and optimizer state* into the teacher, then recovers until the EMA teacher's held-out loss (train prefix when multi-epoch) is back at its pre-reset level. With `train_student`, the LR schedule follows the student's step count. The PTQ bound prices the teacher at k = 4N bits and the requential bound uses k = cumulative KL × tokens/step in nats — [REPO train.py](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/train.py)
- Runtime: "Sampling from the teacher dominates the runtime, since every step autoregressively generates a full synthetic batch." Generation is split across devices. No timings are given — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md)
- Notebooks fig5–fig9 are "self-contained", fetch runs from wandb by sweep tag and write `figures/*.pdf`. My clone shows 0 stored outputs in all five, and there is no notebook for Figs. 1, 4, 10 or 11 (fig7 also draws Fig. 12) — [REPO README](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/README.md)
- License: MIT. The LICENSE text reads "Copyright (c) 2024 Marc Finzi", which suggests it was carried over from an earlier project — [REPO LICENSE](https://github.com/shikaiqiu/requential-coding/blob/58c97962f7d0c265a469a279da175370c2d87bfe/LICENSE)
- History: a single commit `58c97962f7d0c265a469a279da175370c2d87bfe` "Initial release", 2026-07-14 10:03:39 −0400. The repo was created 2026-07-13, last pushed 2026-07-14, and at the time of checking (2026-09-23) had 15 stars, 4 forks and 0 open issues — [GitHub API](https://api.github.com/repos/shikaiqiu/requential-coding)
- Other coverage: the only search hits beyond arXiv were aggregator summaries (arxiviq Substack, Pith). I found no author talk or blog post — [arxiviq](https://arxiviq.substack.com/p/requential-coding-pushing-the-limits), [Pith](https://pith.science/paper/2607.11883)

### Bibliographic data (BibTeX)
The metadata is from the arXiv abstract pages ([2607.11883](https://arxiv.org/abs/2607.11883), [2601.03220](https://arxiv.org/abs/2601.03220), [2504.15208](https://arxiv.org/abs/2504.15208)). Affiliations per the paper: Qiu (NYU), Finzi (CMU), Zheng (CMU), Zhang (CMU), Wilson (NYU) — [REQ](https://arxiv.org/html/2607.11883v1). The requential paper cites 2504.15208 as ICLR 2025 (its ref. [8]) — [REQ refs](https://arxiv.org/html/2607.11883v1#bib).

```bibtex
@article{qiu2026requential,
  title         = {Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data},
  author        = {Qiu, Shikai and Finzi, Marc and Zheng, Yujia and Zhang, Kun and Wilson, Andrew Gordon},
  journal       = {arXiv preprint arXiv:2607.11883},
  year          = {2026},
  month         = jul,
  eprint        = {2607.11883},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.11883},
  url           = {https://arxiv.org/abs/2607.11883},
  note          = {v1 submitted 13 Jul 2026. Code: https://github.com/shikaiqiu/requential-coding}
}

@misc{qiu2026requentialcode,
  title        = {requential-coding (code for ``Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data'')},
  author       = {Qiu, Shikai},
  year         = {2026},
  howpublished = {\url{https://github.com/shikaiqiu/requential-coding}},
  note         = {Commit 58c97962f7d0c265a469a279da175370c2d87bfe (14 Jul 2026), MIT License}
}

@article{finzi2026epiplexity,
  title         = {From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence},
  author        = {Finzi, Marc and Qiu, Shikai and Jiang, Yiding and Izmailov, Pavel and Kolter, J. Zico and Wilson, Andrew Gordon},
  journal       = {arXiv preprint arXiv:2601.03220},
  year          = {2026},
  eprint        = {2601.03220},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2601.03220},
  note          = {v1 6 Jan 2026; v2 16 Mar 2026}
}

@inproceedings{finzi2025computeoptimal,
  title     = {Compute-Optimal {LLM}s Provably Generalize Better with Scale},
  author    = {Finzi, Marc and Kapoor, Sanyam and Granziol, Diego and Gu, Anming and De Sa, Christopher and Kolter, J. Zico and Wilson, Andrew Gordon},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2504.15208},
  archivePrefix = {arXiv},
  doi       = {10.48550/arXiv.2504.15208}
}
```

### Inferences
- The code is tied to autoregressive transformers: `generate`, the per-token KL and the bounds all assume token sequences. An MLP or classification version would be a rewrite of the loop (about 100 lines in PyTorch or JAX), not a configuration change. See Q7.
- Reproducing the paper figures needs wandb and TPU-scale compute (5–20B tokens per run). On one Colab GPU only toy runs of the transformer pipeline are realistic.

### Gaps
- Runtime per run and memory numbers are not given. I did not execute the code, so I have not verified whether it runs on Colab (for example with JAX CUDA 12 wheels).

---

## Q7. For an MLP classifier (label y given x over C classes): what does the requential code reduce to? Is the per-step KL exact? Is sample selection trivial? Obstacles

### Takeaway
None of the sources discusses classification. Everything in this section is my derivation from the paper's general definitions, marked as inference, except where cited. With the inputs x treated as shared side information (the conditional-epiplexity setting S_T(Y|X) of the epiplexity paper), the requential code for a classifier becomes Σ_t [Σ_{b∈batch t} KL(Q_t(·|x_b) ‖ P_t(·|x_b)) + 2 log(1 + ·) + κ]. This per-step KL is exact and cheap (B×C terms). Evaluating the code only requires sampling hard labels ỹ_b ~ Q_t(·|x_b), which is a categorical draw. The main obstacles at small scale are four. The per-message overhead of about 5.2+ bits per step can dominate a small KL. Multi-epoch reuse of small datasets inflates the code, which reflects memorization. The student must train on hard sampled labels, not teacher logits. And results depend on the teacher recipe.

### Cited Findings (general facts the derivation relies on)
- REC is defined for any discrete P and Q on a countable alphabet. The encoder evaluates both likelihoods and the decoder only samples P — [REQ §2](https://arxiv.org/html/2607.11883v1#S2.SS0.SSS0.Px3), [App. B.2](https://arxiv.org/html/2607.11883v1#A2.SS2). The method assumes "the encoder and decoder can sample from the student, and that the encoder can additionally evaluate the likelihoods of both models" — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1)
- Conditional epiplexity is estimated "analogously, providing random variable conditioning as input into the model", and "code length only needs to be computed for the label tokens (tokens contributing to the training loss)", while decoding time counts both input and label tokens — [EPI §4](https://arxiv.org/html/2601.03220v2#S4), [EPI App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)
- Only hard distillation is codable: "hard distillation where only the samples are used, not the logits" — [REQ §3.1 footnote](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px4)
- To evaluate the code length, sample X_t directly from the teacher and accumulate the KL. There is no REC search — [REQ §3.1 "Runtime"](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px2)
- The overhead terms are 2 log(1+KL_t) + κ per REC message with κ < 5.21 (Eq. 2). They are negligible only for large batches (≳1M tokens for LMs) — [REQ §3.1](https://arxiv.org/html/2607.11883v1#S3.SS1.SSS0.Px1). Block-size tradeoff: Eq. 21 — [REQ App. A.2](https://arxiv.org/html/2607.11883v1#A1.SS2)
- Encoding cost is about 2^{KL+ρ} proposals per message with ORC — [REQ App. A.2](https://arxiv.org/html/2607.11883v1#A1.SS2)
- Concentration of the realized length needs BT ≳ 10^6 γ^2 (s′/μ′)^2 samples, with γ < 4 — [REQ App. B.4](https://arxiv.org/html/2607.11883v1#A2.SS4)
- Multi-epoch training makes the code keep growing and the bound turn U-shaped (Fig. 8). "The code length presently only grows with training steps and never decreases", and forgotten information is not credited — [REQ §4.3](https://arxiv.org/html/2607.11883v1#S4.SS3), [§5](https://arxiv.org/html/2607.11883v1#S5)
- The teacher shares the student's init seed and hyperparameters so the KL starts at 0. Useful teacher improvements are EMA smoothing (window ≈ α·t, minimum 50 steps) and iso-loss projection. The epiplexity paper alternatively uses a max-KL threshold with teacher freezing — [REQ App. C](https://arxiv.org/html/2607.11883v1#A3), [EPI App. B.1](https://arxiv.org/html/2601.03220v2#A2.SS1)

### Inferences (my derivation; not in any source)
- **Reduction.** Let the dataset inputs x_1..x_n be known to the decoder, with the minibatch indices at step t fixed by the shared seed. The transmitted object at step t is the label vector ỹ_t ∈ [C]^B. Target: Q_t(ỹ|x_t) = Π_b Q_t(ỹ_b|x_b). Reference: P_t(ỹ|x_t) = Π_b P_t(ỹ_b|x_b). Then
  - KL_t = Σ_b Σ_{c=1}^C Q_t(c|x_b) [log2 Q_t(c|x_b) − log2 P_t(c|x_b)]. This is **exact**, costs one forward pass of each network, and needs no Monte Carlo (in contrast to the repo's sampled-token estimator).
  - L̂_req = Σ_t [KL_t + 2 log2(1+KL_t) + 5.21]. Report Σ_t KL_t and L̂_req separately. Use the EMA student and EMA teacher as P_t and Q_t, as the paper does.
  - Loop. (i) The teacher takes a step on real (x_b, y_b). (ii) Compute Q_t(·|x_b) and P_t(·|x_b) and accumulate KL_t. (iii) Sample ỹ_b ~ Q_t(·|x_b). (iv) The student takes a cross-entropy step on (x_b, ỹ_b). (v) Update the EMAs. (vi) Optionally apply iso-loss projection at steps 100·1.5^k.
  - Cost: about 7/3 of a normal training run. This is trivial on one Colab GPU for MNIST-scale MLPs.
- **Is sample selection trivial?** For evaluation it is: ỹ_b ~ Categorical(Q_t(·|x_b)), with no REC. For actual encoding it is *far* cheaper than in the LM case. Each proposal is just B categorical draws from the fixed P_t(·|x_b) plus a sum of precomputed log-ratios, with no network forward pass per proposal, because the logits for the batch are computed once. It is still exponential in KL per message (about 2^{KL_t}), so real transmission needs blocks with KL ≲ 20–25 bits.
- **Obstacle 1: overhead.** Example: MNIST, batch 128, one epoch = 469 steps. The κ term alone costs ≥ 469 × 5.21 ≈ 2.4 kbit per epoch, plus 2 log2(1+KL_t) per step. If typical per-step KL is ≲ 10 bits, the overhead is comparable to or larger than Σ KL. Remedies: larger batches, or making each REC message cover a large block, since the update rule G may take several optimizer steps on one decoded block (Eq. 21 logic). Always report the Eq. (2) bound, not only Σ KL, at this scale.
- **Obstacle 2: small datasets force multiple epochs.** The teacher then memorizes (for example random or noisy labels) and the KL keeps accumulating, as in Fig. 8. For epiplexity-style estimates, prefer one epoch or a large synthetic stream. Alternatively treat the growth across epochs as the memorization signal.
- **Obstacle 3: the entropy advantage only appears with label entropy.** With clean MNIST, H(Y|X) ≈ 0, so the prequential code has little data-entropy term. Requential and prequential then differ mainly through the incremental-teacher effect. With symmetric label noise ρ over C classes, H(Y|X) = h(ρ) + ρ log2(C−1) per example. That is ≈ 1.36 bits/example for ρ = 0.2, C = 10, or ≈ 81 kbit per 60k-example epoch that prequential pays and requential should not. This is a clean small-scale test of Eq. (1).
- **Obstacle 4: the student must be trained on hard labels.** Soft-label distillation is not covered by the code. Variance and concentration: with small BT, check BT ≥ 10^6 γ^2 (s′/μ′)^2. Otherwise the certified realized length (Thm. B.4) is loose, although the expected-length bound (Thm. A.2) still holds.
- **What is being measured.** A classifier's requential code estimates conditional structural information S(Y|X) for the chosen architecture and training procedure, an upper-bound-style estimate of conditional epiplexity. It does not measure information in the inputs X. For small sequence models over a vocabulary V, the epiplexity paper's exact per-position KL along the teacher's sample (EPI Eq. 44) is the better choice than per-token MC.

### Gaps
- No published source validates requential coding on classification or MLPs, or reports the size of the overhead at small batch sizes. The numbers above are illustrative calculations, not results.
- It is unverified whether iso-loss projection or a KL threshold works better at small scale. Neither paper compares them.
