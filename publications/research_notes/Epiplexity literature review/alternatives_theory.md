# Theoretical alternatives and precursors to epiplexity: formal measures of structural, meaningful or learnable information (as opposed to randomness)

Scope note: research done 2026-09-23. The primary anchor is arXiv:2601.03220v2 (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson; v1 6 Jan 2026, v2 16 Mar 2026). Its full HTML text was downloaded and searched, including Section 2 (Background), Section 7 (Additional Related Work), Appendix A.6 (time-bounded sophistication) and Appendix H (MDL). Definitions of the other measures were taken from the original papers' PDFs (arXiv or author copies) wherever possible. Bibliographic data was checked against Crossref for DOIs where possible. Notation: $K$ = prefix Kolmogorov complexity and $C$ = plain complexity, both on a fixed universal machine $U$. $|p|$ = length of a program in bits. $\log$ = base 2 unless stated.

---

## Q1. What exactly does the epiplexity paper define, and what does it say about each related measure?

### Takeaway
Epiplexity $S_T(X)$ is the length of the program that minimizes the **expected** two-part code $|P| + \mathbb{E}[\log 1/P(X)]$. The minimization runs over prefix-free programs $P$ that can both **evaluate probabilities and sample** within time $T(n)$. Time-bounded entropy $H_T(X)$ is the expected data code length left over under that program. The paper describes epiplexity as "a time-bounded and distributional generalization of sophistication". It argues that sophistication, effective complexity, logical depth, algorithmic sufficient statistics, statistical complexity and excess entropy all fail because they ignore the observer's compute. It treats pseudoentropy (HILL/Yao) and V-entropy as analogues of $H_T$, not of $S_T$: they capture only the random part.

### Cited Findings
**Bibliographic facts**
- Title: "From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence". Authors: Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, Andrew Gordon Wilson. arXiv:2601.03220 [cs.LG]. v1 was posted 6 Jan 2026 and v2 on 16 Mar 2026. Affiliations are Carnegie Mellon University and New York University. Code: github.com/shikaiqiu/epiplexity — [arXiv abs](https://arxiv.org/abs/2601.03220); [HTML v2](https://arxiv.org/html/2601.03220v2)

**Definitions (Section 3)**
- **Definition 7 (T-time probabilistic model).** $P$ is a prefix-free program on a fixed prefix-free UTM $\mathcal U$ with two modes, both of which must halt within $T(n)$ steps:
  - Evaluation: $\mathcal U(P,(0,x))$ outputs $\mathrm{Prob}_P(x)\in[0,1]$.
  - Sampling: $\mathcal U(P,(1,u))$, with $u$ an infinite random tape, outputs a sample.

  The probabilities must be normalized and match the sampler. $\mathcal P_T$ denotes the set of all such programs. The paper notes that $\mathcal P_T$ can be replaced by a function class $\mathcal P_{\mathcal F}$, e.g. "all models reachable by a given optimization procedure with a given neural network architecture" — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Definition 8 (Epiplexity and time-bounded entropy).**
  - $P^\star=\arg\min_{P\in\mathcal P_T}\{|P|+\mathbb E[\log 1/P(X)]\}$, with ties broken by the smallest program.
  - $S_T(X):=|P^\star|$ and $H_T(X):=\mathbb E[\log 1/P^\star(X)]$.
  - $\mathrm{MDL}_T(X):=S_T(X)+H_T(X)$ is the "total time-bounded information content".

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Stated basic properties:**
  - $S_T, H_T \ge 0$.
  - $H(X)\le S_T(X)+H_T(X)\le n+c_1$.
  - $\mathrm{MDL}_{T'}\le \mathrm{MDL}_T$ for $T'\ge T$.
  - $\mathrm{MDL}_{T'}(f^{-1}(X))\le \mathrm{MDL}_T(X)+|f|+c_2$ with $T'=T+\mathrm{Time}(f)$.
  - A short program for $f^{-1}$ does not imply a short one for $f$ under a fixed budget. The paper says this is key to its three "paradoxes".

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Uniform noise and simple patterns.**
  - Uniform noise: $S_T(U_n)\le c_2$, i.e. constant epiplexity, even for a constant time bound.
  - The mixture "0101… w.p. ½ / 1010… w.p. ½" has $S_T=O(1)$ and $H_T=O(1)$ under linear time.

  So epiplexity is small on both pure noise and trivially simple data — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Theorem 9 (PRGs).** For a PRG $G$ that stretches $k$ bits to $n=\mathrm{poly}(k)$ bits with advantage $\varepsilon(k)$:
  - $n-2-n\varepsilon(k)<H_{\mathrm{Poly}}(G(U_k))\le n+c$
  - $S_{\mathrm{Poly}}(G(U_k))\le c+n\varepsilon(k)$

  By contrast, Shannon entropy is $k$, and poly-time-bounded Kolmogorov complexity (and Levin complexity) is at most about $k+c$ — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Theorem 10.** Assuming one-way functions secure against non-uniform PPT adversaries, there are random variables $X_n$ with $S_{\mathrm{Poly}}(X_n)=\Omega(\log n)$. The argument is non-constructive. The paper admits this is "far from the power law scaling we see with some natural data" — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Definition 11 (conditional versions).** $S_T(Y|X)$ and $H_T(Y|X)$ come from the time-bounded MDL over conditional models. In general $H_T(Y,X)-H_T(X)\ne H_T(Y|X)$. The paper also allows conditioning on a deterministic string, e.g. a pretrained model — [arXiv HTML](https://arxiv.org/html/2601.03220v2)

**Estimators (Section 4)**
- **Prequential estimate (heuristic).** $|P_{\mathrm{preq}}|\approx\sum_{i=0}^{M-1}[\log 1/P_i(Z_i)-\log 1/P_M(Z_i)]$, i.e. the "area under the loss curve above the final loss". It relies on symmetry of information, and the paper states it is not a rigorous upper bound — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Requential estimate (rigorous).** $|P_{\mathrm{req}}|\approx\sum_i \mathrm{KL}(P^t_i\|P^s_i)$ (teacher vs. student), coded by relative-entropy coding.
  - Time is counted in FLOPs: $6ND$ for training plus $2N\mathcal D$ for evaluation.
  - Hyperparameters and the $N$/$D$ trade-off are optimized subject to $6ND+2N\mathcal D\le T$.
  - The requential-coding companion paper is now on arXiv: Qiu, Finzi, Zheng, Zhang, Wilson, "Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data", arXiv:2607.11883 (13 Jul 2026).

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2); [arXiv 2607.11883](https://arxiv.org/abs/2607.11883)
- The paper says "the epiplexity of a typical dataset is orders of magnitudes smaller than the random information content" — [arXiv HTML](https://arxiv.org/html/2601.03220v2)

**What the paper says about other measures (Section 2.2, Section 7, Appendices A.6 and H)**
- **Sophistication.**
  - It is the AIT concept that "captures exactly this idea". The paper uses "naive sophistication" (Mota et al. 2013): $\mathrm{nsoph}_c(x)=\min_S\{K(S):K(x|S)>\log|S|-c\}$.
  - Its problems: no specific string can be proven to have high sophistication (Chaitin incompleteness); the optimal programs' runtimes grow faster than any computable function (citing Ay et al. 2010); and with unbounded compute "many complex objects lose their complexity" (fluid mixing, citing Aaronson et al. 2014).

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Appendix A.6 (time-bounded sophistication collapses).** The paper states: "Epiplexity can be seen as a time-bounded and distributional generalization of sophistication."
  - Naive time-bounded sophistication is defined as $\mathrm{soph}^t_c(x):=\min_p\{|p|: p \text{ total}, \exists d\ U(p,d)=x, |p|+|d|\le K^t(x)+c\}$.
  - Lemma 29 shows it is $\le C_t$, a constant, for every $x$: a constant-size clocked universal interpreter is total and absorbs everything into $d$.

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Effective complexity and logical depth.** Effective complexity "aims to separate random from structural content (Gell-Mann and Lloyd, 1996)". Logical depth is "the number of time steps required by a nearly optimal program to produce a given string". The paper says depth "was later shown to be equivalent to sophistication through the busy beaver function (Antunes et al., 2005; Ay et al., 2010)" — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Algorithmic sufficient statistics and statistical complexity.** Algorithmic statistics (Vereshchagin & Vitányi 2004) offers "a principled decomposition of data into regular versus random components". Statistical complexity (Shalizi & Crutchfield 2001) "measures the entropy of causal states in an optimally predictive model". Both are criticized: "these existing notions ... do not account for the limited computation available to the observer ... cannot characterize CSPRNGs or encrypted objects as being random." Swapping $K$ for $K^t$ fails because PRG outputs have small $K^t$ — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Pseudoentropy.** HILL (Håstad et al. 1999), Yao (compression-based), Barak et al. 2003 and Hsiao et al. 2007 are "closely related to time-bounded entropy". The paper adds that "our formulation ... allows for separating out the structural information content, a key contribution" — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **V-entropy (Xu et al. 2020).** Criticized because:
  - "the computational constraint in V-entropy only limits the inference time, and does not account for the time to find such a model", so the minimizer "can be far away from the regime that is practically evaluated";
  - "both pseudoentropy and V-entropy, much like time-bounded entropy, capture only the random component";
  - Xu et al.'s DPI and symmetry violations are "not explicitly proven in the paper".

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Excess entropy (Crutchfield & Packard 1983; Shaw 1984; Grassberger 1986; Feldman 1998).** It is the area between finite-block entropy-density estimates and the asymptotic entropy rate, "an analogous construction to our prequential estimate of epiplexity". The difference is that it is defined for stationary processes and computationally unbounded observers — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Other ML-side relatives named.**
  - Surplus description length (Whitney et al. 2020).
  - Information transfer (Zhang et al. 2020): "more analogous to the spirit of epiplexity".
  - Teacher-size data complexity (Dziugaite & Roy 2025, PAC-Bayes).
  - Learning-curve theory (Hutter 2021).
  - Information bottleneck (Tishby et al. 2000).
  - Speed prior (Schmidhuber 2002).
  - Achille & Soatto 2025 ("information ... reduce[s] the time needed to solve new tasks").
  - Lempel-Ziv/Wolfram-class measures (Zhang et al. 2024).
  - Computational learning limits: Berthet & Rigollet 2013, Steinhardt et al. 2016, Raz 2018.

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Appendix H (MDL).**
  - NML: $P^{\rm NML}_{\mathcal H}(x)=P(x|\hat H(x))/\sum_y P(y|\hat H(y))$, which is intractable for DNNs.
  - Prequential code: $P^{\rm PREQ}(x)=\prod_k P(x_k|\hat H(x_{1:k}))$, where the update rule can be SGD.
  - Regret: $\mathrm{Reg}(Q,\mathcal H,x)=-\log Q(x)-\min_H\{-\log P(x|H)\}$. The two-part regret upper-bounds the model description length.

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Duality framing.** "While MDL is a criterion for model selection given a fixed dataset, epiplexity ... can be viewed as its dual: a criterion for data selection given a fixed computation budget" — [arXiv HTML](https://arxiv.org/html/2601.03220v2)

**Citation errors in the epiplexity paper**
- The bibliography lists "Antunes et al. (2005) Luis Antunes, Lance Fortnow, Dieter van Melkebeek, and N. V. Vinodchandran. Sophistication revisited. Theory of Computing Systems, 38(4):535–555, 2005." — [arXiv HTML](https://arxiv.org/html/2601.03220v2). Crossref shows that "Sophistication Revisited" is by **Antunes & Fortnow only**:
  - ICALP 2003, LNCS 2719, pp. 267–277, DOI 10.1007/3-540-45061-0_23;
  - journal version Theory of Computing Systems 45(1):150–161 (2009), DOI 10.1007/s00224-007-9095-5.

  The four-author paper is "Computational depth: Concept and applications", Theoretical Computer Science 354(3):391–404 (2006), DOI 10.1016/j.tcs.2005.11.033. The paper's entry appears to merge the two — [Crossref: Soph. revisited (journal)](https://api.crossref.org/works/10.1007/s00224-007-9095-5); [Crossref: Soph. revisited (ICALP)](https://api.crossref.org/works/10.1007/3-540-45061-0_23); [Crossref: Computational depth](https://api.crossref.org/works/10.1016/j.tcs.2005.11.033)
- The same entry is cited both for high-sophistication strings existing "by a diagonalization argument" and for the sophistication–busy-beaver-depth equivalence. That equivalence result is in Antunes & Fortnow's "Sophistication Revisited" (Theorem 5.3: $|\mathrm{csoph}(x)-\mathrm{depth_{bb}}(x)|\le O(\log n)$) — [Antunes & Fortnow PDF](https://lance.fortnow.com/papers/files/soph.pdf)
- The Yao 1982 DOI printed as 10.1109/SFCS.1982.95 returned 404 on Crossref, so it could not be verified — [Crossref query result](https://api.crossref.org/works/10.1109/SFCS.1982.95)

### Inferences
- Epiplexity combines three older ideas into one quantity:
  1. the two-part-code decomposition from Kolmogorov's structure function, sophistication and effective complexity;
  2. a resource bound on the model, borrowed from cryptography's indistinguishability-based pseudorandomness;
  3. an operational neural estimator borrowed from prequential MDL.

  Its distinctive choice is to make the model a time-bounded **probabilistic** program that must both evaluate and sample, and to take the expectation over a random variable. The evaluation requirement is plausibly what blocks the Lemma-29 collapse: a clocked universal interpreter is not a normalized, efficiently evaluable density, so it cannot be used to push all bits into the "data" part.
- The paper's statement that logical depth is "equivalent to sophistication through the busy beaver function (… Ay et al., 2010)" is looser than the literature:
  - Antunes & Fortnow prove coarse sophistication ≈ busy-beaver depth to $O(\log n)$.
  - Antunes, Bauwens, Souto & Teixeira (2017) relate sophistication and logical depth through the busy-beaver function with logarithmic precision.
  - Ay et al. (2010) relate **effective complexity**, not sophistication, to logical depth.

  A literature review should state these three results separately.

### Gaps
- No public peer reviews (e.g. OpenReview) of arXiv:2601.03220 were found, so "reviewer-suggested relatives" could not be collected. The acknowledgements thank Scott Aaronson and others for feedback, but that is not a review — [arXiv HTML](https://arxiv.org/html/2601.03220v2); [web search, no OpenReview hits](https://arxiv.org/abs/2601.03220)
- The v1 and v2 related-work sections were not diffed; everything above is quoted from v2.

---

## Q2. Algorithmic-information-theory family: exact definitions, references, properties, computability, relation to epiplexity

### Takeaway
All classic AIT "meaningful information" measures are defined for **individual strings** with an **unbounded** observer, and all are uncomputable:
- Kolmogorov structure function and minimal sufficient statistic;
- sophistication and its naive and coarse variants;
- logical depth and computational depth;
- effective complexity;
- facticity.

Most vanish (up to $O(\log n)$) on both incompressible and very simple strings, which satisfies the "structure is intermediate" desideratum. Epiplexity keeps that desideratum and the two-part-code skeleton. It replaces finite sets, total programs or ensembles with **T-time probabilistic programs** and replaces individual strings with **random variables** (datasets).

### Cited Findings

**2.1 Kolmogorov complexity (plain C, prefix K)**
- **Definition:** $K(x)=\min\{|p|:\mathcal U(p)=x\}$ for a universal prefix-free machine; $K(x|y)$ is the conditional version. The invariance theorem gives $|K_{\mathcal U_1}(x)-K_{\mathcal U_2}(x)|\le C$. A sequence is Martin-Löf random iff $K(x_{1:n})\ge n-c$ for all $n$; randomness deficiency is $\delta(x)=n-K(x)$ — [epiplexity paper, Def. 1–2](https://arxiv.org/html/2601.03220v2)
- **Canonical references:**
  - Kolmogorov, "Three approaches to the quantitative definition of information", Problemy Peredachi Informatsii / Problems Inform. Transmission 1(1):1–7 (1965) — [cited in Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf). Reprinted in Int. J. Computer Mathematics 2(1–4):157–168 (1968), DOI 10.1080/00207166808803030 — [Crossref](https://api.crossref.org/works/10.1080/00207166808803030)
  - Prefix complexity: Chaitin, "A theory of program size formally identical to information theory", JACM 22(3):329–340 (1975) — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
  - Textbook: Li & Vitányi, An Introduction to Kolmogorov Complexity and Its Applications, 4th ed., Springer 2019, DOI 10.1007/978-3-030-11298-1 — [Springer](https://link.springer.com/book/10.1007/978-3-030-11298-1)
- **Relation to epiplexity:**
  - Shannon entropy and $K$ both satisfy information non-increase under deterministic maps ($K(f(x))\le K(x)+K(f)+c$) and invariance to factorization. These are the paper's Paradoxes 1–2.
  - For a PRG output, $H=k$ and $K^{\rm poly}\lesssim k+c$, whereas $H_{\rm Poly}\approx n$.

  — [arXiv HTML](https://arxiv.org/html/2601.03220v2)

**2.2 Time-bounded Kolmogorov complexity $K^t$ and Levin's $Kt$**
- **$K^t$:** $K^t(x)=\min\{|p|: U(p)=x \text{ in at most } t(|x|) \text{ steps}\}$ — [Antunes, Fortnow, van Melkebeek, Vinodchandran, Def. 6](https://lance.fortnow.com/papers/files/depth-j.pdf)
- **Levin complexity:** $Kt(x|y)=\min_p\{|p|+\log t: U(p,y)\text{ halts in at most } t \text{ steps and outputs } x\}$ — [Antunes & Fortnow, Def. 2.7](https://lance.fortnow.com/papers/files/soph.pdf)
- **Canonical references:**
  - Levin, "Universal sequential search problems", Probl. Peredachi Inf. 9(3):115–116 (1973); English translation Problems Inform. Transmission 9(3):265–266 — [Math-Net.ru](https://www.mathnet.ru/eng/ppi914)
  - Levin, "Randomness conservation inequalities; information and independence in mathematical theories", Information and Control 61(1):15–37 (1984), DOI 10.1016/S0019-9958(84)80060-1 — [Crossref](https://api.crossref.org/works/10.1016/S0019-9958(84)80060-1)
  - Survey: Allender, Koucký, Ronneburger, Roy, "The pervasive reach of resource-bounded Kolmogorov complexity in computational complexity theory", JCSS 77(1):14–40 (2011), DOI 10.1016/j.jcss.2010.06.004 — [Crossref](https://api.crossref.org/works/10.1016/j.jcss.2010.06.004)
- **Computability:** Kt is "a computable, time-bounded version of Algorithmic Complexity" — [Scholarpedia: Universal search](http://www.scholarpedia.org/article/Universal_search)
- **Relation to epiplexity:** these are bounded-observer measures of **total** information. On a PRG output they are small ($\le k+c$), while epiplexity's $H_{\rm Poly}$ is near $n$. The epiplexity paper uses this to argue that resource-bounded $K$ does not see pseudorandomness as randomness — [arXiv HTML](https://arxiv.org/html/2601.03220v2)

**2.3 Kolmogorov structure function, algorithmic sufficient statistic, algorithmic statistics**
- **Kolmogorov's proposal:** Kolmogorov proposed the structure function orally (Tallinn 1973/1974). Antunes & Fortnow's form is $H_k(x|n)=\min\{\log|S|:x\in S, C(S|n)\le k\}$ — [Antunes & Fortnow, Def. 2.4](https://lance.fortnow.com/papers/files/soph.pdf)
- **Vereshchagin–Vitányi definitions:**
  - $h_x(\alpha)=\min_S\{\log|S|: S\ni x, K(S)\le\alpha\}$
  - $\beta_x(\alpha)=\min_S\{\delta(x|S): S\ni x, K(S)\le \alpha\}$, where the randomness deficiency is $\delta(x|S)=\log|S|-K(x|S)$
  - $\lambda_x(\alpha)=\min_S\{\Lambda(S)=K(S)+\log|S|: S\ni x, K(S)\le\alpha\}$, described as "the celebrated two-part Minimum Description Length code length ... as a function of α"

  — [Vereshchagin & Vitányi](https://arxiv.org/abs/cs/0204037)
- **Sufficient statistic:** $S$ is a sufficient statistic for $x$ if $K(S)+\log|S|=K(x)+O(1)$. The **minimal sufficient statistic** is the one with the least such $\alpha$ ($\alpha_0$). "The minimal sufficient statistic model expresses all meaningful information in x, and its complexity is the number of bits of meaningful information in the data x. The remainder $h_x(\alpha_0)$ bits ... is the 'noise'" — [Vereshchagin & Vitányi](https://arxiv.org/abs/cs/0204037)
- **Main theorem:** $\beta_x(\alpha)=h_x(\alpha)+\alpha-K(x)=\lambda_x(\alpha)-K(x)$ up to terms logarithmic in $|x|$. Constrained two-part MDL therefore selects best-fitting models "with certainty". Every admissible shape of $h_x$ is realized by some $x$, so non-stochastic strings are common — [Vereshchagin & Vitányi](https://arxiv.org/abs/cs/0204037)
- **Computability:** $h_x$ and $\lambda_x$ are upper semicomputable. $\beta_x$ is neither upper nor lower semicomputable, but is computable with a halting oracle — [Vereshchagin & Vitányi](https://arxiv.org/abs/cs/0204037)
- **Canonical references:**
  - Vereshchagin & Vitányi, "Kolmogorov's structure functions and model selection", IEEE Trans. Inf. Theory 50(12):3265–3290 (2004), DOI 10.1109/TIT.2004.838346, arXiv cs/0204037 — [Crossref](https://api.crossref.org/works/10.1109/TIT.2004.838346)
  - Gács, Tromp & Vitányi, "Algorithmic statistics", IEEE Trans. Inf. Theory 47(6):2443–2463 (2001), DOI 10.1109/18.945257, arXiv math/0006233. They generalize the model class from finite sets to computable probability distributions — [Crossref](https://api.crossref.org/works/10.1109/18.945257); [Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf)
  - Later survey: Vereshchagin & Shen, "Algorithmic statistics: forty years later", arXiv:1607.08077 (Springer chapter DOI 10.1007/978-3-319-50062-1_41) — [arXiv PDF](https://arxiv.org/pdf/1607.08077); [Springer](https://link.springer.com/chapter/10.1007/978-3-319-50062-1_41)

**2.4 Sophistication (Koppel), naive sophistication, coarse sophistication**
- **Sophistication with significance $c$:** $\mathrm{soph}_c(x)=\min\{|p|: p \text{ total}, \exists d:\ U(p,d)=x,\ |p|+|d|\le K(x)+c\}$ — [Antunes & Fortnow, Def. 2.6/3.1](https://lance.fortnow.com/papers/files/soph.pdf); [epiplexity App. A.6, Def. 27](https://arxiv.org/html/2601.03220v2)
- **Koppel's original papers:**
  - Koppel, "Complexity, Depth, and Sophistication", Complex Systems 1(6):1087–1091 (1987). It shows that sophistication (the "projectable part" of the minimal description) and Bennett's depth are equivalent up to a constant for infinite strings, under an appropriate translation — [Complex Systems PDF](https://content.wolfram.com/sites/13/2018/02/01-6-4.pdf); [abstract page](https://www.complex-systems.com/abstracts/v01_i06_a04/)
  - Koppel, "Structure", in R. Herken (ed.), The Universal Turing Machine: A Half-Century Survey, Oxford Univ. Press, 1988, pp. 435–452 — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
  - Koppel & Atlan, "An almost machine-independent theory of program-length complexity, sophistication, and induction", Information Sciences 56(1–3):23–33 (1991), DOI 10.1016/0020-0255(91)90021-L — [Crossref](https://api.crossref.org/works/10.1016/0020-0255(91)90021-L)
- **Naive sophistication:** $\mathrm{nsoph}_c(x)=\min_S\{K(S): x\in S,\ \delta(x|S)\le c\}$. It is "the complexity of the simplest set in which x is a typical element" and was first introduced by Aaronson in a MathOverflow question. It is equivalent to sophistication up to $O(\log|x|)$ terms, via Vereshchagin–Vitányi: $\mathrm{soph}_{c+O(\log|x|)}(x)\le \mathrm{nsoph}_{c+O(\log |x|)}(x)\le \mathrm{soph}_c(x)$. Mota et al. also define naive **coarse** sophistication and relate naive sophistication to lossy compression and busy-beaver depth. Reference: Mota, Aaronson, Antunes, Souto, "Sophistication as Randomness Deficiency", DCFS 2013, LNCS 8031, pp. 172–181, DOI 10.1007/978-3-642-39310-5_17 — [author PDF](https://www.scottaaronson.com/papers/DCFS-Final.pdf); [Springer](https://link.springer.com/chapter/10.1007/978-3-642-39310-5_17)
- **Coarse sophistication:** $\mathrm{csoph}(x)=\min\{2|p|+|d|-C(x): U(p,d)=x,\ p \text{ total}\}$. This is "|p| for sophistication plus a penalty |p|+|d|−C(x)", which removes the unstable significance parameter.
  - Theorem 4.1: $\mathrm{csoph}(x)\le n/2+c$.
  - Theorem 4.2: some $x$ of length $n$ has $\mathrm{csoph}(x)>n/2-4\log n$.
  - Theorem 4.3: given $x$ and $O(\log n)$ bits one can solve the halting problem for all programs shorter than $\mathrm{csoph}(x)-2\log n$.

  — [Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf)
- **Busy-beaver computational depth:** $\mathrm{depth_{bb}}(x)=\min\{|p|-C(x)+k: U(p)=x \text{ in } t \text{ steps}, t\le BB(k)\}$, with $BB(n)=\max_{|p|\le n}\{\text{running time of } U(p)\}$. Theorem 5.3: $|\mathrm{csoph}(x)-\mathrm{depth_{bb}}(x)|\le O(\log n)$ — [Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf)
- **Canonical reference:** Antunes & Fortnow, "Sophistication Revisited". ICALP 2003, LNCS 2719, pp. 267–277, DOI 10.1007/3-540-45061-0_23; Theory of Computing Systems 45(1):150–161 (2009), DOI 10.1007/s00224-007-9095-5 — [Crossref ICALP](https://api.crossref.org/works/10.1007/3-540-45061-0_23); [Crossref TOCS](https://api.crossref.org/works/10.1007/s00224-007-9095-5)
- **Sophistication vs. depth:** Antunes, Bauwens, Souto & Teixeira show that the busy-beaver function of sophistication exceeds logical depth with logarithmically bigger precision, and vice versa. Reference: "Sophistication vs Logical Depth", Theory of Computing Systems 60(2):280–298 (2017), DOI 10.1007/s00224-016-9672-6, arXiv:1304.8046 — [arXiv](https://arxiv.org/abs/1304.8046); [Crossref](https://api.crossref.org/works/10.1007/s00224-016-9672-6)

**2.5 Logical depth (Bennett)**
- **Definition:** $\mathrm{depth}_s(x)=\min\{t: U(p) \text{ halts and outputs } x \text{ in at most } t \text{ steps}, |p|<C(x)+s\}$. This is the time needed to produce $x$ from a near-minimal program.
- **Properties:**
  - "algorithmically random strings are shallow at any significance level", and Chaitin's Ω is shallow;
  - deep strings can be constructed by diagonalization;
  - **slow growth law**: fast deterministic processes cannot turn shallow objects into deep ones, and fast probabilistic ones can do so only with small probability.

  — [Antunes & Fortnow, Def. 2.8](https://lance.fortnow.com/papers/files/soph.pdf)
- **Canonical reference:** C. H. Bennett, "Logical depth and physical complexity", in R. Herken (ed.), The Universal Turing Machine: A Half-Century Survey, Oxford Univ. Press, 1988, pp. 227–257 — [Oxford Academic](https://academic.oup.com/book/54493/chapter-abstract/422572845)

**2.6 Computational depth (Antunes, Fortnow, van Melkebeek, Vinodchandran)**
- **Idea:** "a measure for the amount of 'nonrandom' or 'useful' information in a string by considering the difference of various Kolmogorov complexity measures" — [AFMV PDF](https://lance.fortnow.com/papers/files/depth-j.pdf)
- **Basic computational depth (Def. 13):** $bcd^t(x)=K^t(x)-K(x)$. Theorem 5: there are at least $2^{\epsilon n}$ strings of length $n$ with $bcd^{2^n}(x)\ge(1-\epsilon)n-c\log n$ — [AFMV PDF](https://lance.fortnow.com/papers/files/depth-j.pdf)
- **Levin-based variant:** $Kt(x)-K(x)$, "a variation of Bennett's logical depth" that "does not need a significance level parameter" — [Antunes & Fortnow, Def. 2.9](https://lance.fortnow.com/papers/files/soph.pdf)
- **Other instantiations:**
  - Sublinear-time depth $D^t(x)=C^t(x)-C(x)$, which yields "shallow sets"; "Random sets are also shallow".
  - **Distinguishing computational depth**: polynomial-time distinguishing complexity (Sipser) versus polynomial-time $K$, which "measures the difference between recognizing a string and producing it".

  — [AFMV PDF](https://lance.fortnow.com/papers/files/depth-j.pdf)
- **Canonical reference:** Theoretical Computer Science 354(3):391–404 (2006), DOI 10.1016/j.tcs.2005.11.033 — [Crossref](https://api.crossref.org/works/10.1016/j.tcs.2005.11.033)
- **Related:** Antunes, Matos, Souto, Vitányi, "Depth as Randomness Deficiency" (arXiv:0809.2546); Doty & Moser, "Feasible Depth" (arXiv cs/0701123) — [arXiv 0809.2546](https://arxiv.org/abs/0809.2546); [arXiv cs/0701123](https://arxiv.org/abs/cs/0701123)

**2.7 Effective complexity (Gell-Mann & Lloyd), formalized by Ay, Müller & Szkoła**
- **Definition:**
  - Total information of an ensemble (a computable distribution) $E$: $\Sigma(E):=K(E)+H(E)$.
  - $x$ is $\delta$-typical for $E$ if $E(x)\ge 2^{-H(E)(1+\delta)}$.
  - Effective complexity: $\mathcal E_{\delta,\Delta}(x):=\inf\{K(E): x \text{ is } \delta\text{-typical for } E,\ \Sigma(E)\le K(x)+\Delta\}$, i.e. the complexity of the simplest "good theory". It can be extended with constraints on admissible ensembles. It is "closely related to ... 'Kolmogorov minimal sufficient statistics'".

  — [Ay, Müller, Szkoła](https://arxiv.org/abs/0810.5663)
- **Properties:**
  - Theorem 10: incompressible strings are effectively simple.
  - Theorem 14: strings with effective complexity close to their length exist.
  - Theorems 18–19: above an explicit threshold, effective complexity forces "astronomically large" logical depth; below it, depth can be arbitrarily small (a phase-transition-like relation).

  — [Ay, Müller, Szkoła](https://arxiv.org/abs/0810.5663)
- **Canonical references:**
  - Gell-Mann & Lloyd, "Information measures, effective complexity, and total information", Complexity 2(1):44–52 (1996), DOI 10.1002/(SICI)1099-0526(199609/10)2:1<44::AID-CPLX10>3.0.CO;2-X (verified on Crossref)
  - Ay, Müller, Szkoła, "Effective complexity and its relation to logical depth", IEEE Trans. Inf. Theory 56(9):4593–4607 (2010), DOI 10.1109/TIT.2010.2053892, arXiv:0810.5663

  — [Crossref Ay et al.](https://api.crossref.org/works/10.1109/TIT.2010.2053892); [arXiv 0810.5663](https://arxiv.org/abs/0810.5663)

**2.8 Facticity (Adriaans)**
- **Definition:**
  - Two-part complexity: $K_2(x)=\min_{i,p}\{|i|+|p|: U(\bar i p)=x\}$, where $\bar i$ is a self-delimiting index of a Turing machine $T_i$.
  - **Facticity:** $\varphi_U(x)=\min\{|i|: \exists p\ (|\bar i p|=K_2(x)\ \&\ U(\bar i p)=x)\}$, "the length of the shortest model code of all optimal models under two-part code optimization".
  - It assumes a "faithful" index, i.e. index length ≈ $C$ of the machine; such an index exists but is not recursive.

  — [Adriaans 2012](https://arxiv.org/abs/1203.2245)
- **Properties:** random strings have facticity 0 (Lemma 10), and compressible strings have $0<\varphi(x)<\tfrac12|x|+O(1)$. There is a "facticity threshold" dependent on entropy, and the maximal facticity of stochastic strings of length $n$ is $O(n/\log n)$ — [Adriaans 2012](https://arxiv.org/abs/1203.2245)
- **Resource-bounded variant (Definition 15):** $\varphi^t_U(x)=\min\{|i|:\exists p\,(|\bar i p|=K_2(x)\ \&\ U(\bar i p)=x \text{ in at most } t(|x|) \text{ steps})\}$ — [Adriaans 2012](https://arxiv.org/abs/1203.2245)
- **Canonical references:**
  - Adriaans, "Facticity as the amount of self-descriptive information in a data set", arXiv:1203.2245 (2012)
  - Adriaans, "Between order and chaos: The quest for meaningful information", Theory of Computing Systems 45(4):650–674 (2009), DOI 10.1007/s00224-009-9173-y, which introduces facticity

  — [arXiv 1203.2245](https://arxiv.org/abs/1203.2245); [Springer](https://link.springer.com/article/10.1007/s00224-009-9173-y)

### Inferences
- **Epiplexity as a time-bounded minimal sufficient statistic.** Structurally, $S_T$ is the time-bounded, distributional analogue of the **complexity of the algorithmic minimal sufficient statistic** $\alpha_0$. $\mathrm{MDL}_T=S_T+H_T$ plays the role of $\lambda_x$ at its minimum. "Ties broken by the smallest program" mirrors choosing the *minimal* sufficient statistic among optimal two-part codes. Effective complexity is closer still, because its data part is an ensemble **entropy** $H(E)$, just as $H_T$ is an expected code length. $\Sigma(E)=K(E)+H(E)$ is exactly the unbounded version of $|P|+\mathbb E[\log 1/P(X)]$.
- **Observer bound in facticity.** Adriaans' $t$-facticity is an early, easily overlooked "resource-bounded structure" proposal. It still anchors optimality to the **unbounded** $K_2(x)$, whereas epiplexity anchors to the time-bounded optimum. Facticity is therefore not truly observer-relative in the epiplexity sense.
- **What AIT measures capture that epiplexity does not:**
  - Logical depth, busy-beaver depth and computational depth measure **computation time** ("evolvedness"), not model size. Epiplexity measures model size at a fixed time budget.
  - Bennett's slow growth law, which says depth can only be produced slowly, is a natural counterpart to epiplexity's claim that structural information can be created by computation. Epiplexity makes creation observer-relative, where depth makes it time-costly.
- **What epiplexity captures that AIT measures do not:**
  1. Pseudorandom or encrypted content counts as random. Under $K$, sophistication, sufficient statistics and effective complexity, a PRG output has a short description and is structured or simple.
  2. Estimability from neural training runs.
  3. A dataset-level (distributional) quantity that scales with data size.
- **Existence results line up.** High-sophistication and high-effective-complexity strings exist but cannot be exhibited. Epiplexity's own existence result (Theorem 10) is only $\Omega(\log n)$ and non-constructive. Neither theory has an explicit, provable high-structure example; epiplexity's contribution there is empirical.

### Gaps
- Koppel 1988 ("Structure", Herken volume) was not read directly. Its definition is taken from Antunes & Fortnow's and the epiplexity paper's restatements.
- Gell-Mann & Lloyd 1996 itself was not read; the formal definition used is Ay et al.'s formalization. The original is informal about the "Δ" slack and constraints.
- The exact page range of Kolmogorov 1965 differs by source: Russian original pp. 3–11 versus English translation pp. 1–7. Only the "1:1 (1965), 1–7" form cited by Antunes & Fortnow was verified.
- Chaitin 1966 (JACM 13(4)) and Solomonoff 1964 (Information and Control 7) were not verified in this session.

---

## Q3. Statistical-physics / dynamical-systems family: statistical complexity, excess entropy, predictive information (and apparent complexity)

### Takeaway
These are **Shannon-type, ensemble-level** structure measures for stationary processes. All vanish on i.i.d. noise and on constant sequences, and they are defined for an unbounded, Bayes-optimal observer:
- excess entropy $E$ (the past–future mutual information);
- predictive information $I_{\rm pred}(T)$ (its finite-window, sub-extensive form);
- statistical complexity $C_\mu$ (the entropy of the minimal optimal predictor's causal states), with $E\le C_\mu$.

Excess entropy and predictive information are the closest structural ancestors of epiplexity's **prequential** estimator: both are "area above the asymptotic loss/entropy rate" quantities. Bialek et al. explicitly identify the learning curve as the derivative of predictive information.

### Cited Findings

**3.1 Statistical complexity and ε-machines**
- **Causal states:** pasts are grouped by an equivalence relation $\epsilon$ that maps each past to the set of pasts with the same conditional distribution over futures. The ε-machine is the pair $\{\epsilon, T\}$ of causal-state function and labelled transition matrices $T_{ij}^{(s)}$ — [Shalizi & Crutchfield, Defs. 5, 8, 9](https://arxiv.org/abs/cond-mat/9907176)
- **Statistical complexity of a process:** $C_\mu(\mathcal O)\equiv C_\mu(\mathcal S)=H[\mathcal S]$, the entropy of the causal-state distribution. It measures "the average amount of historical memory stored in the process"; minimality of causal states among prescient rivals (Theorem 2) makes it well defined — [Shalizi & Crutchfield, Def. 12](https://arxiv.org/abs/cond-mat/9907176)
- **Bound:** Theorem 5 ("The Bounds of Excess") states $E\le C_\mu$ — [Shalizi & Crutchfield](https://arxiv.org/abs/cond-mat/9907176)
- **Canonical references:**
  - Crutchfield & Young, "Inferring statistical complexity", Phys. Rev. Lett. 63(2):105–108 (1989), DOI 10.1103/PhysRevLett.63.105 — [Crossref](https://api.crossref.org/works/10.1103/PhysRevLett.63.105)
  - Shalizi & Crutchfield, "Computational mechanics: Pattern and prediction, structure and simplicity", J. Stat. Phys. 104(3–4):817–879 (2001), DOI 10.1023/A:1010388907793, arXiv cond-mat/9907176 — [Crossref](https://api.crossref.org/works/10.1023/A:1010388907793)

**3.2 Excess entropy**
- **Definition:** with block entropy $H(L)$ and $h_\mu(L)=H(L)-H(L-1)$, $E\equiv\sum_{L=1}^{\infty}[h_\mu(L)-h_\mu]$.
- **Equivalent forms:**
  - $E$ is the sub-extensive part of $H(L)$: $H(L)\to E+h_\mu L$ (Prop. 7).
  - $E$ is the mutual information between the semi-infinite past and future (Prop. 8).
- **Examples:**
  - period-$p$ process: $E=\log_2 p$;
  - order-$R$ Markov process: $E=H(R)-Rh_\mu$;
  - processes with finite $E$ are "finitary".
- **Aliases:** "stored information", "effective measure complexity", "predictive information".

— [Crutchfield & Feldman](https://arxiv.org/abs/cond-mat/0102181)
- **Estimation:** finite-$L$ estimates are available as partial sums or as the mutual information between $L/2$-blocks — [Crutchfield & Feldman](https://arxiv.org/abs/cond-mat/0102181)
- **Canonical references:**
  - Crutchfield & Feldman, "Regularities unseen, randomness observed: Levels of entropy convergence", Chaos 13(1):25–54 (2003), DOI 10.1063/1.1530990, arXiv cond-mat/0102181 — [Crossref](https://api.crossref.org/works/10.1063/1.1530990)
  - Earlier: Grassberger, "Toward a quantitative theory of self-generated complexity", Int. J. Theor. Phys. 25(9):907–938 (1986), DOI 10.1007/BF00668821, which introduced "effective measure complexity" — [Crossref](https://api.crossref.org/works/10.1007/BF00668821)
  - The epiplexity paper also credits Crutchfield & Packard 1983 (Physica D 7:201–223) and Shaw 1984 — [arXiv HTML](https://arxiv.org/html/2601.03220v2)

**3.3 Predictive information (Bialek, Nemenman, Tishby)**
- **Definition:** $I_{\rm pred}(T)$ is "the mutual information between the past and the future of a time series". If $S(T)=S_0T+S_1(T)$, extensive terms cancel, so predictive information depends only on the **sub-extensive** term $S_1(T)$: "predictability is a deviation from extensivity" — [Bialek et al.](https://arxiv.org/abs/physics/0007070)
- **Three large-$T$ regimes:** finite; logarithmic, with coefficient counting model dimensionality, $\approx (K/2)\log_2 N$ for $K$-parameter model classes; and fractional power law, as for nonparametric or infinite-parameter models. The authors argue the divergent part of $I_{\rm pred}$ "provides the unique measure for the complexity of dynamics" — [Bialek et al.](https://arxiv.org/abs/physics/0007070)
- **Link to learning and MDL:**
  - "the learning curve is the derivative of the predictive information".
  - The mutual information between data and parameters equals the predictive information about the whole future, "also the 'cumulative information gain' ... or the 'cumulative relative entropy risk'".
  - "the subextensive component of the description length (Rissanen ...) ... also is similar to the predictive information".

  — [Bialek et al.](https://arxiv.org/abs/physics/0007070)
- **Canonical reference:** Bialek, Nemenman, Tishby, "Predictability, complexity, and learning", Neural Computation 13(11):2409–2463 (2001), DOI 10.1162/089976601753195969, arXiv physics/0007070 — [Crossref](https://api.crossref.org/works/10.1162/089976601753195969)

**3.4 Apparent complexity (a coarse-graining relative)**
- **Definition:** apparent complexity of $x$ is $H(f(x))$, where $H$ is an entropy or Kolmogorov measure and $f$ is a "denoising" or "smoothing" function. The random-sequence case "would typically be quite small". It is approximated with gzip in the "coffee automaton" — [Aaronson, Carroll, Ouellette](https://arxiv.org/abs/1405.6903)
- **Why it matters here:** the epiplexity paper cites this work for the point that complex transients (fluid mixing) could be reproduced by short programs given unbounded compute — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Reference:** Aaronson, Carroll, Ouellette, "Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton", arXiv:1405.6903 (2014) — [arXiv](https://arxiv.org/abs/1405.6903)

### Inferences
- **Mathematical link to the prequential estimate.** For an i.i.d. dataset and an ideal Bayesian learner, the cumulative excess log-loss above the asymptote equals the parameter–data mutual information, $\approx (K/2)\log N$. By Bialek et al.'s identification, that is exactly the predictive information. The prequential epiplexity estimate $\sum_i[\log 1/P_i-\log 1/P_M]$ is its compute-bounded, SGD-learner counterpart, which is why the epiplexity paper calls excess entropy "analogous". The key differences:
  - the reference loss is the **final model's** loss at budget $T$, not the true entropy rate;
  - the learner is a specific neural network trained under a FLOP budget, not a Bayes-optimal predictor.
- **Behaviour on pseudorandom data.** $E$, $I_{\rm pred}$ and $C_\mu$ vanish on i.i.d. noise and on constants, and are small ($\log p$) on periodic data, so they satisfy the intermediate-structure desideratum. But on a PRG or cipher stream an unbounded observer sees determinism:
  - the causal-state memory must track the generator state, so $C_\mu$ is large;
  - excess entropy counts the seed/state bits as "stored information".

  A polynomial-time observer sees only noise. This is the exact failure mode the epiplexity paper criticizes.
- **What these measures capture that epiplexity does not:**
  - an intrinsic, model-class-free characterization of stationary processes (memory versus prediction, $E\le C_\mu$);
  - the distinction between information the process *stores* ($C_\mu$) and what it *transmits* ($E$);
  - a classification of growth regimes (log versus power law) that maps onto scaling-law exponents.

  Epiplexity has no analogue of "internal memory versus observable correlation".

### Gaps
- No direct source was fetched on modern neural estimators of excess entropy or statistical complexity, e.g. using transformers' in-context loss-versus-position curves or neural ε-machine reconstruction. Practical algorithms such as CSSR (Shalizi & Klinkner 2004) exist to the best of my knowledge but were not verified in this session.
- Feldman 1998 ("Information theory, excess entropy", lecture notes) and Shaw 1984 were not retrieved.

---

## Q4. Computational pseudo-entropy family: HILL, metric, Yao, next-block pseudoentropy, and relatives

### Takeaway
Cryptographic pseudoentropies are **observer-dependent** (defined relative to a class of efficient tests or compressors). They correctly treat PRG outputs as high-entropy, just like $H_T$. But they quantify only the **random** component: they are maximal on noise, not zero. They are therefore precursors of **time-bounded entropy**, not of epiplexity. Two conceptual parallels stand out:
- next-block pseudoentropy is autoregressive and depends on the block (factorization) order, which mirrors epiplexity's Paradox 2;
- Yao's compression-based pseudoentropy is the closest cryptographic analogue of a code-length definition like $H_T$.

### Cited Findings
- **HILL-type pseudoentropy (Def. 3.1):** $X$ has $\epsilon$-HILL pseudoentropy at least $k$, written $H^{\rm HILL}_\epsilon(X)\ge k$, if there is a $Y$ with statistical min-entropy $H_\infty(Y)\ge k$ whose computational distance from $X$ (with respect to a test class $\mathcal C$) is at most $\epsilon$ — [Barak, Shaltiel, Wigderson](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/BSW03/BSW03.pdf)
  - Original: Håstad, Impagliazzo, Levin, Luby, "A pseudorandom generator from any one-way function", SIAM J. Comput. 28(4):1364–1396 (1999), DOI 10.1137/S0097539793244708 — [Crossref](https://api.crossref.org/works/10.1137/S0097539793244708)
- **Metric-type pseudoentropy (Def. 3.2):** the quantifiers are swapped: for **every** test $f$ there is a $Y$ with $H_\infty(Y)\ge k$ and $\mathrm{bias}_f(X,Y)<\epsilon$. It equals HILL for polynomial-time observers (via the min-max theorem) but drastically differs for logspace observers — [Barak, Shaltiel, Wigderson](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/BSW03/BSW03.pdf)
- **Yao-type (compression) pseudoentropy (Def. 3.4):**
  - A set $D$ is efficiently compressible to $\ell$ bits if efficient $c,d$ exist with $D=\{x: d(c(x))=x\}$.
  - $X$ has $\epsilon$-Yao pseudoentropy at least $k$ if for every $\ell<k$ and every such $D$, $\Pr[X\in D]\le 2^{\ell-k}+\epsilon$.
  - It is motivated as Shannon's source-coding definition with the added constraint that compression and decompression must be efficient.
  - Permissiveness ordering: HILL ≤ metric ≤ Yao. All three are equivalent for poly-size PH-circuits.

  — [Barak, Shaltiel, Wigderson](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/BSW03/BSW03.pdf)
  - Yao's original: A. C. Yao, "Theory and applications of trapdoor functions", FOCS 1982, pp. 80–91 — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
  - Canonical comparison paper: Barak, Shaltiel, Wigderson, "Computational Analogues of Entropy", RANDOM-APPROX 2003, LNCS 2764, pp. 200–215, DOI 10.1007/978-3-540-45198-3_18 — [Crossref](https://api.crossref.org/works/10.1007/978-3-540-45198-3_18)
- **Next-block pseudoentropy (informal Def. 1.1):** $X=(X_1,\dots,X_m)$ has next-block pseudoentropy at least $k$ if there is a jointly distributed $Z=(Z_1,\dots,Z_m)$ such that:
  - $(X_1,\dots,X_{i-1},X_i)\approx_c(X_1,\dots,X_{i-1},Z_i)$ for all $i$; and
  - $\sum_{i=1}^m H(Z_i|X_1,\dots,X_{i-1})\ge k$.

  For a OWF $f$ with input split into $O(\log n)$-bit blocks, $(f(X),X_1,\dots,X_m)$ has next-block pseudoentropy $\ge n+\omega(\log n)$, while its real entropy is $n$ (Vadhan–Zheng) — [Agrawal, Chen, Horel, Vadhan](https://arxiv.org/abs/1902.11202)
- **Next-block pseudoentropy, origin:** it was introduced in Haitner, Reingold, Vadhan, "Efficiency improvements in constructing pseudorandom generators from one-way functions", STOC 2010; SIAM J. Comput. 42(3):1405–1430 (2013), DOI 10.1137/100814421. It was inspired by "inaccessible entropy" (Haitner, Reingold, Vadhan, Wee, STOC 2009) — [Crossref](https://api.crossref.org/works/10.1137/100814421); [Vadhan publications page](https://salil.seas.harvard.edu/publications/efficiency-improvements-constructing-pseudorandom-generators-one-way-functions)
- **Next-block pseudoentropy, from any OWF:** proven in Vadhan & Zheng, "Characterizing pseudoentropy and simplifying pseudorandom generator constructions", STOC 2012, pp. 817–836, DOI 10.1145/2213977.2214051 — [Crossref](https://api.crossref.org/works/10.1145/2213977.2214051)
- **KL-based unification:** relative pseudoentropy requires $\mathrm{KL}(X,Y\|S(Y),Y)\ge\Delta$ for all PPT simulators $S$. "Hardness in relative entropy" unifies next-block pseudoentropy and (in)accessible entropy. Reference: Agrawal, Chen, Horel, Vadhan, "Unifying computational entropies via Kullback–Leibler divergence", CRYPTO 2019, DOI 10.1007/978-3-030-26951-7_28, arXiv:1902.11202 — [arXiv](https://arxiv.org/abs/1902.11202); [Springer](https://link.springer.com/chapter/10.1007/978-3-030-26951-7_28)
- **Conditional computational entropy:** Hsiao, Lu, Reyzin, "Conditional Computational Entropy, or Toward Separating Pseudoentropy from Compressibility", EUROCRYPT 2007, LNCS 4515, pp. 169–186, DOI 10.1007/978-3-540-72540-4_10 — [Crossref](https://api.crossref.org/works/10.1007/978-3-540-72540-4_10)

### Inferences
- $H_T$ is essentially a **Yao-style, code-length pseudoentropy with an MDL model penalty and a sampling requirement**. What distinguishes the epiplexity framework is not $H_T$ but the complementary $S_T$. No pseudoentropy notion has a "structure" counterpart.
- Next-block pseudoentropy can exceed true Shannon entropy and depends on how $X$ is split into blocks. This is a rigorous cryptographic precedent for epiplexity's Paradox 2 (information depends on factorization/ordering) and for the observation that $H_T(Y,X)-H_T(X)\neq H_T(Y|X)$.
- Estimation: pseudoentropies quantify over all efficient tests or distributions, so there is no practical estimator. Any trained model is an efficient distinguisher or compressor, so its held-out log-loss gives only a one-sided bound on the observer's achievable compression, in the Yao sense.

### Gaps
- The exact published venue page for HRV STOC 2010 was not fetched; only the SICOMP 2013 version was verified.
- No source was found that explicitly connects pseudoentropy to neural-network learning curves, beyond the epiplexity paper itself.

---

## Q5. Usable-information and ML-side measures: V-information, PVI, MDL/prequential coding, information in the weights / task complexity, and close relatives

### Takeaway
These measures are the **operational** neighbours of epiplexity. All are estimable with neural networks at small scale:
- V-information and PVI restrict the observer to a predictive family, but measure *usable predictive* information about a target, i.e. a reduction in (V-)entropy. They include no model-description cost.
- MDL/prequential coding supplies the two-part/one-part code machinery that epiplexity reuses.
- Achille et al.'s "information in the weights" is a PAC-Bayes/KL analogue of $S_T$: the KL term plays the role of $|P|$, trading off against training loss. It is task-relative and bounded by model class, not by time.

### Cited Findings

**5.1 Predictive V-information (Xu, Zhao, Song, Stewart, Ermon)**
- **Predictive family (Def. 1):** $\mathcal V\subseteq\Omega=\{f:\mathcal X\cup\{\varnothing\}\to\mathcal P(\mathcal Y)\}$, with the "optional ignorance" condition: for every $f\in\mathcal V$ and $P\in\mathrm{range}(f)$ there is $f'\in\mathcal V$ with $f'[x]=P$ for all $x$ and $f'[\varnothing]=P$ — [Xu et al.](https://arxiv.org/abs/2002.10689)
- **Definitions:**
  - Conditional V-entropy (Def. 2): $H_{\mathcal V}(Y|X)=\inf_{f\in\mathcal V}\mathbb E_{x,y}[-\log f[x](y)]$ and $H_{\mathcal V}(Y|\varnothing)=\inf_{f\in\mathcal V}\mathbb E_y[-\log f[\varnothing](y)]$.
  - Predictive V-information (Def. 3): $I_{\mathcal V}(X\to Y)=H_{\mathcal V}(Y|\varnothing)-H_{\mathcal V}(Y|X)$.
  - With $\mathcal V=\Omega$ these reduce to Shannon quantities.

  — [Xu et al.](https://arxiv.org/abs/2002.10689)
- **Properties:**
  - Proposition 2: monotonicity in $\mathcal V$, non-negativity, and zero when $X\perp Y$.
  - Section 3.2: V-information can be created by computation, violating the DPI (e.g. decryption: $I_{\mathcal V}(t(X)\to Y)>I_{\mathcal V}(X\to Y)$).
  - Section 3.3: it is asymmetric under one-way functions.
  - Section 4: PAC estimation bounds via the Rademacher complexity of $\mathcal V$.

  — [Xu et al.](https://arxiv.org/abs/2002.10689)
- **Canonical reference:** Xu, Zhao, Song, Stewart, Ermon, "A Theory of Usable Information Under Computational Constraints", ICLR 2020 (talk), arXiv:2002.10689 — [arXiv](https://arxiv.org/abs/2002.10689)

**5.2 Pointwise V-information / dataset difficulty (Ethayarajh, Choi, Swayamdipta)**
- **Definition (3.1):** $\mathrm{PVI}(x\to y)=-\log_2 g'[\varnothing](y)+\log_2 g[x](y)$, where $g\in\mathcal V$ is trained (fine-tuned) on $(x,y)$ pairs and $g'\in\mathcal V$ on (null input, $y$) pairs. $I_{\mathcal V}(X\to Y)=\mathbb E[\mathrm{PVI}(x\to y)]$. "PVI is to V-information what PMI is to Shannon information." Dataset difficulty for $\mathcal V$ is framed as the lack of V-usable information — [Ethayarajh et al., ICML 2022](https://proceedings.mlr.press/v162/ethayarajh22a.html)
- **Canonical reference:** "Understanding Dataset Difficulty with V-Usable Information", Proceedings of the 39th ICML, PMLR 162:5988–6008 (2022), arXiv:2110.08420. It won an ICML 2022 Outstanding Paper award — [PMLR](https://proceedings.mlr.press/v162/ethayarajh22a.html); [arXiv](https://arxiv.org/abs/2110.08420)

**5.3 MDL and prequential coding**
- **Two-part MDL:** $L(x)=\min_{H\in\mathcal H}L(H)-\log P(x|H)$ (Rissanen; Grünwald). NML, the prequential code and regret are defined as in the epiplexity paper's Appendix H — [epiplexity paper Def. 6 and App. H](https://arxiv.org/html/2601.03220v2)
- **Canonical references:**
  - Rissanen, "Modeling by shortest data description", Automatica 14(5):465–471 (1978), DOI 10.1016/0005-1098(78)90005-5 — [Crossref](https://api.crossref.org/works/10.1016/0005-1098(78)90005-5)
  - Rissanen, "A universal prior for integers and estimation by minimum description length", Ann. Statist. 11(2) (1983), DOI 10.1214/aos/1176346150 — [Crossref](https://api.crossref.org/works/10.1214/aos/1176346150)
  - Dawid, "Present position and potential developments: Some personal views: Statistical theory: The prequential approach", J. Royal Statistical Society Series A 147(2):278– (1984), DOI 10.2307/2981683. Crossref lists only the start page; the epiplexity paper gives 278–290 — [Crossref](https://api.crossref.org/works/10.2307/2981683); [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
  - Grünwald, The Minimum Description Length Principle, MIT Press, 2007 — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
- **Deep-learning instantiations:**
  - Blier & Ollivier, "The Description Length of Deep Learning Models", NeurIPS 2018, arXiv:1802.07044, showing prequential codes for DNNs — [arXiv](https://arxiv.org/abs/1802.07044)
  - Voita & Titov, "Information-Theoretic Probing with Minimum Description Length", arXiv:2003.12298 — [arXiv](https://arxiv.org/abs/2003.12298)
  - Finzi et al., "Compute-Optimal LLMs Provably Generalize Better With Scale", ICLR 2025, arXiv:2504.15208, which is the source of the prequential model-size heuristic reused in epiplexity — [arXiv](https://arxiv.org/abs/2504.15208)
  - Requential coding (Qiu et al. 2026, arXiv:2607.11883). Its code length "is independent of parameter count and data entropy" and is "often orders of magnitude shorter than the prequential counterpart" — [arXiv](https://arxiv.org/abs/2607.11883)

**5.4 Information in the weights and task complexity (Achille, Paolini, Mbeng, Soatto)**
- **Task complexity (Def. 3.1):** $C(\mathcal D)=\min_{p(y|x)} L_{\mathcal D}(p)+K(p)$, with $L_{\mathcal D}(p)=\sum_i-\log p(y_i|x_i)$ — [Achille et al. 2021](https://arxiv.org/abs/1904.03292)
- **Structure function of a task:** $S_{\mathcal D}(t)=\min_{K(p)\le t}L_{\mathcal D}(p)$.
  - Random labels give the linear $S_{\mathcal D}(t)\approx N\log|\mathcal Y|-t$ (Example 3.5).
  - The Lagrangian family $C_\beta(\mathcal D)=\min_p L_{\mathcal D}(p)+\beta K(p)$ is the Legendre transform of $S_{\mathcal D}$.
  - $\beta=1$ corresponds to Kolmogorov minimal sufficiency (Def. 3.8).

  — [Achille et al. 2021](https://arxiv.org/abs/1904.03292)
- **Computable generalization (Def. 5.1) / "Information in the Weights" (Def. 2.1 in the companion paper):**
  - $C_\beta(\mathcal D;P,Q)=\mathbb E_{w\sim Q(w|\mathcal D)}[L_{\mathcal D}(p_w(y|x))]+\beta\,\mathrm{KL}(Q(w|\mathcal D)\|P(w))$, with an arbitrary "pre-distribution" $P$ and "post-distribution" $Q$.
  - At the minimizing $Q$, $\mathrm{KL}(Q\|P)$ is "the amount of Information in the Weights for the task D at level β".
  - $\beta=1$ formally coincides with the ELBO.
  - It reduces to Shannon mutual information or Fisher information for particular choices.
  - It is linked to generalization through PAC-Bayes and to invariance through "effective information" in activations.
  - The generalized structure function is computed on common datasets.

  — [Achille, Paolini, Soatto](https://arxiv.org/abs/1905.12213); [Achille et al. 2021](https://arxiv.org/abs/1904.03292)
- **Canonical references:**
  - Achille, Paolini, Mbeng, Soatto, "The information complexity of learning tasks, their structure and their distance", Information and Inference: A Journal of the IMA 10(1):51–72 (2021), DOI 10.1093/imaiai/iaaa033, arXiv:1904.03292 — [Crossref](https://api.crossref.org/works/10.1093/imaiai/iaaa033)
  - Achille, Paolini, Soatto, "Where is the Information in a Deep Neural Network?", arXiv:1905.12213 (2019) — [arXiv](https://arxiv.org/abs/1905.12213)
  - Follow-up cited by epiplexity: Achille & Soatto, "AI Agents as Universal Task Solvers", arXiv:2510.12066 (2025) — [arXiv](https://arxiv.org/abs/2510.12066)

**5.5 Other close relatives named by the epiplexity paper**
- **Surplus description length (SDL):** the summed online loss minus the data entropy or a baseline. Reference: Whitney, Song, Brandfonbrener, Altosaar, Cho, "Evaluating representations by the complexity of learning low-loss predictors", arXiv:2009.07368 (2020) — [arXiv](https://arxiv.org/abs/2009.07368); [epiplexity §7](https://arxiv.org/html/2601.03220v2)
- **Information transfer:** Zhang, Li, Dou, Wu, "Measuring Information Transfer in Neural Networks", arXiv:2009.07624 (2020). It sums a loss difference on held-out data and is "more analogous to the spirit of epiplexity" — [arXiv](https://arxiv.org/abs/2009.07624); [epiplexity §7](https://arxiv.org/html/2601.03220v2)
- **Teacher size:** Dziugaite & Roy, "The size of teachers as a measure of data complexity: PAC-Bayes excess risk bounds and scaling laws", AISTATS 2025 — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
- **Speed prior:** Schmidhuber, "The Speed Prior: A New Simplicity Measure Yielding Near-Optimal Computable Predictions", COLT 2002, LNCS 2375, pp. 216–228, DOI 10.1007/3-540-45435-7_15 — [Crossref](https://api.crossref.org/works/10.1007/3-540-45435-7_15)
- **Information bottleneck:** Tishby, Pereira, Bialek, arXiv physics/0004057 (2000) — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)

**5.6 Post-publication follow-ups and critiques of epiplexity (2026)**
- Su, Potapczynski, Qiu, Hughes, Wilson, "Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization", arXiv:2608.11746 (Aug 2026):
  - epiplexity is used as an online signal for domain reweighting, via scaling-law fits to loss curves;
  - a generator is trained with REINFORCE to maximize the learner's epiplexity gain;
  - "higher epiplexity predicts improved downstream performance".

  — [arXiv](https://arxiv.org/abs/2608.11746)
- H. Li, "A Controlled Counterexample to Strong Proxy-Based Explanations of OOD Performance", arXiv:2605.11554 (May 2026):
  - it builds a construction where "a formal structure quantity, its operational proxy, and the task-relevant structure ... separate";
  - the OOD ranking reverses the proxy ranking in 2 of 3 seeds;
  - conclusion: "a proxy for total learned structure can fail to track the task-relevant structure".

  — [arXiv](https://arxiv.org/abs/2605.11554)
- M. Noguer i Alonso, "Financial Epiplexity: A Theory of Learnable Market Structure under Bounded Computation", arXiv:2607.02695 (Jul 2026). It is a time-bounded MDL measure relative to a filtration, representation, model class and budget, and proves "equal entropy need not imply equal epiplexity" — [arXiv](https://arxiv.org/abs/2607.02695)

### Inferences
- **Mapping onto epiplexity:**
  - $H_{\mathcal V}(X)$ with $\mathcal V=\mathcal P_T$ is $H_T$ **without** the $|P|$ penalty and without charging training time.
  - $I_{\mathcal V}$ is a *difference* of such entropies, i.e. usable **task** information, not model structure.

  V-information therefore answers "how much does X help predict Y for this observer?", while epiplexity answers "how many bits of structure must this observer store to model X?". The two can disagree. A dataset with a simple but highly predictive rule has high $I_{\mathcal V}$ and low $S_T$. A dataset with complex regularities that do not reduce loss much at the observer's budget has the reverse.
- **Achille's IW is the closest ML-side precursor of $S_T$.** Both are the "model part" of an optimized two-part (bits-back/ELBO or program-length) code, and both vanish at $\beta=1$ on random labels, where the structure function is linear and there is no shared structure to absorb. Differences:
  1. IW is conditional (a supervised task), while $S_T$ is defined for unconditional $X$ as well as conditional $Y|X$.
  2. IW measures cost in KL to a pre-distribution, while $S_T$ measures it in program bits including the training procedure.
  3. IW has no explicit time bound; the observer bound comes implicitly through architecture and SGD.
  4. IW is presented for generalization and invariance, not for data selection.
- **Achille's task structure function and the VV structure function are the direct bridge** from AIT algorithmic statistics to epiplexity's MDL-with-model-size view. Achille et al.'s computable, generalized structure function on real datasets is an early empirical instance of "structure versus noise in datasets".
- **The requential-coding paper resolves a noted weakness.** The prequential estimate does not upper-bound $K(P_M)$ and does not guarantee runtime. The requential companion paper supplies the explicit code, so a literature review should present prequential as heuristic and requential as rigorous.

### Gaps
- Exact pages for Blier & Ollivier (NeurIPS 2018) and the venue of Voita & Titov (believed to be EMNLP 2020) were not verified; bibliographic lookups were rate-limited.
- Grünwald 2007's ISBN was not verified. The epiplexity paper gives "MIT press, 2007".
- Whether Achille, Paolini & Soatto 2019 ("Where is the information...") was later published at a peer-reviewed venue was not verified; only the arXiv version is confirmed.
- The epiplexity paper does not cite Achille et al. 2019/2021, Ethayarajh et al. 2022, or Bialek et al. 2001 by name. This gap in its related work was found by grepping the full text.

---

## Q6. What does each measure capture that epiplexity does not (and vice versa)? Is it observer-dependent? Does it vanish on pure noise and on trivially simple data?

### Takeaway
Epiplexity's defining combination has no exact predecessor:
- (i) it is observer-dependent through a compute bound on **both** learning and inference;
- (ii) it measures the **model/structure** part rather than the random part;
- (iii) it is distributional, i.e. dataset-level;
- (iv) it vanishes on noise and on trivial data;
- (v) it is estimable from training runs.

Each precursor misses at least one of these:
- AIT measures miss (i) and (v);
- dynamical-systems measures miss (i);
- pseudoentropies and V-entropy miss (ii);
- V-information and PVI are task-relative and lack a model-size term;
- Achille's IW lacks an explicit compute bound.

### Cited Findings
- **Epiplexity itself.** The paper states that information is "observer dependent: the same object may appear random or structured depending on the computational resources of the observer". It gives $S_T(U_n)\le c_2$ for noise and $S_T=O(1)$ for simple periodic mixtures — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **Epiplexity is task-agnostic.** "epiplexity is a measure of information, not a guarantee of OOD generalization to specific tasks ... agnostic to whether these structures are relevant to a specific downstream task". A 2026 counterexample paper shows that total-structure proxies can mis-rank datasets for a specific OOD task — [arXiv HTML](https://arxiv.org/html/2601.03220v2); [Li 2026](https://arxiv.org/abs/2605.11554)
- **Sophistication and relatives.** They vanish on random strings (random strings are shallow and effectively simple, and facticity is 0) — [Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf); [Ay et al.](https://arxiv.org/abs/0810.5663); [Adriaans](https://arxiv.org/abs/1203.2245). They are not observer-bounded, and naive time-bounding collapses sophistication to $O(1)$ — [epiplexity App. A.6](https://arxiv.org/html/2601.03220v2)
- **Structure function / minimal sufficient statistic.** The noise part is $h_x(\alpha_0)$ and the meaningful part is $\alpha_0$ — [Vereshchagin & Vitányi](https://arxiv.org/abs/cs/0204037)
- **Excess entropy, predictive information, statistical complexity.**
  - Excess entropy of a period-$p$ process is $\log_2 p$ — [Crutchfield & Feldman](https://arxiv.org/abs/cond-mat/0102181)
  - Extensive (noise) entropy cancels in predictive information — [Bialek et al.](https://arxiv.org/abs/physics/0007070)
  - $E\le C_\mu$ — [Shalizi & Crutchfield](https://arxiv.org/abs/cond-mat/9907176)
- **Pseudoentropy.** It is observer-relative (a test class $\mathcal C$) but measures randomness, so it is maximal on noise — [Barak, Shaltiel, Wigderson](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/BSW03/BSW03.pdf); the epiplexity paper states this explicitly — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- **V-information.** It is observer-relative through $\mathcal V$ and is zero when $X\perp Y$ (Prop. 2) — [Xu et al.](https://arxiv.org/abs/2002.10689)

### Inferences
- **What precursors capture that epiplexity lacks:**
  - Logical depth and computational depth capture **how long** structure takes to produce. Epiplexity is silent on production time beyond the budget $T$.
  - Statistical complexity separates *stored memory* from *transmitted correlation* ($C_\mu$ versus $E$).
  - Predictive information classifies **growth regimes** (log versus power law) intrinsically.
  - V-information and PVI give **task-relative** and **pointwise** (per-example) values; epiplexity is dataset-level and task-agnostic.
  - Structure functions give the **whole trade-off curve** $h_x(\alpha)$, not a single point. Epiplexity's analogue would be the Pareto frontier of MDL versus compute that the paper estimates, but its model-size versus loss curve at fixed $T$ is not a named object.
- **What epiplexity captures that precursors lack:**
  - PRG and cipher outputs count as random, which unbounded structure measures cannot do.
  - Structural information can be **created** by deterministic computation, e.g. cellular automata and emergent objects, and can depend on ordering.
  - Estimation cost is charged for learning, not only inference.
- **"Structure is intermediate" test.** The classical structure measures all satisfy "≈0 on noise and ≈0 on trivially simple data":
  - AIT: sophistication, coarse and naive sophistication, the minimal-sufficient-statistic complexity, effective complexity, logical and computational depth, facticity (small but >0 on compressible strings, per Adriaans Lemma 11);
  - dynamical: $E$, $I_{\rm pred}$, $C_\mu$.

  Entropy-like quantities fail it: $K$, $K^t$, $Kt$, the pseudoentropies, $H_{\mathcal V}$ and $H_T$ are maximal on noise. V-information vanishes on independent (random) labels but not in the unconditional setting.

### Gaps
- No formal paper was found that proves the behaviour of each measure on PRG outputs in a unified way. The PRG statements for AIT and dynamical measures above are my inferences from their definitions, apart from $K^t$, $Kt$ and Shannon entropy, which the epiplexity paper states.

---

## Q7. Which admit practical estimators with neural networks at small scale (MLP, single GPU)? Which are uncomputable or intractable?

### Takeaway
**Practically estimable with small neural networks:**
- epiplexity (prequential: the area under the loss curve; requential needs a teacher/student pipeline);
- V-information and PVI (two models, with and without input);
- prequential/two-part MDL code lengths;
- Achille's information in the weights (Gaussian/Fisher post-distributions);
- SDL and information transfer.

**Estimable only for small discrete processes (block entropies, ε-machine reconstruction):** excess entropy, predictive information, statistical complexity.

**Uncomputable:** $K$, the structure functions, sophistication (all variants), logical depth, computational depth, effective complexity, facticity. These are usable only through compressor proxies (e.g. gzip).

**Well-defined but intractable to estimate:** the cryptographic pseudoentropies, since they quantify over all efficient tests. $Kt$ is computable but exponential-time.

### Cited Findings
- Epiplexity's prequential estimate uses only the loss curve, which is "particularly convenient when one already has access to the loss curve from an existing training run". The requential estimate needs teacher checkpoints and a student trained on teacher samples. Hyperparameters are tuned with μP — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- V-information "can be reliably estimated from data even in high dimensions with PAC-style guarantees", by optimizing over $\mathcal V$ with gradient descent — [Xu et al.](https://arxiv.org/abs/2002.10689)
- PVI is computed from two fine-tuned models (with and without input) on held-out instances — [Ethayarajh et al.](https://proceedings.mlr.press/v162/ethayarajh22a.html)
- Achille et al. state that the task structure function "cannot be computed in general" but introduce "a generalized version that is computable" and plot it on common datasets. Their IW reduces to Fisher information for Gaussian post-distributions — [Achille et al. 2021](https://arxiv.org/abs/1904.03292); [Achille et al. 2019](https://arxiv.org/abs/1905.12213)
- Prequential code lengths for deep nets are practical (Blier & Ollivier) — [arXiv](https://arxiv.org/abs/1802.07044)
- The structure functions $h_x,\lambda_x$ are only upper semicomputable, and $\beta_x$ is neither upper nor lower semicomputable — [Vereshchagin & Vitányi](https://arxiv.org/abs/cs/0204037)
- No proofs can certify high $K$, and hence high sophistication, beyond a constant $L$ (Chaitin), which is "often not more than a few thousand" bits — [arXiv HTML](https://arxiv.org/html/2601.03220v2)
- The faithful index needed for facticity "is not recursive" — [Adriaans](https://arxiv.org/abs/1203.2245)
- Kt is computable — [Scholarpedia](http://www.scholarpedia.org/article/Universal_search)
- Excess entropy has finite-$L$ block-entropy estimators — [Crutchfield & Feldman](https://arxiv.org/abs/cond-mat/0102181)
- Apparent complexity is approximated via gzip on coarse-grained states — [Aaronson et al.](https://arxiv.org/abs/1405.6903)

### Inferences
- **A single-GPU MLP comparison study** could estimate, on the same datasets:
  1. prequential epiplexity (area under the online loss curve above the final loss, swept over model size and data under a FLOP budget);
  2. V-information and PVI for supervised variants;
  3. Achille-style IW with a diagonal-Gaussian post-distribution;
  4. a total prequential code length as a Kolmogorov-complexity proxy;
  5. block-entropy excess entropy for small-alphabet synthetic sequences such as elementary cellular automata and Markov chains, where ground truth is computable.

  PRG outputs, like the paper's CSPRNG examples, are a crucial control: epiplexity and V-information should be ≈0 there, while block-entropy excess entropy (at feasible $L$) will also be ≈0 because the observer is effectively bounded by $L$.
- The AIT measures can be approximated only through compressor surrogates. Such surrogates themselves define a bounded observer, which the epiplexity paper argues is the right framing anyway.

### Gaps
- No published side-by-side empirical comparison of epiplexity with V-information, IW or excess entropy on the same datasets was found.

---

## Q8. Compact comparison table

### Takeaway
Only epiplexity and Achille-style IW combine "structure (model) part" with "vanishes on noise" and a neural estimator. Only epiplexity adds an explicit compute bound on both training and inference.

### Cited Findings
Cell entries follow the sources in Q2–Q5. Entries marked (I) are my inferences from the definitions rather than statements in the source. "Random" = i.i.d. uniform bits or labels; "constant" = $0^n$ or a constant label.

| Measure | Bounded observer? | ≈0 on random? | ≈0 on constant? | Practical estimator? | Reference |
|---|---|---|---|---|---|
| Epiplexity $S_T$ | Yes (time $T$ for train + infer) | Yes ($\le c$) | Yes ($O(1)$) | Yes (prequential / requential) | [Finzi et al. 2026](https://arxiv.org/abs/2601.03220) |
| Time-bounded entropy $H_T$ | Yes | No (≈ n) | Yes | Yes (final loss × size) | [Finzi et al. 2026](https://arxiv.org/abs/2601.03220) |
| Kolmogorov $K$ | No | No (≈ n) | Yes, up to $O(\log n)$ (I) | No (uncomputable; compressor proxy) | [Kolmogorov 1965/68](https://api.crossref.org/works/10.1080/00207166808803030) |
| $K^t$, Levin $Kt$ | Yes (time) | No (≈ n) | Yes, up to $O(\log n)$ (I) | Computable but exponential | [Levin 1973](https://www.mathnet.ru/eng/ppi914); [Allender et al. 2011](https://api.crossref.org/works/10.1016/j.jcss.2010.06.004) |
| Structure function / minimal sufficient statistic $\alpha_0$ | No | Yes | Yes (I) | No (only upper semicomputable) | [Vereshchagin & Vitányi 2004](https://arxiv.org/abs/cs/0204037) |
| Sophistication $\mathrm{soph}_c$ (Koppel) | No (naive $t$-version collapses) | Yes | Yes (I) | No | [Koppel 1987](https://content.wolfram.com/sites/13/2018/02/01-6-4.pdf); [Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf) |
| Naive / coarse sophistication | No | Yes | Yes (I) | No | [Mota et al. 2013](https://www.scottaaronson.com/papers/DCFS-Final.pdf); [Antunes & Fortnow](https://lance.fortnow.com/papers/files/soph.pdf) |
| Logical depth | No (measures time) | Yes (shallow) | Yes (I) | No | [Bennett 1988](https://academic.oup.com/book/54493/chapter-abstract/422572845) |
| Computational depth $K^t-K$ | Partly (bounded minus unbounded) | Yes (random sets shallow) | Yes (I) | No (involves $K$) | [AFMV 2006](https://lance.fortnow.com/papers/files/depth-j.pdf) |
| Effective complexity | No | Yes (Thm 10) | Yes (I) | No | [Gell-Mann & Lloyd 1996; Ay et al. 2010](https://arxiv.org/abs/0810.5663) |
| Facticity | No ($t$-facticity is a hybrid) | Yes (Lemma 10) | Small but >0 (Lemma 11) | No (faithful index not recursive) | [Adriaans 2009/2012](https://arxiv.org/abs/1203.2245) |
| Statistical complexity $C_\mu$ | No | Yes, for i.i.d. (one causal state) (I) | Yes (I) | Discrete processes only | [Crutchfield & Young 1989](https://api.crossref.org/works/10.1103/PhysRevLett.63.105); [Shalizi & Crutchfield 2001](https://arxiv.org/abs/cond-mat/9907176) |
| Excess entropy $E$ | No | Yes, for i.i.d. (I) | Yes ($\log p$ for period $p$) | Block-entropy estimates | [Crutchfield & Feldman 2003](https://arxiv.org/abs/cond-mat/0102181) |
| Predictive information $I_{\rm pred}$ | No | Yes (extensive part cancels) | Yes (I) | Block entropies / learning curves | [Bialek et al. 2001](https://arxiv.org/abs/physics/0007070) |
| HILL / metric / Yao pseudoentropy | Yes (test class) | No (maximal) | Yes (I) | No (∀ tests) | [HILL 1999](https://api.crossref.org/works/10.1137/S0097539793244708); [BSW 2003](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/BSW03/BSW03.pdf) |
| Next-block pseudoentropy | Yes | No (maximal) | Yes (I) | No | [HRV 2013](https://api.crossref.org/works/10.1137/100814421); [VZ 2012](https://api.crossref.org/works/10.1145/2213977.2214051) |
| V-entropy $H_{\mathcal V}$ | Yes (family $\mathcal V$, inference only) | No (maximal) | Yes | Yes | [Xu et al. 2020](https://arxiv.org/abs/2002.10689) |
| V-information $I_{\mathcal V}(X\to Y)$ | Yes | Yes if $X\perp Y$ | Yes if $Y$ constant (I) | Yes (PAC) | [Xu et al. 2020](https://arxiv.org/abs/2002.10689) |
| PVI | Yes | ≈0 in expectation (I) | Yes (I) | Yes (two fine-tunes) | [Ethayarajh et al. 2022](https://proceedings.mlr.press/v162/ethayarajh22a.html) |
| Two-part MDL model cost $L(H^\star)$ | Partly (model class $\mathcal H$) | Yes (I) | Yes (I) | Yes for simple classes; NML intractable for DNNs | [Rissanen 1978](https://api.crossref.org/works/10.1016/0005-1098(78)90005-5); [Grünwald 2007 via App. H](https://arxiv.org/html/2601.03220v2) |
| Prequential code length (total) | Partly (learner) | No (≈ n) | Yes (I) | Yes | [Dawid 1984](https://api.crossref.org/works/10.2307/2981683); [Blier & Ollivier 2018](https://arxiv.org/abs/1802.07044) |
| Information in the weights $\mathrm{KL}(Q\|P)$ / $C_\beta$ | Partly (model class + SGD, no time bound) | Yes at β=1 on random labels (linear structure function) (I) | Yes (I) | Yes (Gaussian / Fisher) | [Achille et al. 2019](https://arxiv.org/abs/1905.12213); [2021](https://arxiv.org/abs/1904.03292) |
| SDL / information transfer | Partly (learner) | Yes (I) | Yes (I) | Yes | [Whitney et al. 2020](https://arxiv.org/abs/2009.07368); [Zhang et al. 2020](https://arxiv.org/abs/2009.07624) |
| Apparent complexity | Partly (coarse-graining choice) | Yes (smoothing) | Yes (I) | Yes (gzip proxy) | [Aaronson et al. 2014](https://arxiv.org/abs/1405.6903) |

### Inferences
- The table's "≈0 on random" column separates **randomness measures** (K, K^t, pseudoentropies, V-entropy, $H_T$, total prequential length) from **structure measures** (everything else). Epiplexity is the only structure measure with a full compute bound. Achille's IW is the only other structure measure with a neural estimator that is explicitly a model-description cost.

### Gaps
- Behaviour on constant data for several measures is inferred, not stated by the sources. This is flagged (I) in the table.

---

## Q9. Bibliographic data for the BibTeX file (verified where marked)

### Takeaway
DOIs and pages below were verified via Crossref or the arXiv API unless marked "unverified". One known error in the epiplexity paper's bibliography (the "Sophistication revisited" entry) should not be propagated.

### Cited Findings
- Finzi, M.; Qiu, S.; Jiang, Y.; Izmailov, P.; Kolter, J. Z.; Wilson, A. G. "From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence." arXiv:2601.03220 [cs.LG], 2026 (v1 6 Jan, v2 16 Mar) — [arXiv](https://arxiv.org/abs/2601.03220)
- Qiu, S.; Finzi, M.; Zheng, Y.; Zhang, K.; Wilson, A. G. "Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data." arXiv:2607.11883, 2026 — [arXiv](https://arxiv.org/abs/2607.11883)
- Kolmogorov, A. N. "Three approaches to the quantitative definition of information." Problems Inform. Transmission 1(1):1–7, 1965; reprinted Int. J. Comput. Math. 2(1–4):157–168, 1968, doi:10.1080/00207166808803030 — [Crossref](https://api.crossref.org/works/10.1080/00207166808803030)
- Li, M.; Vitányi, P. An Introduction to Kolmogorov Complexity and Its Applications, 4th ed. Springer, 2019, doi:10.1007/978-3-030-11298-1 — [Springer](https://link.springer.com/book/10.1007/978-3-030-11298-1)
- Levin, L. A. "Universal sequential search problems." Problems Inform. Transmission 9(3):265–266, 1973 — [Math-Net.ru](https://www.mathnet.ru/eng/ppi914)
- Levin, L. A. "Randomness conservation inequalities; information and independence in mathematical theories." Information and Control 61(1):15–37, 1984, doi:10.1016/S0019-9958(84)80060-1 — [Crossref](https://api.crossref.org/works/10.1016/S0019-9958(84)80060-1)
- Allender, E.; Koucký, M.; Ronneburger, D.; Roy, S. JCSS 77(1):14–40, 2011, doi:10.1016/j.jcss.2010.06.004 — [Crossref](https://api.crossref.org/works/10.1016/j.jcss.2010.06.004)
- Vereshchagin, N. K.; Vitányi, P. M. B. "Kolmogorov's structure functions and model selection." IEEE Trans. Inf. Theory 50(12):3265–3290, 2004, doi:10.1109/TIT.2004.838346; arXiv:cs/0204037 — [Crossref](https://api.crossref.org/works/10.1109/TIT.2004.838346)
- Gács, P.; Tromp, J. T.; Vitányi, P. M. B. "Algorithmic statistics." IEEE Trans. Inf. Theory 47(6):2443–2463, 2001, doi:10.1109/18.945257; arXiv:math/0006233 — [Crossref](https://api.crossref.org/works/10.1109/18.945257)
- Koppel, M. "Complexity, depth, and sophistication." Complex Systems 1(6):1087–1091, 1987 — [PDF](https://content.wolfram.com/sites/13/2018/02/01-6-4.pdf)
- Koppel, M. "Structure." In R. Herken (ed.), The Universal Turing Machine: A Half-Century Survey, Oxford Univ. Press, 1988, pp. 435–452 — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
- Koppel, M.; Atlan, H. "An almost machine-independent theory of program-length complexity, sophistication, and induction." Information Sciences 56(1–3):23–33, 1991, doi:10.1016/0020-0255(91)90021-L — [Crossref](https://api.crossref.org/works/10.1016/0020-0255(91)90021-L)
- Antunes, L.; Fortnow, L. "Sophistication revisited." ICALP 2003, LNCS 2719, pp. 267–277, doi:10.1007/3-540-45061-0_23; Theory of Computing Systems 45(1):150–161, 2009, doi:10.1007/s00224-007-9095-5 — [Crossref](https://api.crossref.org/works/10.1007/s00224-007-9095-5)
- Antunes, L.; Fortnow, L.; van Melkebeek, D.; Vinodchandran, N. V. "Computational depth: Concept and applications." Theoretical Computer Science 354(3):391–404, 2006, doi:10.1016/j.tcs.2005.11.033 — [Crossref](https://api.crossref.org/works/10.1016/j.tcs.2005.11.033)
- Mota, F.; Aaronson, S.; Antunes, L.; Souto, A. "Sophistication as randomness deficiency." DCFS 2013, LNCS 8031, pp. 172–181, doi:10.1007/978-3-642-39310-5_17 — [Springer](https://link.springer.com/chapter/10.1007/978-3-642-39310-5_17)
- Antunes, L.; Bauwens, B.; Souto, A.; Teixeira, A. "Sophistication vs logical depth." Theory of Computing Systems 60(2):280–298, 2017, doi:10.1007/s00224-016-9672-6; arXiv:1304.8046 — [Crossref](https://api.crossref.org/works/10.1007/s00224-016-9672-6)
- Bennett, C. H. "Logical depth and physical complexity." In R. Herken (ed.), The Universal Turing Machine: A Half-Century Survey, Oxford Univ. Press, 1988, pp. 227–257 — [Oxford Academic](https://academic.oup.com/book/54493/chapter-abstract/422572845)
- Gell-Mann, M.; Lloyd, S. "Information measures, effective complexity, and total information." Complexity 2(1):44–52, 1996, doi:10.1002/(SICI)1099-0526(199609/10)2:1<44::AID-CPLX10>3.0.CO;2-X (verified on Crossref) — [Ay et al. citing it](https://arxiv.org/abs/0810.5663)
- Ay, N.; Müller, M.; Szkoła, A. "Effective complexity and its relation to logical depth." IEEE Trans. Inf. Theory 56(9):4593–4607, 2010, doi:10.1109/TIT.2010.2053892; arXiv:0810.5663 — [Crossref](https://api.crossref.org/works/10.1109/TIT.2010.2053892)
- Adriaans, P. "Between order and chaos: The quest for meaningful information." Theory of Computing Systems 45(4):650–674, 2009, doi:10.1007/s00224-009-9173-y — [Springer](https://link.springer.com/article/10.1007/s00224-009-9173-y)
- Adriaans, P. "Facticity as the amount of self-descriptive information in a data set." arXiv:1203.2245, 2012 — [arXiv](https://arxiv.org/abs/1203.2245)
- Crutchfield, J. P.; Young, K. "Inferring statistical complexity." Phys. Rev. Lett. 63(2):105–108, 1989, doi:10.1103/PhysRevLett.63.105 — [Crossref](https://api.crossref.org/works/10.1103/PhysRevLett.63.105)
- Shalizi, C. R.; Crutchfield, J. P. "Computational mechanics: Pattern and prediction, structure and simplicity." J. Stat. Phys. 104(3–4):817–879, 2001, doi:10.1023/A:1010388907793 — [Crossref](https://api.crossref.org/works/10.1023/A:1010388907793)
- Crutchfield, J. P.; Feldman, D. P. "Regularities unseen, randomness observed: Levels of entropy convergence." Chaos 13(1):25–54, 2003, doi:10.1063/1.1530990 — [Crossref](https://api.crossref.org/works/10.1063/1.1530990)
- Grassberger, P. "Toward a quantitative theory of self-generated complexity." Int. J. Theor. Phys. 25(9):907–938, 1986, doi:10.1007/BF00668821 — [Crossref](https://api.crossref.org/works/10.1007/BF00668821)
- Bialek, W.; Nemenman, I.; Tishby, N. "Predictability, complexity, and learning." Neural Computation 13(11):2409–2463, 2001, doi:10.1162/089976601753195969 — [Crossref](https://api.crossref.org/works/10.1162/089976601753195969)
- Håstad, J.; Impagliazzo, R.; Levin, L. A.; Luby, M. "A pseudorandom generator from any one-way function." SIAM J. Comput. 28(4):1364–1396, 1999, doi:10.1137/S0097539793244708 — [Crossref](https://api.crossref.org/works/10.1137/S0097539793244708)
- Barak, B.; Shaltiel, R.; Wigderson, A. "Computational analogues of entropy." RANDOM-APPROX 2003, LNCS 2764, pp. 200–215, doi:10.1007/978-3-540-45198-3_18 — [Crossref](https://api.crossref.org/works/10.1007/978-3-540-45198-3_18)
- Haitner, I.; Reingold, O.; Vadhan, S. "Efficiency improvements in constructing pseudorandom generators from one-way functions." SIAM J. Comput. 42(3):1405–1430, 2013 (STOC 2010), doi:10.1137/100814421 — [Crossref](https://api.crossref.org/works/10.1137/100814421)
- Vadhan, S.; Zheng, C. J. "Characterizing pseudoentropy and simplifying pseudorandom generator constructions." STOC 2012, pp. 817–836, doi:10.1145/2213977.2214051 — [Crossref](https://api.crossref.org/works/10.1145/2213977.2214051)
- Agrawal, R.; Chen, Y.-H.; Horel, T.; Vadhan, S. "Unifying computational entropies via Kullback–Leibler divergence." CRYPTO 2019, doi:10.1007/978-3-030-26951-7_28; arXiv:1902.11202 — [Springer](https://link.springer.com/chapter/10.1007/978-3-030-26951-7_28)
- Hsiao, C.-Y.; Lu, C.-J.; Reyzin, L. "Conditional computational entropy, or toward separating pseudoentropy from compressibility." EUROCRYPT 2007, LNCS 4515, pp. 169–186, doi:10.1007/978-3-540-72540-4_10 — [Crossref](https://api.crossref.org/works/10.1007/978-3-540-72540-4_10)
- Yao, A. C. "Theory and applications of trapdoor functions." FOCS 1982, pp. 80–91 (DOI unverified) — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
- Xu, Y.; Zhao, S.; Song, J.; Stewart, R.; Ermon, S. "A theory of usable information under computational constraints." ICLR 2020; arXiv:2002.10689 — [arXiv](https://arxiv.org/abs/2002.10689)
- Ethayarajh, K.; Choi, Y.; Swayamdipta, S. "Understanding dataset difficulty with V-usable information." ICML 2022, PMLR 162:5988–6008; arXiv:2110.08420 — [PMLR](https://proceedings.mlr.press/v162/ethayarajh22a.html)
- Rissanen, J. "Modeling by shortest data description." Automatica 14(5):465–471, 1978, doi:10.1016/0005-1098(78)90005-5 — [Crossref](https://api.crossref.org/works/10.1016/0005-1098(78)90005-5)
- Dawid, A. P. "Present position and potential developments: Some personal views: Statistical theory: The prequential approach." JRSS Series A 147(2):278–, 1984, doi:10.2307/2981683 — [Crossref](https://api.crossref.org/works/10.2307/2981683)
- Grünwald, P. D. The Minimum Description Length Principle. MIT Press, 2007 (ISBN unverified) — [epiplexity bibliography](https://arxiv.org/html/2601.03220v2)
- Blier, L.; Ollivier, Y. "The description length of deep learning models." NeurIPS 2018; arXiv:1802.07044 — [arXiv](https://arxiv.org/abs/1802.07044)
- Voita, E.; Titov, I. "Information-theoretic probing with minimum description length." arXiv:2003.12298 (venue unverified; believed EMNLP 2020) — [arXiv](https://arxiv.org/abs/2003.12298)
- Achille, A.; Paolini, G.; Soatto, S. "Where is the information in a deep neural network?" arXiv:1905.12213, 2019 — [arXiv](https://arxiv.org/abs/1905.12213)
- Achille, A.; Paolini, G.; Mbeng, G.; Soatto, S. "The information complexity of learning tasks, their structure and their distance." Information and Inference: A Journal of the IMA 10(1):51–72, 2021, doi:10.1093/imaiai/iaaa033; arXiv:1904.03292 — [Crossref](https://api.crossref.org/works/10.1093/imaiai/iaaa033)
- Achille, A.; Soatto, S. "AI agents as universal task solvers." arXiv:2510.12066, 2025 — [arXiv](https://arxiv.org/abs/2510.12066)
- Whitney, W. F.; Song, M. J.; Brandfonbrener, D.; Altosaar, J.; Cho, K. arXiv:2009.07368, 2020 — [arXiv](https://arxiv.org/abs/2009.07368)
- Zhang, X.; Li, X.; Dou, D.; Wu, J. "Measuring information transfer in neural networks." arXiv:2009.07624, 2020 — [arXiv](https://arxiv.org/abs/2009.07624)
- Schmidhuber, J. "The speed prior." COLT 2002, LNCS 2375, pp. 216–228, doi:10.1007/3-540-45435-7_15 — [Crossref](https://api.crossref.org/works/10.1007/3-540-45435-7_15)
- Aaronson, S.; Carroll, S. M.; Ouellette, L. "Quantifying the rise and fall of complexity in closed systems: The coffee automaton." arXiv:1405.6903, 2014 — [arXiv](https://arxiv.org/abs/1405.6903)
- Finzi, M. et al. "Compute-optimal LLMs provably generalize better with scale." ICLR 2025; arXiv:2504.15208 — [arXiv](https://arxiv.org/abs/2504.15208)
- Su, E.; Potapczynski, A.; Qiu, S.; Hughes, E.; Wilson, A. G. arXiv:2608.11746, 2026 — [arXiv](https://arxiv.org/abs/2608.11746)
- Li, H. arXiv:2605.11554, 2026 — [arXiv](https://arxiv.org/abs/2605.11554)
- Noguer i Alonso, M. arXiv:2607.02695, 2026 — [arXiv](https://arxiv.org/abs/2607.02695)

### Inferences
- For BibTeX, cite "Sophistication Revisited" as the 2009 TOCS journal version by Antunes & Fortnow, with the ICALP 2003 version as an optional note. Cite "Computational depth" separately as AFMV 2006.

### Gaps
- Not verified: the Yao 1982 DOI, the Grünwald 2007 ISBN, the Blier & Ollivier and Voita & Titov page numbers, the Dawid 1984 end page (the epiplexity paper says 278–290), and the Dziugaite & Roy 2025 PMLR volume and pages. Crossref, DBLP and Semantic Scholar lookups were rate-limited or failed in this session.
