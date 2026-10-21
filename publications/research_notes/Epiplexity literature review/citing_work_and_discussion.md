# Work citing, extending, critiquing or reimplementing epiplexity (Jan–Sep 2026), excluding the founding paper and the two eysu35 follow-ups

Scope note: the founding paper (arXiv 2601.03220, code github.com/shikaiqiu/epiplexity) and the eysu35 follow-ups (EpiSelect / EpiGen, arXiv 2608.11746 by Su, Potapczynski, Qiu, Hughes, Wilson) are covered by another researcher. They appear here only where needed for context. Relevance tags (HIGH / MEDIUM / LOW) are judged against the project goal: **estimate epiplexity with MLPs on small datasets on one Colab GPU**. Survey date: 2026-09-23.

Method: I pulled the Semantic Scholar citations endpoint for arXiv:2601.03220, which returned 37 citing records. I downloaded the arXiv HTML full text of every citing paper and extracted each sentence that mentions "epiplexity" or "Finzi". I ran an arXiv API title/abstract search for "epiplexity", which returned 8 hits. I ran GitHub repository searches for "epiplexity" (by name/description: 26 repos; by README text: 87 repos). I also searched the web, Hacker News (via the Algolia API), YouTube metadata, and author homepages and CVs.

---

## Q1. Which papers cite 2601.03220, and what does each do with epiplexity?

### Takeaway
About 40 papers cite the founding paper by late September 2026: 37 in Semantic Scholar, plus at least 2 more that the arXiv search found. Most cite it only in passing. About a dozen do real work with it. They fall into three groups:
- **New or alternative estimators:** requential coding (same group); a closed-form reservoir/ridge estimator (Zhang & Levin); a thermodynamic/WBIC route (Ohzeki). A concurrent quantity, Excess Description Length, is prequential-coding-based and independent.
- **Practical prequential measurements:** self-play curricula (Liu et al.; Dineen et al.) and SSL checkpoint selection (Majchrowska & Teare).
- **A formal critique:** Hongmin Li's counterexample.

For a small-MLP / Colab project, the most useful are:
1. Zhang & Levin's closed-form estimator.
2. The small-data fixes to the prequential estimator in Liu et al.
3. Excess Description Length (Donoway et al.), a finite-data prequential measure with toy models.
4. Requential coding, as the tighter but costlier estimator.
5. Li's counterexample, as a caution on interpreting results.

### Cited Findings

**A. Papers that propose estimators or closely related information measures**

- **HIGH – "Intelligence from Learnable Novelty"** (Yanbo Zhang, Michael Levin). arXiv 2607.18433, 2026-07-20.
  - Their "learnable novelty" is explicitly epiplexity: "the structure a bounded mind genuinely carries away from the data, is the epiplexity recently named by Finzi et al. (2026), and it is what we mean by learnable novelty." Section 3 is titled "A Closed-Form Estimator of Epiplexity."
  - They argue the original estimator "amounts to a full training run for every system scored … computationally expensive but also opaque to gradients."
  - Their replacement: a fixed random reservoir φ with a ridge-regression linear readout W. The model-part description length is C_spec(W) = α·log2 det(I + η·W·Wᵀ), with α = 1/2 and η = 1 (η = 30 for MNIST). Ridge approximates the MDL minimisation.
  - The estimator is cheap, deterministic and differentiable. It ranks Rule 110 highest among the elementary cellular automata (ECAs). Used as an objective, it drives an NCA to solitons and organises an MNIST encoder unsupervised. As an intrinsic RL reward, it improves over the task-reward baseline in 9 of 10 environments.
  - Code: github.com/Zhangyanbo/learnable-novelty.
  - Sources: [arXiv HTML](https://arxiv.org/html/2607.18433), [GitHub](https://github.com/Zhangyanbo/learnable-novelty)
- **HIGH – "Requential Coding: Pushing the Limits of Model Compression with Self-Generated Training Data"** (Shikai Qiu, Marc Finzi, Yujia Zheng, Kun Zhang, Andrew Gordon Wilson). arXiv 2607.11883, 2026-07-13. Same group; a methods follow-up.
  - A teacher trained on real data selects samples from the student's own distribution. The code length is the cumulative teacher–student KL, "independent of parameter count and data entropy, and often orders of magnitude shorter than the prequential counterpart."
  - They say it "isolates the learnable information in a dataset from its unpredictable, random content." Figure 9 ranks datasets by learnable structure; text holds more learnable structure than image data.
  - They call the earlier prequential extrapolation "the non-rigorous prequential heuristic."
  - Code: JAX/Flax GPT-2-style transformer, github.com/shikaiqiu/requential-coding.
  - Sources: [arXiv abs](https://arxiv.org/abs/2607.11883), [arXiv HTML](https://arxiv.org/html/2607.11883), [GitHub](https://github.com/shikaiqiu/requential-coding)
- **HIGH – Excess Description Length (EDL)** (Elizabeth Donoway, Hailey Joren, Fabien Roger, Jan Leike). Concurrent and independent.
  - Extended version: arXiv 2601.04728, 2026-01-08, "Excess Description Length of Learning Generalizable Predictors." Short version: ISIT 2026 (DOI 10.1109/ISIT62367.2026.11654044), which cites epiplexity per Semantic Scholar. The arXiv version's text does not mention epiplexity.
  - EDL is defined via prequential coding. It is "the gap between the bits required to encode training labels sequentially using an evolving model (trained online) and the residual encoding cost under the final trained model."
  - The ISIT abstract calls it "compute-indexed". It "exclud[es] memorization effects on the training set" by comparing to a predictor at the final model's population loss.
  - Stated properties: non-negativity, additivity, monotonicity in compute, and a finite-data saturation bound.
  - Its toy models show "why random labels yield EDL near zero."
  - Essentially the same prequential quantity as epiplexity's estimator, but framed for supervised fine-tuning on finite data. Directly relevant to small-dataset MLP estimation.
  - Sources: [arXiv abs](https://arxiv.org/abs/2601.04728), [Semantic Scholar citation record](https://api.semanticscholar.org/graph/v1/paper/arXiv:2601.03220/citations?fields=title,authors,year,externalIds,abstract&limit=1000)
- **MEDIUM – "Non-Equilibrium Model Selection via Finite-Time Thermodynamics"** (Masayuki Ohzeki). arXiv 2606.16399, 2026-06-15.
  - Gives "a computable thermodynamic contribution to time-bounded MDL." It "identifies the finite-time singular complexity relevant to the structural information measured by epiplexity."
  - Method: extends Watanabe's WBIC / real log canonical threshold to finite-time SGLD ensembles. The author notes the free energy "is not epiplexity itself."
  - Theoretical; a possible Bayesian/Langevin alternative estimator of the model-complexity term.
  - Source: [arXiv HTML](https://arxiv.org/html/2606.16399)

**B. Papers that actually measure epiplexity (prequential-style) in experiments**

- **MEDIUM-HIGH – "Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain"** (Wei Liu, Siya Qi, Yali Du, Yulan He; King's College London). arXiv 2603.02218, v1 2026-02-10, v2 2026-05-15. The GitHub README calls it a "Position:" paper.
  - Adopts epiplexity "as a measurement tool that instantiates an MDL objective under explicit parameter and inference-time budgets." Epiplexity is the prequential code length at the MDL-optimal checkpoint. It frames a "Goldilocks Zone" for self-play data.
  - **Small-data modifications** (Appendix B), stated because "the dataset is relatively small":
    - Re-compute each example's loss after the observer is fully trained, instead of using the final-batch loss.
    - Use per-token averages.
    - For multi-epoch training, define the prequential length as "the difference between the training loss in the first epoch and the loss of the model trained for multiple epochs, evaluated on all data points."
  - All runs reach the MDL optimum at an intermediate point in training.
  - Code: github.com/thinkwee/SelfPlay_SelfEvo_Gap (`epiplexity_mdl.py`).
  - Sources: [arXiv HTML](https://arxiv.org/html/2603.02218), [GitHub](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap)
- **MEDIUM – "Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution"** (Jacob Dineen, Aswin RRV, Zhikun Xu, Ben Zhou; Arizona State University). arXiv 2604.03472, v4 2026-07-22. COLM 2026 according to the official repo.
  - Uses epiplexity as a "functional-level" diversity metric for generated math questions, reported in bits per token.
  - Protocol: fine-tune a fresh LoRA observer (r = 16, lr 1e-4, up to 20 epochs, early stopping, 10% validation split). Epiplexity = (L_online − L_train)/ln 2. The MDL criterion picks the epoch minimising epiplexity/N_train + L_val/(ln 2 · N_val).
  - Finding: epiplexity is higher under vocabulary dropout, and it "lock[s] in" early when the baseline curriculum collapses.
  - Sources: [arXiv HTML](https://arxiv.org/html/2604.03472), [GitHub ARC-ASU/vocab_dropout](https://github.com/ARC-ASU/vocab_dropout)
- **MEDIUM – "Evaluating self-supervised echocardiographic representations across downstream extraction strategies…"** (Sylwia Majchrowska, P. Teare). arXiv 2606.22943, 2026-06-22.
  - Candidate SSL (BYOS) runs "were ranked using the epiplexity criterion … Epiplexity was used as an unsupervised proxy for representation richness and non-collapse during training."
  - A practical small-scale use (checkpoint/hyperparameter selection). The paper gives little detail on the estimator.
  - Source: [arXiv HTML](https://arxiv.org/html/2606.22943)

**C. Critiques / boundary results (details under Q2)**

- **HIGH – "A Controlled Counterexample to Strong Proxy-Based Explanations of OOD Performance: in a Fixed Pretraining-and-Probing Setup"** (Hongmin Li). arXiv 2605.11554, v1 2026-05-12, v2 2026-06-30.
  - Shows that a task-agnostic proxy for total learned structure (a compression-style score from pretraining validation loss) can rank two pretraining corpora opposite to their OOD probe accuracy.
  - Construction: formal proposition plus a synthetic sequence-model experiment; the reversal occurs in 2 of 3 seeds.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.11554), [arXiv abs](https://arxiv.org/abs/2605.11554)

**D. Theoretical extensions or re-framings**

- **LOW-MEDIUM – "Thermodynamic Limits of Physical Intelligence"** (Koichi Takahashi, Yusuke Hayashi). arXiv 2602.05463. Published in the Artificial General Intelligence conference proceedings (DOI 10.1007/978-3-032-33195-3_24).
  - Defines "Thermodynamic Epiplexity per Joule" and distinguishes mutual-information epiplexity from operational compute-bounded MDL epiplexity.
  - Recommends "compute-bounded MDL epiplexity / compression-gain surrogates" when the latent variable is unavailable.
  - Notes the numbers are "benchmark-relative rather than universal."
  - Source: [arXiv HTML](https://arxiv.org/html/2602.05463)
- **LOW – "A Thermodynamic Theory of Learning I: Irreversible Ensemble Transport and Epistemic Costs"**. arXiv 2601.17607, 2026-01-24. Not in the S2 list.
  - Frames epiplexity as "a static, in-principle quantity … independently of how learning is actually carried out." Positions its own "Epistemic Speed Limit" as complementary: "addressing not availability but realization."
  - Source: [arXiv HTML](https://arxiv.org/html/2601.17607v1)
- **LOW – "Financial Epiplexity: A Theory of Learnable Market Structure under Bounded Computation"** (Miquel Noguer i Alonso). arXiv 2607.02695, 2026-07-02.
  - A time-bounded MDL measure of learnable market structure relative to a filtration, representation, model class and budget.
  - Proves "equal entropy need not imply equal epiplexity." Also covers compute non-monotonicity and Kelly/Sharpe bounds in structural bits.
  - Theory only.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2607.02695)
- **LOW – "From Embedding Geometry to Spectral Search: Energy Dispersion Networks For Vector Retrieval"** (Lorenzo Moriondo, Ilias Azizi). arXiv 2606.21535.
  - Uses a two-part MDL "epiplexity" argument as the "theoretical foundation" for Graph Wiring / spectral indexing (arrowspace library).
  - The same author group maintains the PyPI `epiplexity` package (see Q3).
  - Source: [arXiv HTML](https://arxiv.org/html/2606.21535)
- **LOW – "Speculating for Epiplexity: How to Learn the Most from Speculative Design?"** (Botao Amber Hu). arXiv 2602.22132, 2026-02-25. A design-theory paper: splits a speculative artifact's knowledge into structured epistemic information versus entropic noise, and adds a self-assessment questionnaire. Source: [arXiv HTML](https://arxiv.org/html/2602.22132)
- **LOW-MEDIUM – "Seeking the Unfamiliar but Memorable: Conceptual Creativity as Meta-Learning"** (Meng-Ye Ren). arXiv 2605.16477.
  - Calls epiplexity "most closely related." Its Creator–Appraiser reward (Appraiser improvement after a few adaptation steps) measures "observer-relative bounded learnability" as a generation objective. Contrast drawn: "epiplexity is a measurement on existing" data.
  - Experiments: MNIST autoencoder Appraiser and CLIP + LoRA.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.16477)

**E. Citing papers that use epiplexity as motivation or an interpretive lens (no computation)**

- **MEDIUM – "Training Language Models via Neural Cellular Automata"** (Daniel Lee, Seungwook Han, Akarsh Kumar, Pulkit Agrawal). arXiv 2603.10055.
  - Cites epiplexity as "a secondary motivation for transfer": deterministic NCA data can carry structure for bounded observers.
  - Measures complexity with gzip and names "epiplexity" as a future axis for tuning synthetic data.
  - Source: [arXiv HTML](https://arxiv.org/html/2603.10055)
- **LOW – "Data-efficient pre-training by scaling synthetic megadocs"** (Kim, Kotha, Choi, Hashimoto, Haber, Liang). arXiv 2603.18534. Says that placing the real document last may "be related to epiplexity," via the inverse-is-harder-than-forward argument. Source: [arXiv HTML](https://arxiv.org/html/2603.18534)
- **LOW – "Group Distributionally Robust Optimization-Driven RL for LLM Reasoning"** (Panaganti et al.). arXiv 2601.19280. Uses epiplexity as a data-value motivation for non-uniform prompt sampling, and cites Finzi's X thread (x.com/m_finzi/status/2008934727156453661). Source: [arXiv HTML](https://arxiv.org/html/2601.19280)
- **LOW – "Human-like autonomy emerges from self-play and a pinch of human data"** (Cornelisse et al.). arXiv 2606.19370. Says epiplexity "is a theoretical measure that we cannot yet compute or apply to data selection in practice." Source: [arXiv HTML](https://arxiv.org/html/2606.19370)
- **LOW – "Interestingness as an Inductive Heuristic for Future Compression Progress"** (Vincent Herrmann, Jürgen Schmidhuber). arXiv 2605.14831; also on OpenReview. Only: "A rich description of compute-bounded and observer-dependent MDL models has recently been presented by Finzi et al. (2026)." Sources: [arXiv HTML](https://arxiv.org/html/2605.14831), [OpenReview PDF](https://openreview.net/attachment?id=6GTlSlWW9C&name=pdf)
- **LOW – brief mentions:**
  - Momennejad & Raileanu, "A Compositional Framework for Open-ended Intelligence," 2606.15386 (epiplexity as learnability vs noisy-TV novelty).
  - Gauderis et al., "From Mechanistic to Compositional Interpretability," 2605.08934 (epiplexity as a bounded-observer alternative complexity measure for interpretability).
  - Grosso, Mikuni, Heinrich, VERaiPHY physics review, 2607.10039 (surrogates "preserving the epiplexity relevant for learning").
  - Zhang, "Mirror Horizon," 2607.11937.
  - Chen, "Wide Learning," 2608.29608.
  - Xu, "Predicting LLM Compression Degradation from Spectral Statistics," 2604.18085.
  - Sonoda et al., "Agentic theorem prover" theory, 2602.10538.
  - Rathi & Radford, "Shaping capabilities with token-level data filtering," 2601.21571 (cited in §7 only).
  - Sources: arXiv HTML of each (e.g. [2606.15386](https://arxiv.org/html/2606.15386), [2605.08934](https://arxiv.org/html/2605.08934), [2607.10039](https://arxiv.org/html/2607.10039), [2601.21571](https://arxiv.org/html/2601.21571))
- **LOW – reference-list-only or one-line citations:**
  - 2608.25745 (Iacovissi, Derr, Williamson, constrained data processing inequality; HTML not retrievable)
  - 2607.06442 (SIEVE)
  - 2606.23371 (TSD)
  - 2606.20999 (Inductive Generalization for Robotic Manipulation)
  - 2605.30160 (distributional RL in chaos)
  - 2605.08613 and 2605.05861 (Xiao et al., emergent communication)
  - 2602.14481 (rate–distortion–complexity)
  - 2603.10234 (I2X)
  - 2603.05290 (X-RAY, KDD 2026)
  - Source: [Semantic Scholar citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2601.03220/citations?fields=title,authors,year,externalIds,abstract&limit=1000)
- arXiv title/abstract search for "epiplexity" returned only 8 papers: 2601.03220, 2602.05463, 2602.22132, 2605.11554, 2606.16399, 2606.21535, 2607.02695, 2608.11746. — [arXiv API query](https://export.arxiv.org/api/query?search_query=abs:epiplexity+OR+ti:epiplexity&max_results=100)

### Inferences
- Beyond the founding group, the only substantive new *estimator* is Zhang & Levin's reservoir/ridge log-det score. It is the most Colab-friendly option: closed-form, needs no training loop, and is differentiable. It uses a linear readout on random features, not a trained MLP, so it measures a different observer class than a gradient-trained MLP. It would be a useful cheap baseline next to prequential MLP estimates.
- For small datasets, the prequential estimator needs changes: multi-epoch training, recomputing per-example losses after training, and choosing the MDL-optimal checkpoint with a validation split. Liu et al. and Dineen et al. each wrote down concrete versions, and EDL's definition (compare to a predictor at the final model's *population* loss) handles the same memorisation issue. A small-MLP project should pick one of these conventions explicitly.
- No independent group has yet published a careful reproduction of the founding paper's headline experiments (ECA rules, chess ordering, OOD correlation). The empirical uses are mostly LLM/LoRA self-play settings.

### Gaps
- Google Scholar "cited by" could not be queried programmatically. The Semantic Scholar list (37) is probably incomplete: it missed 2606.16399 and 2601.17607, which I found by other means. Papers posted in September 2026 may not be indexed yet.
- The HTML for 2608.25745 could not be retrieved, so how it uses epiplexity is unknown.
- The exact epiplexity estimator used by Majchrowska & Teare (2606.22943) is not described in the text I extracted.

---

## Q2. Are there critiques or technical objections?

### Takeaway
There is little public adversarial critique: no LessWrong/Alignment Forum posts, Reddit threads or OpenReview reviews were found, and the Hacker News posts drew essentially no discussion. The substantive objections come from papers and repos:
1. **Task-relevance / proxy validity:** more total learnable structure need not mean better downstream OOD performance. Evidence: Li's formal counterexample and a negative GRPO-transfer result.
2. **Estimator cost and non-differentiability** (Zhang & Levin).
3. **Looseness of the prequential estimator**, which the founding group itself acknowledges in the requential-coding paper.
4. **Small-data / multi-epoch invalidity** of the single-epoch prequential assumptions (Liu et al.).
5. **Observer- and benchmark-relativity** of the numbers.

### Cited Findings
- **Proxy vs. task-relevant structure (Hongmin Li, 2605.11554).**
  - The paper argues that proxy-based explanations "conflate three distinct objects": a formal structure quantity, its operational proxy, and task-relevant structure.
  - It proves (Proposition 1) that there exist pretraining distributions D_A, D_B with proxy S(D_A) > S(D_B) but OOD performance reversed.
  - Experiment: a causal LM with a background generator ("abundant but task-irrelevant structure") and a relevance generator. The OOD ranking reverses the proxy ranking in 2 of 3 seeds.
  - Scope, in the author's words: "does not reject structure-based explanations in general; it identifies a boundary on strong proxy-based explanations."
  - Sources: [arXiv HTML](https://arxiv.org/html/2605.11554), [arXiv abs](https://arxiv.org/abs/2605.11554)
- **Negative empirical result (sunnydigital repo).**
  - Epiplexity measured by a Qwen2.5-1.5B probe across 8 benchmark datasets was used to weight GRPO curricula for Qwen2.5-3B. "Epiplexity shows essentially zero correlation with transfer (Spearman ρ = −0.17)."
  - Reward variance was the dominant factor.
  - Unreviewed GitHub result. — [GitHub sunnydigital/epiplexity-curriculum-post-training](https://github.com/sunnydigital/epiplexity-curriculum-post-training)
- **Estimator cost / not usable as an objective.** Zhang & Levin say the original estimator requires "a full training run for every system scored … computationally expensive but also opaque to gradients … serves as a measure but not as an objective." — [arXiv 2607.18433](https://arxiv.org/html/2607.18433)
- **Prequential looseness (founding group).**
  - The requential-coding paper says prequential coding "codes the exact data sequence regardless of how much the model learns, yielding large codes when the data has high entropy."
  - It calls the earlier extrapolation "the non-rigorous prequential heuristic."
  - It warns that "a weak [compressor] renders the theory vacuous or even misleading, for example making larger models appear more complex when they in fact generalize better."
  - Source: [arXiv 2607.11883](https://arxiv.org/html/2607.11883)
- **Small-data assumptions.** Liu et al. note that the original computation "assumes a sufficiently large dataset, such that training for a single epoch suffices." They changed the procedure for small datasets and multi-epoch training. — [arXiv 2603.02218](https://arxiv.org/html/2603.02218)
- **Practicality.** Cornelisse et al.: epiplexity "in its current form, is a theoretical measure that we cannot yet compute or apply to data selection in practice." — [arXiv 2606.19370](https://arxiv.org/html/2606.19370)
- **Benchmark-relativity.** Takahashi & Hayashi say the resulting numbers "are benchmark-relative rather than universal." — [arXiv 2602.05463](https://arxiv.org/html/2602.05463)
- **Static vs. realised information.** "Epiplexity is fundamentally a static, in-principle quantity … independently of how learning is actually carried out." — [arXiv 2601.17607](https://arxiv.org/html/2601.17607v1)
- **Credit for irrelevant structure (toy example).** The paperfoot/fti repo reports: "Epiplexity ranks the agent that learns everything highest and credits the noise memoriser with 0.8 bits," whereas their readout-relative usable-information measure zeroes it. — [GitHub paperfoot/fti](https://github.com/paperfoot/fti)
- **Repos that disclaim measuring "true" epiplexity:**
  - "This is not an estimate of theoretical epiplexity. It measures the area of a training loss curve above final validation loss…" — [GitHub edirent/audio_epiplexity_test](https://github.com/edirent/audio_epiplexity_test)
  - "computing it requires a converged model … It is an online proxy inspired by the paper, not the quantity itself." — [GitHub ckgresla/cyoa-public](https://github.com/ckgresla/cyoa-public)
- **Weak signal in a small reproduction.** A Kaggle-T4 test comparing aligned vs obfuscated code found epiplexity 65.53 vs 62.59 and time-bounded entropy 3.32 vs 3.91, "not that great, but seem directionally correct." — [GitHub aryansi225/Epiplexity_for_Weak_Learners](https://github.com/aryansi225/Epiplexity_for_Weak_Learners)
- **Theoretical reformulation.**
  - asving's draft "Computational Information: A Generalization of Shannon Entropy" defines subjective entropy for a priced program class, with an amortisation parameter. It derives a computationally bounded data-processing inequality.
  - It "situates epiplexity (Finzi et al. 2025) and predictive V-information (Xu et al. 2020) as the two axes of the framework."
  - Sources: [GitHub asving/Epiplexity](https://github.com/asving/Epiplexity), [PDF mirror](https://asving.com/uploads/2026/01/epiplexity-reformulation.pdf)
- **Hacker News.** The launch post (item 46542217, 2026-01-08) had 5 points and 0 comments. A repost (47426916, 2026-03-18) had 2 points and 2 comments. No substantive technical discussion. — [HN 46542217](https://news.ycombinator.com/item?id=46542217); [HN Algolia search](https://hn.algolia.com/api/v1/search?query=epiplexity&tags=story)
- **Community pages.** The Universal Algorithmic Intelligence group (Cole Wyeth) hosted a Finzi talk on 2026-02-09. The page has "no substantive critiques" in its comments. — [uaiasi.com](https://uaiasi.com/2026/02/08/marc-finzi-on-epiplexity/)

### Inferences
- The strongest objection for a small-dataset project is Li's. Epiplexity measures *total* learnable structure, including structure irrelevant to any downstream task. The paperfoot toy example and the sunnydigital null result point the same way. A project that validates its estimates against downstream performance should expect cases where they disagree, and should consider task-conditioned variants (e.g. bossman-lab's "task-relevant epiplexity" draft).
- The founding group itself treats the prequential area-under-curve as a heuristic upper bound. Small-MLP estimates should be reported as estimator-, optimizer- and schedule-dependent, and compared only within a fixed pipeline. That is also Li's "fixed pipeline" point.
- The paper was widely cited but little debated. The lack of critique probably reflects this quiet reception, not that the paper settled the open questions.

### Gaps
- X/Twitter could not be searched or read directly. Replies to the launch threads by Wilson (x.com/andrewgwils/status/2008938365010563106) and Finzi (x.com/m_finzi/status/2008934727156453661) may contain critiques I could not see.
- No LessWrong / Alignment Forum or r/MachineLearning discussion was found (web search returned none). This is absence of evidence, not a confirmed absence.
- No OpenReview reviews of the founding paper were found, consistent with no venue submission being publicly visible (see Q4).
- No formal-methods critique was found from the algorithmic-information-theory side (sophistication, Kolmogorov structure function, logical depth: Antunes/Fortnow, Aaronson, Hutter). The founding paper cites these works, but I found no public response from those authors.

---

## Q3. Independent reimplementations, notebooks, small-scale reproductions and libraries

### Takeaway
There are roughly 25–30 public GitHub repos. Most are small, 0–30 stars. None is an established, validated library. The most relevant to a small-MLP Colab project:
- the official Zhang & Levin reservoir estimator code;
- bossman-lab's numpy/MLP "task-relevant epiplexity" experiments;
- the Liu et al. `epiplexity_mdl.py` pipeline;
- lcrh's browser (TF.js) estimator experiments;
- a PRNG learnability-boundary study (lmxxf);
- three "library" attempts (RichardScottOZ, Taiyou, and the PyPI `epiplexity` package from tuned-org-uk).

The package name `epiplexity` is claimed by several READMEs. On PyPI it belongs to the tuned-org-uk project.

### Cited Findings

**Libraries / general estimators**

- **MEDIUM – RichardScottOZ/Epiplexity** (Python, created 2026-01-16). "Epiplexity implementation first pass."
  - Claims a framework-agnostic core with scikit-learn, TensorFlow/Keras and PyTorch wrappers.
  - Offers prequential and requential coding and knowledge-distillation tracking. Aimed at geoscience/scientific ML.
  - Source: [GitHub](https://github.com/RichardScottOZ/Epiplexity)
- **MEDIUM – Taiyou/Epiplexity** (Python/PyTorch, 2026-01-25; Japanese README). Classes `PrequentialEpiplexityEstimator`, `RequentialEpiplexityEstimator` and `ScalingLawEpiplexityEstimator`. Its cost table: prequential 1x, requential 2–10x. Source: [GitHub](https://github.com/Taiyou/Epiplexity)
- **MEDIUM/LOW – tuned-org-uk/graph-wiring-epiplexity**, the PyPI `epiplexity` package (v0.1.0–0.5.0).
  - "Measure any algorithm's epiplexity … for *any* algorithm wrapped as a T-time probabilistic model."
  - Provides `EpiplexityEngine`, a `TTimeProbabilisticModel` ABC, adapters (PyTorch classifiers, transformer LMs, GNNs, ArrowSpace), property tests, and a scikit-learn worked example. Notebooks include a CVE-1999–2025 case study.
  - Source: [GitHub](https://github.com/tuned-org-uk/graph-wiring-epiplexity), [PyPI JSON](https://pypi.org/pypi/epiplexity/json)
- **LOW-MEDIUM – victor0777/epiplexity** ("Epiplex", 2026-01-10). A prequential-code-length metrics tracker for LLM training (`PrequentialMeter` with DDP, JSONL logger). Its README also says `pip install epiplexity`, which conflicts with the PyPI owner. Source: [GitHub](https://github.com/victor0777/epiplexity)

**Estimators in official or companion code**

- **HIGH – Zhangyanbo/learnable-novelty** (official code for 2607.18433; Python, uv; 29 stars). The closed-form score S^φ: "a ridge readout on a fixed random reservoir, cheap to compute, deterministic, and differentiable." Covers ECA ranking, inverse NCA, MNIST encoder and RL intrinsic reward. Source: [GitHub](https://github.com/Zhangyanbo/learnable-novelty)
- **MEDIUM-HIGH – shikaiqiu/requential-coding** (official code for 2607.11883). JAX/Flax GPT-2-style transformer, Hydra configs, one notebook per paper figure; runs on GPU/TPU. Source: [GitHub](https://github.com/shikaiqiu/requential-coding)
- **MEDIUM – thinkwee/SelfPlay_SelfEvo_Gap** (official code for 2603.02218). `epiplexity_mdl.py` computes "S … the prequential code length at the MDL-optimal checkpoint" for coding tasks from Absolute Zero Reasoner. Source: [GitHub](https://github.com/thinkwee/SelfPlay_SelfEvo_Gap)
- **MEDIUM – ARC-ASU/vocab_dropout** (official code for 2604.03472, COLM 2026). LoRA-observer epiplexity computation. Source: [GitHub](https://github.com/ARC-ASU/vocab_dropout)

**Small-scale experiments and reproductions (most relevant scale)**

- **HIGH – bossman-lab/task-relevant-epiplexity** (2026-05-24). A draft paper extending epiplexity to "task-relevant, observer-dependent transfer." Includes a synthetic PCA transfer experiment and a "small next-token MLP transfer experiment" (`sequence_mlp_transfer.py`). Needs only numpy and pandas, plus a one-command reproduction script. Source: [GitHub](https://github.com/bossman-lab/task-relevant-epiplexity)
- **MEDIUM – lcrh/epiplexity** (JavaScript / TensorFlow.js in the browser).
  - Discrete four-state NCA with trained conv or transformer observers and learning-curve estimates.
  - Continuous NCA with a frozen random CNN reservoir, ridge fit (λ = 0.3) and score S = ½ log2 det(I + WᵀW), optimised by gradient ascent.
  - Source: [GitHub](https://github.com/lcrh/epiplexity)
- **LOW-MEDIUM – lcrh/walker-epiplexity** (2026-09-14). A video-first reproduction of Zhang & Levin's epiplexity-only Walker2d experiment, using an unmodified vendor snapshot of the authors' code. PPO default MLP, frozen 32-feature reservoir, horizon 16; CPU. Source: [GitHub](https://github.com/lcrh/walker-epiplexity)
- **MEDIUM – the-puzzler/epijepa** (2026-09-18).
  - Uses the Zhang–Levin reservoir estimator as an anti-collapse regulariser for joint-embedding SSL.
  - Backbone probe accuracy: CIFAR-10 76.64% vs 77.53% for SIGReg; Imagenette 87.16% vs 90.22%.
  - Source: [GitHub](https://github.com/the-puzzler/epijepa)
- **MEDIUM – eberlful/LearnableNovelty** (2026-07-23). Applies the reservoir epiplexity S^φ to NASA CMAPSS turbofan time series. A 3-layer 1D-CNN encoder is trained by maximising S^φ. Source: [GitHub](https://github.com/eberlful/LearnableNovelty)
- **MEDIUM – lmxxf/grokking-train-learnability.** "Validating the core prediction of … From Entropy to Epiplexity": finds the learnability phase transition for pseudorandom sequence generators with a small transformer.
  - The boundary lies between state spaces of 256 and 2^31.
  - LFSR-31 is learnable (after a context increase) but a glibc LCG is not, even with 20x more data. Larger models did not help.
  - Unreviewed. Source: [GitHub](https://github.com/lmxxf/grokking-train-learnability)
- **MEDIUM – edirent/audio_epiplexity_test** (2026-06-16).
  - A controlled prequential proxy for audio: S_hat = Σ max(loss_i − final_loss, 0)·tokens_i, in nats.
  - Uses a shared k-means tokenizer and synthetic controls (silence, sine, white/pink noise, uniform/Zipf random tokens) versus ESC-50, UrbanSound8K, GTZAN, etc. Targets a single RTX 4090.
  - Source: [GitHub](https://github.com/edirent/audio_epiplexity_test)
- **LOW-MEDIUM – aryansi225/Epiplexity_for_Weak_Learners** (2026-05-28). A Kaggle T4 notebook testing whether a weak learner separates aligned from obfuscated code by epiplexity. Weak separation (see Q2). Source: [GitHub](https://github.com/aryansi225/Epiplexity_for_Weak_Learners)
- **LOW-MEDIUM – EricBoittier/epiplexity_mlip.** Data-selection experiments for machine-learned interatomic potentials on rMD17 aspirin, with a teacher-noise option, KL/gzip dataset diagnostics and Snakemake. Source: [GitHub](https://github.com/EricBoittier/epiplexity_mlip)

**Applied or LLM-scale projects**

- **LOW – sunnydigital/epiplexity-curriculum-post-training.** GRPO curriculum weighted by probe-model epiplexity; null result (see Q2). Source: [GitHub](https://github.com/sunnydigital/epiplexity-curriculum-post-training)
- **LOW – ckgresla/cyoa-public** (2026-08-23). RLVR bandit curricula with an "online epiplexity-motivated learnability signal": the fresh-sample loss minus an EMA convergence baseline. Source: [GitHub](https://github.com/ckgresla/cyoa-public)
- **LOW – DTennant/ideation-epiplexity.** Proposes epiplexity (area between the loss curve and the converged loss) to separate "ideation" from "optimization" in self-improving agents. Source: [GitHub](https://github.com/DTennant/ideation-epiplexity)
- **LOW – 6tizer/epiplexity.** Despite the name, it measures a 1B-vs-3B/8B perplexity gap as "learnability"; not the paper's estimator. Source: [GitHub](https://github.com/6tizer/epiplexity)

**Educational, seminar and placeholder repos**

- **LOW – chris-alexiuk/epiplexity.** An interactive Svelte teaching site (PRNG, cellular automata, chess demos, quizzes). Source: [GitHub](https://github.com/chris-alexiuk/epiplexity)
- **LOW – yhjeon-nxt/epiplexity-seminar** (seminar slides/HTML). Source: [GitHub](https://github.com/yhjeon-nxt/epiplexity-seminar)
- **LOW – JasonGross/epiplexity-curriculum-visualizer** (2026-09-13). "Visualizing curricula optimizations based on epiplexity and MDL"; currently only a README. Source: [GitHub](https://github.com/JasonGross/epiplexity-curriculum-visualizer)
- **LOW – Godsend/reasoning-core.** A position draft; epiplexity estimation is "planned." Source: [GitHub](https://github.com/Godsend/reasoning-core)

**Other**

- **LOW – DuoNeural/lab**, "P23 — DHP Epiplexity" (Zenodo). Claims "τ*/τ_L=0.72 is MDL epiplexity boundary." Source: [GitHub](https://github.com/DuoNeural/lab)
- **LOW – RandMan444/epiplexity-alchemy.** A notebook posted to HN claiming to beat DeepMind's Alchemy meta-RL benchmark with epiplexity. The repo now returns 404. Source: [HN 46544216](https://news.ycombinator.com/item?id=46544216)
- **Blog-style explainers** (summaries, not reimplementations):
  - arxiviq substack — [link](https://arxiviq.substack.com/p/from-entropy-to-epiplexity-rethinking)
  - "Epiplexity: When Computation Creates Information" (aiwithmike substack) — [link](https://aiwithmike.substack.com/p/epiplexity-when-computation-creates)
  - Lixin Xu paper notes, Feb 2026 — [link](https://davidlxu.github.io/posts/2026/02/epiplexity-paper-notes/)
  - AkihikoWatanabe paper_notes #4145 — [link](https://github.com/AkihikoWatanabe/paper_notes/issues/4145)
  - note.com AI Nest (Japanese) — [link](https://note.com/ainest/n/n977eab5f1dcc?hl=en)
  - Andy Trattner, "Epiplexity," 2026-06-03 (speculative essay linking it to the free-energy principle and AI coding agents) — [link](https://andys.blog/epiplexity/)

### Inferences
- No existing repo does what the project targets: prequential or requential epiplexity with trained MLPs on small datasets, calibrated against known ground truths such as PRNG vs structured data. The nearest are:
  - bossman-lab (numpy MLP transfer);
  - lmxxf (PRNG learnability boundary, but with transformers and accuracy rather than code length);
  - edirent (clean synthetic-control design for a prequential proxy).
- The Zhang–Levin reservoir estimator has already been reused independently three times (lcrh, epijepa, eberlful) within two months. That suggests it is easy to implement on one GPU or even in a browser.

### Gaps
- GitHub *code* search (files that mention epiplexity inside code) needs authentication and was not run, so repos that mention epiplexity only in code files are missed.
- None of the repos has been independently verified, and most report no uncertainty estimates.
- No Colab notebook explicitly labelled as a reproduction of the founding paper's ECA or chess experiments was found.

---

## Q4. Did the founding paper appear at a venue? Talks, slides, videos

### Takeaway
As of 2026-09-23, 2601.03220 appears to be **arXiv-only**:
- v1 2026-01-06, v2 2026-03-16;
- no journal-ref, and the arXiv comment only gives the code URL;
- author pages (Izmailov CV and publications page, Shikai Qiu's homepage) list it as "arXiv."

The core idea also appears as the final chapter of Yiding Jiang's August 2025 CMU PhD thesis. There are several public talks: an MIT IAIFI colloquium, a talk on Michael Levin's channel, a UAI reading-group talk, a Cornell AI-MI seminar, and a September 2026 Wilson lecture.

### Cited Findings
- **arXiv metadata:** v1 published 2026-01-06, updated 2026-03-16 (v2). Comment: "Code available at https://github.com/shikaiqiu/epiplexity"; no journal reference. — [arXiv API](https://export.arxiv.org/api/query?id_list=2601.03220); [alphaXiv](https://www.alphaxiv.org/abs/2601.03220)
- **Author listings:** Pavel Izmailov's CV lists it under 2026 as "arxiv." The same CV lists him as Area Chair for ICML 2026 and NeurIPS 2026. His publications page shows no conference. Shikai Qiu's homepage lists the venue as "arXiv." — [Izmailov CV](https://izmailovpavel.github.io/files/CV.pdf); [Izmailov publications](https://izmailovpavel.github.io/publications.html); [Shikai Qiu homepage](https://shikaiqiu.github.io/)
- **Talks and videos:**
  - IAIFI Colloquium (MIT, Kolker Room 26-414), Andrew Gordon Wilson, 2026-02-27, "From Entropy to Epiplexity…"; uploaded 2026-02-28. — [YouTube 2XBqlpi4fNk](https://www.youtube.com/watch?v=2XBqlpi4fNk)
  - "'From Entropy to Epiplexity' by Andrew Wilson and Marc Finzi," about 1 h 7 min talk plus Q&A, on "Michael Levin's Academic Content" channel, uploaded 2026-04-11. It connects to Zhang & Levin's later paper. — [YouTube _U8AwUq_aJQ](https://www.youtube.com/watch?v=_U8AwUq_aJQ)
  - Marc Finzi at the Universal Algorithmic Intelligence weekly meeting, 2026-02-09. Finzi is listed as a Research Scientist at OpenAI. The recording was shared only in a WhatsApp group. — [uaiasi.com](https://uaiasi.com/2026/02/08/marc-finzi-on-epiplexity/)
  - Cornell AI-MI Seminar Series, "From Entropy to Epiplexity…". — [aimi.cornell.edu](https://aimi.cornell.edu/event/ai-mi-seminar-series-from-entropy-to-epiplexity-rethinking-information-for-computationally-bounded-intelligence/)
  - Andrew Gordon Wilson, "The Foundations of Modern AI: Generalization, Data Selection, and Epiplexity," on his own channel, uploaded 2026-09-07. A two-part talk: soft inductive biases / model selection, then data selection / epiplexity. — [YouTube lKoJJxjUfdw](https://www.youtube.com/watch?v=lKoJJxjUfdw)
- **Launch threads and posts:**
  - Wilson on X: "We have been working on this for almost 2 years." — [X](https://x.com/andrewgwils/status/2008938365010563106)
  - Finzi's X thread defines epiplexity as "the info in its weights, while time-bounded entropy is the leftover info in the data given the model (the NLL)." — [X](https://x.com/m_finzi/status/2008934733108154622)
  - Wilson's LinkedIn announcement. — [LinkedIn](https://www.linkedin.com/posts/andrew-gordon-wilson-46645b177_i-am-excited-to-announce-our-new-paper-where-activity-7414712093234339840-KPEu)
  - AI Weekly news item. — [AI Weekly](https://aiweekly.co/alerts/nyus-wilson-introduces-epiplexity-to-guide-ai-data-selection)
  - A podcast episode, likely an automated paper summary. — [Apple Podcasts](https://podcasts.apple.com/us/podcast/from-entropy-to-epiplexity-rethinking-information-for/id1802074035?i=1000744246692)

### Inferences
- The founding paper may be under review at NeurIPS 2026 or another venue, but nothing public confirms any submission or acceptance. For now, cite it as an arXiv preprint (v2).
- The Wilson and Finzi talks (IAIFI, Levin channel) are the best sources for the authors' informal statements about estimator caveats and intended use.

### Gaps
- No slides were found publicly posted.
- The Cornell AI-MI seminar date and recording status were not verified.
- NeurIPS 2026 decisions may not yet be public, and ICML 2026 workshop proceedings were not exhaustively searched.

---

## Q5. Earlier or concurrent work by the same authors that led to epiplexity

### Takeaway
The clearest precursor is Yiding Jiang's CMU PhD thesis (August 2025, CMU-ML-25-115). Its final part introduces epiplexity, "built on time-bounded probability and V-Entropy," with an estimator from learning curves and scaling laws. That chapter is adapted from the then-in-preparation Finzi/Qiu/Jiang manuscript.

Other technical ingredients come from the group's 2024–2025 work:
- Adaptive Data Optimization (per-domain loss-curve scaling laws);
- Compute-optimal LLMs provably generalize better (prequential per-token complexity);
- the Kolmogorov / soft-inductive-bias program (Goldblum/Finzi/Wilson; Wilson ICML 2025);
- scaling-collapse loss-curve universality (Qiu et al.).

The main methods follow-up is requential coding (July 2026). Concurrent and independent: Excess Description Length (Donoway, Joren, Roger, Leike, January 2026).

### Cited Findings
- **Yiding Jiang, "Quantifying, Understanding, and Improving Generalization in Deep Learning,"** CMU PhD thesis, August 2025, CMU-ML-25-115 (advisor Zico Kolter).
  - The final part introduces "epiplexity, a measure of the structural information that a time-bounded observer can extract from data, built on time-bounded probability and V-Entropy." It "resolves three apparent paradoxes" and proposes "a practical estimator derived from learning curves and scaling laws," validated on cellular automata, chess, synthetic benchmarks, language and vision.
  - The chapter is "adapted from a manuscript in preparation … 'From Entropy to Epiplexity…'."
  - It shows the ECA ranking is robust across Mamba, TCN and transformer observers.
  - Liu et al. (2603.02218) cite it as "Jiang, 2025" together with Finzi et al. 2026.
  - Source: [CMU thesis PDF](https://ml.cmu.edu/research/phd-dissertation-pdfs/yidingji_phd_mld_2025.pdf)
- **Jiang, Zhou, Feng, Malladi, Kolter, "Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws."** arXiv 2410.11820, ICLR 2025. Models each domain's loss curve with a scaling law to set data mixtures online. This is the mechanism behind the thesis's "estimator … from learning curves and scaling laws" and the later scaling-law-based epiplexity selection. — [arXiv](https://arxiv.org/abs/2410.11820); [thesis](https://ml.cmu.edu/research/phd-dissertation-pdfs/yidingji_phd_mld_2025.pdf)
- **Finzi, Kapoor, Granziol, Gu, De Sa, Kolter, Wilson, "Compute-Optimal LLMs Provably Generalize Better With Scale."** arXiv 2504.15208, ICLR 2025. Introduced the per-token-complexity bound and the prequential-heuristic extrapolation. Requential coding (2607.11883) later "certifies the same trend with a rigorous code." — [arXiv](https://arxiv.org/abs/2504.15208); [2607.11883](https://arxiv.org/html/2607.11883)
- **Andrew Gordon Wilson, "Deep Learning is Not So Mysterious or Different."** arXiv 2503.02113, ICML 2025. The soft-inductive-bias / Kolmogorov framing that Wilson's September 2026 talk pairs with epiplexity. — [arXiv](https://arxiv.org/abs/2503.02113); [YouTube lKoJJxjUfdw](https://www.youtube.com/watch?v=lKoJJxjUfdw)
- **Qiu, Xiao, Wilson, Pennington, Agarwala, "Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks."** arXiv 2507.02119, ICML 2025. Loss-curve universality work by the co-first author; code at github.com/shikaiqiu/supercollapse. — [arXiv](https://arxiv.org/abs/2507.02119)
- **Pre-2025 background from the same group** (listed for completeness; outside the 2025–2026 window):
  - Goldblum, Finzi, Rowan, Wilson, "The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning," arXiv 2304.05366 (ICML 2024). — [arXiv](https://arxiv.org/abs/2304.05366)
  - Lotfi, Finzi, Kuang, Rudner, Goldblum, Wilson, "Non-Vacuous Generalization Bounds for Large Language Models," arXiv 2312.17173 (ICML 2024). — [arXiv](https://arxiv.org/abs/2312.17173)
- **Follow-up (same group): Requential Coding**, arXiv 2607.11883, 2026-07-13 (Q1). — [arXiv](https://arxiv.org/abs/2607.11883)
- **Concurrent / independent: Excess Description Length.** Donoway, Joren, Roger, Leike, arXiv 2601.04728 (2026-01-08); ISIT 2026 short version by Donoway. Prequential-coding-based measure of learned generalizable structure; posted two days after epiplexity's v1. — [arXiv](https://arxiv.org/abs/2601.04728)

### Inferences
- The thesis chapter shows the estimator's lineage: learning-curve / scaling-law fits (ADO), then prequential area-under-curve, then requential coding. A small-MLP project can follow the same path: cheap prequential first, requential or a reservoir estimator as a cross-check.
- EDL and epiplexity converged independently on the same prequential "online minus final loss" quantity. That supports the estimator's naturalness. EDL's explicit handling of finite-data memorisation, via population loss rather than training loss, is a useful correction to borrow for small datasets.
- (Inference, not confirmed by sources) The scaling-collapse loss-curve universality results could justify fitting parametric loss curves to extrapolate the final loss in short MLP runs.

### Gaps
- I did not verify whether Jiang's thesis chapter differs substantively from the arXiv paper's estimator (e.g. the scaling-law estimator in the thesis vs the prequential/requential emphasis on arXiv). Only the thesis abstract and a few passages were checked.
- Marc Finzi's and Andrew Gordon Wilson's homepages were not checked individually for additional unpublished precursors, workshop versions or slides.
