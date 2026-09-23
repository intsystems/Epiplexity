# Estimating Epiplexity at Small Scale: MLPs and Small Datasets on One Colab GPU

Scope and conventions for these notes (researched 2026-09-23):
- "Paper" means arXiv 2601.03220v2 (16 Mar 2026), read in full from https://arxiv.org/html/2601.03220v2. The figure values below were read off the paper's SVG figures after rendering them locally, so they are approximate.
- "Repo" means https://github.com/shikaiqiu/epiplexity (main branch, downloaded 2026-09-23). Code facts come from reading the files.
- Evidence tags:
  - **[VERIFIED-code]**: read directly in the source.
  - **[VERIFIED-local]**: measured in this session on a CPU-only Windows machine (numpy 2.0.2, torch 2.8.0+cpu). This is not Colab hardware.
  - **[FIGURE~]**: read approximately off a paper figure.
  - **[ESTIMATE]**: my own arithmetic or assumption. None of these numbers were measured on Colab.

## Q1. Which experiments in the paper are already small, which architectures were used, and where is each qualitative claim shown at its smallest?

### Takeaway
Every trained experiment in the paper uses a GPT-2-style transformer. There are no MLPs or CNNs. Still, the synthetic experiments are already "small" by LLM standards: models of about 3k to 28M parameters, binary vocabularies, sequences of 64 to 512 tokens, and compute of roughly 1e11 to 2e17 FLOPs per frontier. The central claims (structured > trivial > pseudorandom, induction, emergence) are shown only at this synthetic scale. The claim that *epiplexity* (not just entropy) depends on ordering is shown only on chess, with 1M to 160M-parameter models trained on 5B tokens. At small scale the paper shows ordering dependence only for time-bounded *entropy*, using the rule-30 one-way-function experiment.

### Cited Findings
- **Architecture.** "Unless otherwise stated, we use the GPT-2 … transformer architecture trained with Adam optimizer." The learning rate is tuned on a small model and transferred with µP and CompleteP. The paper has no MLP or CNN experiments. The only CNN mentioned is AlphaZero's policy/value network, in the Appendix F discussion. — [Paper, App. C and App. F](https://arxiv.org/html/2601.03220v2)
- **ECA structured/trivial/random (Fig. 3, §5.1, App. C.1).**
  - Task: predict Y = ECA^48(X) from X on 64 cells. X is burned in for 1000 steps from a uniform random state.
  - Grid: width ∈ {16, 32, 64, 128, 256, 512} × depth ∈ {1, 2, 4, 6, 9}. Batch 1536 sequences, base LR 0.03, 100 warmup steps, EMA time scale 50.
  - Test set size 𝒟 = 100M tokens, counting Y only.
  - Results: rule 15 gives low H_T and low S_T; rule 30 gives maximal H_T and no S_T; rule 54 gives high S_T.
  - [Paper §5.1, App. C.1](https://arxiv.org/html/2601.03220v2)
- **Fig. 3 values [FIGURE~].**
  - The compute axis runs from about 1e11 to 2e17 FLOPs. It includes the 2N𝒟 test-evaluation cost with 𝒟 = 1e8.
  - S_T(Y|X), rule 54: rises to about 5.4e6 bits at about 1e17 FLOPs.
  - S_T(Y|X), rule 15: peaks at about 1.1e6 around 1e12 FLOPs and then settles near 0.2e6.
  - S_T(Y|X), rule 30: about 0.
  - H_T, rule 30: stays near 1e8 bits, i.e. about 1 bit per token.
  - H_T, rule 54: falls to about 0.15e8 bits.
  - At the lowest compute (about 1e12), rule 15's S_T is *above* rule 54's. The ordering 54 > 15 > 30 only settles from about 1e13 FLOPs.
  - Source: [Paper Fig. 3 (eca_rules_measurements.svg)](https://arxiv.org/html/2601.03220v2/eca_rules_measurements.svg)
- **ECA, more rules (Fig. 2c, App. C.7).**
  - Rules {0, 32, 4, 15, 22, 30, 41, 54, 106, 110}, which cover all four Wolfram classes.
  - Grid: width ∈ {16, 32, 64, 128} × depth ∈ {1, 2, 3}, up to 10,000 steps, batch 384, 𝒟 = 250M tokens.
  - "For each rule, we report the maximum epiplexity over the resulting compute range."
  - [Paper App. C.7](https://arxiv.org/html/2601.03220v2)
- **Hard induction (Fig. 5b, App. C.3).**
  - Rule 30 iterated 4 steps on 32 cells, with h ∈ {0..5} input bits hidden.
  - One model: 3 layers, width 256, batch 1536, 20,000 steps, max teacher-student KL 0.03 nats/token.
  - "Further increasing model size or training data led to no improvement in the loss."
  - [Paper App. C.3](https://arxiv.org/html/2601.03220v2)
- **Hard induction values [FIGURE~].** Compute axis about 5e13 to 2e16 FLOPs. Measured epiplexity rises monotonically from about 2.4e6 (h=0) to about 5.5e6 (h=5). The notebook plots K_req/1e6 without converting nats to bits, so these units are nats. — [Fig. 5 (induction_hard.svg)](https://arxiv.org/html/2601.03220v2/induction_hard.svg); [notebooks/induction_hard.ipynb](https://github.com/shikaiqiu/epiplexity/blob/main/notebooks/induction_hard.ipynb)
- **Easy induction (Fig. 5c, App. C.2).**
  - 8-symbol random Markov chains with h hidden rows; sequence length 512.
  - One model: 3 layers, width 128, LR 0.03, batch 384, 3,000 steps.
  - "Further increasing the model size led to negligible improvement in the loss."
  - [FIGURE~] Compute runs about 1e13 to 2.5e15 FLOPs. Epiplexity is about 1.0e6 (h=0), 2.7e6 (h=2), 4.0e6 (h=4), 4.7e6 (h=6) and 3.2e6 (h=8), so intermediate h has the highest value.
  - [Paper App. C.2](https://arxiv.org/html/2601.03220v2); [induction_easy.svg](https://arxiv.org/html/2601.03220v2/induction_easy.svg)
- **Factorization / one-way function (Fig. 4a).**
  - Rule 30 for 8 steps, state size n ∈ {16, 24, 32, 48, 64}, forward vs reverse direction.
  - Repo config: 8 layers, width 128 (n ≤ 32) or 192; batch 512; 50k or 100k steps; seeds {68, 37}.
  - Only H_T is reported. The forward direction reaches the Shannon entropy; the reverse direction keeps a gap of roughly 2 to 12 bits [FIGURE~]. No epiplexity is reported.
  - [experiments/soi.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/soi.py); [owf_scaling.svg](https://arxiv.org/html/2601.03220v2/owf_scaling.svg)
- **Chess ordering (the epiplexity-order claim).** "We train models of varying sizes from 1M to 160M parameters with depth between 3 and 24 layers … The teacher models are trained for 5B tokens … The test set size is set to 5B tokens." This is also the only OOD-transfer experiment. — [Paper App. C.4](https://arxiv.org/html/2601.03220v2)
- **Emergence (Fig. 6, App. C.8).**
  - ECA rule 54, t = 64 steps.
  - Grid: widths {16, 32, 64, 128}, depths {1, 2, 4, 8, 16, 32}, loops ℓ ∈ {1, 2, 4, 8, 16}.
  - Batch 147,456 tokens; 𝒟 = 100M final-state tokens.
  - The repo sets T = 1e9 // (96·4·64·6) ≈ 6.8k steps, i.e. about 1e9 training tokens per run [VERIFIED-code].
  - [Paper App. C.8](https://arxiv.org/html/2601.03220v2); [experiments/eca_emergence.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_emergence.py)
- **Natural data (Fig. 8a).** OpenWebText (character level), Lichess, and CIFAR-5M (greyscale, 1024-pixel raster sequences). "time-bound of 6×10^18 FLOPs, by training models of up to 160M parameters on at most 5B tokens using requential coding." The scaling-law estimates (Fig. 8b) reuse published scaling laws and involve no training. — [Paper §6.2–6.3, App. C.5–C.6, C.9](https://arxiv.org/html/2601.03220v2)
- **Prequential vs requential magnitudes (Fig. 2c) [FIGURE~]**, all in "MB" on the axes:

  | Group | S_req | S_preq |
  |---|---|---|
  | ECA | 0 to 0.9 | 0 to 3.9 |
  | Easy induction | 0.15 to 0.93 | 4.1 to 6.3 |
  | Hard induction | 0.45 to 1.0 | 3.8 to 6.3 |
  | Natural | about 11 to 45 | about 10 to 70 |

  The paper says "the prequential estimate is typically several times larger than the requential estimate, the two estimates correlate well, particularly within each group." — [req_vs_preq_scatter.svg](https://arxiv.org/html/2601.03220v2/req_vs_preq_scatter.svg); [Paper §4.3](https://arxiv.org/html/2601.03220v2)

### Inferences
- **Smallest demonstration of each claim.**
  - *Structured > trivial > random:* the ECA experiment. Frontier models go down to about 3k parameters (1 layer, width 16). The ordering is visible on the frontier at about 1e13 to 1e14 FLOPs. That compute is tiny, but the paper still ran about 24 to 30 model configurations per rule to get the frontier.
  - *Deterministic transformations create information:* shown at the same scale, since ECA is the deterministic transformation.
  - *Induction:* one mid-size model per condition, about 1e15 to 2e16 FLOPs.
  - *Emergence:* small models (width ≤ 128) with a large depth and loop sweep.
  - *Order dependence of epiplexity:* never shown at small scale. This is an open slot a Colab-scale project could fill, for example forward vs reverse ECA or OWF data with S_T reported.
- **Units in the paper's figures are inconsistent.** The ECA notebook divides K by ln 2 (bits). The hard-induction notebook plots raw K_req (nats). Fig. 2c is labelled "MB". Hard-induction S_req of 0.45 to 1.0 MB equals 3.6 to 8 Mbit, or 2.5 to 5.5 M nats, which matches the Fig. 5 bars. So "MB" appears to mean megabytes and the Fig. 5 bars are in nats. A small-scale study should state its units explicitly.
- **Anchor magnitudes for small-scale work.** Synthetic S_T values in the paper are about 1e5 to 1e7 bits, with test sets of 1e8 to 2.5e8 label tokens. Scaling 𝒟 down by 100× to 1000× (e.g. MNIST-sized 1e4 to 1e5 examples) should shrink S by roughly 𝒟^(1−β) (paper eq. 59), so small datasets will have S in the kbit to 100 kbit range. Absolute noise control therefore matters more (see Q5).

### Gaps
- The paper gives no wall-clock times or hardware per experiment beyond acknowledging Google TPU Research Cloud support. Per-run GPU-hours for the synthetic experiments could not be found and are estimated in Q3.
- It is unclear whether the ECA Fig. 3 frontier points at the lowest compute come from the smallest models or from early checkpoints of larger ones. The raw wandb runs (project "shikai/requential") are not public in the repo.

## Q2. How is the compute bound T chosen and swept, how many training runs does one epiplexity value need, and what is the minimal protocol that preserves the definition?

### Takeaway
T is a FLOP budget covering both training (decoding the model) and evaluation on the test set: T = 6ND + 2N𝒟. S_T is read off a compute-optimal Pareto frontier. To build it, the paper sweeps model size and shape, and treats every logged checkpoint of every run (constant LR plus EMA) as a candidate (N, D) point. A full S_T(X) curve used about 24 to 30 runs per dataset for ECA and about 48 configurations per dataset for natural data. A single model per condition was used only when training converged to a known theoretical floor (the induction tasks). In that case the paper argues that S_T stabilizes, so no frontier is needed.

### Cited Findings
- **Time bound.** "training a model with N parameters on D tokens takes time approximately 6ND … while evaluating it on X takes time 2N𝒟." The two-part code "runs in time 6ND+2N𝒟". Hyperparameters and the N/D trade-off are optimized "subject to the time bound 6ND+2N𝒟 ≤ T." — [Paper §4, §4.1](https://arxiv.org/html/2601.03220v2)
- **Frontier construction.** "we first identify a good learning rate for a small model size and use … µP … and CompleteP … We then train models of various depths and widths to simultaneously sweep over model size and width-depth ratios, for a total number of training tokens chosen to be larger than the test dataset size 𝒟 … we record an EMA of the iterates … under a constant learning rate schedule … Each training run traces a curve in the |P|+E[log 1/P(X)] vs T plane … The Pareto frontier of all such curves yields the optimal hyperparameters." — [Paper App. B.1](https://arxiv.org/html/2601.03220v2)
- **Frontier smoothing.** The paper uses "the lower convex hull of the resulting curves" and "retain[s] only the median point (ordered by compute) per training run … on the lower convex hull". Without this, finite sweeps produce "noisy, oscillatory trends in the estimated epiplexity" (Fig. 10). — [Paper App. B.1](https://arxiv.org/html/2601.03220v2)
- **Implementation [VERIFIED-code].**
  - `compute_lower_convex_hull(..., reduced=True, tol=2e-2)` in the notebook implements the hull and the median point per run.
  - Requential: K(M) = K_req/ln 2; K(X|M) = student_loss × test_tokens/ln 2; total compute = 6·P·student_tokens + 2·P·test_tokens.
  - Prequential: K(M) = K_auc/ln 2 and K(X|M) = train_loss × test_tokens/ln 2.
  - [notebooks/eca_3rules.ipynb](https://github.com/shikaiqiu/epiplexity/blob/main/notebooks/eca_3rules.ipynb)
- **Run counts [VERIFIED-code].**
  - ECA-3-rules script: 5 depths × 6 widths = 30 configs per rule. The notebook output reports "73 runs loaded" for the 3 rules.
  - eca_rules: 3 × 4 = 12 configs per rule.
  - Induction notebooks: "6 runs loaded" each, i.e. one run per h value.
  - Natural-data sweep (`picodo/sweeps/requential.yaml`): model.N ∈ {3, 6, 12, 24} × model.P ∈ {1, …, 160} (12 values) × 4 datasets, each run to T = 5e9 tokens with seed ∈ {0}.
  - [experiments/](https://github.com/shikaiqiu/epiplexity/tree/main/experiments); [picodo/sweeps/requential.yaml](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/sweeps/requential.yaml)
- **Single-model shortcut.** For easy and hard induction, "the compute budget T and test set size 𝒟 need not be precisely specified … as the epiplexity stabilizes as T and 𝒟 increase due to the convergent training dynamics." The shortcut is justified because the models "converge by the end of training … to the theoretical minimum values." — [Paper App. C.2, C.3, C.7](https://arxiv.org/html/2601.03220v2)
- **Scaling behaviour.**
  - In the large-compute limit, D*(T) → 𝒟 and S_∞(X) = β/(1−β) · D0^β · 𝒟^(1−β).
  - In the small-compute regime, N* ≈ T/𝒟 ("the model size is constrained by the need to evaluate on δ tokens") and S_T ∝ T^(α(1−β)/(β+1)).
  - [Paper App. B.3](https://arxiv.org/html/2601.03220v2)
- **Stated error sources.** The paper lists (1) the convex hull and median-point heuristics, (2) a fixed architecture and optimizer "rather than considering all possible programs", and (3) suboptimal LR and Adam betas. It judges these "sub-leading corrections … unlikely to alter the ordering between datasets if the estimated epiplexity gap is already significant." — [Paper App. B.1 "Sources of errors"](https://arxiv.org/html/2601.03220v2)
- **Practical recommendation.** "we recommend using prequential coding for crudely estimating epiplexity and ranking the epiplexity of different datasets … and requential coding for obtaining the most accurate estimates otherwise." Requential is "typically 2× to 10× slower than prequential coding." — [Paper §4.3](https://arxiv.org/html/2601.03220v2)
- **Lightweight follow-up estimators.**
  - EpiSelect (arXiv 2608.11746) fits power-law scaling laws to per-domain loss curves and differentiates the implied prequential area. Overhead is "roughly 12 sec … per fit", and the fit reaches R² of about 0.9 "within the first 4k steps".
  - EpiGen uses the per-step "difference between the learner loss on the samples in the buffer before and after training on the current batch."
  - Both run at 124M to 1.3B scale. Neither reports MLP or toy experiments or seed-level error bars.
  - [EpiSelect/EpiGen paper](https://arxiv.org/html/2608.11746); [EpiSelect repo](https://github.com/eysu35/EpiSelect); [EpiGen repo](https://github.com/eysu35/EpiGen)
- **Third-party package.** A framework-agnostic "first pass" implementation exists with prequential and requential wrappers for sklearn, Keras and PyTorch. It reports no results or benchmarks. — [RichardScottOZ/Epiplexity](https://github.com/RichardScottOZ/Epiplexity)

### Inferences
- **The minimal protocol that keeps the meaning of S_T (not just "area under one loss curve")** [ESTIMATE/design]:
  1. Fix and report 𝒟, the size of the dataset whose information is measured, plus the time bound T or a small grid of T values.
  2. Run a small width sweep (about 4 to 6 widths, 1 to 2 depths) with µP-style LR transfer, a constant LR and an EMA of the weights. Log the prequential area and the EMA held-out loss at geometrically spaced steps so each run yields a curve of (T, |P|+H) points.
  3. Take the lower convex hull with a median point per run, and report S_T at a few T values.

  This is about 5 to 12 runs per dataset per seed, compared with the paper's 24 to 30.
- **When a single run is enough.** If the task has a known loss floor that the model provably reaches (deterministic targets, Markov chains with known entropy rate, parity), the paper's own induction argument allows one well-tuned model per condition. Report it as "S at convergence" rather than S_T.
- **Ranking-only mode.** If only a ranking is needed, one fixed architecture and one prequential run per dataset (the paper's "crude" mode) is acceptable. It measures |P_preq| at a fixed (N, D), not S_T. Fig. 3 shows why it can mislead: at about 1e12 FLOPs rule 15 ranks above rule 54, so rankings should be checked at 2 to 3 compute levels.
- **Test-set size matters at small scale.** With small 𝒟 the 2N𝒟 evaluation term pushes the optimum toward small N (N* ≈ T/𝒟 in the small-compute regime). Datasets compared against each other must share the same 𝒟 and T grid, otherwise S values are not comparable.
- **Cheap analytic estimator.** Fit L(D) = E + (D0/D)^β to one large-model learning curve and use S ≈ β/(1−β) · D0^β · D^(1−β) (paper eq. 54/59). This is essentially what EpiSelect does, and it could serve as a cross-check at Colab scale. It inherits every error in the power-law fit, especially for plateau-then-drop curves (Q5).

### Gaps
- The paper never reports how much S_T changes with the number or spacing of runs in the sweep (for example 30 vs 10 configurations). This sensitivity is unquantified.
- I found no published small-scale ablation of the convex-hull and median-point heuristic.

## Q3. What does the repo actually require, would it run on a Colab T4/L4/A100, and what must be rewritten for MLPs?

### Takeaway
The synthetic experiments (`experiments/` + `soph/`) are plain PyTorch. They need only torch, numpy, wandb, tqdm, fire, pandas and plum-dispatch, and the models fit easily in 16 GB. I ran `soph.train.train` on CPU with wandb disabled and it works as a library call that returns a DataFrame [VERIFIED-local]. The natural-data code (`picodo/`, JAX) is TPU-scale, with sweeps of 5B tokens per run, and is out of scope for Colab.

The main practical obstacles:
- ECA data generation runs synchronously on CPU in numpy and costs 0.15 to 0.5 s per batch.
- Plotting notebooks pull from a private wandb project.
- The published scripts differ from the paper's stated hyperparameters in several places.
- The prequential area K_auc counts all sequence tokens, including unscored input tokens.

An MLP port needs a new model class, a new synthetic-sampling function for requential coding, and corrected token accounting. The training and coding logic can stay.

### Cited Findings
- **Environment.** README: "conda create -n epi python=3.10 … pip install torch numpy wandb tqdm fire pandas plum-dispatch". The JAX environment is `jax[cuda12] flax optax chex wandb hydra-core …`. The README says: "Set `debug = True` at the top of each script for a quick single-point test run." — [README](https://github.com/shikaiqiu/epiplexity/blob/main/README.md)
- **Local run [VERIFIED-local].**
  - I installed `fire` and `plum-dispatch` and called `soph.train.train(... device='cpu', dtype='float32', wandb_log=False, requential=False)` on ECA rule 54: width 64, 48 steps, B = 384, 1-layer width-32 GPT (0.0125M non-embedding params), T = 200 steps.
  - Wall-clock was 143 s (0.72 s/step on CPU). It returned a DataFrame with iter, tokens, compute, train_loss, K_auc, and so on.
  - Loss fell from 0.693 to about 0.40 nats/token; K_auc was about 8.3e5 nats at step 200; logged compute was 9.8e11 FLOPs.
  - Only deprecation warnings appeared (`torch.cuda.amp.GradScaler`).
  - Code used: [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py), [experiments/ca_shared.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/ca_shared.py)
- **CPU data-generation cost [VERIFIED-local].** `soph.datasets.CA.generate` (numpy, 1000-step burn-in plus 48 steps) took 0.146 s per batch of 384×64 and 0.488 s per batch of 1536×64. Without burn-in, 4 steps on 1536×32 took 0.0016 s. `get_batch` is called inside the training loop, so this cost is paid every step. — [soph/datasets/CA.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/datasets/CA.py), [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py)
- **Precision and multi-GPU handling [VERIFIED-code].**
  - `dtype='auto'` picks bfloat16 only if the GPU name contains "a100" or "h100". Otherwise it uses float16 with GradScaler, which is the path on a T4 or L4.
  - `dispatch_multigpu` starts one worker per visible GPU, so on a single Colab GPU the grid runs sequentially.
  - [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py), [soph/utils/executor.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/utils/executor.py)
- **Logged quantities [VERIFIED-code].**
  - K_auc = `np.trapz(train_loss_history − current_train_loss, tokens)`. Here `train_loss` is the pre-update loss on fresh batches, averaged over each logging interval, and `tokens = step × A·B·S`, where S = full sequence length.
  - K_req accumulates `kl.mean() × token_count` over *masked (label) tokens* only.
  - Hidden-matrix learning rates are `learning_rate / fan_in` (a µP-style rule); embeddings and LayerNorm use `vec_lr = 6e-4`; AdamW with eps = 1e-20.
  - Requential training uses an EMA teacher for synthetic targets and freezes the teacher when the student's KL exceeds `max_kl`.
  - [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py), [soph/model.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/model.py)
- **Token-accounting mismatch [VERIFIED-code].** For the ECA and hard-induction tasks the loss is a mean over the Y half only (`predict_half_forward` / loss_mask), but `tokens_per_iter = A*B*S` counts all 2×width tokens. K_auc is therefore about 2× the area per label token, while K_req counts label tokens only. — [experiments/ca_shared.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/ca_shared.py), [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py)
- **Scripts vs paper discrepancies [VERIFIED-code].**
  - `eca_3rules.py` uses B = 384 and tag 'arxiv_3rules'. The paper's App. C.1 says batch 1536, and the notebook filters on tag 'test'.
  - `eca_rules.py` sets `'requential': False` and ECA steps = [128], yet Fig. 2c plots K_req, and App. C.7 says the other hyperparameters are identical to C.1 (48 steps).
  - [experiments/eca_3rules.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_3rules.py), [experiments/eca_rules.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/eca_rules.py), [Paper App. C.1/C.7](https://arxiv.org/html/2601.03220v2)
- **Figures depend on private logs.** Every notebook calls `wandb.Api().runs("shikai/requential", …)`, so figures cannot be re-plotted without re-running the experiments. — [notebooks/](https://github.com/shikaiqiu/epiplexity/tree/main/notebooks)
- **Natural-data scale.** The sweep uses up to 160M-parameter models, 5B tokens per run, sequence length 512, and requential training on all 4 datasets. — [picodo/sweeps/requential.yaml](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/sweeps/requential.yaml)
- **GPU and Colab specs.**
  - T4: "65 FP16 TFLOPS" mixed precision, 16 GB GDDR6, 300 GB/s. — [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
  - Colab free notebooks "can run for at most 12 hours"; Pro+ allows up to 24 h "if you have sufficient compute units". GPU type is not guaranteed. — [Hivenet summary of the Colab FAQ (secondary)](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations)

### Inferences
- **Per-run compute from the configs [ESTIMATE].**
  - *Hard induction:* about 2.4M params × 1536 × 64 tokens × 20k steps ≈ 2.8e16 FLOPs for the teacher (the figure's x-axis ends at about 2e16). At an assumed 10 to 20 TFLOP/s sustained on a T4 (15 to 30% of the fp16 peak; not measured), that is about 25 to 50 min of teacher-only GPU time. Requential training would take about 1 to 8 h, using the paper's 2 to 10× requential overhead. Numpy data generation with burn-in (about 0.25 s/step, extrapolated from my measurement) adds about 1.4 h of CPU time.
  - *Easy induction:* about 0.6M params × 384 × 512 × 3000 ≈ 2e15 FLOPs, i.e. minutes of GPU time per run.
  - *ECA-3-rules grid, one rule:* the sum over 30 configs is about 12·ΣL·ΣD² ≈ 9.2e7 params, times 6 × 4.9e8 tokens ≈ 2.7e17 FLOPs (teacher only). All 3 rules are about 8e17 FLOPs, or roughly 2e18 with requential. That is about 1.5 days on a T4 at the assumed rate. The numpy data pipeline adds 0.146 s × 10k steps × 90 runs ≈ 37 h of CPU time.

  Conclusion: a full Fig. 3 reproduction takes about 2 to 4 Colab T4 days. An A100 cuts the GPU part about 5 to 10× but not the CPU data part.
- **Colab-scale subset [ESTIMATE].** Widths {16, 32, 64, 128} × depths {1, 2}, 3 rules, prequential only, with GPU-side data generation. Each run is ≤ 1e15 FLOPs, so wall-clock is dominated by per-step overhead. That is roughly minutes per run and under 1 to 2 h per rule including 3 seeds. The paper's Fig. 3 frontier already shows the 54 > 15 > 30 ordering by about 1e13 to 1e14 FLOPs.
- **Rewrites needed for an MLP port:**
  1. *Model:* an MLP class exposing `forward(inputs, targets, loss_mask) → (logits, loss)` and `configure_optimizers` with µP-style `lr/fan_in` for hidden matrices, to keep LR transfer across widths.
  2. *Output factorization:* for Y|X tasks such as ECA, parity or labels, use independent Bernoulli or categorical heads. For deterministic Y the optimum needs no output correlations. For generative S_T(X) of images, an autoregressive MLP (MADE-style) or per-pixel conditional model is needed. The "tokens" unit then becomes output units.
  3. *Requential sampling:* replace `generate_synthetic` (autoregressive KV-cache sampling) with a single forward pass plus sampling of each output. KL(teacher‖student) is then closed-form per output. This should bring the requential overhead close to 2× rather than the paper's 2 to 10× for autoregressive transformers.
  4. *Token accounting:* compute K_auc over scored label units only.
  5. *Data:* move ECA/GoL generation to the GPU (`torch.roll` plus a lookup table), or pre-generate a large pool.
  6. *FLOPs:* 6N per training example and 2N per evaluation example remain valid for dense MLPs.
  7. *Logging:* call `train(..., wandb_log=False)` and keep the returned DataFrame; reuse the notebook's hull code offline.
- **Token-count effect on Fig. 2c.** Because of the token-count mismatch, part of the "prequential is several times larger than requential" gap for ECA and induction in Fig. 2c may be a factor of about 2 from counting input tokens. This comes from reading the code only and is not confirmed with the authors. Recompute both on the same token basis before comparing.

### Gaps
- No Colab measurements were possible here (the machine has no GPU). All GPU throughput numbers above are assumptions, and step-overhead timings for tiny models on a T4/L4 remain to be measured.
- I did not run the requential path locally, so its real overhead for tiny models is unmeasured.
- I did not test `picodo/` (JAX); its GPU memory needs were not checked.

## Q4. Candidate small testbeds: what is known about learnability and sample complexity, and how good is each as an epiplexity testbed?

### Takeaway
The best Colab testbeds have (a) a tunable structure knob, (b) a known entropy or loss floor, so the "final loss" baseline and H_T can be checked, (c) cheap exact data generation, and (d) known learnability for MLPs.

| Rank | Testbed | Why |
|---|---|---|
| Strong | ECA (rules 15, 30, 54, 110) | Paper baseline; the MLP version is untested |
| Strong | Markov chains / n-gram sources with known entropy rate, including hidden-row induction | Known floor; paper's easy induction |
| Strong | Sparse parity (n, k) | Known n^O(k) SGD time; plateau-then-drop |
| Strong | Random Hierarchy Model | Known sample complexity; built for CNNs/MLPs |
| Strong | Random vs true labels and shuffled vs random pixels on MNIST/CIFAR-10 | Epiplexity analogue of the Zhang et al. 2017 randomization tests |
| Middle | Game of Life | Learnable only with heavy overparameterization and seed luck |
| Middle | Grokking / modular arithmetic | Tiny finite dataset; train and test diverge by design |
| Middle | LCG vs cryptographic PRNG | Clear theory; learning even LCGs needed large transformers |
| Middle | Teacher-student MLPs | Clean theory but little "structure" beyond the teacher |

### Cited Findings
- **ECA (paper).** Class II rule 15 → low S_T, low H_T; class III rule 30 → high H_T, no S_T; class IV rule 54 → high S_T (Fig. 3). ECA data from class IV rules also transferred best downstream in Zhang et al. 2024, where GPT-2 models of 85M ("Small") and 708M ("Large") parameters were trained on ECA windows of 60 time steps × 100 cells on 12 H100 GPUs. "Class IV rules especially outperform the other classes on the chess move prediction task." — [Paper §5.1, §6](https://arxiv.org/html/2601.03220v2); [Zhang et al. 2024, Intelligence at the Edge of Chaos](https://arxiv.org/abs/2410.02536)
- **Game of Life (Springer & Kenyon).**
  - A minimal CNN for n-step Life has "2n+1 layer[s]" and "23n+2 trainable parameters"; one step needs 25 weights.
  - Success at minimal size for 1-step Life was 4.7%. Over 50% success needed ≥ 3× overcompleteness for 1 step and 4× for 2 steps; 3 to 5 steps needed more than the maximum 24× tested.
  - Training data: "1 million randomly generated training examples … 100 epochs of 10,000 … batch size of 8", 32×32 boards. Initial density d ≈ 0.38 helped.
  - "4-6 sign perturbations" of the initial weights reduced success below 50%.
  - [It's Hard for Neural Networks to Learn the Game of Life](https://arxiv.org/abs/2009.01398)
- **Sparse parity (Barak et al.).**
  - SGD on 2-layer MLPs (widths 10, 100, 1000) learns (n, k)-parity with n ∈ {10, 20, 30}, k ∈ {2, 3, 4} in "c·n^(αk)" steps, with c·n^(αk) ≤ 1e5 iterations.
  - Loss curves show a "long plateau followed by sharp phase transition"; for disjoint-PolyNets the error stays ≥ 49% for a "1−o(1) fraction" of the time before convergence.
  - Success was reported as "at least 20% of 25 random trials" per configuration. No wall-clock numbers are given.
  - [Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit](https://arxiv.org/abs/2207.08799)
- **Random Hierarchy Model (Cagnetta et al.).**
  - Parameters v, n_c, m, s, L, with input dimension d = s^L.
  - Deep CNNs need P* ≃ n_c m^L samples (polynomial in d). Two-layer fully connected networks need about n_c m^((d−1)/(s−1)), exponential in d.
  - Architectures tested were CNNs and MLPs, not transformers.
  - [How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model](https://arxiv.org/abs/2307.02129)
- **Randomization tests (Zhang et al. 2017).**
  - Variants: true labels, partially corrupted, random labels, shuffled pixels (one fixed permutation for all images), random pixels (a different permutation per image), and Gaussian.
  - "training time increases only by a small constant factor compared with training on the true labels."
  - CIFAR-10 MLP 1x512 (1,209,866 params): train 100.0% / test 50.51% on true labels, and 99.34% / 10.61% on random labels.
  - CIFAR-10 MLP 3x512 (1,735,178 params): 100.0% / 52.39% on true labels, and 100.0% / 10.48% on random labels.
  - Fitting random labels requires multiple passes: "the network starts fitting after going through the training set multiple times."
  - [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)
- **Markov chains / statistical induction heads (Edelman et al.).** In-context Markov chains with k = 2 or 3 states, transition matrices drawn from a Dirichlet prior, and a 2-layer attention-only transformer. Learning goes through uniform, then unigram, then bigram stages with "long plateaus and sudden drops"; "a one-layer transformer fails". The paper's easy-induction task is a modified version with V = 8 and a known theoretical minimum loss. — [The Evolution of Statistical Induction Heads](https://arxiv.org/abs/2402.11004); [Paper §5.3.1](https://arxiv.org/html/2601.03220v2)
- **Modular arithmetic / grokking.**
  - Nanda et al.: P = 113, 30% of the 12,769 pairs for training, a 1-layer ReLU transformer (d = 128, 4 heads, MLP 512), AdamW with lr 1e-3 and weight decay 1, full batch, 40,000 epochs, 5 seeds. Test accuracy rises after about 10,000 epochs.
  - Gromov: a 2-layer MLP without biases (3Np parameters, quadratic activation) groks modular addition (for example p = 97). There is a critical training fraction α_c, and AdamW and weight decay make grokking faster and more data-efficient.
  - [Progress Measures for Grokking](https://arxiv.org/abs/2301.05217); [Grokking Modular Arithmetic (Gromov)](https://arxiv.org/abs/2301.02679)
- **PRNGs.**
  - Theory: CSPRNG outputs have nearly maximal time-bounded entropy and nearly constant epiplexity (Theorem 9). JAX's generator is Threefish/Threefry encryption of counters (§2.1). — [Paper §3](https://arxiv.org/html/2601.03220v2)
  - Empirically, transformers can learn LCGs in context. With a fixed modulus m = 2048, "only one layer and one attention head" (d_model 768) suffices, trained on 100,000 sequences. Unseen moduli need at least 3 layers, and scaling experiments used 6 layers with d_model 1024. m = 2^32 took "Four NVIDIA A100 GPUs … 21.82 hours". Training runs were 100k to 200k steps, and loss curves show copying first and later "grokking". — [(How) Can Transformers Predict Pseudo-Random Numbers?](https://arxiv.org/abs/2502.10390)
- **Teacher-student MLPs (Goldt et al.).** Online SGD dynamics of two-layer student/teacher networks are captured by ODEs that are "asymptotically exact" in high input dimension. Generalization with over-parameterized students (K > M) depends on which layers are trained. — [Dynamics of SGD for two-layer networks in the teacher-student setup](https://arxiv.org/abs/1906.08632)
- **MNIST-1D (small-scale benchmark).**
  - 4,000/1,000 train/test examples of length 40.
  - Test accuracy: logistic regression 32±1%, MLP 68±2%, CNN 94±2%, GRU 91±2%.
  - "less than a minute of walltime" to train a strong classifier. It "differentiates more clearly between linear, nonlinear, and convolutional models" than MNIST.
  - [Scaling Down Deep Learning with MNIST-1D](https://arxiv.org/abs/2011.14439)

### Inferences
- **ECA with MLPs (best first target).**
  - Y = ECA^t(X) on w cells, predicted with a factorized-Bernoulli MLP.
  - H(Y|X) = 0, since Y is deterministic given X. Any residual loss is time-bounded entropy, which gives a clean check.
  - The rule class is the structure knob; steps t and width w are the difficulty knobs.
  - Risk: MLPs lack the transformer's locality and weight sharing, so rule 54 may look closer to rule 30 than it does for GPTs. A 1-D CNN (circular padding) is a natural second architecture that shows the observer dependence.
- **Markov / n-gram sources with a known entropy rate.** The ideal calibration testbed. The exact entropy rate gives the true H floor, so estimator bias in the "final loss" can be measured directly. Hidden-row variants reproduce the paper's easy-induction result, where epiplexity is non-monotone in h. An MLP over a fixed context window (n-gram MLP) works for a fixed order, but in-context induction needs a sequence model.
- **Sparse parity.**
  - Strengths: a clean contrast between k = 1 to 2 (easy) and larger k (hard). Planted-structure epiplexity is small in K-complexity terms (the subset S costs about k·log2 n bits) but the learned network may be large.
  - Weaknesses: the plateau-then-drop dynamics make the prequential area highly sensitive to *when* the drop happens (seed variance; only ≥ 20% of 25 trials succeed in some settings). Truncation also matters: in the plateau phase S ≈ 0 even though the structure is learnable.
  - Good for testing estimator variance, not for a first demo.
- **Random Hierarchy Model.** Arguably the best "natural-data-like" synthetic source for MLPs vs CNNs. Structure is tunable through m and L, sample complexity is known and differs between architectures, and it links epiplexity to the observer (CNN vs MLP). It needs a custom generator.
- **Random vs true labels (MNIST / CIFAR-10 / MNIST-1D).**
  - The prediction is S(true) > 0 and S(random) ≈ 0 when the baseline is held-out loss in a single pass.
  - Pitfall: in multi-epoch training with a *train*-loss baseline, memorized random labels would give spuriously large "epiplexity" (up to 𝒟·log2 10 ≈ 199 kbit for 60k labels; see Q5).
  - Magnitudes are small. For MNIST labels given images, the Blier & Ollivier prequential total is 4.10 kbits, which upper-bounds S_preq, versus the paper's ECA S of about 5e6 bits. The signal is real but needs careful noise control.
  - Partial label corruption p ∈ [0, 1] gives a dose-response knob.
- **Shuffled vs random pixels.**
  - For an MLP with i.i.d. initialization, one fixed pixel permutation is an exact symmetry of the architecture, so the epiplexity distribution should be unchanged.
  - For a CNN it destroys locality, so S_T should drop and/or shift to higher compute.
  - A different random permutation per image destroys spatial structure for every architecture.
  - This is a clean small-scale demonstration of observer dependence (my inference; not tested in any source found).
- **Game of Life.** A rich emergence story (the paper's §5.3.2 and App. E motivation), but the learnability literature shows extreme seed and initialization sensitivity and a need for 3× to more than 24× overparameterization for multi-step prediction. Expect high variance in S. Use 1 to 2 steps, a CNN observer, and many seeds, or treat it as a stretch goal.
- **Modular arithmetic.**
  - The dataset is tiny and finite (p² examples, e.g. 12,769). Epiplexity is capped by the table's entropy of p² · log2 p ≈ 87 kbit for p = 113.
  - Grokking relies on multi-epoch training, so the paper's one-pass/fresh-data estimator does not apply directly. A block-prequential (Blier & Ollivier) scheme is required, and train and test losses diverge by construction.
  - Interesting but methodologically hard; not a first testbed.
- **PRNG outputs.**
  - Theory predicts S ≈ 0 for a CSPRNG (e.g. counter-mode Threefry or ChaCha) and positive S for weak generators.
  - Background knowledge, not verified in this session: the low-order bits of power-of-two-modulus LCGs have short periods, so predicting the low bits is easy while the high bits are hard. That gives a within-generator structure knob.
  - The empirical LCG-learning results needed transformers and A100-hours, so small MLPs will likely see only the easy low bits. This makes a good "trivial vs pseudorandom" pair but a weak "rich structure" example.
- **Teacher-student MLPs with a planted teacher.** Useful for verifying that S grows with teacher width M and saturates, since structure is exactly controlled. But the structure is "one smooth function" with no emergent sub-circuits, so it tests the estimator more than the paper's thesis.

### Gaps
- I found no published work that estimates epiplexity (or the paper's prequential/requential S_T) with MLPs or CNNs on any of these testbeds. A search for "epiplexity MNIST MLP" returned nothing relevant.
- I found no sample-complexity results for MLPs (as opposed to transformers or CNNs) on ECA prediction tasks.
- I did not locate quantitative shuffled-pixel test accuracies from Zhang et al. 2017.

## Q5. Statistical issues at small scale: seeds, optimizer sensitivity, truncation, the final-loss baseline, units, and error bars

### Takeaway
The paper reports no seeds, error bars or variance for any epiplexity estimate. Its synthetic figures are single-seed per configuration; only the entropy-only OWF experiment uses 2 seeds. At small scale the dominant risks are:
1. **Baseline noise amplified by D.** Noise in the final-loss baseline is multiplied by the number of training tokens D. For example, 0.001 nats/token × 2.5e8 tokens ≈ 3.6e5 bits, which is larger than rule 15's entire S_T.
2. **Memorization under finite data.** With multi-epoch training, a train-loss baseline turns memorized noise into fake epiplexity.
3. **Plateau-then-drop dynamics.** These make the prequential area depend on seed and truncation point.
4. **Hyperparameter dependence** (LR, batch size, EMA).

Related MDL literature suggests code lengths are more seed-stable than accuracy, but uses 4 to 8 seeds.

### Cited Findings
- **No error bars in the paper.** Searching the paper text for "error bar", "standard deviation", "confidence interval", "seeds" and "variance" finds no discussion of estimator variance. The repo's training function fixes `seed = 1337 + seed_offset`. Only `soi.py` sweeps seeds ([68, 37]). The natural-data sweep has `seed: values: [0]`. — [Paper full text](https://arxiv.org/html/2601.03220v2); [soph/train.py](https://github.com/shikaiqiu/epiplexity/blob/main/soph/train.py); [experiments/soi.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/soi.py); [picodo/sweeps/requential.yaml](https://github.com/shikaiqiu/epiplexity/blob/main/picodo/sweeps/requential.yaml)
- **Final-loss baseline.** The paper assumes one-epoch, i.i.d. training and estimates Σ log 1/P_M(Z_i) as M·log 1/P_M(Z_M), "a rescaled loss for P_M on unseen data". For the ADO experiment it computes the sum exactly because i.i.d. breaks. It warns: "In cases where train and test diverge, such as when there is overfitting, this difference could become important." — [Paper §4.1, App. B.1](https://arxiv.org/html/2601.03220v2)
- **Rigor caveats of prequential.** "both L(Z,P_M) and L(Z|P_M) can only upper-bound the respective Kolmogorov complexities, and thus their difference does not yield an upper bound for K(P_M)", and the runtime of the implied program is not guaranteed. Prequential "can be viewed as an approximation of requential coding with a static teacher", so "we expect the prequential estimate to be an overestimate." — [Paper §4.1, App. B.2](https://arxiv.org/html/2601.03220v2)
- **Requential tuning.** Two interventions reduce the requential code length: "distilling from an exponential moving average (EMA) of teacher checkpoints" and "imposing a maximum KL threshold between teacher and student". "The EMA time scale and the maximum KL threshold are additional hyperparameters." The max-KL values used were 0.03 (hard induction), 0.1 (natural), and none for ECA; the repo's easy-induction script uses 0.005. — [Paper App. B.1, C.1–C.4](https://arxiv.org/html/2601.03220v2); [experiments/induction_easy.py](https://github.com/shikaiqiu/epiplexity/blob/main/experiments/induction_easy.py)
- **Optimizer dependence.** The estimate depends on "using a fixed architecture … and learning algorithm (e.g., requential training with Adam)" and on "suboptimality of other hyperparameters, such as the learning rate, Adam (β1, β2)". The authors expect "only sub-leading corrections". — [Paper App. B.1](https://arxiv.org/html/2601.03220v2)
- **Truncation and plotting.** In the ECA notebook, a frontier that stops before the maximum compute is extended flat to `global_max_x` (`pareto_y.append(pareto_y[-1])`). — [notebooks/eca_3rules.ipynb](https://github.com/shikaiqiu/epiplexity/blob/main/notebooks/eca_3rules.ipynb)
- **Units [VERIFIED-code].** The repo logs per-token losses in nats. The README says "K(X) … (Mbits)" for picodo. The ECA notebook converts K_req/ln 2 to bits. The hard-induction notebook plots `K_req/1e6` without converting. Fig. 2c's axes are labelled "MB". — [README](https://github.com/shikaiqiu/epiplexity/blob/main/README.md); [notebooks](https://github.com/shikaiqiu/epiplexity/tree/main/notebooks); [Fig. 2c](https://arxiv.org/html/2601.03220v2/req_vs_preq_scatter.svg)
- **Seed stability in MDL probing (Voita & Titov).** "using accuracy can lead to different rankings of layers depending on a random seed," whereas "the MDL results are stable and the scores given to different layers are well separated" (seeds 0 to 4). Control tasks with random labels had "codelengths … substantially larger than for the linguistic task (at least twice larger)." — [Information-Theoretic Probing with MDL](https://arxiv.org/abs/2003.12298)
- **Seeds and dataset-size dependence (Whitney et al.).** 8 seeds per representation per dataset size for MNIST, and 4 for part-of-speech. "if H(Y|φ(X)) > 0, MDL grows without bound as the size of the evaluation dataset n increases." SDL instead subtracts a fixed baseline ε: m_SDL = Σ_i [L(A,i) − ε]_+. — [Evaluating representations by the complexity of learning low-loss predictors](https://arxiv.org/abs/2009.07368)
- **Early-block overfitting (Blier & Ollivier).** "Large architectures might overfit during the first steps of the prequential encoding … the encoding cost of the first packs of data might be worse than with the uniform code." They address this with "switching" between architectures and "self-switching" between stopping times. — [The Description Length of Deep Learning Models](https://arxiv.org/abs/1802.07044)
- **Plateau and seed variance on hard tasks.** Sparse parity shows long plateaus and sharp transitions, with success in "at least 20% of 25 random trials" [Barak et al.](https://arxiv.org/abs/2207.08799). Game of Life minimal networks fail about 20% of the time after "a 1-sign perturbation" of the initial weights [Springer & Kenyon](https://arxiv.org/abs/2009.01398).
- **The EpiSelect paper's floor.** It defines the prequential area relative to "the floor set by the best model obtainable under the compute budget", and reports scaling-law fit R² of about 0.88 to 0.9. It gives no seed-level uncertainty. — [arXiv 2608.11746](https://arxiv.org/html/2608.11746)

### Inferences
- **Baseline noise is the number-one small-scale error source** [ESTIMATE, arithmetic]. With S = Σ(L_i − L_final) over D units, any bias δ in L_final shifts S by D·δ. For the paper's ECA (about 2.5e8 label tokens per run), δ = 0.001 nats gives about 3.6e5 bits, comparable to rule 15's S_T of about 0.2e6 bits.

  Mitigations:
  - (a) Evaluate the EMA model's final loss on a large held-out set, sized to make the standard error ≪ S/D.
  - (b) Where the true floor is known (deterministic Y, Markov entropy rate), report S with both the empirical and the theoretical floor.
  - (c) Integrate only over label units, with consistent token counts.
- **Finite datasets (MNIST, CIFAR-10, modular arithmetic).**
  - Options: (i) single-pass prequential with D ≤ 𝒟 fresh examples, which is valid but gives small S; or (ii) block-prequential (Blier & Ollivier timesteps 8, 16, …, 32768 for MNIST), retraining on the transmitted prefix. The second is a valid code, with T including all retraining FLOPs.
  - In both cases the "final loss" should be *held-out*, never the training loss. For random labels, a train-loss baseline makes S ≈ 𝒟·log2 C, the opposite of the theory. This is where the method distinguishes itself from the naive "area under the train curve".
- **Seeds and error bars.** Use at least 5 seeds per configuration (following Voita & Titov's 5 and Whitney's 4 to 8). Vary init and data order jointly, and additionally check data-sample seeds. Report the median and IQR (or a bootstrap CI) of S_T at each T, computed per seed after that seed's frontier. Also report the between-dataset difference with a paired bootstrap over seeds. This matters because rankings can flip at low T (Fig. 3).
- **Optimizer sensitivity.** Run S at LR × {0.5, 1, 2} of the tuned value and at 2 batch sizes on one reference dataset. If the ranking survives, keep a fixed recipe across datasets. Use constant LR plus EMA as in the repo, so checkpoints along a run are valid points on the D axis. A decaying schedule makes intermediate checkpoints non-comparable.
- **Truncation.** If a run's loss is still falling at the maximum step, the prequential S at that point is correct for that (N, D), but S_∞ is underestimated. Report whether each frontier point's run has plateaued. Do not extend frontiers flat without flagging it.
- **Units checklist.**
  - Report S_T and H_T in bits (divide nats by ln 2) as totals for the stated 𝒟.
  - Also report per-unit bits (bits per label or output bit) and the ratio S_T/(S_T+H_T).
  - State T in FLOPs using 6N·D_train + 2N·𝒟, with N as total parameters (the repo's "P").

### Gaps
- There is no empirical measurement of seed variance for S_T or S_preq in the paper, follow-ups or third-party code. This must be measured.
- The sensitivity of S_T to learning rate and batch size is not quantified anywhere I found.
- I found no guidance on how to set max-KL or EMA for tiny MLPs.

## Q6. Related small-scale work measuring prequential code length or learning-curve area with MLPs, and their runtimes

### Takeaway
Prequential and online MDL with small MLPs is a mature, cheap technique:
- Blier & Ollivier (2018) coded MNIST and CIFAR-10 labels with MLPs and CNNs.
- Voita & Titov (2020) used 2-layer MLP probes with an 11-block online code.
- Whitney et al. (2020) built JAX tooling that computes full loss-data curves for MLP probes in about 2 minutes on one GPU.

These give ready-made block schedules, seed protocols and magnitudes (kbits for 50k to 60k labels) for a Colab study. None separates "model bits" at a compute-optimal frontier as epiplexity does. The closest precursors the paper itself cites are Zhang et al. 2020 ("information transfer") and Whitney et al.'s SDL.

### Cited Findings
- **Blier & Ollivier 2018.**
  - MNIST (60k) architectures: an "MLP with two hidden layers of dimension 256" and a VGG-like CNN.
  - CIFAR-10 (50k) architectures: an MLP with 512 hidden units, a 5000-unit shallow network, a "tinyCNN", and VGG-like networks.
  - Prequential timesteps: MNIST "8, 16, 32, …, 32768"; CIFAR-10 "10, 20, …, 40960".
  - Training: Adam with lr 0.001 and dropout 0.5. The model is re-estimated at each timestep from the accumulated data, and the first block is sent with the uniform code.
  - Code lengths: MNIST uniform 199 kbits, variational 22.2 kbits, prequential 4.10 kbits. CIFAR-10 uniform 166 kbits, variational 89.0 kbits, prequential 45.3 kbits.
  - No runtime or seed-variance statements.
  - [arXiv 1802.07044](https://arxiv.org/abs/1802.07044)
- **Voita & Titov 2020.** Online code over "0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100 percent of the dataset". Probes include "MLP-2" with hidden size 1000 and no dropout. MDL was stable across seeds 0 to 4, and control tasks gave at least 2× longer codes. Units are kbits. — [arXiv 2003.12298](https://arxiv.org/abs/2003.12298)
- **Whitney et al. 2020.** Surplus description length (SDL) and ε-sample complexity. MNIST probes were "two-hidden-layer MLPs with hidden dimension 512". They used 20 dataset sizes log-uniform in [10, 50000] with 8 seeds each. A JAX package parallelizes probe training "on a single accelerator", producing "loss-data curves in around two minutes on one GPU". Part-of-speech took "about an hour". — [arXiv 2009.07368](https://arxiv.org/abs/2009.07368)
- **Paper's positioning.**
  - Excess entropy (area between block-entropy estimates and the entropy rate) is "an analogous construction to our prequential estimate" but lacks the compute dependence.
  - SDL is "the summed online loss … with either the entropy of the data or a fixed baseline performance subtracted out".
  - "More analogous to the spirit of epiplexity is information transfer from Zhang et al. (2020), which sums a variant of a loss difference, adapted to held out test data and for the classification setting."
  - The paper does not cite Blier & Ollivier or Voita & Titov.
  - [Paper §7](https://arxiv.org/html/2601.03220v2)
- **MNIST-1D.** Designed for small-scale science: "less than a minute of walltime" per strong classifier, and results reproducible "in your browser in a few minutes". — [arXiv 2011.14439](https://arxiv.org/abs/2011.14439)

### Inferences
- **A Colab adaptation of the Whitney-style protocol is cheap** [ESTIMATE]. Take 20 dataset sizes × 8 seeds of 2-layer MLPs on MNIST (about 2 min on one GPU per their report), vmap-parallelized. It yields held-out loss-data curves from which (a) an SDL-like area and (b) a block-prequential code follow. Adding the paper's compute axis (6ND+2N𝒟 over a width sweep) and the held-out final-loss baseline turns it into an S_T estimate for Y|X on MNIST or CIFAR-10 in minutes to an hour on a T4.
- **Expected magnitudes (label-conditional).** For MNIST, S_preq ≤ 4.10 kbits (the Blier & Ollivier prequential total bounds the area above a non-negative final loss). The CIFAR-10 total is 45.3 kbits. These are four to three orders of magnitude below the paper's synthetic S values, which argues for synthetic tasks with large 𝒟 (ECA, Markov, RHM) as primary testbeds. Image classification then serves as the "real data" check.
- **Block schedule as the finite-data answer.** The Blier & Ollivier and Voita & Titov block schedules are the natural answer to "what if the dataset is finite", because retraining on transmitted data is allowed in the code. Their "switching" trick shows that the choice of architecture per block matters at small D. This is the same effect the paper handles by sweeping N on a frontier.

### Gaps
- Blier & Ollivier and Voita & Titov do not report runtimes. Whitney et al.'s "two minutes" is for their JAX package on an unspecified GPU, not a T4.
- I did not retrieve Zhang et al. 2020 ("Measuring information transfer in neural networks"), which the paper calls the closest precursor. Its protocol and runtimes are unverified here.
- I found no small-scale study that compares a compute-optimal-frontier S_T against single-run prequential areas for MLPs.
