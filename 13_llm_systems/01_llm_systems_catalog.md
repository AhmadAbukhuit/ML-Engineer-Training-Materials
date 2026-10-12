# Complete Catalog of Generative AI & LLM Systems Engineering

Large Language Models (LLMs) have evolved beyond traditional statistical NLP into high-throughput systems engineering problems involving memory bandwidth limits, parameter quantization, preference alignment, and hybrid retrieval.

---

## 1. Computational Bottlenecks of Autoregressive LLM Inference

In transformer decoder generation, inference proceeds in two distinct stages:

### 1.1 The Prefill Phase (Compute-Bound)

- The entire prompt sequence $x_{1:T}$ is processed concurrently in a single forward pass.
- Highly parallel matrix-matrix multiplications ($\text{GEMM}$); compute utilization of GPU Tensor Cores approaches peak TFLOPs.

### 1.2 The Decoding Phase (Memory Bandwidth-Bound)

- Tokens are generated sequentially: $x_{T+1}, x_{T+2}, \dots$, one token per forward step.
- Matrix-vector multiplications ($\text{GEMV}$); computational intensity (FLOPs per byte transferred from HBM) is extremely low.
- Without caching past tokens, step $t$ must recalculate Key and Value projections for all prior $t-1$ tokens, scaling generation runtime to quadratic $O(T^2)$.

### 1.3 The Key-Value (KV) Cache

- Caches the Key ($K \in \mathbb{R}^{B \times H \times t \times D}$) and Value ($V \in \mathbb{R}^{B \times H \times t \times D}$) tensors across all layers.
- For each new token at step $t$, computes only the new query $q_t \in \mathbb{R}^{1 \times D}$, appends new $(k_t, v_t)$ to memory, and performs attention across the cached buffer in $O(T)$ time.

---

## 2. Post-Training Quantization (PTQ)

Quantization compresses 16-bit floating point parameters ($\text{FP16} / \text{BF16}$) into lower bit-width representations (e.g. `INT8` or `INT4`), slashing memory bandwidth bottlenecks and VRAM footprint by $2\times$ to $4\times$.

### 2.1 Uniform Integer Quantization (INT8)

Maps continuous real values $x \in [x_{\min}, x_{\max}]$ to discrete integers $q \in [q_{\min}, q_{\max}]$:

$$q = \text{clip}\left( \left\lfloor \frac{x}{S} \right\rceil + Z, \, q_{\min}, \, q_{\max} \right)$$
$$\tilde{x} = S \cdot (q - Z)$$

- **Symmetric Quantization ($Z = 0$)**:
  $$S = \frac{\max(|x|)}{127}$$
  Zero-point is fixed at 0. Optimal for zero-centered weights; eliminates zero-point subtractive arithmetic during matrix multiplication.
- **Asymmetric Quantization**:
  $$S = \frac{x_{\max} - x_{\min}}{255}, \quad Z = \left\lfloor -\frac{x_{\min}}{S} \right\rceil$$
  Preserves non-negative distribution ranges, crucial for activations after ReLU.

### 2.2 NormalFloat4 (NF4 - Dettmers et al., 2023)

- Neural network weights initialized with Gaussian distributions retain near-normal empirical distributions throughout pre-training.
- Standard uniform integer quantization assigns excessive resolution to improbable tail values while wasting resolution near the dense mean.
- **NF4 Formulation**: An information-theoretically optimal non-linear quantile codebook with 16 discrete bins $q_i$, such that each bin holds equal probability mass under a standard normal distribution $\mathcal{N}(0, 1)$.

---

## 3. Preference Alignment: Direct Preference Optimization (DPO)

RLHF historically required a multi-stage pipeline: training a supervised policy, training an auxiliary reward model $r_\psi(x, y)$, and optimizing the policy via PPO with high training instability.

### 3.1 The DPO Mathematical Formulation (Rafailov et al., 2023)

By parameterizing the optimal reward function directly in terms of the underlying language model policy:
$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$
DPO substitutes this closed-form solution into the Bradley-Terry preference likelihood:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

- $y_w$: Preferred (winning) completion.
- $y_l$: Dispreferred (losing) completion.
- $\beta$: KL penalty temperature controlling trust-region deviation from reference model $\pi_{\text{ref}}$.
- **Benefits**: Completely eliminates the reward model network, the PPO actor-critic loop, and the complex training dynamics of value estimation.

---

## 4. Advanced RAG: Hybrid Retrieval & Reciprocal Rank Fusion

Single-modality vector search fails on keyword matching, part numbers, and exact technical terminology. Hybrid retrieval fuses sparse lexical inverted indexes with dense semantic embeddings.

### 4.1 Reciprocal Rank Fusion (RRF - Cormack et al., 2009)

Instead of normalizing and adding disparate raw distance scores (which have different scales and distributions), RRF merges results based exclusively on relative **rank orders**:

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

- $M$: Set of ranking algorithms (e.g., BM25 lexical ranker + Dense cosine vector ranker).
- $r_m(d)$: Position rank of document $d$ in the output of retrieval algorithm $m$ ($1, 2, \dots$).
- $k$: Smoothing constant (typically $k = 60$) ensuring top-ranked documents receive high weight without letting outliers dominate.
