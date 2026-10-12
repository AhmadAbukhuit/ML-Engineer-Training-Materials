# Complete Catalog of Transfer Learning Models

Transfer Learning is the paradigm wherein knowledge acquired from solving a source task $\mathcal{T}_S$ on a source domain $\mathcal{D}_S = \{\mathcal{X}_S, P(X_S)\}$ is leveraged and transferred to improve learning of a predictive function $f_T(\cdot)$ on a target task $\mathcal{T}_T$ within a target domain $\mathcal{D}_T = \{\mathcal{X}_T, P(X_T)\}$.

$$\min_{\boldsymbol{\theta}} \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}_T} \left[ \mathcal{L}_T(f(\mathbf{x}; \boldsymbol{\theta}_0 + \Delta \boldsymbol{\theta}), y) \right] + \Omega(\Delta \boldsymbol{\theta})$$

---

## 1. Taxonomy & Core Mechanics of Transfer Learning

| Setting | Domain Relationship | Task Relationship | Common Techniques |
| :--- | :--- | :--- | :--- |
| **Inductive Transfer** | $\mathcal{D}_S = \mathcal{D}_T$ or $\mathcal{D}_S \ne \mathcal{D}_T$ | $\mathcal{T}_S \ne \mathcal{T}_T$ (target labels available) | Feature extraction, Fine-tuning, PEFT |
| **Transductive Transfer** | $\mathcal{D}_S \ne \mathcal{D}_T$ (covariate shift) | $\mathcal{T}_S = \mathcal{T}_T$ (target labels unavailable) | Domain Adaptation, Maximum Mean Discrepancy |
| **Unsupervised Transfer** | $\mathcal{D}_S \ne \mathcal{D}_T$ | $\mathcal{T}_S \ne \mathcal{T}_T$ (no labels in either domain) | Self-supervised cross-domain alignment |

---

## 2. Feature Extraction vs Full Fine-Tuning

### 2.1 Linear Probing (Feature Extraction)

- **Mechanics**: The pretrained backbone weights $\boldsymbol{\theta}_{\text{backbone}}$ are completely frozen ($\nabla_{\boldsymbol{\theta}} \mathcal{L} = \mathbf{0}$).
- A new task-specific classification head $g_\phi(\mathbf{h}) = \mathbf{W}_h \mathbf{h} + \mathbf{b}_h$ is attached and trained on target representations $\mathbf{h} = f(\mathbf{x})$.
- **Advantages**: Minimal compute, zero risk of catastrophic forgetting of base representations, convex optimization when using linear/logistic models.
- **Disadvantages**: Suboptimal when target domain distribution differs substantially from source domain.

### 2.2 End-to-End Fine-Tuning

- The entire network (backbone + head) is updated with target gradients.
- **Catastrophic Forgetting Risk**: High learning rates overwrite general semantic features learned during large-scale pretraining.
- **Mitigations**:
  - **Gradual Unfreezing (Howard & Ruder - ULMFiT)**: Freeze all layers initially; train head for 1 epoch; unfreeze top layer; gradually unfreeze earlier layers backward.
  - **Discriminative Learning Rates (Layer-wise Learning Rate Decay / LLRD)**: Apply geometrically decaying learning rates to deeper/earlier layers:
    $$\eta_l = \eta \cdot \lambda^{L - l}, \quad \lambda \approx 0.85$$
    Early generic feature detectors (edges, textures) update minimally; higher semantic layers adapt aggressively.
  - **Weight Decay toward Initialization ($L_2$-SP)**: Penalizes divergence from pretrained weights $\boldsymbol{\theta}_0$:
    $$\Omega(\boldsymbol{\theta}) = \frac{\alpha}{2} \|\boldsymbol{\theta} - \boldsymbol{\theta}_0\|_2^2$$

---

## 3. Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of modern multi-billion parameter foundation models (LLMs, Vision Transformers) requires prohibitive GPU memory and creates massive storage overhead per downstream task. PEFT methods freeze $\approx 99\%$ of pretrained weights and train a tiny fraction ($< 1\%$) of parameters.

### 3.1 Low-Rank Adaptation (LoRA - Hu et al., 2021)

- **Core Hypothesis**: The weight updates $\Delta \mathbf{W}$ during task adaptation have a low "intrinsic rank" $r \ll \min(d_{\text{in}}, d_{\text{out}})$.
- **Parameterization**:
  Decomposes the weight update into two low-rank matrices $\mathbf{A} \in \mathbb{R}^{d_{\text{in}} \times r}$ and $\mathbf{B} \in \mathbb{R}^{r \times d_{\text{out}}}$:
  $$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{A} \mathbf{B})$$
  - $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ remains frozen during training.
  - $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$ is initialized with Gaussian noise.
  - $\mathbf{B} = \mathbf{0}$ is initialized to zero, ensuring $\Delta \mathbf{W} = \mathbf{0}$ at step 0.
  - Scaling factor $\frac{\alpha}{r}$ prevents hyperparameter re-tuning when varying rank $r$.
- **Zero-Latency Inference (Weight Merging)**:
  During deployment, $\Delta \mathbf{W} = \frac{\alpha}{r} \mathbf{A} \mathbf{B}$ is added directly to $\mathbf{W}_0$ offline:
  $$\mathbf{W}_{\text{merged}} = \mathbf{W}_0 + \Delta \mathbf{W}$$
  Zero extra inference latency or memory overhead compared to the original foundation model.

### 3.2 QLoRA (Quantized LoRA - Dettmers et al., 2023)

- Backbones quantized to 4-bit NormalFloat (NF4), an information-theoretically optimal quantile distribution for normally distributed weights.
- **Double Quantization (DQ)**: Quantizes the quantization constants, saving 0.37 bits per parameter.
- **Paged Optimizers**: Manages CUDA memory spikes by paging optimizer states across CPU and GPU RAM.
- Enables fine-tuning a 65B parameter LLM on a single consumer 48GB GPU without performance degradation.

### 3.3 Bottleneck Adapters (Houlsby et al. & Pfeiffer et al.)

- Inserts tiny feed-forward bottleneck modules after multi-head attention and feed-forward layers:
  $$\text{Adapter}(\mathbf{h}) = \sigma(\mathbf{h} \mathbf{W}_{\text{down}}) \mathbf{W}_{\text{up}} + \mathbf{h}$$
  where $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times m}$, $\mathbf{W}_{\text{up}} \in \mathbb{R}^{m \times d}$, $m \ll d$.

### 3.4 Prefix Tuning & Prompt Tuning

- **Prompt Tuning (Lester et al.)**: Prepends $k$ learnable virtual token embeddings $\mathbf{P} \in \mathbb{R}^{k \times d}$ to the input sequence.
- **Prefix Tuning (Li & Liang)**: Prepends learnable continuous vectors directly to Keys and Values in every Transformer attention layer:
  $$\text{Attention}(\mathbf{Q}, [\mathbf{P}_K; \mathbf{K}], [\mathbf{P}_V; \mathbf{V}])$$

### 3.5 $(IA)^3$ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

- Rescales inner activations with learned vectors: $\mathbf{l} \odot \mathbf{x}$. Requires under 0.02% trainable parameters.

---

## 4. Domain Adaptation (Transductive Transfer)

Domain adaptation addresses covariate shift where input distributions differ between source and target ($P(X_S) \ne P(X_T)$) while the label space remains identical ($Y_S = Y_T$).

### 4.1 Maximum Mean Discrepancy (MMD)

- Computes distribution divergence in a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_k$:
  $$\text{MMD}^2(\mathcal{D}_S, \mathcal{D}_T) = \left\| \frac{1}{N_S}\sum_{i=1}^{N_S} \phi(\mathbf{x}_i^S) - \frac{1}{N_T}\sum_{j=1}^{N_T} \phi(\mathbf{x}_j^T) \right\|_{\mathcal{H}}^2$$
- Unbiased kernel expansion:
  $$\text{MMD}^2 = \frac{1}{N_S(N_S-1)} \sum_{i \ne i'} k(\mathbf{x}_i^S, \mathbf{x}_{i'}^S) + \frac{1}{N_T(N_T-1)} \sum_{j \ne j'} k(\mathbf{x}_j^T, \mathbf{x}_{j'}^T) - \frac{2}{N_S N_T} \sum_{i, j} k(\mathbf{x}_i^S, \mathbf{x}_j^T)$$
- Joint objective: $\mathcal{L} = \mathcal{L}_{\text{supervised}}(X_S, Y_S) + \lambda \text{MMD}^2(f(X_S), f(X_T))$.

### 4.2 Adversarial Domain Adaptation (DANN - Ganin et al.)

- **Architecture**:
  - Feature Extractor $G_f(\mathbf{x}; \theta_f)$.
  - Label Predictor $G_y(\mathbf{f}; \theta_y)$.
  - Domain Discriminator $G_d(\mathbf{f}; \theta_d)$ classifying whether representation $\mathbf{f}$ originates from source or target.
- **Gradient Reversal Layer (GRL)**:
  Acts as identity in the forward pass, multiplies gradient by $-\lambda$ during backward pass:
  $$R(\mathbf{x}) = \mathbf{x}, \quad \frac{d R}{d \mathbf{x}} = -\lambda \mathbf{I}$$
- Minimax objective forces $G_f$ to learn representations that are discriminative for the main task while invariant across domains.

---

## 5. Summary Comparison Matrix

| Transfer Technique | Trainable Parameters | Memory Overhead | Inference Overhead | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Probing** | $< 0.1\%$ (Head only) | Lowest | 0% | Fast baseline, small target data |
| **Full Fine-Tuning** | 100% | Extremely High | 0% | Ample target data, domain divergence |
| **LoRA** | $\approx 0.05\% - 0.5\%$ | Low | **0%** (via weight merge) | LLMs, Diffusion, Vision Transformers |
| **QLoRA** | $\approx 0.05\% - 0.5\%$ | Lowest GPU VRAM | 0% (after dequant/merge) | Resource-constrained LLM training |
| **Adapters** | $\approx 1\% - 3\%$ | Low | $+10\% - 20\%$ FLOPs | Multi-task parameter sharing |
| **MMD Domain Adaptation** | 100% | Moderate | 0% | Distribution shift without target labels |
| **DANN (Adversarial)** | 100% + Discriminator | Moderate | 0% | Complex cross-domain covariate shift |
