# Complete Catalog of Self-Supervised Learning Models

Self-Supervised Learning (SSL) is the machine learning paradigm where supervised learning targets (pseudo-labels or predictive targets) are autonomously generated directly from the inherent structure, geometry, or temporal ordering of unlabeled data $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$. The model solves a **pretext task** to build invariant, discriminative, and transferable representations $\mathbf{z} = f_\theta(\mathbf{x})$ that empower downstream transfer and linear probing tasks without requiring human annotations.

---

## 1. Contrastive Learning Architectures

Contrastive methods optimize an objective that pulls representations of augmented views of the same instance (positives) together in latent space while pushing representations of different instances (negatives) apart.

### 1.1 SimCLR (Simple Framework for Contrastive Learning of Visual Representations)
- **Key Components**:
  1. **Stochastic Data Augmentations**: Random cropping & resizing, color jittering, Gaussian blur, random flipping.
  2. **Base Encoder**: Deep neural network $f(\cdot)$ mapping input $\mathbf{x}$ to representation $\mathbf{h} = f(\mathbf{x}) \in \mathbb{R}^{d_h}$.
  3. **Non-linear Projection Head**: Multi-layer perceptron $g(\cdot)$ mapping representation $\mathbf{h}$ to projection $\mathbf{z} = g(\mathbf{h}) \in \mathbb{R}^{d_z}$.
  4. **NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss**:
     For a batch of $N$ examples yielding $2N$ augmented views, the loss for positive pair $(i, j)$ is:
     $$\ell_{i, j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \ne i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$
     where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ is cosine similarity and $\tau > 0$ is the temperature hyperparameter.
- **Critical Insight**: The projection head $g(\cdot)$ discards nuisance information (e.g., color, crop location); using $\mathbf{h}$ directly for downstream tasks significantly outperforms using $\mathbf{z}$.

### 1.2 MoCo (Momentum Contrast)
- **Problem in SimCLR**: SimCLR requires immense batch sizes ($N = 4096$) to supply sufficient negative samples in GPU memory.
- **MoCo Innovation (He et al.)**:
  - Treats contrastive learning as dynamic dictionary look-up.
  - **Dynamic FIFO Memory Queue**: Maintains a large queue of negative keys ($K = 65,536$) decoupled from mini-batch size.
  - **Momentum Encoder**: The key encoder $\theta_k$ is updated via an exponential moving average (EMA) of the query encoder $\theta_q$:
    $$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q, \quad m \approx 0.999$$
    This prevents representation collapse and ensures dictionary keys remain mutually consistent across gradient update steps.
- **MoCo v2 & v3**: Incorporates cosine decay learning rates, MLP projection heads, and Vision Transformer (ViT) backbones.

### 1.3 CPC (Contrastive Predictive Coding)
- Self-supervised learning on sequential/temporal signals (audio, text, video).
- Uses an autoregressive model $g_{\text{ar}}$ to summarize past latents $\mathbf{z}_{\le t}$ into context $\mathbf{c}_t$.
- Predicts future latents $\mathbf{z}_{t+k}$ via InfoNCE loss:
  $$\mathcal{L}_N = -\mathbb{E}_{\mathcal{X}} \left[ \log \frac{\exp(\mathbf{z}_{t+k}^T \mathbf{W}_k \mathbf{c}_t)}{\exp(\mathbf{z}_{t+k}^T \mathbf{W}_k \mathbf{c}_t) + \sum_{\mathbf{z}_j \in \mathcal{N}} \exp(\mathbf{z}_j^T \mathbf{W}_k \mathbf{c}_t)} \right]$$
- InfoNCE provides a rigorous lower bound on the mutual information $I(\mathbf{x}_{t+k}; \mathbf{c}_t) \ge \log(N) - \mathcal{L}_N$.

---

## 2. Non-Contrastive / Bootstrap Representation Learning

These methods eliminate the requirement for negative pairs entirely, utilizing architectural asymmetries or information-theoretic regularizations to prevent **dimensional collapse** (where the network outputs a constant trivial embedding $\mathbf{z} = \mathbf{c}$).

### 2.1 BYOL (Bootstrap Your Own Latent)
- **Architecture**:
  - **Online Network**: Parameterized by $\theta$; consists of encoder $f_\theta$, projector $g_\theta$, and an additional **predictor** $q_\theta$.
  - **Target Network**: Parameterized by $\xi$; consists of encoder $f_\xi$ and projector $g_\xi$. Updates via EMA: $\xi \leftarrow \tau \xi + (1 - \tau) \theta$.
- **Loss Function**: Mean squared error between normalized online prediction and target projection:
  $$\mathcal{L}_{\text{BYOL}} = 2 - 2 \frac{\langle q_\theta(z_\theta), z'_\xi \rangle}{\|q_\theta(z_\theta)\|_2 \|z'_\xi\|_2}$$
- **Asymmetry**: Online network has gradients backpropagated; target network has **stop-gradient** ($\text{sg}$). The predictor + EMA target acts as an optimal Wiener filter, preventing representational collapse without negatives.

### 2.2 SimSiam (Simple Siamese)
- Proves that the EMA target network in BYOL is not strictly mandatory; the key collapse-preventing mechanism is the **stop-gradient operation** coupled with a predictor MLP:
  $$\mathcal{L}(p_1, z_2) = -\frac{p_1}{\|p_1\|_2} \cdot \text{stop\_gradient}\left(\frac{z_2}{\|z_2\|_2}\right)$$
- Interpreted mathematically as an Expectation-Maximization-like alternating optimization between representation coordinates and cluster centers.

### 2.3 SwAV (Swapped Assignments between Views)
- Blends contrastive learning with online clustering.
- Projects embeddings onto a set of $K$ trainable prototype vectors $\mathbf{C} = [\mathbf{c}_1, \dots, \mathbf{c}_K]$.
- Assigns features to prototypes using the **Sinkhorn-Knopp algorithm** (solving an optimal transport problem under equipartition constraints to avoid cluster collapse).
- Swapped loss: Predicts the cluster assignment $\mathbf{q}_1$ of view 1 from the representation $\mathbf{z}_2$ of view 2:
  $$\mathcal{L}_{\text{SwAV}} = \ell(\mathbf{z}_1, \mathbf{q}_2) + \ell(\mathbf{z}_2, \mathbf{q}_1)$$

### 2.4 Barlow Twins & VICReg (Information-Theoretic Regularization)
- **Barlow Twins**:
  - Evaluates the cross-correlation matrix $\mathcal{C} \in \mathbb{R}^{d \times d}$ between normalized outputs of two identical sub-networks:
    $$\mathcal{C}_{jk} = \frac{\sum_b z_{b, j}^A z_{b, k}^B}{\sqrt{\sum_b (z_{b, j}^A)^2} \sqrt{\sum_b (z_{b, k}^B)^2}}$$
  - **Objective Function**: Forces $\mathcal{C}$ towards the identity matrix $\mathbf{I}$:
    $$\mathcal{L}_{\text{BT}} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \ne i} \mathcal{C}_{ij}^2$$
  - Diagonal terms enforce **invariance**; off-diagonal terms enforce **redundancy reduction** (decorrelation of dimensions).
- **VICReg (Variance-Invariance-Covariance Regularization)**:
  - Explicitly decomposes the loss into three terms:
    1. *Invariance*: MSE between representations $s(Z, Z') = \frac{1}{N} \sum_i \|z_i - z'_i\|^2$.
    2. *Variance*: Hinge loss ensuring embedding variance along each dimension is above threshold $\gamma$: $v(Z) = \frac{1}{d}\sum_j \max(0, \gamma - \sqrt{\text{Var}(z^j) + \epsilon})$.
    3. *Covariance*: Off-diagonal covariance penalty decorrelating embedding features: $c(Z) = \frac{1}{d} \sum_{i \ne j} C_{ij}^2$.

---

## 3. Masked Modeling & Reconstruction

### 3.1 MAE (Masked Autoencoders - Vision)
- **Paradigm (He et al., 2022)**:
  - Divides an input image into non-overlapping patches (e.g., $16 \times 16$).
  - Randomly masks a high fraction (e.g., 75% to 80%) of patches.
  - Passes *only unmasked patches* through a heavy Vision Transformer (ViT) encoder.
  - Appends lightweight learnable mask tokens and passes all tokens through a shallow ViT decoder.
  - Objective: Reconstruct pixel values of masked patches using MSE loss:
    $$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2$$
  - Asymmetric design achieves up to $4\times$ to $10\times$ faster pretraining while learning dense spatial representations.

### 3.2 BEiT (BERT Pre-training of Image Transformers)
- Tokenizes image patches into discrete visual tokens using a pretrained discrete VQ-VAE codebook.
- Masks image patches and trains the Vision Transformer to predict the discrete token IDs (Masked Image Modeling / MIM).

### 3.3 Masked Language Modeling (BERT, RoBERTa)
- Corrupts 15% of input tokens (80% replaced with `[MASK]`, 10% random token, 10% kept unchanged).
- Predicts original vocabulary token ID via cross-entropy:
  $$\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\backslash \mathcal{M}}; \theta)$$

### 3.4 Causal Auto-regressive Modeling (GPT Series)
- Pretrains on sequence $\mathbf{x} = (x_1, \dots, x_T)$ by maximizing forward autoregressive log-likelihood:
  $$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{t=1}^T \log P(x_t | x_1, \dots, x_{t-1}; \theta)$$

### 3.5 ELECTRA (Replaced Token Detection)
- Replaces MLM with a Generator-Discriminator setup:
  - Generator predicts masked tokens.
  - Discriminator solves binary classification: was token $x_t$ original or replaced by the generator?
  - Every single token in the sequence provides supervisory gradient, greatly increasing sample efficiency.

---

## 4. Autoencoders & Heuristic Pretext Tasks

### 4.1 Vanilla Autoencoder
- Minimizes reconstruction error: $\min_{\theta, \phi} \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i - g_\phi(f_\theta(\mathbf{x}_i))\|^2$.
- Bottleneck latent dimension $d_z \ll D$ forces network to learn principal non-linear manifold.

### 4.2 Denoising Autoencoder (DAE)
- Corrupts inputs $\tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}} | \mathbf{x})$ via additive Gaussian noise or masking dropout.
- Network learns score matching: the reconstruction vector $g_\phi(f_\theta(\tilde{\mathbf{x}})) - \tilde{\mathbf{x}}$ points in the direction of the score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$.

### 4.3 Variational Autoencoder (VAE)
- Probabilistic latent variable model optimizing Evidence Lower Bound (ELBO):
  $$\mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$$
- Uses reparameterization trick $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ for end-to-end backpropagation.

### 4.4 Heuristic Pretext Tasks
- **Rotation Prediction (RotNet)**: Rotates images by $0^\circ, 90^\circ, 180^\circ, 270^\circ$; model classifies angle.
- **Jigsaw Puzzles**: Splits image into $3 \times 3$ grid, permutes patches; model predicts permutation index.
- **Colorization**: Converts RGB to Lab space; given L channel, model predicts $a, b$ color channels.

---

## 5. Summary Comparison Matrix

| Model Family | Key Mechanism | Requires Negatives? | Prevents Collapse Via | Primary Application Domains |
| :--- | :--- | :--- | :--- | :--- |
| **SimCLR** | Contrastive (NT-Xent) | Yes (large batches) | Repulsion from negatives | Computer Vision |
| **MoCo** | Contrastive (InfoNCE) | Yes (memory queue) | Negative memory queue + EMA | Computer Vision |
| **BYOL** | Bootstrap target | No | Stop-gradient + Predictor + EMA | Computer Vision, Audio |
| **SimSiam** | Siamese predictor | No | Stop-gradient operation | Computer Vision |
| **Barlow Twins** | Cross-correlation decorrelation | No | Decorrelation penalty on $\mathcal{C}_{ij}$ | Vision, Speech |
| **VICReg** | Variance / Covariance regularization | No | Explicit variance hinge + covariance penalty | Tabular, Vision |
| **MAE** | Masked patch autoencoding | No | Reconstructing raw pixel patches | Vision Transformers |
| **BERT / GPT** | Masked / Causal token prediction | No | Cross-entropy vocabulary prediction | Natural Language Processing |
| **DAE / VAE** | Denoising / Generative ELBO | No | Dimensional bottleneck + KL prior | Unsupervised feature extraction |
