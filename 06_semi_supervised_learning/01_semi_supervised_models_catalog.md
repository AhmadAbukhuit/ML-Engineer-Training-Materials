# Complete Catalog of Semi-Supervised Learning Models

Semi-Supervised Learning (SSL) addresses the fundamental setting where labeled data $\mathcal{D}_L = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N_L}$ is scarce and costly to acquire, while unlabeled data $\mathcal{D}_U = \{\mathbf{u}_j\}_{j=1}^{N_U}$ is plentiful ($N_U \gg N_L$). The goal is to leverage the geometric, topological, and manifold structure of $\mathcal{D}_U$ to improve generalization performance over learning exclusively from $\mathcal{D}_L$.

$$\min_\theta \sum_{i=1}^{N_L} \mathcal{L}_s(f_\theta(\mathbf{x}_i), y_i) + \lambda(t) \sum_{j=1}^{N_U} \mathcal{L}_u(f_\theta(\mathbf{u}_j), \mathcal{D}_U)$$

---

## 1. Foundational Assumptions of Semi-Supervised Learning

Unlabeled data provides value only if the underlying marginal distribution $p(\mathbf{x})$ contains information about the posterior distribution $p(y|\mathbf{x})$. SSL relies on four core assumptions:

1. **Smoothness Assumption**: If two points $\mathbf{x}_1, \mathbf{x}_2$ reside in a high-density region and are close to each other, their conditional outputs $y_1, y_2$ are likely identical.
2. **Cluster Assumption**: Points belonging to the same cluster share the same class label; equivalently, the decision boundary between classes traverses low-density regions.
3. **Low-Density Separation Assumption**: The optimal classifier decision boundary passes through sparse, low-density regions of the input space (minimizing Shannon conditional entropy).
4. **Manifold Assumption**: High-dimensional observations $\mathbf{x} \in \mathbb{R}^D$ lie concentrated on or near a lower-dimensional Riemannian manifold $\mathcal{M} \subset \mathbb{R}^d$ ($d \ll D$). Distances and geodesics on the manifold govern class transitions.

---

## 2. Pseudo-Labeling & Self-Training

### 2.1 Classic Self-Training
- **Procedure**:
  1. Train base classifier $f_\theta$ on labeled set $\mathcal{D}_L$.
  2. Compute predictions $\hat{y}_j = \arg\max_c P(y=c|\mathbf{u}_j)$ on unlabeled pool $\mathcal{D}_U$.
  3. Filter samples whose confidence $\max_c P(y=c|\mathbf{u}_j) \ge \tau$ meets threshold $\tau$.
  4. Augment labeled set: $\mathcal{D}_L \leftarrow \mathcal{D}_L \cup \{(\mathbf{u}_j, \hat{y}_j)\}$.
  5. Retrain until convergence or maximum iterations.
- **Confirmation Bias Risk**: Erroneously pseudo-labeled points enter the training set and propagate catastrophic errors. Alleviated via high thresholds ($\tau \ge 0.95$), student-teacher distillation, or sample reweighting.

### 2.2 FixMatch
- **Coupling Pseudo-labeling with Consistency (Sohn et al., 2020)**:
  1. For each unlabeled sample $\mathbf{u}_b$, generate a **weakly augmented** view $\alpha(\mathbf{u}_b)$ (e.g., standard flip/crop) and a **strongly augmented** view $\mathcal{A}(\mathbf{u}_b)$ (e.g., RandAugment / CTAugment).
  2. Model predicts probability distribution on weak view: $q_b = P(y | \alpha(\mathbf{u}_b))$.
  3. If $\max(q_b) \ge \tau$, compute pseudo-label $\hat{q}_b = \arg\max(q_b)$.
  4. Enforce cross-entropy loss on the strong view using the weak pseudo-label:
     $$\mathcal{L}_u = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\max(q_b) \ge \tau) \cdot H\left(\hat{q}_b, P(y | \mathcal{A}(\mathbf{u}_b))\right)$$

### 2.3 FlexMatch (Curriculum Pseudo-Labeling)
- Addresses class-imbalance failure in FixMatch: easy classes dominate pseudo-labels early while difficult classes are ignored due to fixed threshold $\tau$.
- Introduces class-adaptive dynamic thresholds $\tau_t(c) = \beta_t(c) \cdot \tau$, where $\beta_t(c)$ tracks the learning status of class $c$ over training progression.

---

## 3. Consistency Regularization

Consistency regularization enforces that a realistic perturbation $\delta$ (data augmentation, dropout, adversarial noise) applied to an unlabeled sample $\mathbf{u}$ should not change the model's prediction:

$$\min_\theta d(f_\theta(\mathbf{u}), f_\theta(\mathbf{u} + \delta))$$

### 3.1 $\Pi$-Model
- Passes each unlabeled instance $\mathbf{u}$ through the stochastic network twice under different dropout masks and data augmentations:
  $$\mathcal{L}_u = \|f_\theta(\mathbf{u}, \epsilon_1) - f_\theta(\mathbf{u}, \epsilon_2)\|_2^2$$

### 3.2 Temporal Ensembling (Laine & Aila)
- Maintains an exponential moving average (EMA) of network predictions over training epochs:
  $$Z_i \leftarrow \alpha Z_i + (1 - \alpha) f_\theta(\mathbf{u}_i)$$
- Enforces consistency between current prediction $f_\theta(\mathbf{u}_i)$ and historical consensus $Z_i$.
- Dramatically reduces noise, but requires updating predictions once per epoch (cumbersome for large datasets).

### 3.3 Mean Teacher (Tarvainen & Valpola)
- Replaces EMA on outputs with an **EMA on model weights**:
  - **Student Network**: Parameterized by $\theta$, updated via SGD gradients.
  - **Teacher Network**: Parameterized by $\theta'$, updated via EMA:
    $$\theta'_t = \alpha \theta'_{t-1} + (1 - \alpha) \theta_t, \quad \alpha \approx 0.999$$
- Consistency loss is computed between student predictions on perturbed input and teacher predictions on clean/differently perturbed input:
  $$\mathcal{L}_{\text{cons}} = \|f_\theta(\mathbf{u}) - f_{\theta'}(\mathbf{u}')\|_2^2$$

### 3.4 UDA (Unsupervised Data Augmentation)
- Uses state-of-the-art domain augmentations (RandAugment for images, Back-Translation for NLP) and minimizes KL divergence:
  $$\mathcal{L}_u = D_{\text{KL}}\left( f_{\theta^*}(\mathbf{u}) \parallel f_\theta(\mathcal{A}(\mathbf{u})) \right)$$
  combined with Target Sharpening: $q_c = \frac{p_c^{1/T}}{\sum_k p_k^{1/T}}$ with $T < 1$.

### 3.5 MixMatch & ReMixMatch
- **MixMatch**:
  1. Average predictions across $K$ augmentations of unlabeled samples.
  2. Sharpen target distribution via temperature $T$.
  3. Apply **MixUp** convex linear interpolation across concatenated labeled and unlabeled batches:
     $$x' = \lambda x_1 + (1 - \lambda) x_2, \quad y' = \lambda y_1 + (1 - \lambda) y_2, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$
- **ReMixMatch**: Augments MixMatch with distribution alignment and self-supervised rotation loss.

---

## 4. Graph-Based Semi-Supervised Learning

Models instances as vertices in an affinity graph $G = (V, E)$, with edge weights $W_{ij}$ representing similarity.

### 4.1 Label Propagation (Zhu & Ghahramani, 2002)
- **Affinity Matrix**: $W_{ij} = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ for $i \ne j$, $W_{ii} = 0$.
- **Transition Probability**: $P = D^{-1} W$, where $D_{ii} = \sum_j W_{ij}$.
- **Iteration**: $Y^{(t+1)} = P Y^{(t)}$.
- **Hard Clamping**: At each step, original labeled instances are reset to their true labels:
  $$Y_L^{(t+1)} = Y_L^{(0)}$$
- **Convergence / Harmonic Function**:
  Decomposing $P = \begin{bmatrix} P_{LL} & P_{LU} \\ P_{UL} & P_{UU} \end{bmatrix}$, the unconstrained equilibrium on unlabeled nodes has closed-form:
  $$Y_U^* = (I - P_{UU})^{-1} P_{UL} Y_L^{(0)}$$
  This forms a harmonic function minimizing Dirichlet energy on the graph.

### 4.2 Label Spreading (Zhou et al., 2004)
- Uses symmetric normalized Laplacian: $S = D^{-1/2} W D^{-1/2}$.
- **Soft Clamping** with parameter $\alpha \in (0, 1)$:
  $$Y^{(t+1)} = \alpha S Y^{(t)} + (1 - \alpha) Y^{(0)}$$
  Allows labeled nodes to retain a fraction of their label while adapting to structural neighbor consensus, providing high resilience to label noise.

### 4.3 Graph Convolutional Networks (GCNs - Kipf & Welling)
- First-order localized spectral convolution:
  $$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$
  where $\tilde{A} = A + I_N$ and $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. Trains end-to-end transductively with cross-entropy evaluated only on $\mathcal{D}_L$.

---

## 5. Semi-Supervised Generative Models & Transductive SVMs

### 5.1 Semi-Supervised VAE (Kingma et al., 2014)
- **M1 Model**: Standard unsupervised VAE extracts latent code $\mathbf{z}$; supervised classifier trains on $\mathbf{z}$.
- **M2 Model**: Joint generative model $p_\theta(\mathbf{x}, y, \mathbf{z}) = p(y) p(\mathbf{z}) p_\theta(\mathbf{x}|y, \mathbf{z})$.
  - For labeled data: ELBO includes observed $y$.
  - For unlabeled data: $y$ is treated as an unobserved discrete latent variable and marginalized over: $\sum_y \mathcal{L}(\mathbf{x}, y)$.

### 5.2 Semi-Supervised GANs (SGAN / CatGAN)
- Discriminator output space expanded from binary (real/fake) to $K + 1$ classes:
  - Classes $1, \dots, K$: ground truth categorical classes for labeled real data.
  - Class $K + 1$: generated fake data from Generator $G(\mathbf{z})$.
- Unlabeled real samples are trained via unsupervised cross-entropy loss: $\log(1 - P(y = K+1 | \mathbf{x}))$.

### 5.3 Transductive Support Vector Machines ($S^3\text{VM}$ / TSVM - Vapnik)
- Finds hyper-plane $(\mathbf{w}, b)$ and binary label assignments $y_j^* \in \{-1, +1\}$ for all unlabeled points $\mathbf{u}_j$ maximizing the margin:
  $$\min_{\mathbf{w}, b, y^*, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i \in L} \xi_i + C^* \sum_{j \in U} \xi_j^*$$
  $$\text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad y_j^*(\mathbf{w}^T \mathbf{u}_j + b) \ge 1 - \xi_j^*$$

---

## 6. Summary Comparison Matrix

| Algorithm Family | Paradigm | Core Mechanism | Handling of Unlabeled Data | Resilience to Noise |
| :--- | :--- | :--- | :--- | :--- |
| **Self-Training** | Inductive | Greedy iterative pseudo-labeling | Threshold confidence $\ge \tau$ | Low (confirmation bias) |
| **FixMatch / FlexMatch** | Inductive | Dual augmentation consistency | Weakly-augmented pseudo-label on strong view | High |
| **Mean Teacher** | Inductive | Parameter EMA consistency | Student-teacher prediction agreement | High |
| **MixMatch** | Inductive | Holistic data mixing | MixUp interpolation across labeled/unlabeled | High |
| **Label Propagation** | Transductive | Random walk on affinity graph | Harmonic Dirichlet minimization with hard clamping | Moderate |
| **Label Spreading** | Transductive | Normalized graph Laplacian | Soft clamping with parameter $\alpha$ | High |
| **Transductive SVM** | Transductive | Low-density separation | Margin maximization over unlabeled assignments | Moderate |
