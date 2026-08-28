# 📐 Phase 0: Mathematical Foundations & Numerical Computing

Mastering the mathematical foundations behind machine learning is what separates engineers who merely call high-level library APIs from those who can diagnose training instabilities (exploding/vanishing gradients), optimize memory consumption, and design custom loss functions or architectures.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. Translate mathematical equations directly into efficient, vectorized `NumPy` code without relying on slow Python `for` loops.
2. Calculate and derive gradients analytically using the multivariate chain rule across computational graphs.
3. Understand probability distributions and formulate loss functions from first principles using Maximum Likelihood Estimation (MLE).
4. Implement fundamental optimization algorithms (SGD, Momentum, RMSprop, Adam) and evaluate their behavior across convex and non-convex loss surfaces.

---

## 📚 Curated Study Resources

### 🎥 Video Lectures & Playlists

* **3Blue1Brown - Essence of Linear Algebra:** Geometric intuitions for vectors, matrices, dot/cross products, determinants, eigenvalues, and coordinate transformations.
* **3Blue1Brown - Essence of Calculus:** Visual foundation for derivatives, chain rule, partial derivatives, and gradients.
* **Andrej Karpathy - Building micrograd:** The definitive step-by-step walkthrough of building a scalar-valued automatic differentiation engine from scratch.
* **StatQuest with Josh Starmer:** Clear, visual explanations of Maximum Likelihood, Cross-Entropy, PCA, and gradient descent mechanics.

### 📖 Textbooks & Reference Papers

* **[Mathematics for Machine Learning](https://mml-book.github.io/)** (Marc Peter Deisenroth et al.) — Free PDF: Chapters 2–7 cover Linear Algebra, Analytic Geometry, Matrix Decompositions, Vector Calculus, Probability, and Continuous Optimization.
* **[The Matrix Calculus You Need for Deep Learning](https://arxiv.org/abs/1802.01528)** (Terence Parr & Jeremy Howard) — Essential guide for matrix/tensor derivatives and dimension matching.
* **[Deep Learning (Chapter 2-4)](https://www.deeplearningbook.org/)** (Ian Goodfellow, Yoshua Bengio, Aaron Courville) — Clear overview of applied math for deep learning.

### 💻 GitHub Repositories to Study

* `karpathy/micrograd`: A tiny, clean autograd engine implementing backpropagation over a dynamically built DAG.
* `joelgrus/joelnet`: A minimal deep learning library built from scratch in pure Python to understand computational layers.
* `numpy/numpy`: Examine how core mathematical operations and vector broadcasting are implemented in Python/C.

---

## 🔬 Core Mathematical Concepts & Formulas

### 1. Linear Algebra & Matrix Decompositions

* **Vector Dot Product & Cosine Similarity:**
  $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2 \cos(\theta)$$
  $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{}_2 \Vert{}\mathbf{v}\Vert{}_2}$$

* **Matrix-Vector Multiplication:**
  $$y_i = \sum_{j=1}^{d} A_{ij} x_j \iff \mathbf{y} = A\mathbf{x}$$

* **Eigenvalues and Eigenvectors:**
  $$A\mathbf{v} = \lambda\mathbf{v} \iff (A - \lambda I)\mathbf{v} = \mathbf{0}$$

* **Singular Value Decomposition (SVD):**
  $$A = U \Sigma V^T$$
  Where $U$ and $V$ are orthogonal matrices containing singular vectors, and $\Sigma$ is a diagonal matrix containing singular values $\sigma_i$.

---

### 2. Multivariate Calculus & Computational Graphs

* **Gradient Vector:**
  $$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \end{bmatrix}^T$$

* **Multivariate Chain Rule (Scalar to Vector/Matrix):**
  Given $z = f(y)$ and $y = g(x)$:
  $$\frac{\partial z}{\partial x_i} = \sum_{j} \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

* **Jacobian Matrix (Vector-Valued Functions):**
  For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:
  $$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

---

### 3. Probability, Statistics & Loss Functions

* **Bayes' Theorem:**
  $$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

* **Gaussian (Normal) Distribution:**
  $$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

* **Mean Squared Error (derived from Gaussian MLE):**
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

* **Binary Cross-Entropy Loss (derived from Bernoulli MLE):**
  $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

* **Categorical Cross-Entropy with Softmax:**
  $$\hat{y}_k = \frac{\exp(z_k)}{\sum_{j=1}^{C} \exp(z_j)}, \quad \mathcal{L}_{\text{CCE}} = -\sum_{k=1}^{C} y_k \log(\hat{y}_k)$$

---

### 4. Optimization Algorithms

| Optimizer | Update Equation | Primary Advantage |
| :--- | :--- | :--- |
| **Standard SGD** | $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ | Simple baseline; noisy gradient updates. |
| **SGD with Momentum** | $v_{t+1} = \beta v_t + \eta \nabla L(\theta_t)$$\theta_{t+1} = \theta_t - v_{t+1}$ | Dampens oscillations in high-curvature valleys. |
| **RMSprop** | $s_{t+1} = \beta s_t + (1 - \beta) (\nabla L(\theta_t))^2$, \ $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1} + \epsilon}} \nabla L(\theta_t)$ | Adapts learning rate per parameter based on gradient magnitude. |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$, $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ | Combines Momentum and RMSprop with bias correction; default standard for deep learning. |

---

## 🛠️ Practical Coding Tasks (Pure NumPy)

Complete these four implementation tasks inside `phase_0_math_foundations/src/` and verify your solutions with tests in `phase_0_math_foundations/tests/`.

---

### Task 0.1: Vectorized Loss & Metrics Engine

* **File:** `src/metrics.py`
* **Goal:** Implement a fully vectorized suite of metrics and loss functions without using any Python `for` loops.
* **Requirements:**
  1. `mse_loss(y_true, y_pred)` and its gradient `mse_gradient(y_true, y_pred)`.
  2. `binary_cross_entropy(y_true, y_pred, eps=1e-15)` with epsilon clipping to prevent `log(0)`.
  3. `softmax(z, axis=-1)` with numerical stability trick: $\exp(z_i - \max(z))$.
  4. `categorical_cross_entropy(y_true_one_hot, y_pred_probs)`.
  5. `cosine_similarity_matrix(A, B)`: Compute the pairwise cosine similarity matrix between two sets of vectors $A \in \mathbb{R}^{N \times D}$ and $B \in \mathbb{R}^{M \times D}$.
* **Deliverable:** Passing unit tests comparing your results against `scikit-learn` and `torch.nn.functional` outputs.

---

### Task 0.2: Micro-Autograd Engine & 2-Layer MLP

* **File:** `src/autograd.py` and `src/mlp.py`
* **Goal:** Implement a scalar or tensor-level computational graph supporting automatic reverse-mode differentiation.
* **Requirements:**
  1. Build a `Value` (or `Tensor`) class that tracks `.data`, `.grad`, `._prev` (children nodes), and `._op`.
  2. Implement forward and backward mathematical operations: `+`, `-`, `*`, `/`, `**` (power), `tanh`, and `relu`.
  3. Implement the topological sort method inside `Value.backward()` to traverse the computational graph in reverse dependency order.
  4. Build a 2-layer Multi-Layer Perceptron (MLP) using your autograd engine and train it to solve the non-linear XOR classification problem.
* **Deliverable:** A training notebook `notebooks/01_autograd_xor.ipynb` showing loss decreasing to $< 0.01$.

---

### Task 0.3: Custom Optimizer Suite & Loss Surface Visualizer

* **File:** `src/optimizers.py`
* **Goal:** Implement and compare 4 optimization algorithms on standard non-convex optimization benchmarks (e.g., Rosenbrock function or Rastrigin function).
* **Requirements:**
  1. Implement an abstract base class `Optimizer` with an `.update(params, grads)` interface.
  2. Implement concrete classes: `SGD`, `SGDMomentum`, `RMSprop`, and `Adam`.
  3. Include bias corrections in `Adam` for both first and second moments.
  4. Plot the 2D trajectory of each optimizer traversing the Rosenbrock banana function:
     $$f(x, y) = (a - x)^2 + b(y - x^2)^2 \quad \text{where } a=1, b=100$$
* **Deliverable:** An evaluation notebook `notebooks/02_optimizer_comparison.ipynb` with contour plots showing the convergence path of each optimizer.

---

### Task 0.4: Principal Component Analysis (PCA) via SVD

* **File:** `src/pca.py`
* **Goal:** Implement Principal Component Analysis from scratch using both Covariance Eigendecomposition and SVD.
* **Requirements:**
  1. Standardize and center the input matrix $X \in \mathbb{R}^{N \times D}$ to have zero mean.
  2. Method A: Compute sample covariance matrix $C = \frac{1}{N-1}X^T X$, then compute eigenvalues/eigenvectors using `np.linalg.eigh`.
  3. Method B: Compute SVD directly on $X = U \Sigma V^T$.
  4. Calculate explained variance and cumulative explained variance ratio for top $k$ components.
  5. Project high-dimensional data onto the top $k$ principal components.
* **Deliverable:** Verification script ensuring your transformed coordinates and explained variance match `sklearn.decomposition.PCA` up to sign ambiguity.

---

## 📁 Directory Structure for Phase 0

```text
phase_0_math_foundations/
├── README.md
├── notebooks/
│   ├── 01_autograd_xor.ipynb
│   └── 02_optimizer_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── autograd.py
│   ├── metrics.py
│   ├── mlp.py
│   ├── optimizers.py
│   └── pca.py
└── tests/
    ├── __init__.py
    ├── test_autograd.py
    ├── test_metrics.py
    └── test_pca.py
```
