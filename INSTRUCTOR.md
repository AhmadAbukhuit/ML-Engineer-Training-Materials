# 👨‍🏫 Instructor Guide: Teaching the ML Engineering Curriculum

Welcome to the instructional guide for the **Machine Learning Engineer Training Materials**. This curriculum is designed to bridge the gap between academic theory and real-world production engineering across all modern learning paradigms.

As an instructor, your mission is not merely to teach students how to call library functions, but to build engineers who understand algorithmic mechanics from first principles, write robust vectorized code, enforce data contracts, and deploy scalable inference microservices.

---

## 🎯 Pedagogical Philosophy

### 1. First Principles First, Frameworks Second
Students should **never** import a high-level abstraction (e.g., `nn.Linear`, `sklearn.tree.DecisionTreeClassifier`, or `FastAPI`) before they have built or analyzed the computational mechanics underneath:
* Before using PyTorch Autograd $\longrightarrow$ Build a scalar reverse-mode computational graph and compute gradients manually.
* Before tuning XGBoost $\longrightarrow$ Implement Gini Impurity splitting and a recursive decision tree node.
* Before launching a vector database $\longrightarrow$ Build an in-memory cosine similarity matrix and nearest-neighbor search.
* Before using PEFT/LoRA $\longrightarrow$ Implement low-rank weight factorization $\frac{\alpha}{r}AB$ and offline weight merging.

### 2. Code as the Ultimate Proof
Mathematics without implementation leads to fragile understanding. Every theoretical derivation presented on the whiteboard must be grounded in an executable, vectorized Python implementation.

### 3. Test-Driven Learning (TDD)
In production, machine learning code without unit tests is technical debt. Throughout this course, students utilize `pytest` not just as a testing tool, but as an **automated feedback loop**:
* Tests define the contract (shapes, edge cases, numerical tolerances, mathematical invariances).
* A green test suite gives students immediate, objective confirmation of mastery.

### 4. Zero-Loop Vectorization Policy
Enforce a strict anti-pattern rule: **No explicit Python `for`-loops over multidimensional array samples** where vectorized operations (in `NumPy`, `Polars`, or `DuckDB`) apply. Explicit loops in Python run at interpreter speed; vectorized code runs at compiled SIMD/BLAS speed.

---

## 🧩 Phase-by-Phase Teaching Guide & Pitfalls

### Phase 1: Mathematical Foundations & Numerical Computing
* **Focus:** Vectorization, reverse-mode autodiff, numerical stability, loss surface geometry.
* **Key Derivations:** Chain rule backpropagation through DAG nodes, numerically stable softmax with max-shift, PCA via SVD right singular vectors.
* **Common Student Misconceptions:**
  * *Broadcasting bugs:* Forgetting that `(N,)` and `(N, 1)` broadcast to `(N, N)` in NumPy operations, causing silent memory blowups.
  * *Numerical underflow:* Directly exponentiating large logits without max-shifting.
* **Instructor Tip:** Have students inspect tensor shapes at every single layer (`print(tensor.shape)` or debugger breakpoints) during Autograd derivations.

### Phase 2: Big Data Engineering, Feature Stores & Exploratory Data Analysis
* **Focus:** Columnar formats (Parquet, Polars, DuckDB), streaming ETL, point-in-time correctness, and EDA with NumPy, Pandas, and Scikit-Learn.
* **Key Derivations:** Interquartile Range (IQR) outlier bounds, Robust scaling vs Standard scaling, out-of-fold smoothed target encoding, temporal point-in-time `ASOF` joins.
* **Common Student Misconceptions:**
  * *Data Leakage in Preprocessing:* Fitting transformers on the combined dataset before splitting train/test sets.
  * *Lookahead Leakage:* Joining future feature states onto historical event timestamps.
* **Instructor Tip:** Have students write a script that scales a dataset before vs after train/test split to demonstrate how evaluation metrics become falsely optimistic.

### Phase 3: Supervised Learning & Tree Ensembles
* **Focus:** Empirical risk minimization, hinge loss, Gaussian log-likelihoods, tree splitting, and time series validation.
* **Key Derivations:** Hinge Loss subgradient updates: $\nabla_w \mathcal{L} = w - C y_i x_i \cdot \mathbb{I}(y_i (w^T x_i + b) < 1)$, Gaussian Naive Bayes closed-form parameters, recursive Gini Impurity minimization, XGBoost second-order Taylor expansion.
* **Common Student Misconceptions:**
  * *Standard K-Fold on Time Series:* Shuffling temporal sequences during CV. Emphasize why Walk-Forward Expanding-Window Validation is required.
  * *Unbounded Tree Depth:* Overfitting on noise ($R^2 = 1.0$) while collapsing on out-of-sample data.
* **Instructor Tip:** Live-code an overfitted Decision Tree vs a Random Forest on noisy data to physically visualize variance reduction via bagging.

### Phase 4: Unsupervised Learning & Dimensionality Reduction
* **Focus:** Density estimation, manifold discovery, probabilistic mixtures, and anomaly detection.
* **Key Derivations:** Lloyd's alternating minimization, K-Means++ $D(x)^2$ sampling proof, DBSCAN density-reachability chains, Gaussian Mixture Expectation-Maximization E/M steps, Isolation Forest average BST path length $c(n)$.
* **Common Student Misconceptions:**
  * *Assuming K-Means fits arbitrary shapes:* Show students how K-Means fails completely on concentric circles or two moons, where DBSCAN succeeds effortlessly.
  * *Singular Covariance in GMM:* Emphasize why adding diagonal regularization ($\epsilon I$) is mandatory to prevent collapsing components.
* **Instructor Tip:** Plot cluster boundaries for K-Means vs DBSCAN vs GMM on non-spherical datasets side-by-side.

### Phase 5: Self-Supervised Learning & Contrastive Dynamics
* **Focus:** Learning invariant representations from unlabeled data via pretext tasks and contrastive losses.
* **Key Derivations:** NT-Xent / InfoNCE loss as a lower bound on mutual information, temperature scaling $\tau$, Wang & Isola alignment and uniformity metrics, Deep Autoencoder backpropagation.
* **Common Student Misconceptions:**
  * *Dimensional Collapse:* Students often wonder why networks don't simply output a constant zero vector; explain how negative pairs in InfoNCE or stop-gradients in BYOL prevent trivial solutions.
  * *Using Projections for Downstream Tasks:* Teach students that the MLP projection head discards invariant nuisance factors, so the encoder representation $h$ must be used instead.
* **Instructor Tip:** Have students visualize the cosine similarity matrix before and after contrastive training.

### Phase 6: Semi-Supervised Learning & Graph Propagation
* **Focus:** Combining limited labeled data with abundant unlabeled data; low-density separation.
* **Key Derivations:** Self-training confidence thresholding, Graph Dirichlet energy minimization, Harmonic function equilibrium $Y_U^* = (I - P_{UU})^{-1} P_{UL} Y_L$, probability temperature sharpening $q_i \propto p_i^{1/T}$.
* **Common Student Misconceptions:**
  * *Confirmation Bias:* Students often set the pseudo-labeling threshold too low ($\tau = 0.5$), contaminating the training set with mislabeled points. Emphasize high thresholds ($\tau \ge 0.85$).
* **Instructor Tip:** Live-code label propagation on 100 points with only 2 labeled samples to demonstrate how graph geometry diffuses supervision across clusters.

### Phase 7: Deep Learning Architectures & Attention Mechanisms
* **Focus:** Deep representations, gradient highway preservation, spatial detection metrics, and multi-head attention.
* **Key Derivations:** ResNet gradient highway $\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y}(\frac{\partial \mathcal{F}}{\partial x} + I)$, Scaled Dot-Product Attention $\text{softmax}(QK^T / \sqrt{d_k})V$, vectorized IoU and NMS candidate deduplication.
* **Common Student Misconceptions:**
  * *Attention Dimension Scaling:* Forgetting why dividing by $\sqrt{d_k}$ is essential (large dot products push softmax into saturated vanishing gradient regions).
* **Instructor Tip:** Print parameter counts and attention weight heatmaps to show which tokens attend to which context positions.

### Phase 8: Reinforcement Learning & Sequential Decisions
* **Focus:** Markov Decision Processes, value-based learning, deep Q-networks, and policy gradients.
* **Key Derivations:** Bellman expectation and optimality equations, Tabular Q-Learning off-policy update, Deadly Triad mitigation (Replay Buffer + Target Network), Policy Gradient Theorem, baseline variance reduction.
* **Common Student Misconceptions:**
  * *On-Policy vs Off-Policy:* Clarify why SARSA evaluates the exploratory policy while Q-Learning evaluates the greedy optimal policy.
  * *Target Instability in DQN:* Show what happens when updating Q against a rapidly shifting online network without a frozen target network.
* **Instructor Tip:** Train tabular Q-learning on the GridWorld environment live and display the converged Q-table arrows indicating optimal navigation.

### Phase 9: Transfer Learning, PEFT & Domain Adaptation
* **Focus:** Foundation model adaptation, parameter-efficient fine-tuning, and cross-domain alignment.
* **Key Derivations:** Catastrophic forgetting, Low-Rank Adaptation factorization $\Delta W = \frac{\alpha}{r}AB$, offline weight merging $W_{\text{merged}} = W_0 + \Delta W$, Maximum Mean Discrepancy (MMD) in RKHS.
* **Common Student Misconceptions:**
  * *LoRA Inference Overhead Myth:* Many students assume LoRA adds latency during production; prove mathematically that weight folding yields exactly 0ms added inference time.
* **Instructor Tip:** Print the trainable vs frozen parameter counts ($< 0.1\%$ trainable) to physically illustrate why LoRA democratizes LLM fine-tuning on consumer hardware.

### Phase 10: Production APIs, Containerization & MLOps
* **Focus:** Microservice latency, Pydantic schemas, ONNXRuntime acceleration, Docker multi-stage builds, MLflow experiment tracking.
* **Key Derivations:** In-memory caching vs disk I/O, P95/P99 latency distribution analysis, ONNX graph constant folding and operator fusion.
* **Common Student Misconceptions:**
  * *Model Reloading on Request:* Loading model artifacts inside API route handlers rather than `@asynccontextmanager` startup lifespans.
  * *Container Bloat:* Copying development caches and virtualenvs into production Docker images.
* **Instructor Tip:** Run a benchmark comparing naive disk loading (500ms) against cached ONNX runtime inference (2ms).

---

## 🛠️ Classroom Delivery & Session Structure

Each weekly 3-to-4 hour class session should follow a structured rhythm:

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. Motivation   │ ──► │ 2. Mathematical │ ──► │ 3. Live Coding  │ ──► │ 4. Lab & Test   │
│ & Industry Context│   │    Derivation   │     │    & Walkthrough│     │    Execution    │
│  (20-30 mins)   │     │   (45-60 mins)  │     │   (45-60 mins)  │     │   (60-90 mins)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Motivation & Industry Context:** Frame the problem through a real-world engineering failure (e.g., lookahead leakage causing financial backtest blowups, confirmation bias in pseudo-labeling, or unmerged adapter weights slowing down real-time latency).
2. **Mathematical Derivation:** Work through equations step-by-step on the board. Do not skip algebra or matrix dimension checks.
3. **Live Coding & Walkthrough:** Review the core architecture together, emphasizing clean type hints, docstrings, and vectorization.
4. **Lab & Test Execution:** Students complete hands-on tasks independently or in pairs, running `pytest` to verify their code against unit tests.

---

## 📋 Evaluation & Grading Rubric

| Category | Weight | Description |
| :--- | :---: | :--- |
| **Weekly Coding Tasks & Pytest Suites** | **40%** | Green test passes across all 10 modules, vectorization quality, absence of anti-patterns, adherence to `ruff` linting standards. |
| **Milestone Exams & Code Walkthroughs** | **30%** | Individual oral code defense: Students must explain any function, its gradient derivation, and time complexity on demand. |
| **End-to-End Capstone Project** | **20%** | Complete system integrating raw data ingestion, feature validation, model training, ONNX export, FastAPI serving, and Docker containerization. |
| **Code Review & Engineering Rigor** | **10%** | Git commit hygiene, meaningful commit messages, clear documentation, and test coverage. |
