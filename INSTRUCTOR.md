# 👨‍🏫 Instructor Guide: Teaching the ML Engineering Curriculum

Welcome to the instructional guide for the **Machine Learning Engineer Training Materials**. This curriculum is designed to bridge the gap between academic theory and real-world production engineering.

As an instructor, your mission is not merely to teach students how to call library functions, but to build engineers who understand algorithmic mechanics from first principles, write robust vectorized code, enforce data contracts, and deploy scalable inference microservices.

---

## 🎯 Pedagogical Philosophy

### 1. First Principles First, Frameworks Second

Students should **never** import a high-level abstraction (e.g., `nn.Linear`, `sklearn.tree.DecisionTreeClassifier`, or `FastAPI`) before they have built or analyzed the computational mechanics underneath:

* Before using PyTorch Autograd $\longrightarrow$ Build a scalar reverse-mode computational graph and compute gradients manually.
* Before tuning XGBoost $\longrightarrow$ Implement Gini Impurity splitting and a recursive decision tree node.
* Before launching a vector database $\longrightarrow$ Build an in-memory cosine similarity matrix and nearest-neighbor search.

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
* **Key Derivations:** 
  * Chain rule backpropagation through DAG nodes.
  * Numerically stable softmax: $\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$.
  * Singular Value Decomposition (SVD) projection for PCA.
* **Common Student Misconceptions:**
  * **Broadcasting bugs:** Forgetting that `(N,)` and `(N, 1)` broadcast to `(N, N)` in NumPy operations, creating silent out-of-memory errors or incorrect metrics.
  * **Numerical underflow:** Directly computing exponentiations in log-loss or softmax without max-shifting.
* **Instructor Tip:** Have students inspect tensor shapes at every single layer (`print(tensor.shape)` or debugger breakpoints) during Autograd derivations.

### Phase 2: Big Data Engineering & Feature Store Pipelines

* **Focus:** High-throughput columnar formats (Parquet, Polars, DuckDB), streaming ETL, point-in-time correctness.
* **Key Derivations:**
  * Memory layouts: Row-oriented (CSV) vs Columnar (Parquet).
  * Out-of-fold smoothed target encoding: $\hat{x}_i = \frac{n \cdot \bar{y}_{\text{cat}} + m \cdot \bar{y}_{\text{global}}}{n + m}$.
  * Point-in-time temporal joins (`ASOF` merges) to eliminate lookahead data leakage.
* **Common Student Misconceptions:**
  * **Lookahead Data Leakage:** Joining future feature snapshots onto historical target events. Teach students why standard SQL `JOIN` on customer ID produces unrealistic backtest metrics.
  * **Memory Exhaustion:** Trying to parse massive JSON files with `json.load()` all at once instead of line-by-line generators or streaming parsers.
* **Instructor Tip:** Run live benchmarks comparing DuckDB against Pandas on a 500MB Parquet query to visually demonstrate the 10x–50x speedup of columnar execution.

### Phase 3: Classical Machine Learning & Statistical Learning

* **Focus:** Loss formulation, structural risk minimization, inductive biases of linear, margin, and tree models.
* **Key Derivations:**
  * Hinge Loss subgradient descent: $\nabla_w \mathcal{L} = w - C y_i x_i \cdot \mathbb{I}(y_i (w^T x_i + b) < 1)$.
  * Gaussian Naive Bayes log-likelihood closed-form updates.
  * Recursive Gini Impurity minimization for binary decision splits.
* **Common Student Misconceptions:**
  * **Standard K-Fold on Time Series:** Shuffling temporal data during cross-validation. Emphasize why Walk-Forward Expanding-Window Validation is non-negotiable for temporal sequences.
  * **Default Hyperparameter Syndrome:** Believing tree depth should be unbounded. Show how unconstrained trees memorize training noise ($R^2 = 1.0$) while failing catastrophically on test data.
* **Instructor Tip:** Live-code an overfitted Decision Tree vs a 100-tree Random Forest on a noisy synthetic dataset to demonstrate variance reduction via bagging.

### Phase 4: Deep Learning Architectures & Modern Foundation Methods

* **Focus:** Deep representations, gradient flow through skip connections, self-attention routing, parameter-efficient adaptation (LoRA), and retrieval grounding (RAG).
* **Key Derivations:**
  * ResNet gradient highway: $\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y}( \frac{\partial \mathcal{F}}{\partial x} + I )$.
  * Scaled Dot-Product Attention: $\text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$.
  * LoRA rank factorization: $\Delta W = \frac{\alpha}{r} (A \cdot B)$, proving parameter reduction ratios.
* **Common Student Misconceptions:**
  * **Attention Dimension Scaling:** Forgetting why dividing by $\sqrt{d_k}$ is essential (large dot products push softmax into regions with vanishing gradients).
  * **Trainable vs Frozen Weights:** Confusion over which parameters accumulate gradients during transfer learning and LoRA fine-tuning.
* **Instructor Tip:** Have students print the parameter count comparison before and after LoRA ($r=4$ vs full rank) to physically visualize why fine-tuning 7B models fits in commodity GPU VRAM.

### Phase 5: Production APIs, Containerization & MLOps

* **Focus:** Microservice latency, data contracts, hardware compilation (ONNX), containerization, and experiment governance (MLflow).
* **Key Derivations:**
  * Amdahl's Law in model serving (disk I/O vs memory-resident inference).
  * P95 and P99 latency percentile distributions vs deceptive mean latencies.
  * Computational graph fusion in ONNX Runtime.
* **Common Student Misconceptions:**
  * **Model Loading Inside Route Handlers:** Loading `.joblib` or `.onnx` files from disk on every incoming HTTP request. Emphasize `@asynccontextmanager` singleton memory caching.
  * **Container Bloat:** Installing full development toolchains inside production Docker images rather than using multi-stage builds and slim base images.
* **Instructor Tip:** Run an Apache Bench (`ab`) or `hey` concurrency test on a naive endpoint (disk reload) versus the cached endpoint to demonstrate the 500ms $\to$ 2ms latency drop.

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

1. **Motivation & Industry Context:** Frame the problem through a real-world engineering failure (e.g., lookahead leakage causing financial backtest blowups, or unmerged adapter weights slowing down real-time latency).
2. **Mathematical Derivation:** Work through equations step-by-step on the board. Do not skip algebra or matrix dimension checks.
3. **Live Coding & Walkthrough:** Review the core architecture together, emphasizing clean type hints, docstrings, and vectorization.
4. **Lab & Test Execution:** Students complete hands-on tasks independently or in pairs, running `pytest` to verify their code against unit tests.

---

## 📋 Evaluation & Grading Rubric

| Category | Weight | Description |
| :--- | :---: | :--- |
| **Weekly Coding Tasks & Pytest Suites** | **40%** | Green test passes, vectorization quality, absence of anti-patterns, adherence to `ruff` linting standards. |
| **Milestone Exams & Code Walkthroughs** | **30%** | Individual oral code defense: Students must explain any function, its gradient derivation, and time complexity on demand. |
| **End-to-End Capstone Project** | **20%** | Complete system integrating raw data ingestion, feature validation, model training, ONNX export, FastAPI serving, and Docker containerization. |
| **Code Review & Engineering Rigor** | **10%** | Git commit hygiene, meaningful commit messages, clear documentation, and test coverage. |

---

## 💡 Instructor Checklist

### Before Class

* [ ] Ensure `.venv` has all dependencies installed (`pip install -r requirements.txt`).
* [ ] Run `pytest` on the relevant module to verify reference solutions are clean.
* [ ] Prepare whiteboard diagrams for tensor dimensions and operational data flows.

### During Class

* [ ] Challenge students to predict tensor shapes before running forward passes.
* [ ] Highlight performance bottlenecks in standard Python loops.
* [ ] Encourage students to read error tracebacks and Pytest failure diffs carefully.

### After Class

* [ ] Review student Pull Requests and check for vectorized implementations.
* [ ] Provide constructive feedback on algorithmic efficiency and schema design.
