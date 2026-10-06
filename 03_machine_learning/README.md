# 🤖 Phase 2: Core Machine Learning & Statistical Modeling

Before training multi-billion parameter neural networks, a production-grade ML Engineer must master classical machine learning algorithms. Classical models provide fast, interpretable baselines, often outperform deep learning on tabular data, and form the statistical intuition required for diagnosing complex production issues.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. **Understand Algorithmic Tradeoffs:** Understand the Bias-Variance tradeoff, the No Free Lunch theorem, and when a linear model, tree-based ensemble, or non-parametric clusterer is the optimal tool.
2. **Implement from First Principles:** Build core classical algorithms (`LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, and `KMeans`) from scratch in pure NumPy.
3. **Master Tree Ensembles:** Understand bagging, feature subsampling, and sequential gradient boosting (Random Forests and XGBoost / LightGBM).
4. **Evaluate with Rigor:** Move beyond simple accuracy to evaluate real-world skewed distributions using Precision-Recall curves, F1-scores, ROC-AUC, and stratified k-fold cross-validation.
5. **Serialize & Version Baselines:** Train, tune hyperparameters, and serialize deployable model artifacts using `joblib`.

---

## 📚 Curated Study Resources

### 📖 Textbooks & Reading
* **[An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/)** (James, Witten, Hastie, Tibshirani) — The definitive, intuitive guide to regression, classification, resampling, and tree methods.
* **[The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/)** — Advanced theoretical foundations for gradient boosting and statistical learning theory.
* **[Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)** — Comprehensive engineering documentation covering loss formulations, solvers, and pipeline best practices.

### 🎥 Video Lectures

* **StatQuest with Josh Starmer:** Step-by-step visual breakdowns of Logistic Regression, Decision Trees, Random Forests, XGBoost, and ROC/AUC.
* **Andrew Ng - Machine Learning Specialization (Coursera/YouTube):** Intuitive explanations of cost functions, gradient descent, and regularization.

---

## 🔬 Core Algorithms & Mathematical Foundations

### 1. Linear & Logistic Regression

* **Ordinary Least Squares (OLS) Normal Equation:**
  $$\theta = (X^T X)^{-1} X^T \mathbf{y}$$
* **Ridge Regularization (L2 Penalty):**
  $$\mathcal{L}_{\text{Ridge}}(\theta) = \frac{1}{2N} \Vert{}X\theta - \mathbf{y}\Vert{}_2^2 + \frac{\lambda}{2} \Vert{}\theta\Vert{}_2^2$$
* **Logistic Regression (Sigmoid Hypothesis & Log-Loss):**
  $$h_\theta(\mathbf{x}) = \sigma(\theta^T \mathbf{x}) = \frac{1}{1 + e^{-\theta^T \mathbf{x}}}$$
  $$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log(h_\theta(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(\mathbf{x}^{(i)})) \right]$$

### 2. Decision Trees & Ensembles

* **Gini Impurity for a Node $S$:**
  $$I_G(S) = 1 - \sum_{k=1}^C p_k^2$$
  Where $p_k$ is the fraction of items labeled with class $k$ in node $S$.
* **Information Gain via Split on Feature $j$ with Threshold $t$:**
  $$\Delta I_G(S, j, t) = I_G(S) - \left( \frac{|S_L|}{|S|} I_G(S_L) + \frac{|S_R|}{|S|} I_G(S_R) \right)$$
* **Random Forest:** Bagging (Bootstrap Aggregation) + random feature subset selection at each split to decorrelate individual trees.
* **Gradient Boosting:** Fits trees sequentially on the negative gradient (pseudo-residuals) of the loss function:
  $$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)} \right]_{f = f_{m-1}}$$

### 3. Unsupervised Clustering: K-Means

* **Objective Function (Inertia / Within-Cluster Sum of Squares):**
  $$J = \sum_{j=1}^k \sum_{\mathbf{x} \in S_j} \Vert{}\mathbf{x} - \boldsymbol{\mu}_j\Vert{}^2$$
* **Silhouette Coefficient for sample $i$:**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  Where $a(i)$ is the mean intra-cluster distance, and $b(i)$ is the mean nearest-cluster distance.

---

## 🛠️ Practical Coding Tasks

Complete these implementation tasks in `03_machine_learning/src/` and verify your solutions with the test suite in `03_machine_learning/tests/`.

---

### Task 2.1: Parametric Linear Models from Scratch

* **File:** `src/linear_models.py`
* **Requirements:**
  1. Implement `LinearRegression` supporting both the analytical closed-form solution (`fit_normal_equation`) and mini-batch gradient descent (`fit_gradient_descent`) with L2 regularization ($\lambda$).
  2. Implement `LogisticRegression` for binary classification using vector dot-products, sigmoid activation, and gradient descent optimization.
  3. Support `.fit(X, y)`, `.predict(X)`, and `.predict_proba(X)`.
* **Deliverable:** Unit tests proving accuracy matches `scikit-learn` on synthetic datasets.

---

### Task 2.2: Recursive Decision Tree Classifier

* **File:** `src/decision_tree.py`
* **Requirements:**
  1. Implement a binary `DecisionTreeClassifier` from scratch using pure NumPy.
  2. Find the optimal split across all continuous features using vectorized Gini Impurity.
  3. Support stopping criteria: `max_depth`, `min_samples_split`, and purity thresholds.
  4. Implement `.predict(X)` by recursively traversing the learned tree nodes.
* **Deliverable:** Passing unit tests achieving $>90\%$ accuracy on classification datasets.

---

### Task 2.3: K-Means Clustering & Silhouette Validation

* **File:** `src/kmeans.py`
* **Requirements:**
  1. Implement `KMeans` with K-Means++ initialization to prevent poor local minima.
  2. Implement iterative Expectation-Maximization: assignment step (Voronoi partition) and update step (centroid recomputation).
  3. Implement vectorized `compute_inertia(X, centroids)` and `silhouette_score(X, labels)`.
* **Deliverable:** Unit tests comparing custom cluster inertia and silhouette scores against `sklearn.cluster.KMeans`.

---

### Task 2.4: Production Training & Model Registry Serialization
* **File:** `src/train_eval.py`
* **Requirements:**
  1. Load processed telemetry or tabular dataset.
  2. Perform Stratified K-Fold cross validation.
  3. Compute confusion matrix, ROC-AUC, Precision, and Recall.
  4. Serialize the winning model artifact to `models/baseline_classifier.joblib`.
* **Deliverable:** A reproducible script outputting validation metrics and serialized weights.

---

## 📁 Directory Structure for Phase 2

```text
03_machine_learning/
├── README.md
├── 01_supervised_algorithms.md
├── 02_unsupervised_and_evaluation.md
├── notebooks/
│   └── 01_classical_ml_benchmarks.ipynb
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── linear_models.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   └── train_eval.py
└── tests/
    ├── __init__.py
    ├── test_linear_models.py
    ├── test_decision_tree.py
    └── test_kmeans.py
```
