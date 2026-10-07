# 🤖 Phase 2: Core Machine Learning, Probabilistic Models & Ensembles

Before training multi-billion parameter neural networks, a production-grade ML Engineer must master classical statistical algorithms. Classical models provide fast, interpretable baselines, excel on tabular datasets, and form the statistical intuition required for diagnosing complex production issues.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. **Understand Algorithmic Tradeoffs:** Master the Bias-Variance tradeoff, the No Free Lunch theorem, and when linear, non-parametric, probabilistic, or tree-based ensembles are optimal.
2. **Implement Core Classifiers from First Principles:** Build `LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`, `KMeans`, `LinearSVM`, and `GaussianNaiveBayes` from scratch in pure NumPy.
3. **Master Tree Ensembles & Boosting:** Understand bagging (Random Forests) and sequential gradient boosting (**XGBoost** 2nd-order Taylor expansion).
4. **Time Series Forecasting:** Build lag features, rolling windows, and evaluate temporal models using **Walk-Forward (Rolling-Origin) Cross-Validation** to eliminate lookahead bias.
5. **Evaluate with Rigor:** Move beyond simple accuracy to evaluate real-world skewed distributions using Precision-Recall curves, F1-scores, ROC-AUC, and stratified k-fold cross-validation.
6. **Serialize & Version Baselines:** Train, tune hyperparameters, and serialize deployable model artifacts using `joblib`.

---

## 📚 Curated Study Resources

### 📖 Textbooks & Reading
* **[An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/)** (James, Witten, Hastie, Tibshirani) — Intuitive guide to regression, classification, resampling, and tree methods.
* **[The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/)** — Advanced theoretical foundations for gradient boosting and statistical learning theory.
* **[Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)** — Comprehensive engineering documentation covering loss formulations, solvers, and pipeline best practices.

### 📄 Included Deep-Dive Guides
* [`01_supervised_algorithms.md`](01_supervised_algorithms.md): Linear/Logistic Regression, Bias-Variance, Decision Trees, and Ensembles.
* [`02_unsupervised_and_evaluation.md`](02_unsupervised_and_evaluation.md): K-Means EM, DBSCAN, Confusion Matrix, Precision/Recall, ROC.
* [`03_advanced_classical_and_time_series.md`](03_advanced_classical_and_time_series.md): Soft-Margin SVMs, Hinge Loss, Gaussian Naive Bayes, and Time Series Walk-Forward Validation.

---

## 🛠️ Practical Coding Tasks

Complete these implementation tasks in `03_machine_learning/src/` and verify your solutions with the test suite in `03_machine_learning/tests/`.

---

### Task 2.1: Parametric Linear Models from Scratch
* **File:** `src/linear_models.py`
* **Goal:** Implement `LinearRegression` supporting both the analytical closed-form solution (`fit_normal_equation`) and mini-batch gradient descent with L2 penalty ($\lambda$), plus `LogisticRegression` with Sigmoid activation.

---

### Task 2.2: Recursive Decision Tree Classifier
* **File:** `src/decision_tree.py`
* **Goal:** Implement a binary `DecisionTreeClassifier` from scratch using pure NumPy and Gini Impurity split optimization.

---

### Task 2.3: K-Means Clustering & Silhouette Validation
* **File:** `src/kmeans.py`
* **Goal:** Implement `KMeans` with K-Means++ initialization and Expectation-Maximization centroid updates, plus `silhouette_score`.

---

### Task 2.4: Production Training & Model Registry Serialization
* **File:** `src/train_eval.py`
* **Goal:** Train a production Random Forest classifier on telemetry data, run Stratified 5-Fold cross-validation, and export weights to `models/baseline_classifier.joblib`.

---

### Task 2.5: Soft-Margin Support Vector Machine & Naive Bayes
* **Files:** `src/svm.py`, `src/naive_bayes.py`
* **Goal:** Implement `LinearSVM` using subgradient descent on Hinge Loss, and `GaussianNaiveBayes` with log-space joint probability computation.

---

### Task 2.6: Time Series Forecasting & Walk-Forward Validation
* **File:** `src/time_series.py`
* **Goal:** Build lag features, rolling statistics, and a `WalkForwardValidator` cross-validation harness ensuring test splits are strictly in the future.

---

### Task 2.7: Ensemble Benchmarks (Random Forest vs XGBoost)
* **File:** `src/ensemble_benchmark.py`
* **Goal:** Benchmark training and inference latency, F1-scores, and ROC-AUC between Random Forest, Gradient Boosting, and XGBoost.

---

## 📁 Directory Structure for Phase 2

```text
03_machine_learning/
├── README.md
├── 01_supervised_algorithms.md
├── 02_unsupervised_and_evaluation.md
├── 03_advanced_classical_and_time_series.md
├── notebooks/
│   └── 01_classical_ml_benchmarks.ipynb
├── models/
│   └── baseline_classifier.joblib
├── src/
│   ├── __init__.py
│   ├── linear_models.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   ├── train_eval.py
│   ├── svm.py
│   ├── naive_bayes.py
│   ├── time_series.py
│   └── ensemble_benchmark.py
└── tests/
    ├── __init__.py
    ├── test_linear_models.py
    ├── test_decision_tree.py
    ├── test_kmeans.py
    ├── test_svm.py
    ├── test_naive_bayes.py
    ├── test_time_series.py
    └── test_ensemble_benchmark.py
```
