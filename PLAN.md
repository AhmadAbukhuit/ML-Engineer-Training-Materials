# 📅 Machine Learning Engineering Curriculum & Teaching Plan

An exhaustive, battle-tested teaching plan and week-by-week curriculum taking engineers from numerical and mathematical foundations to production MLOps, modern deep learning architectures, and cutting-edge systems engineering across all 18 curriculum phases.

---

## 🗺️ Curriculum Structure & Progression Tracks

This curriculum is structured into two complementary stages:

1. **The Core Engineering Curriculum (Weeks 1–20 / Phases 1–10):** Covers mathematical foundations, big data engineering, classical and modern learning paradigms (supervised, unsupervised, self-supervised, semi-supervised, reinforcement, transfer learning), deep learning, and production containerized MLOps.
2. **The Advanced Systems Specialization (Weeks 21–28 / Phases 11–18):** Advanced industry specializations in explainability, production drift monitoring, LLM systems, recommender architectures, conformal prediction, graph ML, distributed cluster parallelism, and safety/fairness auditing.

```text
 ═══════════════════════════════════════════════════════════════════════════════════════════════
                    CORE CURRICULUM: FOUNDATIONS TO PRODUCTION (WEEKS 1–20)
 ═══════════════════════════════════════════════════════════════════════════════════════════════
  Weeks 1–3   │ Phase 1: Mathematical Foundations & Numerical Computing (01_math_foundations)
  Weeks 4–6   │ Phase 2: Data Science & Big Data Engineering Pipelines (02_data_engineering)
  Weeks 7–8   │ Phase 3: Supervised Learning, Ensembles & Evaluation Metrics (03_supervised_learning)
  Weeks 9–10  │ Phase 4: Unsupervised Learning & Dimensionality Reduction (04_unsupervised_learning)
  Week 11     │ Phase 5: Self-Supervised Learning & Contrastive Dynamics (05_self_supervised_learning)
  Week 12     │ Phase 6: Semi-Supervised Learning & Graph Propagation (06_semi_supervised_learning)
  Weeks 13–14 │ Phase 7: Deep Learning Architectures & Attention Mechanisms (07_deep_learning)
  Weeks 15–16 │ Phase 8: Reinforcement Learning & Sequential Decisions (08_reinforcement_learning)
  Weeks 17–18 │ Phase 9: Transfer Learning, PEFT & Domain Adaptation (09_transfer_learning)
  Weeks 19–20 │ Phase 10: Production MLOps, Containerization & CI/CD (10_mlops_production)
 ═══════════════════════════════════════════════════════════════════════════════════════════════
               ADVANCED SPECIALIZATION & SYSTEMS MASTERCLASSES (WEEKS 21–28)
 ═══════════════════════════════════════════════════════════════════════════════════════════════
  Week 21     │ Phase 11: Explainable AI (XAI) & Model Interpretability (11_explainable_ai)
  Week 22     │ Phase 12: Production Observability & Drift Detection (12_model_monitoring_drift)
  Week 23     │ Phase 13: Generative AI & LLM Systems Engineering (13_llm_systems)
  Week 24     │ Phase 14: Recommendation Systems & Collaborative Filtering (14_recommender_systems)
  Week 25     │ Phase 15: Uncertainty Quantification & Conformal Prediction (15_uncertainty_quantification)
  Week 26     │ Phase 16: Graph Machine Learning & Graph Neural Networks (16_graph_neural_networks)
  Week 27     │ Phase 17: Distributed Deep Learning & Cluster Parallelism (17_distributed_training)
  Week 28     │ Phase 18: AI Safety, Privacy & Algorithmic Fairness (18_ai_safety_privacy)
 ═══════════════════════════════════════════════════════════════════════════════════════════════
```

---

## 📐 Phase 1: Mathematical Foundations & Numerical Computing (Weeks 1–3)

### Phase Description

The objective of Phase 1 is to dismantle the "black box" mentality. Students master the foundational numerical machinery governing modern machine learning algorithms: vectorized linear algebra in NumPy, reverse-mode automatic differentiation, multivariate calculus, statistical estimation, and gradient-based optimization mechanics.

### Week 1: Vectorized Linear Algebra & Matrix Decompositions

* **Theory:**
  * Vector spaces, inner products, Cauchy-Schwarz inequality, orthogonal projections, matrix rank.
  * Matrix factorizations: Spectral Eigendecomposition, Positive Semi-Definiteness (PSD), and Singular Value Decomposition (SVD: $A = U \Sigma V^T$).
  * Principal Component Analysis (PCA) derivation via covariance matrix decomposition and maximization of projected variance.
* **Practice & Labs:**
  * Repo Directory: [`01_math_foundations/`](01_math_foundations/)
  * Guides: [`01_math_foundations/01_linear_algebra.md`](01_math_foundations/01_linear_algebra.md)
  * Implement PCA from scratch using SVD in [`01_math_foundations/src/pca.py`](01_math_foundations/src/pca.py).
* **Deliverable & Tests:**
  * Pass [`01_math_foundations/tests/test_pca.py`](01_math_foundations/tests/test_pca.py).

### Week 2: Multivariate Calculus & Reverse-Mode Autograd Engine

* **Theory:**
  * Partial derivatives, gradient vectors, Jacobian matrices, and Hessian curvature.
  * Directed Acyclic Graphs (DAG) for mathematical operations and topological node sorting.
  * Reverse-mode automatic differentiation (vector-Jacobian products) vs. forward-mode and numerical finite differences.
* **Practice & Labs:**
  * Guide: [`01_math_foundations/02_multivariate_calculus.md`](01_math_foundations/02_multivariate_calculus.md)
  * Implement the scalar Autograd computation graph in [`01_math_foundations/src/autograd.py`](01_math_foundations/src/autograd.py).
* **Deliverable & Tests:**
  * Pass [`01_math_foundations/tests/test_autograd.py`](01_math_foundations/tests/test_autograd.py).

### Week 3: Probability Theory, Statistical Estimation & Gradient Optimizers

* **Theory:**
  * Probability spaces, Bayes' Theorem, Gaussian, Bernoulli, and Categorical distributions.
  * Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP).
  * Information Theory: Shannon Entropy, Binary/Categorical Cross-Entropy, KL Divergence.
  * First-order optimizers: Stochastic Gradient Descent (SGD), Momentum, RMSProp, and Adam mechanics (first/second moment exponential moving averages with bias correction).
* **Practice & Labs:**
  * Guide: [`01_math_foundations/03_probability_statistics.md`](01_math_foundations/03_probability_statistics.md)
  * Implement vectorized loss functions and metrics in [`01_math_foundations/src/metrics.py`](01_math_foundations/src/metrics.py).
  * Implement modular optimizer classes in [`01_math_foundations/src/optimizers.py`](01_math_foundations/src/optimizers.py).
* **Deliverable & Tests:**
  * Pass [`01_math_foundations/tests/test_autograd.py`](01_math_foundations/tests/test_autograd.py) and [`test_metrics.py`](01_math_foundations/tests/test_metrics.py).

---

## 💾 Phase 2: Data Science & Big Data Engineering Pipelines (Weeks 4–6)

### Phase Description

Students learn production-grade data engineering: exploratory data analysis with NumPy and Pandas, composable preprocessing pipelines with Scikit-Learn, high-throughput analytical query processing with Polars and DuckDB, preventing temporal lookahead leakage, and runtime data contract validation using Pandera.

### Week 4: Exploratory Data Analysis & Preprocessing Pipelines (NumPy, Pandas, Scikit-Learn)

* **Theory:**
  * Vectorized array broadcasting and boolean masking in NumPy.
  * Missing data audits (MCAR, MAR, MNAR) and imputation strategies in Pandas.
  * Outlier detection via the Interquartile Range (IQR) rule: $[Q_1 - 1.5 \cdot \text{IQR}, Q_3 + 1.5 \cdot \text{IQR}]$.
  * Pearson correlation matrix computation for multicollinearity detection.
  * Composable Scikit-Learn pipelines: `RobustScaler`, `SimpleImputer`, `OneHotEncoder`, and `ColumnTransformer` without train/test data leakage.
* **Practice & Labs:**
  * Repo Directory: [`02_data_engineering/`](02_data_engineering/)
  * Implement EDA and preprocessing pipeline helpers in [`02_data_engineering/src/data_analysis.py`](02_data_engineering/src/data_analysis.py).
* **Deliverable & Tests:**
  * Pass [`02_data_engineering/tests/test_data_analysis.py`](02_data_engineering/tests/test_data_analysis.py).

### Week 5: Columnar Storage, Analytical SQL & Semi-Structured Parsing

* **Theory:**
  * Row-oriented (CSV) vs. columnar storage formats (Apache Parquet, Arrow).
  * Analytical vector execution with `DuckDB` and SIMD-accelerated lazy query plans in `Polars`.
  * Streaming JSON ingestion, nested object flattening, and schema normalization.
* **Practice & Labs:**
  * Ingest columnar Parquet with DuckDB in [`02_data_engineering/src/duckdb_parquet.py`](02_data_engineering/src/duckdb_parquet.py).
  * Implement semi-structured JSON telemetry parsing in [`02_data_engineering/src/data_engineering.py`](02_data_engineering/src/data_engineering.py).
* **Deliverable & Tests:**
  * Pass [`02_data_engineering/tests/test_duckdb_parquet.py`](02_data_engineering/tests/test_duckdb_parquet.py) and [`test_data_engineering.py`](02_data_engineering/tests/test_data_engineering.py).

### Week 6: Leakage-Free Transformations, Point-in-Time Joins & Data Contracts

* **Theory:**
  * Out-of-fold smoothed empirical Bayes target encoding with smoothing parameter $m$.
  * Yeo-Johnson power transformations for variance stabilization across negative/zero values.
  * Point-in-time temporal `ASOF` joins: matching feature events to predictions without lookahead leakage.
  * Runtime schema enforcement and statistical constraint contracts with `Pandera`.
* **Practice & Labs:**
  * Build out-of-fold encoders and power transforms in [`02_data_engineering/src/advanced_transforms.py`](02_data_engineering/src/advanced_transforms.py).
  * Build temporal `ASOF` joins in [`02_data_engineering/src/temporal_joins.py`](02_data_engineering/src/temporal_joins.py).
  * Write Pandera data contract validation schemas in [`02_data_engineering/src/data_validation.py`](02_data_engineering/src/data_validation.py).
* **Deliverable & Tests:**
  * Pass [`02_data_engineering/tests/test_advanced_transforms.py`](02_data_engineering/tests/test_advanced_transforms.py), [`test_temporal_joins.py`](02_data_engineering/tests/test_temporal_joins.py), and [`test_data_validation.py`](02_data_engineering/tests/test_data_validation.py).

---

## 🎯 Phase 3: Supervised Learning, Ensembles & Evaluation Metrics (Weeks 7–8)

### Phase Description

Comprehensive coverage of supervised machine learning: regularized linear models, support vector machines, naive Bayes, metric-based non-parametric models, recursive CART trees, gradient boosting ensembles, time-series forecasting, and exhaustive diagnostic evaluation metrics.

### Week 7: Linear Models, SVMs, Naive Bayes & Metric Learning

* **Theory:**
  * Vectorized Ordinary Least Squares (OLS), Ridge ($L_2$ Tikhonov), Lasso ($L_1$ coordinate descent), and ElasticNet.
  * Soft-margin Linear Support Vector Machines with subgradient hinge loss optimization: $\min \frac{1}{2}\|w\|_2^2 + C \sum \max(0, 1 - y_i(w^Tx_i + b))$.
  * Gaussian Naive Bayes: conditional independence assumptions and numerical log-likelihood stabilization.
  * K-Nearest Neighbors (k-NN) with Euclidean, Manhattan, and Minkowski metrics.
* **Practice & Labs:**
  * Repo Directory: [`03_supervised_learning/`](03_supervised_learning/)
  * Model Catalog: [`03_supervised_learning/01_supervised_models_catalog.md`](03_supervised_learning/01_supervised_models_catalog.md)
  * Implement models in [`03_supervised_learning/src/linear_models.py`](03_supervised_learning/src/linear_models.py), [`src/svm.py`](03_supervised_learning/src/svm.py), [`src/naive_bayes.py`](03_supervised_learning/src/naive_bayes.py), and [`src/knn.py`](03_supervised_learning/src/knn.py).
* **Deliverable & Tests:**
  * Pass [`03_supervised_learning/tests/test_linear_models.py`](03_supervised_learning/tests/test_linear_models.py), [`test_svm.py`](03_supervised_learning/tests/test_svm.py), [`test_naive_bayes.py`](03_supervised_learning/tests/test_naive_bayes.py), and [`test_knn.py`](03_supervised_learning/tests/test_knn.py).

### Week 8: Decision Trees, Gradient Boosting, Time-Series & Evaluation Metrics

* **Theory:**
  * Classification and Regression Trees (CART): recursive binary splitting via Gini Impurity and MSE reduction.
  * Bagging vs. Boosting: Random Forests, LightGBM, and XGBoost second-order Taylor expansions.
  * Time-Series Forecasting: lag features, rolling statistics, and Walk-Forward expanding-window cross-validation without lookahead leakage.
  * Comprehensive Evaluation Metrics: Accuracy Paradox, Precision, Recall, Specificity, F1, $F_\beta$, Balanced Accuracy, Matthews Correlation Coefficient (MCC), ROC-AUC, PR-AUC, Brier Score, MAE, MSE, RMSE, MAPE, and $R^2$.
* **Practice & Labs:**
  * Metrics Guide: [`03_supervised_learning/02_evaluation_metrics_guide.md`](03_supervised_learning/02_evaluation_metrics_guide.md)
  * Implement CART tree from scratch in [`03_supervised_learning/src/decision_tree.py`](03_supervised_learning/src/decision_tree.py).
  * Build ensemble benchmarks in [`03_supervised_learning/src/ensemble_benchmark.py`](03_supervised_learning/src/ensemble_benchmark.py).
  * Build time-series validation in [`03_supervised_learning/src/time_series.py`](03_supervised_learning/src/time_series.py).
  * Implement complete vectorized metrics in [`03_supervised_learning/src/evaluation_metrics.py`](03_supervised_learning/src/evaluation_metrics.py).
* **Deliverable & Tests:**
  * Pass [`03_supervised_learning/tests/test_decision_tree.py`](03_supervised_learning/tests/test_decision_tree.py), [`test_ensemble_benchmark.py`](03_supervised_learning/tests/test_ensemble_benchmark.py), [`test_time_series.py`](03_supervised_learning/tests/test_time_series.py), and [`test_evaluation_metrics.py`](03_supervised_learning/tests/test_evaluation_metrics.py).

---

## 🔍 Phase 4: Unsupervised Learning & Dimensionality Reduction (Weeks 9–10)

### Phase Description

Explores discovering latent structures without supervisory labels: centroid clustering, density-based clustering, probabilistic mixtures via EM, manifold discovery, and unsupervised anomaly detection.

### Week 9: K-Means++, DBSCAN & Gaussian Mixture Models (EM)

* **Theory:**
  * Lloyd's alternating minimization algorithm and K-Means++ probabilistic $D(x)^2$ centroid seeding.
  * Density-Based Spatial Clustering of Applications with Noise (DBSCAN): core points, $\epsilon$-neighborhoods, border points, and arbitrary-shape cluster expansion.
  * Gaussian Mixture Models (GMM) and Expectation-Maximization (EM): E-step posterior responsibilities $\gamma_{ik}$ and M-step parameter re-estimation with diagonal covariance regularization.
* **Practice & Labs:**
  * Repo Directory: [`04_unsupervised_learning/`](04_unsupervised_learning/)
  * Model Catalog: [`04_unsupervised_learning/01_unsupervised_models_catalog.md`](04_unsupervised_learning/01_unsupervised_models_catalog.md)
  * Implement K-Means in [`04_unsupervised_learning/src/kmeans.py`](04_unsupervised_learning/src/kmeans.py), DBSCAN in [`src/dbscan.py`](04_unsupervised_learning/src/dbscan.py), and GMM in [`src/gmm.py`](04_unsupervised_learning/src/gmm.py).
* **Deliverable & Tests:**
  * Pass [`04_unsupervised_learning/tests/test_kmeans.py`](04_unsupervised_learning/tests/test_kmeans.py), [`test_dbscan.py`](04_unsupervised_learning/tests/test_dbscan.py), and [`test_gmm.py`](04_unsupervised_learning/tests/test_gmm.py).

### Week 10: Dimensionality Reduction & Isolation Forests

* **Theory:**
  * Principal Component Analysis (PCA): covariance eigendecomposition, singular values, explained variance ratios, and low-rank reconstruction.
  * Unsupervised Outlier Detection: Isolation Forests, recursive random hyperplanes, and anomaly scoring based on average Binary Search Tree path length $c(n)$.
* **Practice & Labs:**
  * Implement PCA in [`04_unsupervised_learning/src/pca.py`](04_unsupervised_learning/src/pca.py) and Isolation Forest in [`src/isolation_forest.py`](04_unsupervised_learning/src/isolation_forest.py).
* **Deliverable & Tests:**
  * Pass [`04_unsupervised_learning/tests/test_pca.py`](04_unsupervised_learning/tests/test_pca.py) and [`test_isolation_forest.py`](04_unsupervised_learning/tests/test_isolation_forest.py).

---

## 🔄 Phase 5: Self-Supervised Learning & Contrastive Dynamics (Week 11)

### Phase Description

Students learn representation learning on unlabeled data using pretext tasks, contrastive objectives, and bottleneck autoencoders.

### Week 11: Contrastive InfoNCE Loss, SimCLR & Deep Autoencoders

* **Theory:**
  * Self-supervised paradigms: generative vs. contrastive representation learning.
  * Normalized Temperature-scaled Cross Entropy (NT-Xent / InfoNCE) loss: mutual information lower bounds and temperature scaling $\tau$.
  * Dimensional collapse prevention; Wang & Isola geometric Alignment and Uniformity evaluation metrics on hyperspheres.
  * Deep Bottleneck Autoencoders and Denoising Autoencoders (DAE) with Gaussian noise corruption.
  * SimCLR framework: stochastic augmentations, base encoder, and non-linear MLP projection head.
* **Practice & Labs:**
  * Repo Directory: [`05_self_supervised_learning/`](05_self_supervised_learning/)
  * Model Catalog: [`05_self_supervised_learning/01_self_supervised_models_catalog.md`](05_self_supervised_learning/01_self_supervised_models_catalog.md)
  * Implement `NTXentLoss` and Alignment/Uniformity in [`05_self_supervised_learning/src/contrastive_loss.py`](05_self_supervised_learning/src/contrastive_loss.py).
  * Implement `DeepAutoencoder` in [`src/autoencoder.py`](05_self_supervised_learning/src/autoencoder.py) and `SimCLRFramework` in [`src/simclr_framework.py`](05_self_supervised_learning/src/simclr_framework.py).
* **Deliverable & Tests:**
  * Pass [`05_self_supervised_learning/tests/test_contrastive_loss.py`](05_self_supervised_learning/tests/test_contrastive_loss.py), [`test_autoencoder.py`](05_self_supervised_learning/tests/test_autoencoder.py), and [`test_simclr_framework.py`](05_self_supervised_learning/tests/test_simclr_framework.py).

---

## 🤝 Phase 6: Semi-Supervised Learning & Graph Propagation (Week 12)

### Phase Description

Combines small labeled datasets with massive pools of unlabeled data using pseudo-labeling, graph Markov walks, and consistency regularization.

### Week 12: Self-Training, Label Propagation & Consistency Regularization

* **Theory:**
  * Semi-supervised learning assumptions: smoothness assumption, cluster assumption, and manifold assumption.
  * Self-Training / Pseudo-Labeling: confidence thresholding $\tau$, iterative pool expansion, and confirmation bias mitigation.
  * Graph-Based Label Propagation: RBF affinity kernel matrices, transition probability normalization, harmonic Dirichlet energy minimization, and clamped ground-truth nodes.
  * Consistency Regularization: perturbation invariance, probability temperature sharpening ($q_i \propto p_i^{1/T}$), and ramp-up unsupervised loss scheduling.
* **Practice & Labs:**
  * Repo Directory: [`06_semi_supervised_learning/`](06_semi_supervised_learning/)
  * Model Catalog: [`06_semi_supervised_learning/01_semi_supervised_models_catalog.md`](06_semi_supervised_learning/01_semi_supervised_models_catalog.md)
  * Implement `SelfTrainingClassifier` in [`06_semi_supervised_learning/src/pseudo_labeling.py`](06_semi_supervised_learning/src/pseudo_labeling.py), `LabelPropagation` in [`src/label_propagation.py`](06_semi_supervised_learning/src/label_propagation.py), and sharpening functions in [`src/consistency_regularization.py`](06_semi_supervised_learning/src/consistency_regularization.py).
* **Deliverable & Tests:**
  * Pass [`06_semi_supervised_learning/tests/test_pseudo_labeling.py`](06_semi_supervised_learning/tests/test_pseudo_labeling.py), [`test_label_propagation.py`](06_semi_supervised_learning/tests/test_label_propagation.py), and [`test_consistency_regularization.py`](06_semi_supervised_learning/tests/test_consistency_regularization.py).

---

## 🧠 Phase 7: Deep Learning Architectures & Attention Mechanisms (Weeks 13–14)

### Phase Description

Focuses on deep representations, residual gradient highways, spatial object detection metrics, and self-attention Transformer mechanisms.

### Week 13: Residual Networks (ResNet) & Object Detection Metrics

* **Theory:**
  * Degradation problem in deep neural networks and vanishing gradients.
  * Residual learning: identity shortcut connections ($\mathcal{F}(x) + x$) and projection blocks preserving gradient highways.
  * Vectorized spatial Intersection over Union (IoU) calculation and Non-Maximum Suppression (NMS) bounding box deduplication.
* **Practice & Labs:**
  * Repo Directory: [`07_deep_learning/`](07_deep_learning/)
  * Implement ResNet blocks in [`07_deep_learning/src/resnet.py`](07_deep_learning/src/resnet.py).
  * Implement IoU and NMS in [`07_deep_learning/src/detection_metrics.py`](07_deep_learning/src/detection_metrics.py).
* **Deliverable & Tests:**
  * Pass [`07_deep_learning/tests/test_resnet.py`](07_deep_learning/tests/test_resnet.py) and [`test_detection_metrics.py`](07_deep_learning/tests/test_detection_metrics.py).

### Week 14: Transformers, Multi-Head Attention & RAG Vector Search

* **Theory:**
  * Scaled Dot-Product Attention: query, key, value projections and $\text{softmax}(QK^T / \sqrt{d_k})V$.
  * Multi-Head Attention (MHA) with causal autoregressive decoding masks.
  * Dense vector embeddings, in-memory cosine vector store, semantic document chunking, and Retrieval-Augmented Generation (RAG) context synthesis.
* **Practice & Labs:**
  * Implement `MultiHeadAttention` in [`07_deep_learning/src/attention.py`](07_deep_learning/src/attention.py).
  * Build vector store and RAG pipeline in [`07_deep_learning/src/rag.py`](07_deep_learning/src/rag.py).
* **Deliverable & Tests:**
  * Pass [`07_deep_learning/tests/test_attention.py`](07_deep_learning/tests/test_attention.py) and [`test_rag.py`](07_deep_learning/tests/test_rag.py).

---

## 🎮 Phase 8: Reinforcement Learning & Sequential Decisions (Weeks 15–16)

### Phase Description

Covers sequential decision-making under uncertainty, Markov Decision Processes, value-based methods, deep Q-networks, and policy gradient optimization.

### Week 15: MDPs, GridWorlds & Tabular Q-Learning

* **Theory:**
  * Markov Decision Processes (MDPs): states, actions, transition probabilities, discount factor $\gamma$, and reward signals.
  * Bellman Expectation and Optimality Equations for $V(s)$ and $Q(s, a)$.
  * Exploration vs. exploitation: $\epsilon$-greedy strategy with exponential decay.
  * Off-policy Tabular Q-Learning update rule: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
* **Practice & Labs:**
  * Repo Directory: [`08_reinforcement_learning/`](08_reinforcement_learning/)
  * Model Catalog: [`08_reinforcement_learning/01_reinforcement_learning_models_catalog.md`](08_reinforcement_learning/01_reinforcement_learning_models_catalog.md)
  * Implement GridWorld environment and tabular agent in [`08_reinforcement_learning/src/q_learning.py`](08_reinforcement_learning/src/q_learning.py).
* **Deliverable & Tests:**
  * Pass [`08_reinforcement_learning/tests/test_q_learning.py`](08_reinforcement_learning/tests/test_q_learning.py).

### Week 16: Deep Q-Networks (DQN) & Policy Gradients (REINFORCE)

* **Theory:**
  * Function approximation instabilities (the Deadly Triad: function approximation + bootstrapping + off-policy training).
  * Deep Q-Networks (DQN): Cyclic Experience Replay Buffer and periodic Target Network weight synchronization.
  * Policy Gradient Theorem: optimizing parameterized policies $\pi_\theta(a|s)$ directly via expected cumulative returns.
  * REINFORCE algorithm: Monte Carlo trajectory sampling and baseline subtraction for variance reduction.
* **Practice & Labs:**
  * Implement `DQNAgent` in [`08_reinforcement_learning/src/dqn_agent.py`](08_reinforcement_learning/src/dqn_agent.py).
  * Implement `REINFORCEAgent` in [`08_reinforcement_learning/src/policy_gradient.py`](08_reinforcement_learning/src/policy_gradient.py).
* **Deliverable & Tests:**
  * Pass [`08_reinforcement_learning/tests/test_dqn_agent.py`](08_reinforcement_learning/tests/test_dqn_agent.py) and [`test_policy_gradient.py`](08_reinforcement_learning/tests/test_policy_gradient.py).

---

## 🔀 Phase 9: Transfer Learning, PEFT & Domain Adaptation (Weeks 17–18)

### Phase Description

Adapting foundation models to downstream domains efficiently: backbone freezing, parameter-efficient fine-tuning (LoRA), and distribution alignment.

### Week 17: Backbone Freezing, Fine-Tuning & LoRA Adaptation

* **Theory:**
  * Inductive transfer, catastrophic forgetting, feature extraction vs. full fine-tuning.
  * Low-Rank Adaptation (LoRA - Hu et al., 2021): factorizing weight updates into rank-$r$ matrices $\Delta W = \frac{\alpha}{r} A B$.
  * Trainable parameter accounting ($< 0.1\%$ trainable) and zero-latency inference weight merging ($W_{\text{merged}} = W_0 + \Delta W$).
* **Practice & Labs:**
  * Repo Directory: [`09_transfer_learning/`](09_transfer_learning/)
  * Model Catalog: [`09_transfer_learning/01_transfer_learning_models_catalog.md`](09_transfer_learning/01_transfer_learning_models_catalog.md)
  * Implement two-stage transfer model in [`09_transfer_learning/src/transfer_learning.py`](09_transfer_learning/src/transfer_learning.py).
  * Implement `LoRALinear` in [`09_transfer_learning/src/lora.py`](09_transfer_learning/src/lora.py).
* **Deliverable & Tests:**
  * Pass [`09_transfer_learning/tests/test_transfer_learning.py`](09_transfer_learning/tests/test_transfer_learning.py) and [`test_lora.py`](09_transfer_learning/tests/test_lora.py).

### Week 18: Domain Adaptation & Maximum Mean Discrepancy (MMD)

* **Theory:**
  * Covariate shift between source $P(X_S)$ and target $P(X_T)$ domains under identical label conditional distribution $P(Y|X)$.
  * Reproducing Kernel Hilbert Space (RKHS) mean embeddings.
  * Maximum Mean Discrepancy (MMD) non-parametric two-sample test statistic using multi-kernel Gaussian RBFs.
* **Practice & Labs:**
  * Implement `MaximumMeanDiscrepancy` in [`09_transfer_learning/src/domain_adaptation.py`](09_transfer_learning/src/domain_adaptation.py).
* **Deliverable & Tests:**
  * Pass [`09_transfer_learning/tests/test_domain_adaptation.py`](09_transfer_learning/tests/test_domain_adaptation.py).

---

## 🚀 Phase 10: Production MLOps, APIs & Deployment (Weeks 19–20)

### Phase Description

Hardens models into production microservices with FastAPI, strict Pydantic schemas, ONNXRuntime hardware acceleration, MLflow experiment tracking, Docker containerization, and CI/CD quality gates.

### Week 19: REST APIs with FastAPI, Pydantic & ONNX Acceleration

* **Theory:**
  * Asynchronous Server Gateway Interface (ASGI) concurrency and event loop execution.
  * Strict input data validation and sanitization via Pydantic models.
  * Open Neural Network Exchange (ONNX) specification, operator fusion, constant folding, and multi-threaded CPU inference via `ONNXRuntime`.
* **Practice & Labs:**
  * Repo Directory: [`10_mlops_production/`](10_mlops_production/)
  * Build inference endpoints in [`10_mlops_production/app/main.py`](10_mlops_production/app/main.py) with schemas in [`app/schemas.py`](10_mlops_production/app/schemas.py).
  * Build ONNX exporter and benchmark engine in [`10_mlops_production/app/onnx_converter.py`](10_mlops_production/app/onnx_converter.py).
* **Deliverable & Tests:**
  * Pass [`10_mlops_production/tests/test_api.py`](10_mlops_production/tests/test_api.py) and [`test_onnx.py`](10_mlops_production/tests/test_onnx.py).

### Week 20: MLflow Lifecycle Governance, Docker & Capstone Defense

* **Theory:**
  * Experiment tracking, parameter and metric logging, run tagging, and automated artifact registration using `MLflow`.
  * Production containerization: multi-stage `Dockerfile` builds, dependency isolation, and non-root security principles.
  * Multi-container orchestration via `docker-compose.yml`.
* **Practice & Labs:**
  * Implement MLflow run management in [`10_mlops_production/app/mlflow_tracker.py`](10_mlops_production/app/mlflow_tracker.py).
  * Build multi-stage [`10_mlops_production/Dockerfile`](10_mlops_production/Dockerfile) and [`docker-compose.yml`](10_mlops_production/docker-compose.yml).
  * Assemble end-to-end Capstone project integrating data ingestion, model training, ONNX serialization, and containerized deployment.
* **Deliverable & Assessment:**
  * Pass [`10_mlops_production/tests/test_mlflow.py`](10_mlops_production/tests/test_mlflow.py) and defend Capstone project.

---

## 🔍 Phase 11: Explainable AI (XAI) & Model Interpretability (Week 21)

### Phase Description

Black-box machine learning models deployed in regulated domains (healthcare, banking, criminal justice) require actionable, mathematically grounded interpretations. This phase equips students with local and global explainability algorithms based on coalitional game theory and surrogate modeling.

### Week 21: Shapley Values, LIME, Permutation Importance & PDP

* **Theory:**
  * Coalitional game theory and the Shapley value formulation:
    $$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
  * Proof of uniqueness under the 4 Shapley axioms: Efficiency, Symmetry, Dummy Player, and Additivity.
  * KernelSHAP and Monte Carlo Sampling Shapley approximations.
  * Local Interpretable Model-agnostic Explanations (LIME): exponential similarity kernels $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$ and weighted ridge regression surrogate models.
  * Vectorized Permutation Feature Importance: measuring empirical score drop under feature shuffling.
  * Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE): marginalizing out nuisance feature distributions.
* **Practice & Labs:**
  * Repo Directory: [`11_explainable_ai/`](11_explainable_ai/)
  * Catalog: [`11_explainable_ai/01_explainable_ai_catalog.md`](11_explainable_ai/01_explainable_ai_catalog.md)
  * Implement exact and sampling Shapley explainers in [`11_explainable_ai/src/shapley_values.py`](11_explainable_ai/src/shapley_values.py).
  * Implement LIME in [`src/lime_explainer.py`](11_explainable_ai/src/lime_explainer.py).
  * Implement Permutation Importance in [`src/permutation_importance.py`](11_explainable_ai/src/permutation_importance.py).
  * Implement PDP and ICE in [`src/partial_dependence.py`](11_explainable_ai/src/partial_dependence.py).
* **Deliverable & Tests:**
  * Pass [`11_explainable_ai/tests/test_shapley_values.py`](11_explainable_ai/tests/test_shapley_values.py), [`test_lime_explainer.py`](11_explainable_ai/tests/test_lime_explainer.py), [`test_permutation_importance.py`](11_explainable_ai/tests/test_permutation_importance.py), and [`test_partial_dependence.py`](11_explainable_ai/tests/test_partial_dependence.py).

---

## 📈 Phase 12: Production Observability & Drift Detection (Week 22)

### Phase Description

Once deployed, machine learning models decay silently due to changes in real-world data distributions. This phase covers statistical testing, distance metrics, and streaming drift algorithms to detect covariate shift and concept drift before business KPIs degrade.

### Week 22: Covariate Shift, PSI, KS-Tests & Streaming Drift Algorithms

* **Theory:**
  * Taxonomy of drift: Covariate shift ($P(X)$ changes), Concept drift ($P(Y|X)$ changes), and Prior probability shift ($P(Y)$ changes).
  * Population Stability Index (PSI): symmetric relative entropy over quantile bins $\text{PSI} = \sum_{k} (P_k - Q_k) \ln(P_k / Q_k)$ with industry action thresholds ($<0.1$ stable, $>0.2$ significant shift).
  * Two-sample Kolmogorov-Smirnov (KS) test: non-parametric supremum test statistic $D = \sup_x |F_1(x) - F_2(x)|$.
  * Wasserstein-1 Earth Mover's Distance: minimum cost to transport probability mass between distributions.
  * Streaming Adaptive Windowing (ADWIN): maintaining variable-length windows and evaluating Hoeffding bound cutoffs.
  * Page-Hinkley cumulative mean deviation test: sub-millisecond detection of sudden concept shifts in streaming data.
* **Practice & Labs:**
  * Repo Directory: [`12_model_monitoring_drift/`](12_model_monitoring_drift/)
  * Catalog: [`12_model_monitoring_drift/01_drift_monitoring_catalog.md`](12_model_monitoring_drift/01_drift_monitoring_catalog.md)
  * Implement PSI, KS-test, and Wasserstein distance in [`12_model_monitoring_drift/src/data_drift.py`](12_model_monitoring_drift/src/data_drift.py).
  * Implement ADWIN and Page-Hinkley in [`src/concept_drift.py`](12_model_monitoring_drift/src/concept_drift.py).
* **Deliverable & Tests:**
  * Pass [`12_model_monitoring_drift/tests/test_data_drift.py`](12_model_monitoring_drift/tests/test_data_drift.py) and [`test_concept_drift.py`](12_model_monitoring_drift/tests/test_concept_drift.py).

---

## ⚡ Phase 13: Generative AI & LLM Systems Engineering (Week 23)

### Phase Description

Modern LLM deployments are governed by hardware memory bandwidth limits rather than raw compute capacity. This phase covers systems engineering principles: KV-cache memory optimization, post-training quantization, direct preference alignment, and hybrid search.

### Week 23: KV-Cache Bottlenecks, Quantization, DPO Loss & Hybrid Search

* **Theory:**
  * Transformer decoder inference bottlenecks: Prefill compute-bound GEMM vs. Decoding memory-bandwidth-bound GEMV.
  * Key-Value (KV) Cache: caching past tokens to reduce autoregressive attention from $O(T^2)$ to $O(T)$; calculating VRAM footprint: $2 \cdot 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot T$.
  * Post-Training Quantization (PTQ): Symmetric and Asymmetric INT8 scale $S$ and zero-point $Z$ calibration.
  * NormalFloat4 (NF4 - QLoRA): information-theoretically optimal non-linear quantile codebook for Gaussian-distributed neural network weights.
  * Direct Preference Optimization (DPO - Rafailov et al., 2023): closed-form Bradley-Terry log-ratio preference loss eliminating separate reward models and unstable PPO loops.
  * Hybrid Retrieval: Reciprocal Rank Fusion (RRF: $\sum \frac{1}{k + r(d)}$) combining dense cosine semantic search and sparse lexical BM25 matching.
* **Practice & Labs:**
  * Repo Directory: [`13_llm_systems/`](13_llm_systems/)
  * Catalog: [`13_llm_systems/01_llm_systems_catalog.md`](13_llm_systems/01_llm_systems_catalog.md)
  * Implement `KVCache` and cached MHA in [`13_llm_systems/src/kv_cache.py`](13_llm_systems/src/kv_cache.py).
  * Implement uniform INT8 and NF4 quantizers in [`src/quantization.py`](13_llm_systems/src/quantization.py).
  * Implement DPO loss in [`src/dpo_loss.py`](13_llm_systems/src/dpo_loss.py).
  * Implement RRF and `HybridSearchEngine` in [`src/hybrid_retrieval.py`](13_llm_systems/src/hybrid_retrieval.py).
* **Deliverable & Tests:**
  * Pass [`13_llm_systems/tests/test_kv_cache.py`](13_llm_systems/tests/test_kv_cache.py), [`test_quantization.py`](13_llm_systems/tests/test_quantization.py), [`test_dpo_loss.py`](13_llm_systems/tests/test_dpo_loss.py), and [`test_hybrid_retrieval.py`](13_llm_systems/tests/test_hybrid_retrieval.py).

---

## 🎯 Phase 14: Recommendation Systems & Collaborative Filtering (Week 24)

### Phase Description

Real-world recommendation architectures operate under a two-stage funnel: candidate retrieval filtering millions of items down to hundreds under strict sub-10ms SLAs, followed by ranking models capturing non-linear feature interactions for CTR prediction.

### Week 24: Implicit ALS, Two-Tower Dual Encoders & Factorization Machines

* **Theory:**
  * Implicit feedback formulation (Hu, Koren, Volinsky): preference $p_{ui} \in \{0, 1\}$ and confidence $c_{ui} = 1 + \alpha r_{ui}$.
  * Alternating Least Squares (ALS) optimization: exploiting sparse interactions $Y^TY + Y_{\text{active}}^T(C-I)Y_{\text{active}}$ for $O(f^2 n_u + f^3)$ solves per user.
  * Two-Tower Dual-Encoder architectures: projecting sparse user and item metadata into a normalized metric space with in-batch contrastive softmax learning.
  * Factorization Machines (FM - Rendle, 2010): second-order feature interaction ranking in linear $O(kd)$ time using the algebraic identity:
    $$\sum_{i < j} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left[ \left(\sum_i v_{if} x_i\right)^2 - \sum_i v_{if}^2 x_i^2 \right]$$
* **Practice & Labs:**
  * Repo Directory: [`14_recommender_systems/`](14_recommender_systems/)
  * Catalog: [`14_recommender_systems/01_recommender_systems_catalog.md`](14_recommender_systems/01_recommender_systems_catalog.md)
  * Implement `ImplicitALS` in [`14_recommender_systems/src/als_recommender.py`](14_recommender_systems/src/als_recommender.py).
  * Implement `TwoTowerModel` in [`src/two_tower.py`](14_recommender_systems/src/two_tower.py).
  * Implement `FactorizationMachine` in [`src/factorization_machine.py`](14_recommender_systems/src/factorization_machine.py).
* **Deliverable & Tests:**
  * Pass [`14_recommender_systems/tests/test_als_recommender.py`](14_recommender_systems/tests/test_als_recommender.py), [`test_two_tower.py`](14_recommender_systems/tests/test_two_tower.py), and [`test_factorization_machine.py`](14_recommender_systems/tests/test_factorization_machine.py).

---

## 🎲 Phase 15: Uncertainty Quantification & Conformal Prediction (Week 25)

### Phase Description

Deploying machine learning models in safety-critical domains requires calibrated, mathematically grounded measures of uncertainty. Students learn distribution-free finite-sample conformal prediction and Bayesian variance decomposition.

### Week 25: Conformal Prediction Sets, Bayesian Regression & Gaussian Processes

* **Theory:**
  * Inductive (Split) Conformal Prediction: non-conformity scores $s_i = |y_i - \hat{\mu}(x_i)|$ and adjusted quantiles $\hat{q} = \text{Quantile}(s_{1:n}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n})$.
  * Mathematical marginal coverage guarantee: $1 - \alpha \le \mathbb{P}(Y_{n+1} \in C(X_{n+1})) \le 1 - \alpha + \frac{1}{n+1}$ without distribution or model assumptions.
  * Conformal classification prediction sets $\mathcal{C}(x) = \{k : \hat{P}(Y=k|x) \ge 1 - \hat{q}\}$.
  * Bayesian Linear Regression: conjugate Gaussian prior $\mathcal{N}(0, \alpha^{-1} I)$, analytical posterior $S_N^{-1} = \alpha I + \beta X^TX$, and predictive variance decomposition:
    $$\sigma_N^2(x) = \underbrace{\frac{1}{\beta}}_{\text{Aleatoric (noise)}} + \underbrace{x^T S_N x}_{\text{Epistemic (model knowledge)}}$$
  * Gaussian Process Regression (GPR): Squared Exponential (RBF) kernel and exact analytical inference via Cholesky decomposition $L L^T = K + \sigma_n^2 I$.
* **Practice & Labs:**
  * Repo Directory: [`15_uncertainty_quantification/`](15_uncertainty_quantification/)
  * Catalog: [`15_uncertainty_quantification/01_uncertainty_quantification_catalog.md`](15_uncertainty_quantification/01_uncertainty_quantification_catalog.md)
  * Implement `SplitConformalRegressor` and `ConformalClassifier` in [`15_uncertainty_quantification/src/conformal_prediction.py`](15_uncertainty_quantification/src/conformal_prediction.py).
  * Implement `BayesianLinearRegression` in [`src/bayesian_linear_regression.py`](15_uncertainty_quantification/src/bayesian_linear_regression.py).
  * Implement `GaussianProcessRegressor` in [`src/gaussian_process.py`](15_uncertainty_quantification/src/gaussian_process.py).
* **Deliverable & Tests:**
  * Pass [`15_uncertainty_quantification/tests/test_conformal_prediction.py`](15_uncertainty_quantification/tests/test_conformal_prediction.py), [`test_bayesian_linear_regression.py`](15_uncertainty_quantification/tests/test_bayesian_linear_regression.py), and [`test_gaussian_process.py`](15_uncertainty_quantification/tests/test_gaussian_process.py).

---

## 🕸️ Phase 16: Graph Machine Learning & Graph Neural Networks (Week 26)

### Phase Description

Relational domains (molecules, social networks, fraud detection, knowledge graphs) cannot be represented on Euclidean grids. This phase covers spatial and spectral graph neural networks operating on arbitrary topologies.

### Week 26: MPNN Framework, Graph Convolutions (GCN) & Graph Attention (GAT)

* **Theory:**
  * Spatial deep learning over non-Euclidean graphs.
  * Message Passing Neural Networks (MPNN - Gilmer et al., 2017): message computation $m_{ij} = M(h_i, h_j)$, permutation-invariant aggregation $\bigoplus \in \{\text{sum}, \text{mean}, \text{max}\}$, and node update $h_i^{(l+1)} = U(h_i^{(l)}, m_i)$.
  * Graph Convolutional Networks (GCN - Kipf & Welling, 2017): localized first-order spectral Chebyshev approximation, renormalization trick $\tilde{A} = A + I_N$, normalized symmetric Laplacian $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$, and layer propagation $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$.
  * Graph Attention Networks (GAT - Veličković et al., 2018): anisotropic self-attention over node neighborhoods $e_{ij} = \text{LeakyReLU}(a^T [Wh_i \,\|\, Wh_j])$, masked softmax coefficients $\alpha_{ij}$, and multi-head attention aggregation.
* **Practice & Labs:**
  * Repo Directory: [`16_graph_neural_networks/`](16_graph_neural_networks/)
  * Catalog: [`16_graph_neural_networks/01_gnn_catalog.md`](16_graph_neural_networks/01_gnn_catalog.md)
  * Implement `GraphConvolutionalNetwork` in [`16_graph_neural_networks/src/gcn.py`](16_graph_neural_networks/src/gcn.py).
  * Implement `GraphAttentionLayer` in [`src/gat.py`](16_graph_neural_networks/src/gat.py).
  * Implement `MessagePassingLayer` in [`src/mpnn.py`](16_graph_neural_networks/src/mpnn.py).
* **Deliverable & Tests:**
  * Pass [`16_graph_neural_networks/tests/test_gcn.py`](16_graph_neural_networks/tests/test_gcn.py), [`test_gat.py`](16_graph_neural_networks/tests/test_gat.py), and [`test_mpnn.py`](16_graph_neural_networks/tests/test_mpnn.py).

---

## 🖥️ Phase 17: Distributed Deep Learning & Cluster Parallelism (Week 27)

### Phase Description

Training billion-parameter foundation models exceeds single accelerator limits. This phase explores collective cluster communication algorithms, GPU memory sharding with ZeRO, and numerical precision stability.

### Week 27: Ring All-Reduce, ZeRO Memory Sharding & Dynamic Loss Scaling

* **Theory:**
  * Distributed Data Parallel (DDP) communication bottlenecks; parameter server $O(P \cdot N)$ limitations.
  * Ring All-Reduce communication protocol (Patarasuk & Yuan, 2009): logical ring topology, Scatter-Reduce phase ($P-1$ steps), and All-Gather phase ($P-1$ steps).
  * Communication volume invariant: $2 \frac{P-1}{P} N$, strictly independent of cluster size $P$.
  * Static GPU memory accounting under mixed precision Adam (16 bytes per parameter baseline).
  * ZeRO (Zero Redundancy Optimizer - Rajbhandari et al., 2020) partitioning:
    * ZeRO Stage 1 ($P_{os}$): Optimizer state partitioning ($4\times$ memory reduction).
    * ZeRO Stage 2 ($P_{os+g}$): Gradient and optimizer partitioning ($8\times$ memory reduction).
    * ZeRO Stage 3 ($P_{os+g+p}$): Parameter, gradient, and optimizer partitioning ($N_d \times$ linear memory scaling).
  * Dynamic Loss Scaling for FP16 mixed precision: preventing gradient underflow without manual scale tuning via adaptive overflow detection and scale backoff/growth.
* **Practice & Labs:**
  * Repo Directory: [`17_distributed_training/`](17_distributed_training/)
  * Catalog: [`17_distributed_training/01_distributed_training_catalog.md`](17_distributed_training/01_distributed_training_catalog.md)
  * Implement `ring_allreduce` in [`17_distributed_training/src/ring_allreduce.py`](17_distributed_training/src/ring_allreduce.py).
  * Implement `ZeROMemoryEstimator` in [`src/zero_memory.py`](17_distributed_training/src/zero_memory.py).
  * Implement `DynamicLossScaler` in [`src/loss_scaler.py`](17_distributed_training/src/loss_scaler.py).
* **Deliverable & Tests:**
  * Pass [`17_distributed_training/tests/test_ring_allreduce.py`](17_distributed_training/tests/test_ring_allreduce.py), [`test_zero_memory.py`](17_distributed_training/tests/test_zero_memory.py), and [`test_loss_scaler.py`](17_distributed_training/tests/test_loss_scaler.py).

---

## 🛡️ Phase 18: AI Safety, Privacy & Algorithmic Fairness (Week 28)

### Phase Description

Deploying machine learning models in society exposes organizations to adversarial manipulation, data leakage, and discriminatory societal harms. Students master adversarial robustness, differential privacy, and mathematical fairness audits.

### Week 28: Adversarial Robustness, DP-SGD & Algorithmic Bias Auditing

* **Theory:**
  * Adversarial vulnerability in deep neural networks: linear behavior in high-dimensional manifolds.
  * Fast Gradient Sign Method (FGSM - Goodfellow et al., 2014): single-step $L_\infty$ perturbation $x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$.
  * Projected Gradient Descent (PGD - Madry et al., 2018): multi-step iterative attack with random restart and projection onto the $L_\infty$ $\epsilon$-ball.
  * Differential Privacy: $(\epsilon, \delta)$-DP mathematical definition over adjacent datasets.
  * Differentially Private SGD (DP-SGD - Abadi et al., 2016): per-sample gradient clipping with $L_2$ norm threshold $C$ and calibrated Gaussian mechanism noise injection: $\tilde{g} = \frac{1}{B} (\sum \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I))$.
  * Algorithmic Fairness Metrics:
    * Demographic Parity (Statistical Parity): $\mathbb{P}(\hat{Y}=1 | A=0) = \mathbb{P}(\hat{Y}=1 | A=1)$ and Disparate Impact ratio (US EEOC 80% four-fifths rule).
    * Equal Opportunity: True Positive Rate (TPR) equality $\mathbb{P}(\hat{Y}=1 | A=0, Y=1) = \mathbb{P}(\hat{Y}=1 | A=1, Y=1)$.
    * Equalized Odds: Simultaneous TPR and FPR parity across sensitive demographic groups.
* **Practice & Labs:**
  * Repo Directory: [`18_ai_safety_privacy/`](18_ai_safety_privacy/)
  * Catalog: [`18_ai_safety_privacy/01_safety_privacy_catalog.md`](18_ai_safety_privacy/01_safety_privacy_catalog.md)
  * Implement FGSM and PGD in [`18_ai_safety_privacy/src/adversarial.py`](18_ai_safety_privacy/src/adversarial.py).
  * Implement `DPSGDStep` in [`src/differential_privacy.py`](18_ai_safety_privacy/src/differential_privacy.py).
  * Implement `FairnessAuditor` in [`src/fairness_audit.py`](18_ai_safety_privacy/src/fairness_audit.py).
* **Deliverable & Tests:**
  * Pass [`18_ai_safety_privacy/tests/test_adversarial.py`](18_ai_safety_privacy/tests/test_adversarial.py), [`test_differential_privacy.py`](18_ai_safety_privacy/tests/test_differential_privacy.py), and [`test_fairness_audit.py`](18_ai_safety_privacy/tests/test_fairness_audit.py).

---

## 📋 Comprehensive Repository Test Verification

To achieve certification across the complete curriculum, students must achieve a 100% test pass rate across all 18 modules (175 test functions):

```bash
# Run complete test harness across all 18 phases
pytest

# Verify code style and linting standards
ruff check .
```
