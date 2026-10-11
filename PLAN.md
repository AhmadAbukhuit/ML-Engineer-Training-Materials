# 📅 20-Week Machine Learning Engineering Curriculum Plan

This curriculum provides a structured, rigorous, 20-week roadmap designed to take students from numerical and mathematical foundations to production-grade MLOps deployment across all modern learning paradigms.

---

## 🗺️ Curriculum Overview & Phase Timeline

```text
 Weeks 1-3    │ Phase 1: Mathematical Foundations & Numerical Computing (01_math_foundations)
 Weeks 4-6    │ Phase 2: Big Data Engineering & Feature Store Pipelines (02_data_engineering)
 Weeks 7-8    │ Phase 3: Supervised Learning & Tree Ensembles (03_supervised_learning)
 Weeks 9-10   │ Phase 4: Unsupervised Learning & Dimensionality Reduction (04_unsupervised_learning)
 Week 11      │ Phase 5: Self-Supervised Learning & Contrastive Dynamics (05_self_supervised_learning)
 Week 12      │ Phase 6: Semi-Supervised Learning & Graph Propagation (06_semi_supervised_learning)
 Weeks 13-14  │ Phase 7: Deep Learning Architectures & Attention Mechanisms (07_deep_learning)
 Weeks 15-16  │ Phase 8: Reinforcement Learning & Decision Processes (08_reinforcement_learning)
 Weeks 17-18  │ Phase 9: Transfer Learning, PEFT & Domain Adaptation (09_transfer_learning)
 Weeks 19-20  │ Phase 10: Production MLOps, APIs & Deployment (10_mlops_production)
```

---

## 📐 Phase 1: Mathematical Foundations & Numerical Computing (Weeks 1–3)

### Phase Description
The objective of Phase 1 is to eliminate the "black box" mentality. Students master the mathematical machinery governing modern machine learning algorithms: vectorized linear algebra in NumPy, reverse-mode automatic differentiation, multivariate calculus, statistical estimation, and gradient-based optimization mechanics.

### Week 1: Vectorized Linear Algebra & Matrix Decompositions
* **Theory:**
  * Vector spaces, inner products, projections, matrix multiplication algorithms.
  * Matrix factorizations: Eigenvalues, Eigenvectors, and Singular Value Decomposition (SVD).
  * Principal Component Analysis (PCA) derivation via covariance matrix decomposition.
* **Practice & Labs:**
  * Repo Directory: `01_math_foundations/`
  * Implement PCA from scratch using SVD in [`01_math_foundations/src/pca.py`](01_math_foundations/src/pca.py).
* **Deliverable & Tests:**
  * Pass [`01_math_foundations/tests/test_pca.py`](01_math_foundations/tests/test_pca.py).

### Week 2: Multivariate Calculus & Reverse-Mode Autograd Engine
* **Theory:**
  * Partial derivatives, gradient vectors, Jacobians, and Hessians.
  * Directed Acyclic Graphs (DAG) for mathematical operations.
  * Reverse-mode automatic differentiation vs numerical differentiation.
* **Practice & Labs:**
  * Implement the scalar Autograd engine in [`01_math_foundations/src/autograd.py`](01_math_foundations/src/autograd.py).
* **Deliverable & Tests:**
  * Pass [`01_math_foundations/tests/test_autograd.py`](01_math_foundations/tests/test_autograd.py).

### Week 3: Probability Theory, Statistical Estimation & Gradient Optimizers
* **Theory:**
  * Random variables, Bayes' Theorem, Maximum Likelihood Estimation (MLE).
  * Shannon Entropy, Cross-Entropy, Kullback-Leibler (KL) Divergence.
  * SGD, Momentum, RMSProp, and Adam optimization updates.
* **Practice & Labs:**
  * Implement vectorized metrics in [`01_math_foundations/src/metrics.py`](01_math_foundations/src/metrics.py).
  * Implement custom optimizers in [`01_math_foundations/src/optimizers.py`](01_math_foundations/src/optimizers.py).
* **Deliverable & Tests:**
  * Pass [`01_math_foundations/tests/test_metrics.py`](01_math_foundations/tests/test_metrics.py).

---

## 💾 Phase 2: Big Data Engineering & Feature Store Pipelines (Weeks 4–6)

### Phase Description
Students learn production-grade data engineering: processing multi-modal datasets without memory exhaustion using Polars and DuckDB, preventing data leakage, and enforcing data contracts with Pandera.

### Week 4: Columnar Data Ingestion & Analytical SQL Engines
* **Theory:** Row-oriented vs columnar storage (Parquet), vectorized SIMD query execution in Polars and DuckDB.
* **Practice & Labs:** Ingest datasets with `Polars` and `DuckDB` in [`02_data_engineering/src/`](02_data_engineering/src/).
* **Deliverable & Tests:** Pass [`02_data_engineering/tests/test_duckdb_parquet.py`](02_data_engineering/tests/test_duckdb_parquet.py).

### Week 5: Semi-Structured & Unstructured Multimodal Pipelines
* **Theory:** Streaming JSON parsing, image color spaces, histogram normalization, YOLO bounding boxes.
* **Practice & Labs:** Implement JSON parsing and image transforms in [`02_data_engineering/src/`](02_data_engineering/src/).
* **Deliverable & Tests:** Pass [`02_data_engineering/tests/test_data_engineering.py`](02_data_engineering/tests/test_data_engineering.py).

### Week 6: Feature Engineering, Leakage Prevention & Data Contracts
* **Theory:** Out-of-fold target encoding, Yeo-Johnson transforms, point-in-time temporal `ASOF` joins, Pandera schemas.
* **Practice & Labs:** Build leak-free transforms and validation contracts in [`02_data_engineering/src/`](02_data_engineering/src/).
* **Deliverable & Tests:** Pass [`02_data_engineering/tests/test_advanced_transforms.py`](02_data_engineering/tests/test_advanced_transforms.py), [`test_temporal_joins.py`](02_data_engineering/tests/test_temporal_joins.py), and [`test_data_validation.py`](02_data_engineering/tests/test_data_validation.py).

---

## 🎯 Phase 3: Supervised Learning & Tree Ensembles (Weeks 7–8)

### Phase Description
Covers linear models, maximum-margin classifiers, probabilistic models, decision trees, gradient boosting, and time-series forecasting.

### Week 7: Linear Models, SVMs, Naive Bayes & k-NN
* **Theory:** OLS, Ridge ($L_2$), Lasso ($L_1$), Soft-margin SVM hinge loss, Gaussian Naive Bayes, k-NN metrics.
* **Practice & Labs:** Implement models in [`03_supervised_learning/src/`](03_supervised_learning/src/).
* **Deliverable & Tests:** Pass [`03_supervised_learning/tests/test_linear_models.py`](03_supervised_learning/tests/test_linear_models.py), [`test_svm.py`](03_supervised_learning/tests/test_svm.py), [`test_naive_bayes.py`](03_supervised_learning/tests/test_naive_bayes.py), and [`test_knn.py`](03_supervised_learning/tests/test_knn.py).

### Week 8: Decision Trees, Gradient Boosting & Time Series
* **Theory:** CART trees (Gini impurity), Random Forest, XGBoost second-order Taylor expansion, lag features, Walk-Forward validation.
* **Practice & Labs:** Build decision trees, ensemble benchmark harness, and time-series cross-validation.
* **Deliverable & Tests:** Pass [`03_supervised_learning/tests/test_decision_tree.py`](03_supervised_learning/tests/test_decision_tree.py), [`test_ensemble_benchmark.py`](03_supervised_learning/tests/test_ensemble_benchmark.py), and [`test_time_series.py`](03_supervised_learning/tests/test_time_series.py).

---

## 🔍 Phase 4: Unsupervised Learning & Dimensionality Reduction (Weeks 9–10)

### Phase Description
Explores discovering latent structures without supervisory signals: centroid clustering, density-based clustering, probabilistic mixtures, and anomaly detection.

### Week 9: K-Means++, DBSCAN & Gaussian Mixture Models (EM)
* **Theory:** Lloyd's algorithm, K-Means++ probabilistic initialization, DBSCAN density reachability, GMM Expectation-Maximization.
* **Practice & Labs:** Implement K-Means, DBSCAN, and GMM from scratch in [`04_unsupervised_learning/src/`](04_unsupervised_learning/src/).
* **Deliverable & Tests:** Pass [`04_unsupervised_learning/tests/test_kmeans.py`](04_unsupervised_learning/tests/test_kmeans.py), [`test_dbscan.py`](04_unsupervised_learning/tests/test_dbscan.py), and [`test_gmm.py`](04_unsupervised_learning/tests/test_gmm.py).

### Week 10: Dimensionality Reduction & Isolation Forests
* **Theory:** PCA via SVD / Covariance eigendecomposition, reconstruction error, Isolation Forest random BST path length anomaly scoring.
* **Practice & Labs:** Implement PCA and Isolation Forest in [`04_unsupervised_learning/src/`](04_unsupervised_learning/src/).
* **Deliverable & Tests:** Pass [`04_unsupervised_learning/tests/test_pca.py`](04_unsupervised_learning/tests/test_pca.py) and [`test_isolation_forest.py`](04_unsupervised_learning/tests/test_isolation_forest.py).

---

## 🔄 Phase 5: Self-Supervised Learning & Contrastive Dynamics (Week 11)

### Phase Description
Students learn representation learning from unlabeled data using pretext tasks, contrastive objectives, and autoencoders.

### Week 11: Contrastive InfoNCE Loss, SimCLR & Autoencoders
* **Theory:** Contrastive representation learning, NT-Xent / InfoNCE loss, temperature scaling $\tau$, dimensional collapse prevention, Deep Autoencoders.
* **Practice & Labs:** Implement `NTXentLoss`, `SimCLRFramework`, and `DeepAutoencoder` in [`05_self_supervised_learning/src/`](05_self_supervised_learning/src/).
* **Deliverable & Tests:** Pass [`05_self_supervised_learning/tests/test_contrastive_loss.py`](05_self_supervised_learning/tests/test_contrastive_loss.py), [`test_autoencoder.py`](05_self_supervised_learning/tests/test_autoencoder.py), and [`test_simclr_framework.py`](05_self_supervised_learning/tests/test_simclr_framework.py).

---

## 🤝 Phase 6: Semi-Supervised Learning & Graph Propagation (Week 12)

### Phase Description
Combines small labeled datasets with massive unlabeled pools using pseudo-labeling, graph Markov walks, and consistency regularization.

### Week 12: Self-Training, Label Propagation & Consistency Regularization
* **Theory:** Smoothness, cluster, and manifold assumptions; self-training confidence thresholding; graph harmonic Dirichlet energy minimization; temperature sharpening.
* **Practice & Labs:** Implement `SelfTrainingClassifier`, `LabelPropagation`, and consistency functions in [`06_semi_supervised_learning/src/`](06_semi_supervised_learning/src/).
* **Deliverable & Tests:** Pass [`06_semi_supervised_learning/tests/test_pseudo_labeling.py`](06_semi_supervised_learning/tests/test_pseudo_labeling.py), [`test_label_propagation.py`](06_semi_supervised_learning/tests/test_label_propagation.py), and [`test_consistency_regularization.py`](06_semi_supervised_learning/tests/test_consistency_regularization.py).

---

## 🧠 Phase 7: Deep Learning Architectures & Attention (Weeks 13–14)

### Phase Description
Focuses on deep representations, residual gradient highways, spatial object detection metrics, and self-attention Transformer mechanisms.

### Week 13: Residual Networks (ResNet) & Object Detection Metrics
* **Theory:** Gradient highway preservation $\mathcal{F}(x) + x$, vectorized IoU calculation, Non-Maximum Suppression (NMS).
* **Practice & Labs:** Implement ResNet blocks and detection metrics in [`07_deep_learning/src/`](07_deep_learning/src/).
* **Deliverable & Tests:** Pass [`07_deep_learning/tests/test_resnet.py`](07_deep_learning/tests/test_resnet.py) and [`test_detection_metrics.py`](07_deep_learning/tests/test_detection_metrics.py).

### Week 14: Transformers, Multi-Head Attention & RAG
* **Theory:** Scaled Dot-Product Attention $\text{softmax}(QK^T / \sqrt{d_k})V$, causal decoding masks, dense vector indexing, RAG synthesis.
* **Practice & Labs:** Implement `MultiHeadAttention` and vector search in [`07_deep_learning/src/`](07_deep_learning/src/).
* **Deliverable & Tests:** Pass [`07_deep_learning/tests/test_attention.py`](07_deep_learning/tests/test_attention.py) and [`test_rag.py`](07_deep_learning/tests/test_rag.py).

---

## 🎮 Phase 8: Reinforcement Learning & Decision Processes (Weeks 15–16)

### Phase Description
Covers sequential decision-making, Markov Decision Processes, value-based methods, deep Q-networks, and policy gradients.

### Week 15: MDPs, GridWorlds & Tabular Q-Learning
* **Theory:** Bellman expectation and optimality equations, exploration vs exploitation ($\epsilon$-greedy), off-policy Q-Learning updates.
* **Practice & Labs:** Build GridWorld environment and Q-Learning agent in [`08_reinforcement_learning/src/`](08_reinforcement_learning/src/).
* **Deliverable & Tests:** Pass [`08_reinforcement_learning/tests/test_q_learning.py`](08_reinforcement_learning/tests/test_q_learning.py).

### Week 16: Deep Q-Networks (DQN) & Policy Gradient (REINFORCE)
* **Theory:** Deadly triad, Experience Replay Buffer, Target Network synchronization, Policy Gradient Theorem, baseline variance reduction.
* **Practice & Labs:** Implement `DQNAgent` and `REINFORCEAgent` in [`08_reinforcement_learning/src/`](08_reinforcement_learning/src/).
* **Deliverable & Tests:** Pass [`08_reinforcement_learning/tests/test_dqn_agent.py`](08_reinforcement_learning/tests/test_dqn_agent.py) and [`test_policy_gradient.py`](08_reinforcement_learning/tests/test_policy_gradient.py).

---

## 🔀 Phase 9: Transfer Learning, PEFT & Domain Adaptation (Weeks 17–18)

### Phase Description
Addresses adapting foundation models to downstream domains efficiently: backbone freezing, parameter-efficient fine-tuning (LoRA), and domain alignment.

### Week 17: Backbone Freezing, Fine-Tuning & LoRA Adaptation
* **Theory:** Catastrophic forgetting, gradual unfreezing, Low-Rank Adaptation ($\Delta W = \frac{\alpha}{r}AB$), offline weight merging.
* **Practice & Labs:** Implement `TransferLearningModel` and `LoRALinear` in [`09_transfer_learning/src/`](09_transfer_learning/src/).
* **Deliverable & Tests:** Pass [`09_transfer_learning/tests/test_transfer_learning.py`](09_transfer_learning/tests/test_transfer_learning.py) and [`test_lora.py`](09_transfer_learning/tests/test_lora.py).

### Week 18: Domain Adaptation & Maximum Mean Discrepancy (MMD)
* **Theory:** Covariate shift $P(X_S) \ne P(X_T)$, RKHS embeddings, multi-kernel Gaussian RBF discrepancy.
* **Practice & Labs:** Implement `MaximumMeanDiscrepancy` in [`09_transfer_learning/src/`](09_transfer_learning/src/).
* **Deliverable & Tests:** Pass [`09_transfer_learning/tests/test_domain_adaptation.py`](09_transfer_learning/tests/test_domain_adaptation.py).

---

## 🚀 Phase 10: Production MLOps, APIs & Deployment (Weeks 19–20)

### Phase Description
Hardens models into production microservices with FastAPI, ONNXRuntime acceleration, MLflow lifecycle tracking, Docker, and CI/CD quality gates.

### Week 19: REST APIs with FastAPI, Pydantic & ONNX Acceleration
* **Theory:** ASGI concurrency, strict Pydantic data schemas, ONNX graph optimization, operator fusion.
* **Practice & Labs:** Build inference endpoints and ONNX converter in [`10_mlops_production/`](10_mlops_production/).
* **Deliverable & Tests:** Pass [`10_mlops_production/tests/test_api.py`](10_mlops_production/tests/test_api.py) and [`test_onnx.py`](10_mlops_production/tests/test_onnx.py).

### Week 20: MLflow Lifecycle Governance, Docker & Capstone Defense
* **Theory:** Run tracking, model staging, multi-stage Docker builds, automated test verification.
* **Practice & Labs:** Build MLflow tracker, Dockerfile, docker-compose, and assemble full end-to-end pipeline.
* **Deliverable & Assessment:** Pass [`10_mlops_production/tests/test_mlflow.py`](10_mlops_production/tests/test_mlflow.py), pass entire repository test suite (98/98 tests), and defend Capstone project.
