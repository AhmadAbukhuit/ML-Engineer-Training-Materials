# 🚀 Machine Learning Engineer Roadmap

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

An end-to-end engineering roadmap and practice repository covering mathematical foundations, big data engineering, classical and modern learning paradigms (supervised, unsupervised, self-supervised, semi-supervised, reinforcement learning, transfer learning), deep learning architectures, and production MLOps deployments.

---

## 📌 Overview & Philosophy

Building real-world AI systems requires more than importing high-level frameworks. A production-ready Machine Learning Engineer must master:

1. **The Underlying Mathematics:** Understanding vectors, gradients, loss surfaces, and statistical uncertainty to diagnose why models fail.
2. **Data & Systems Engineering:** Parsing, cleaning, and transforming massive multi-modal datasets (tabular, images, hierarchical formats) reliably.
3. **Model Mastery Across All Learning Paradigms:** Mastering supervised, unsupervised, self-supervised, semi-supervised, reinforcement, and transfer learning models with complete mathematical formulations.
4. **Deep Learning Architectures:** Residual networks, multi-head attention, vector search, and parameter-efficient fine-tuning (LoRA).
5. **Production Operations (MLOps):** Exposing serialized weights via low-latency REST APIs (`FastAPI`), isolating dependencies in containers (`Docker`), and optimizing hardware inference via `ONNXRuntime`.

---

## 📋 Curriculum Progress Tracker

### Phase 1: Mathematical Foundations & Numerical Computing

- [x] **1.1 Linear Algebra for ML:** Matrix/vector operations, dot products, matrix factorizations, eigenvalues/eigenvectors, and PCA via SVD in `NumPy`.
- [x] **1.2 Multivariate Calculus:** Partial derivatives, chain rule for computational graphs, gradients, and custom reverse-mode Autograd engine.
- [x] **1.3 Probability & Statistics:** Random variables, probability distributions (Gaussian, Bernoulli), Bayes' Theorem, MLE, and vectorized evaluation metrics.
- [x] **1.4 Optimization Theory:** Convex functions, SGD, Momentum, RMSProp, and Adam optimization mechanics with modular MLP implementations.

### Phase 2: Data Science & Big Data Engineering

- [x] **2.1 Structured Ingestion:** Query columnar Parquet datasets and execute analytical SQL using `DuckDB` and `Polars`.
- [x] **2.2 Semi-Structured Parsing:** Streaming JSON telemetry parser with schema normalization, nested dictionary extraction, and missing-value handling.
- [x] **2.3 Unstructured Handling:** Image processing pipeline with contrast normalization, data augmentation, and Pascal VOC / YOLO bounding box parsers.
- [x] **2.4 Advanced Feature Engineering:** Out-of-fold smoothed target encoding, Yeo-Johnson transformations, and Point-in-Time temporal `ASOF` joins.
- [x] **2.5 Data Quality & Validation:** Runtime data contracts enforcing column constraints and distribution bounds using `Pandera`.
- [x] **2.6 Exploratory Data Analysis & Preprocessing Pipelines:** NumPy vectorization, Pandas missingness audits, IQR outlier filtering, correlation matrices, and composable Scikit-Learn pipelines (`RobustScaler`, `ColumnTransformer`).

### Phase 3: Supervised Learning

- [x] **3.1 Linear & Regularized Models:** Vectorized OLS Linear Regression, Ridge ($L_2$), Lasso ($L_1$), ElasticNet, and Logistic Regression with Gradient Descent.
- [x] **3.2 Support Vector Machines & Margin Classifiers:** Soft-margin Linear SVM with hinge loss optimization and kernel SVM formulations.
- [x] **3.3 Probabilistic Classification:** Gaussian Naive Bayes from scratch with log-likelihood numerical stability.
- [x] **3.4 Non-Parametric & Metric Learning:** K-Nearest Neighbors (k-NN) classifier and regressor from first principles with Euclidean, Manhattan, and Minkowski distance metrics.
- [x] **3.5 Tree & Ensemble Benchmarking:** CART Decision Tree from scratch (Gini impurity), alongside Random Forest, LightGBM, and XGBoost comparative benchmarks.
- [x] **3.6 Time Series Forecasting:** Lag feature engineering, rolling window statistics, and Walk-Forward cross-validation without temporal lookahead leakage.

### Phase 4: Unsupervised Learning

- [x] **4.1 Centroid & Partitioning Clustering:** Vectorized K-Means with K-Means++ probabilistic initialization, inertia computation, and silhouette scoring.
- [x] **4.2 Density-Based Clustering:** DBSCAN from scratch identifying core points, border points, and arbitrary-shaped clusters with noise isolation.
- [x] **4.3 Distribution-Based Mixture Models:** Gaussian Mixture Models (GMM) trained via the Expectation-Maximization (EM) algorithm.
- [x] **4.4 Dimensionality Reduction:** Principal Component Analysis (PCA) from scratch via SVD and covariance eigendecomposition, explained variance ratios, and reconstruction.
- [x] **4.5 Unsupervised Outlier & Anomaly Detection:** Isolation Forest algorithm from scratch with average BST path length anomaly scoring.

### Phase 5: Self-Supervised Learning

- [x] **5.1 Contrastive Representation Learning:** Normalized Temperature-scaled Cross Entropy (NT-Xent / InfoNCE) loss from scratch with temperature scaling $\tau$.
- [x] **5.2 Representation Evaluation:** Alignment and Uniformity geometric evaluation metrics.
- [x] **5.3 Deep & Denoising Autoencoders:** Bottleneck Autoencoder with analytical backpropagation, reconstruction MSE loss, and Gaussian noise corruption (DAE).
- [x] **5.4 Contrastive Pipelines:** SimCLR framework with stochastic multi-view augmentations, base feature encoder, and non-linear MLP projection head.

### Phase 6: Semi-Supervised Learning

- [x] **6.1 Self-Training & Pseudo-Labeling:** Iterative pseudo-labeling classifier wrapper with confidence thresholding $\tau$ and dynamic unlabeled pool expansion.
- [x] **6.2 Graph-Based Learning:** Label Propagation algorithm from first principles via RBF kernel affinity matrices, Markov random walk transitions, and ground-truth node clamping.
- [x] **6.3 Consistency Regularization:** Probability distribution temperature sharpening ($q_i \propto p_i^{1/T}$), Gaussian ramp-up loss schedules, and consistency loss metrics.

### Phase 7: Deep Learning & Modern Architectures

- [x] **7.1 Foundations & Training Loops:** Modular deep learning training engine featuring validation tracking, learning rate scheduling, and early stopping.
- [x] **7.2 Vision Architectures & Object Detection:** Residual blocks with identity and projection skip connections (`F(x) + x`), vectorized IoU matrices, and Non-Maximum Suppression (NMS).
- [x] **7.3 Attention Mechanisms:** Scaled Dot-Product Self-Attention and Multi-Head Attention (MHA) from scratch supporting causal decoding masks.
- [x] **7.4 Vector Search & RAG:** Normalized dense embeddings, in-memory cosine vector store, semantic document chunking, and grounded prompt synthesis.

### Phase 8: Reinforcement Learning

- [x] **8.1 Markov Decision Processes:** Discrete 2D GridWorld environment with obstacles, step penalties, and goal rewards.
- [x] **8.2 Tabular Value-Based Learning:** Off-policy Q-Learning agent with $\epsilon$-greedy exploration decay and Bellman optimality updates.
- [x] **8.3 Deep Q-Networks (DQN):** Deep Q-Network agent in NumPy featuring cyclic Experience Replay Buffer and periodic Target Network weight synchronization.
- [x] **8.4 Policy Gradients:** REINFORCE Monte Carlo policy gradient agent with softmax action distribution and baseline subtraction variance reduction.

### Phase 9: Transfer Learning & Fine-Tuning

- [x] **9.1 Two-Stage Transfer Learning:** Feature extractor backbone parameter freezing, target classification head replacement, and fine-tuning.
- [x] **9.2 Parameter-Efficient Fine-Tuning (LoRA):** Low-Rank Adaptation ($\Delta W = \frac{\alpha}{r} A B$) with base weight freezing, adapter training, parameter reduction accounting, and zero-latency inference weight merging.
- [x] **9.3 Domain Adaptation:** Maximum Mean Discrepancy (MMD) distribution alignment with multi-kernel Gaussian RBF in RKHS.

### Phase 10: Production APIs & MLOps

- [x] **10.1 REST API Serving:** Asynchronous microservice using `FastAPI` and `Uvicorn` with single-sample and batch prediction endpoints.
- [x] **10.2 Data Contracts & Schemas:** Strict input validation and sanitization using `Pydantic` models.
- [x] **10.3 Containerization & Orchestration:** Production multi-stage `Dockerfile` and `docker-compose.yml` service configurations.
- [x] **10.4 Hardware Acceleration & Model Compilation:** High-performance ONNX model export and multi-threaded `ONNXRuntime` inference with latency benchmarking.
- [x] **10.5 Lifecycle Management & Governance:** Standardized experiment tracking, metric and parameter logging, and automated best-run model selection with `MLflow`.

---

## 📂 Repository Structure

```text
ML-Engineer-Training-Materials/
├── .gitignore                   # Ignore heavy datasets, checkpoints, and weights
├── README.md                    # Root syllabus & progress tracker
├── INSTRUCTOR.md                # Instructor teaching notes & pedagogical strategies
├── PLAN.md                      # 20-Week structured curriculum teaching plan
├── requirements.txt             # Core dependencies
├── pyproject.toml               # Multi-module pytest test harness & ruff configurations
│
├── data/                        # Local data directory
│   ├── raw/                     # Original CSVs, JSON, XML, images
│   └── processed/               # Pipeline outputs ready for training
│
├── 00_Introduction/             # ML Engineering career overview, scope, and industry roles
│   └── README.md
│
├── 01_math_foundations/         # Vectorized math, autograd engine, custom optimizers & PCA
│   ├── README.md
│   ├── 01_linear_algebra.md
│   ├── 02_multivariate_calculus.md
│   ├── 03_probability_statistics.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 02_data_engineering/         # High-performance ETL, Polars, DuckDB, Pandera, Scikit-Learn
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 03_supervised_learning/      # Linear, SVM, Naive Bayes, k-NN, CART Trees, Ensembles & Time Series
│   ├── README.md
│   ├── 01_supervised_models_catalog.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 04_unsupervised_learning/    # K-Means, DBSCAN, GMM (EM), PCA (SVD), Isolation Forest
│   ├── README.md
│   ├── 01_unsupervised_models_catalog.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 05_self_supervised_learning/ # SimCLR, NT-Xent/InfoNCE, Deep Autoencoders, Alignment/Uniformity
│   ├── README.md
│   ├── 01_self_supervised_models_catalog.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 06_semi_supervised_learning/ # Self-Training / Pseudo-Labeling, Label Propagation, Consistency
│   ├── README.md
│   ├── 01_semi_supervised_models_catalog.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 07_deep_learning/            # ResNet, Multi-Head Attention, Object Detection Metrics, RAG Vector Search
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 08_reinforcement_learning/   # GridWorld, Tabular Q-Learning, Deep Q-Networks (DQN), REINFORCE
│   ├── README.md
│   ├── 01_reinforcement_learning_models_catalog.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
├── 09_transfer_learning/        # Pretrained Backbones, LoRA PEFT, Weight Merging, MMD Domain Adaptation
│   ├── README.md
│   ├── 01_transfer_learning_models_catalog.md
│   ├── notebooks/
│   ├── src/
│   └── tests/
│
└── 10_mlops_production/         # FastAPI, Pydantic, ONNXRuntime, MLflow, Docker & CI/CD
    ├── README.md
    ├── app/
    ├── tests/
    ├── Dockerfile
    └── docker-compose.yml
```

## 🛠️ Environment Setup

### Clone the repository

```bash
git clone https://github.com/AhmadAbukhuit/ML-Engineer-Training-Materials.git
cd ML-Engineer-Training-Materials
```

### Initialize Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies & Run Tests

```bash
pip install --upgrade pip
pip install -r requirements.txt
pytest
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make to this roadmap are **greatly appreciated**.

Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct, formatting rules, and the process for submitting pull requests.

## 📄 License

This project is distributed under the MIT License. See the [`LICENSE`](LICENSE) file for more information.
