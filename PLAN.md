# 📅 20-Week Machine Learning Engineering Curriculum Plan

This curriculum provides a structured, rigorous, 20-week roadmap designed to take students from numerical and mathematical foundations to production-grade MLOps deployment.

---

## 🗺️ Curriculum Overview & Phase Timeline

```text
 Weeks 1-4    │ Phase 0: Mathematical Foundations & Numerical Computing
 Weeks 5-8    │ Phase 1: Big Data Engineering & Feature Store Pipelines
 Weeks 9-12   │ Phase 2: Classical Machine Learning & Statistical Modeling
 Weeks 13-16  │ Phase 3: Deep Learning Architectures, Transformers & Foundation Models
 Weeks 17-20  │ Phase 4: Production MLOps, High-Performance APIs & Deployment
```

---

## 📐 Phase 0: Mathematical Foundations & Numerical Computing (Weeks 1–4)

### Phase Description
The objective of Phase 0 is to eliminate the "black box" mentality. Students master the mathematical machinery governing modern machine learning algorithms: vectorized linear algebra in NumPy, reverse-mode automatic differentiation, multivariate calculus, statistical estimation, and gradient-based optimization mechanics.

---

### Week 1: Course Onboarding & Vectorized Linear Algebra
* **Theory:**
  * Vector spaces, inner products, projections, matrix multiplication algorithms.
  * Matrix factorizations: Eigenvalues, Eigenvectors, and Singular Value Decomposition (SVD).
  * Principal Component Analysis (PCA) derivation via covariance matrix decomposition.
* **Practice & Labs:**
  * Repo Directory: `01_math_foundations/`
  * Implement PCA from scratch using SVD in [`src/pca.py`](01_math_foundations/src/pca.py).
  * Reconstruct compressed data matrices and quantify explained variance ratios.
* **Deliverable & Tests:**
  * Run and pass [`tests/test_pca.py`](01_math_foundations/tests/test_pca.py).

---

### Week 2: Multivariate Calculus & Reverse-Mode Autograd Engine
* **Theory:**
  * Partial derivatives, gradient vectors, Jacobians, and Hessians.
  * Directed Acyclic Graphs (DAG) for mathematical operations.
  * Reverse-mode automatic differentiation vs numerical differentiation.
* **Practice & Labs:**
  * Implement the scalar Autograd engine in [`src/autograd.py`](01_math_foundations/src/autograd.py).
  * Build topological sort and backpropagation traversal through addition, multiplication, ReLU, and power nodes.
* **Deliverable & Tests:**
  * Run and pass [`tests/test_autograd.py`](01_math_foundations/tests/test_autograd.py).

---

### Week 3: Probability, Information Theory & Vectorized Metrics
* **Theory:**
  * Random variables, Bayes' Theorem, Maximum Likelihood Estimation (MLE).
  * Information theory: Shannon Entropy, Cross-Entropy, Kullback-Leibler (KL) Divergence.
  * Classification and regression metric mathematics: Precision, Recall, F1, ROC-AUC, Log-Loss, and MSE.
* **Practice & Labs:**
  * Implement vectorized evaluation metrics in [`src/metrics.py`](01_math_foundations/src/metrics.py).
  * Ensure numerically stable log-loss with epsilon clipping.
* **Deliverable & Tests:**
  * Run and pass [`tests/test_metrics.py`](01_math_foundations/tests/test_metrics.py).

---

### Week 4: Optimization Mechanics & Multi-Layer Perceptrons
* **Theory:**
  * Convex vs non-convex loss surfaces, saddle points, and ill-conditioned curvature.
  * Optimization algorithms: Stochastic Gradient Descent (SGD), Momentum, RMSProp, and Adam.
  * Weight initialization theory: Xavier/Glorot vs He/Kaiming normal distributions.
* **Practice & Labs:**
  * Implement modular `DenseLayer`, activation functions, and optimizers in [`src/mlp.py`](01_math_foundations/src/mlp.py).
  * Train an MLP from scratch on synthetic non-linear classification datasets.
* **Deliverable & Tests:**
  * Verify convergence curves and loss minimization without framework abstractions.

---

## 🗄️ Phase 1: Big Data Engineering & Feature Store Pipelines (Weeks 5–8)

### Phase Description
Real-world machine learning systems spend 80% of their operational lifecycle ingesting, cleaning, and validating data. Phase 1 equips students with high-throughput data processing skills: columnar storage with Parquet, in-process analytical SQL with DuckDB, lightning-fast transformations with Polars, multi-modal image pipelines, and point-in-time correctness.

---

### Week 5: Columnar Storage & In-Process Analytical SQL (DuckDB & Parquet)
* **Theory:**
  * Columnar vs Row-oriented storage formats (Parquet vs CSV).
  * Dictionary encoding, run-length encoding (RLE), and Snappy compression.
  * Vectorized query execution engines and predicate pushdown.
* **Practice & Labs:**
  * Repo Directory: `02_data_engineering/`
  * Complete [`notebooks/03_sql_and_parquet_benchmarks.ipynb`](02_data_engineering/notebooks/03_sql_and_parquet_benchmarks.ipynb).
  * Implement analytical SQL aggregation and Parquet export in [`src/duckdb_parquet.py`](02_data_engineering/src/duckdb_parquet.py).
* **Deliverable & Tests:**
  * Run and pass [`tests/test_duckdb_parquet.py`](02_data_engineering/tests/test_duckdb_parquet.py).

---

### Week 6: High-Performance Tabular & Semi-Structured ETL
* **Theory:**
  * Memory layouts in Apache Arrow and Polars.
  * Lazy vs Eager execution frames and query plan optimization.
  * Streaming JSON parsing vs memory-exhaustive bulk loading.
* **Practice & Labs:**
  * Implement streaming JSON telemetry parser in [`src/telemetry_parser.py`](02_data_engineering/src/telemetry_parser.py).
  * Complete Exploratory Data Analysis in [`notebooks/01_telemetry_eda.ipynb`](02_data_engineering/notebooks/01_telemetry_eda.ipynb).
* **Deliverable & Tests:**
  * Run and pass [`tests/test_data_engineering.py`](02_data_engineering/tests/test_data_engineering.py).

---

### Week 7: Computer Vision Ingestion & Annotation Parsers
* **Theory:**
  * 2D image matrix representation, color spaces (RGB, BGR, Grayscale).
  * Contrast stretching, histogram equalization, and spatial data augmentations.
  * Spatial annotation formats: Pascal VOC XML vs YOLO normalized coordinates.
* **Practice & Labs:**
  * Implement OpenCV/Pillow image preprocessing in [`src/image_pipeline.py`](02_data_engineering/src/image_pipeline.py).
  * Complete image exploration in [`notebooks/02_image_visualization.ipynb`](02_data_engineering/notebooks/02_image_visualization.ipynb).
* **Deliverable & Tests:**
  * Verify image augmentations and bounding box coordinate conversions.

---

### Week 8: Advanced Feature Engineering, Leakage Prevention & Data Contracts
* **Theory:**
  * Target encoding pitfalls and out-of-fold smoothed empirical Bayes regularization.
  * Non-linear variance stabilizing transforms (Yeo-Johnson power transform).
  * Point-in-time correctness: eliminating lookahead bias using temporal `ASOF` joins.
  * Data contracts and runtime schema validation with Pandera.
* **Practice & Labs:**
  * Implement smoothed target encoding and Yeo-Johnson in [`src/advanced_transforms.py`](02_data_engineering/src/advanced_transforms.py).
  * Implement temporal point-in-time join in [`src/temporal_joins.py`](02_data_engineering/src/temporal_joins.py).
  * Build Pandera validation schemas in [`src/data_validation.py`](02_data_engineering/src/data_validation.py).
* **Deliverable & Tests (Milestone 1):**
  * Pass [`tests/test_advanced_transforms.py`](02_data_engineering/tests/test_advanced_transforms.py), [`tests/test_temporal_joins.py`](02_data_engineering/tests/test_temporal_joins.py), and [`tests/test_data_validation.py`](02_data_engineering/tests/test_data_validation.py).
  * **Milestone 1 Defense:** Code review of an automated ETL pipeline with strict data contracts.

---

## 🤖 Phase 2: Classical Machine Learning & Statistical Modeling (Weeks 9–12)

### Phase Description
Students master fundamental supervised and unsupervised learning algorithms, formulating objective functions, optimization mechanics, bias-variance tradeoffs, tree ensembles, and temporal cross-validation.

---

### Week 9: Supervised Linear & Maximum-Margin Classifiers
* **Theory:**
  * Ordinary Least Squares (OLS) normal equations vs Gradient Descent.
  * Logistic Regression with L2 Ridge regularization and cross-entropy loss.
  * Support Vector Machines (SVM): Maximum margin separation, Hinge loss, and soft-margin subgradient updates.
* **Practice & Labs:**
  * Repo Directory: `03_machine_learning/`
  * Implement Linear & Logistic Regression in [`src/linear_models.py`](03_machine_learning/src/linear_models.py).
  * Implement Linear SVM with subgradient descent in [`src/svm.py`](03_machine_learning/src/svm.py).
* **Deliverable & Tests:**
  * Pass [`tests/test_linear_models.py`](03_machine_learning/tests/test_linear_models.py) and [`tests/test_svm.py`](03_machine_learning/tests/test_svm.py).

---

### Week 10: Probabilistic & Tree-Based Classifiers
* **Theory:**
  * Generative vs Discriminative modeling paradigms.
  * Gaussian Naive Bayes: Class priors, feature Gaussian likelihoods, and log-space inference.
  * Decision Trees: Recursive binary splitting, Gini Impurity vs Information Gain (Entropy), stopping criteria, and pruning.
* **Practice & Labs:**
  * Implement Gaussian Naive Bayes from scratch in [`src/naive_bayes.py`](03_machine_learning/src/naive_bayes.py).
  * Implement Decision Tree Classifier from scratch in [`src/decision_tree.py`](03_machine_learning/src/decision_tree.py).
* **Deliverable & Tests:**
  * Pass [`tests/test_naive_bayes.py`](03_machine_learning/tests/test_naive_bayes.py) and [`tests/test_decision_tree.py`](03_machine_learning/tests/test_decision_tree.py).

---

### Week 11: Tree Ensembles & Head-to-Head Benchmarking Harness
* **Theory:**
  * Bagging vs Boosting: Variance reduction (Random Forest) vs Bias reduction (Gradient Boosting).
  * Extreme Gradient Boosting (XGBoost): Second-order Taylor expansion, shrinkage, column subsampling, and regularization.
  * Evaluating model tradeoffs: Training duration, inference latency (p50/p95), ROC-AUC, and F1 score.
* **Practice & Labs:**
  * Build the comparative benchmarking engine in [`src/ensemble_benchmark.py`](03_machine_learning/src/ensemble_benchmark.py).
  * Complete [`notebooks/01_classical_ml_models.ipynb`](03_machine_learning/notebooks/01_classical_ml_models.ipynb).
* **Deliverable & Tests:**
  * Pass [`tests/test_ensemble_benchmark.py`](03_machine_learning/tests/test_ensemble_benchmark.py).

---

### Week 12: Unsupervised Pattern Mining & Time Series Forecasting
* **Theory:**
  * Partitioning clustering: Lloyd's K-Means algorithm, centroid updates, inertia metric.
  * Time series dynamics: Stationarity, autocorrelation, lag features, and rolling statistical moments.
  * Temporal validation: Why random K-Fold splits corrupt time series backtests; Walk-Forward Expanding-Window Validation.
* **Practice & Labs:**
  * Implement K-Means clustering from scratch in [`src/kmeans.py`](03_machine_learning/src/kmeans.py).
  * Implement temporal lag extraction and Walk-Forward split engine in [`src/time_series.py`](03_machine_learning/src/time_series.py).
* **Deliverable & Tests (Milestone 2):**
  * Pass [`tests/test_kmeans.py`](03_machine_learning/tests/test_kmeans.py) and [`tests/test_time_series.py`](03_machine_learning/tests/test_time_series.py).
  * **Milestone 2 Defense:** Present benchmark findings comparing classical models against tree ensembles.

---

## 🧠 Phase 3: Deep Learning Architectures, Transformers & Foundation Models (Weeks 13–16)

### Phase Description
Phase 3 transitions from classical statistical learning to deep representations. Students master deep residual networks, object detection spatial metrics, self-attention mechanisms in Transformers, transfer learning adaptation, and modern foundation model techniques (LoRA fine-tuning and RAG retrieval).

---

### Week 13: Deep Learning Dynamics & Residual Networks (ResNet)
* **Theory:**
  * Vanishing and exploding gradients in deep networks.
  * Residual learning formulation: $\mathcal{F}(x) + x$, gradient highway preservation.
  * Identity shortcuts vs 1x1 projection shortcuts.
  * Modular training loops, learning rate schedules (Cosine Annealing), and early stopping.
* **Practice & Labs:**
  * Repo Directory: `04_deep_learning/`
  * Implement Residual Block from scratch in [`src/resnet.py`](04_deep_learning/src/resnet.py).
  * Implement Early Stopping and Training Engine in [`src/training_engine.py`](04_deep_learning/src/training_engine.py).
* **Deliverable & Tests:**
  * Pass [`tests/test_resnet.py`](04_deep_learning/tests/test_resnet.py).

---

### Week 14: Computer Vision & Object Detection Mechanics (YOLO)
* **Theory:**
  * Object detection formulation: Simultaneous bounding box regression and classification.
  * Spatial metrics: Intersection over Union (IoU) calculation.
  * Candidate deduplication: Greedy Non-Maximum Suppression (NMS) algorithm.
* **Practice & Labs:**
  * Implement vectorized IoU matrix calculation in [`src/detection_metrics.py`](04_deep_learning/src/detection_metrics.py).
  * Implement Non-Maximum Suppression (NMS) handling multi-box candidate clusters.
* **Deliverable & Tests:**
  * Pass [`tests/test_detection_metrics.py`](04_deep_learning/tests/test_detection_metrics.py).

---

### Week 15: Transformers & Attention Mechanisms
* **Theory:**
  * Limitations of recurrence (RNN/LSTM) in long-range sequence modeling.
  * Scaled Dot-Product Attention: Query, Key, Value routing and $\frac{1}{\sqrt{d_k}}$ temperature scaling.
  * Multi-Head Attention (MHA): Splitting representations across parallel projection subspaces.
  * Causal autoregressive decoding masks.
* **Practice & Labs:**
  * Implement scaled dot-product attention and `MultiHeadAttention` in [`src/attention.py`](04_deep_learning/src/attention.py).
  * Complete [`notebooks/01_deep_learning_architectures.ipynb`](04_deep_learning/notebooks/01_deep_learning_architectures.ipynb).
* **Deliverable & Tests:**
  * Pass [`tests/test_attention.py`](04_deep_learning/tests/test_attention.py).

---

### Week 16: Foundation Model Adaptation: LoRA & Vector Search RAG
* **Theory:**
  * Transfer Learning: Pretrained backbone feature extraction and classification head replacement.
  * Parameter-Efficient Fine-Tuning (PEFT): Low-Rank Adaptation (LoRA) formulation ($\Delta W = \frac{\alpha}{r} A B$), weight merging for zero-latency deployment.
  * Retrieval-Augmented Generation (RAG): Hallucination mitigation, document chunking, dense semantic vector search, and prompt augmentation.
* **Practice & Labs:**
  * Implement backbone freezing and fine-tuning in [`src/transfer_learning.py`](04_deep_learning/src/transfer_learning.py).
  * Implement LoRA linear layer and weight merging in [`src/lora.py`](04_deep_learning/src/lora.py).
  * Build dense vector index and RAG pipeline in [`src/rag_vector_search.py`](04_deep_learning/src/rag_vector_search.py).
* **Deliverable & Tests (Milestone 3):**
  * Pass [`tests/test_transfer_learning.py`](04_deep_learning/tests/test_transfer_learning.py), [`tests/test_lora.py`](04_deep_learning/tests/test_lora.py), and [`tests/test_rag.py`](04_deep_learning/tests/test_rag.py).
  * **Milestone 3 Defense:** Demonstrate a working RAG retrieval pipeline and LoRA parameter efficiency calculations.

---

## 🚀 Phase 4: Production MLOps, High-Performance APIs & Deployment (Weeks 17–20)

### Phase Description
The final phase prepares students for engineering in production environments. Students transition trained models into hardened, containerized, low-latency microservices with automated experiment tracking, hardware acceleration (ONNX), strict data contracts, and integration testing.

---

### Week 17: Production Inference Microservices with FastAPI & Pydantic
* **Theory:**
  * Asynchronous request processing: ASGI, event loops, and Uvicorn workers.
  * Strict data validation contracts: Pydantic schemas, physical range limits, NaN rejection.
  * In-memory model caching: Eliminating disk I/O with lifespan managers (`@asynccontextmanager`).
* **Practice & Labs:**
  * Repo Directory: `05_mlops_production/`
  * Define input/output contracts in [`app/schemas.py`](05_mlops_production/app/schemas.py).
  * Implement thread-safe model manager in [`app/model_loader.py`](05_mlops_production/app/model_loader.py).
  * Build `/health`, `/predict`, and `/batch_predict` endpoints in [`app/main.py`](05_mlops_production/app/main.py).
* **Deliverable & Tests:**
  * Pass [`tests/test_api.py`](05_mlops_production/tests/test_api.py).

---

### Week 18: Model Compilation, Optimization & ONNX Runtime Acceleration
* **Theory:**
  * Model optimization bottlenecks: Python GIL contention, framework overhead.
  * Open Neural Network Exchange (ONNX) representation and protobuf graph format.
  * Graph optimizations: Constant folding, node elimination, operator fusion.
  * Quantization and execution providers (CPU vs CUDA vs TensorRT).
* **Practice & Labs:**
  * Implement model conversion in [`app/onnx_converter.py`](05_mlops_production/app/onnx_converter.py).
  * Benchmark latency distributions (mean, p50, p95, p99) between Scikit-Learn and ONNX Runtime.
* **Deliverable & Tests:**
  * Pass [`tests/test_onnx.py`](05_mlops_production/tests/test_onnx.py).

---

### Week 19: Experiment Tracking, Model Registry & Docker Containerization
* **Theory:**
  * Reproducibility crisis in ML: Hyperparameter tracking, run lineage, and artifact logging.
  * MLflow architecture: Tracking Server, Model Artifacts, and Model Registry staging.
  * Container virtualization: Multi-stage Docker builds, layer caching, non-root user execution, and orchestration with Docker Compose.
* **Practice & Labs:**
  * Implement experiment tracker and best-run query in [`app/mlflow_tracker.py`](05_mlops_production/app/mlflow_tracker.py).
  * Build and test [`Dockerfile`](05_mlops_production/Dockerfile) and [`docker-compose.yml`](05_mlops_production/docker-compose.yml).
* **Deliverable & Tests:**
  * Pass [`tests/test_mlflow.py`](05_mlops_production/tests/test_mlflow.py).
  * Successful container build and local healthcheck query via `curl`.

---

### Week 20: Capstone Project Defense & Automated Quality Gates (CI/CD)
* **Theory:**
  * Continuous Integration for ML: Pre-commit hooks, linting gates (`ruff`), automated test runners (`pytest`).
  * Monitoring deployed models: Concept drift, covariate shift, latency degradation.
* **Practice & Labs:**
  * Assemble the full end-to-end pipeline:
    1. Ingest raw data with DuckDB/Polars.
    2. Validate with Pandera contracts.
    3. Train and track models with MLflow.
    4. Compile best model to ONNX.
    5. Serve via FastAPI inside a Docker container.
* **Deliverable & Assessment:**
  * **Final Capstone Defense:** Comprehensive demonstration of the live microservice under concurrent load with complete test suite passing.
  * Complete repository-wide verification: 100% test pass rate across all 58 tests.
