# 🚀 Phase 4: Production APIs, Containerization & MLOps

A high-accuracy model has zero business value until it is securely deployed, versioned, monitored, and exposed through low-latency microservices that downstream applications can reliably query.

---

## 🎯 Learning Objectives

By the end of this phase, you will:

1. **Build Production REST APIs:** Develop high-performance, asynchronous inference services using `FastAPI` and `Uvicorn`.
2. **Enforce Strict Data Contracts:** Use `Pydantic` models to validate input schemas, reject malformed payloads, and handle edge cases at runtime.
3. **Implement Thread-Safe Model Ingestion:** Cache serialized model weights in memory during application startup to achieve sub-millisecond inference latency.
4. **Package Reproducible Containers:** Write lean, multi-stage `Dockerfile` definitions that encapsulate OS dependencies, Python runtimes, and model artifacts.
5. **Orchestrate Multi-Service Stacks:** Configure `docker-compose.yml` for local service simulation (API + Redis cache or Prometheus monitoring).
6. **Automate Quality Gates (CI/CD):** Set up automated testing pipelines that run linting and unit test suites before any code or weights can be merged.

---

## 📚 Curated Study Resources

### 📖 Framework Documentation & Runbooks
* **[FastAPI Official Documentation](https://fastapi.tiangolo.com/)**: Asynchronous route handling, dependency injection, and Pydantic integration.
* **[Docker Documentation](https://docs.docker.com/get-started/)**: Container virtualization, multi-stage builds, and layer caching.
* **[Full Stack Deep Learning](https://fullstackdeeplearning.com/)**: Production ML system design, deployment patterns, and monitoring.
* **[MLflow Documentation](https://mlflow.org/docs/latest/index.html)**: Experiment tracking, model registry, and artifact staging.

---

## 🔬 Core Production Concepts & Architecture

### 1. The Inference Microservice Architecture

```text
[ Client Request ]
        │
        ▼ (HTTP POST /predict)
┌──────────────────────────────────────────────┐
│  FastAPI Application                         │
│  ├─ 1. Pydantic Request Validation           │
│  │     (Type check, Range check, NaN reject) │
│  ├─ 2. Feature Extraction & Alignment        │
│  ├─ 3. In-Memory Model Scoring (.predict)    │
│  │     (Cached .joblib model)                │
│  └─ 4. Pydantic Response Serialization       │
└──────────────────────────────────────────────┘
        │
        ▼ (JSON: class, probability, latency_ms)
[ Response ]
```

### 2. In-Memory Weight Caching
* **Bad Practice:** Loading the `.joblib` or `.pt` model file from disk inside the `/predict` route handler. This adds hundreds of milliseconds of disk I/O to every individual HTTP request.
* **Production Best Practice:** Load model weights once during the FastAPI lifespan (`@asynccontextmanager`) and keep the weights resident in process memory.

### 3. Containerization Principles for ML
* **Base Image:** Use slim Debian-based Python images (e.g., `python:3.11-slim`) rather than bloated default images.
* **Layer Caching:** Copy `requirements.txt` and install dependencies *before* copying application code so Docker reuses cached dependency layers across code edits.
* **Non-Root Execution:** Always create and run under a non-root system user inside the container for security isolation.

---

## 🛠️ Practical Coding Tasks

Complete these implementation tasks in `05_mlops_production/` and verify your API with the integration test suite in `05_mlops_production/tests/`.

---

### Task 4.1: Pydantic Data Contracts & Schemas
* **File:** `app/schemas.py`
* **Requirements:**
  1. Define `TelemetryFeatures` representing the 10 numerical features expected by the trained baseline model.
  2. Implement input validators enforcing realistic physical limits (e.g., non-negative sensor readings, rejection of infinite/NaN values).
  3. Define `PredictionResponse` returning predicted class, probability distribution, and execution latency.
  4. Define `BatchPredictionRequest` supporting vectorized multi-sample scoring.

---

### Task 4.2: Thread-Safe Model Manager
* **File:** `app/model_loader.py`
* **Requirements:**
  1. Implement a `ModelManager` singleton that caches the loaded model artifact in RAM.
  2. Provide a clean `.predict(features)` and `.predict_proba(features)` interface.
  3. Provide a fallback mock model when running tests or if artifacts are not yet generated.

---

### Task 4.3: High-Performance FastAPI Service
* **File:** `app/main.py`
* **Requirements:**
  1. Endpoint `GET /health`: Liveness and readiness probe reporting status and model load state.
  2. Endpoint `POST /predict`: Scores a single sample with sub-millisecond execution time.
  3. Endpoint `POST /batch_predict`: Vectorized batch scoring for high-throughput client requests.
  4. Endpoint `GET /model_info`: Exposes metadata (model type, feature count, version).

---

### Task 4.4: Production Containerization & Orchestration
* **Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`
* **Requirements:**
  1. Build a multi-stage or lean `Dockerfile` running Uvicorn workers.
  2. Configure `docker-compose.yml` to launch the API service and expose port 8000.

---

### Task 4.5: Integration Test Suite
* **File:** `tests/test_api.py`
* **Requirements:**
  1. Use FastAPI `TestClient` to verify `/health` returns `200 OK`.
  2. Verify `/predict` correctly parses valid input and returns probabilities summing to 1.
  3. Verify input validation properly returns `422 Unprocessable Entity` when fields are missing or invalid.
  4. Verify `/batch_predict` handles multiple records simultaneously.

---

### Task 4.6: Model Compilation & Latency Optimization with ONNX
* **File:** `app/onnx_converter.py`
* **Requirements:**
  1. Implement `export_to_onnx` converting trained models to portable ONNX graphs.
  2. Build `ONNXInferenceSession` for high-performance, multithreaded CPU scoring.
  3. Implement `benchmark_inference` quantifying latency distributions (mean, p50, p95, p99).
* **Deliverable:** Unit tests verifying prediction parity between Scikit-Learn and ONNX Runtime.

---

### Task 4.7: Experiment Tracking & Governance with MLflow
* **File:** `app/mlflow_tracker.py`
* **Requirements:**
  1. Implement `MLflowTracker` encapsulating experiment setup, parameter logging, and metric tracking.
  2. Support model artifact versioning and run tagging.
  3. Implement `get_best_run` to automatically retrieve top candidate models for production promotion.
* **Deliverable:** Unit tests verifying metric persistence and best candidate selection.

---

## 📁 Directory Structure for Phase 4

```text
05_mlops_production/
├── README.md
├── 01_onnx_and_mlflow.md
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── onnx_converter.py
│   └── mlflow_tracker.py
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_onnx.py
    └── test_mlflow.py
```

