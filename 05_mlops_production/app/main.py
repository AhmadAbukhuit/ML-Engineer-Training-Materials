"""
Production FastAPI Inference Microservice.

Endpoints:
- GET  /health: Liveness and readiness probe.
- GET  /model_info: Metadata regarding loaded model artifact.
- POST /predict: Single-instance inference with Pydantic validation.
- POST /batch_predict: High-throughput batch inference.
"""

from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException, status
import numpy as np

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from .model_loader import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload model into process memory
    model_manager.load_model()
    yield
    # Teardown logic if needed


app = FastAPI(
    title="ML Engineer Production Inference Microservice",
    description="Low-latency REST API for serving serialized machine learning models.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Kubernetes / Docker health check and readiness probe.
    """
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        version=model_manager.version
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Monitoring"])
def get_model_info():
    """
    Returns metadata about the active model.
    """
    return ModelInfoResponse(
        model_name=model_manager.model_name,
        n_features=model_manager.n_features,
        version=model_manager.version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Generates prediction and probability for a single 10-feature sample.
    """
    start_time = time.perf_counter()
    try:
        # Convert to 2D numpy array: shape (1, 10)
        features = np.array(request.features, dtype=np.float32).reshape(1, -1)

        # In-memory inference
        pred = int(model_manager.predict(features)[0])
        probs = model_manager.predict_proba(features)[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return PredictionResponse(
            prediction=pred,
            probability=float(probs[1]) if len(probs) > 1 else float(probs[0]),
            class_probabilities={f"class_{i}": float(p) for i, p in enumerate(probs)},
            latency_ms=round(latency_ms, 3)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e!s}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict(request: BatchPredictionRequest):
    """
    Vectorized batch inference for multiple samples simultaneously.
    """
    start_time = time.perf_counter()
    try:
        features = np.array(request.instances, dtype=np.float32)

        preds = [int(p) for p in model_manager.predict(features)]
        probs_matrix = model_manager.predict_proba(features)

        # Positive class probabilities (or top probability)
        if probs_matrix.shape[1] > 1:
            positive_probs = [float(p[1]) for p in probs_matrix]
        else:
            positive_probs = [float(p[0]) for p in probs_matrix]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return BatchPredictionResponse(
            predictions=preds,
            probabilities=positive_probs,
            count=len(preds),
            latency_ms=round(latency_ms, 3)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {e!s}"
        )
