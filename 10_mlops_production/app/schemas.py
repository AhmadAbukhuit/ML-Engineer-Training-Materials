"""
Pydantic Request & Response Data Contracts for Inference Microservice.
"""

from pydantic import BaseModel, Field, field_validator
import math


class PredictionRequest(BaseModel):
    """
    Single sample inference payload. Expects 10 continuous numeric features.
    """
    features: list[float] = Field(
        ...,
        description="List of 10 numeric feature values for classification.",
        min_length=10,
        max_length=10,
        examples=[[0.5, -1.2, 3.4, 0.0, 1.1, -0.4, 2.2, -0.9, 0.1, 1.5]]
    )

    @field_validator("features")
    @classmethod
    def validate_finite_numbers(cls, v: list[float]) -> list[float]:
        for val in v:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Feature values must be finite real numbers (NaN and Inf are not permitted).")
        return v


class PredictionResponse(BaseModel):
    """
    Prediction outcome for a single sample.
    """
    prediction: int = Field(..., description="Predicted class index (0 or 1).")
    probability: float = Field(..., description="Predicted probability of the positive class.")
    class_probabilities: dict[str, float] = Field(..., description="Probabilities for each class.")
    latency_ms: float = Field(..., description="Inference execution latency in milliseconds.")


class BatchPredictionRequest(BaseModel):
    """
    Batch inference request containing multiple samples.
    """
    instances: list[list[float]] = Field(
        ...,
        description="List of feature vectors, each containing 10 numbers.",
        min_length=1,
        examples=[[[0.5, -1.2, 3.4, 0.0, 1.1, -0.4, 2.2, -0.9, 0.1, 1.5],
                   [1.0, 0.2, -0.4, 0.5, -1.1, 0.4, -0.2, 0.9, -0.1, -0.5]]]
    )

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, instances: list[list[float]]) -> list[list[float]]:
        for i, sample in enumerate(instances):
            if len(sample) != 10:
                raise ValueError(f"Instance at index {i} must have exactly 10 features, got {len(sample)}.")
            for val in sample:
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Instance at index {i} contains non-finite number (NaN or Inf).")
        return instances


class BatchPredictionResponse(BaseModel):
    """
    Batch inference response.
    """
    predictions: list[int]
    probabilities: list[float]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    """
    Service health and readiness probe payload.
    """
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """
    Model metadata payload.
    """
    model_name: str
    n_features: int
    version: str
