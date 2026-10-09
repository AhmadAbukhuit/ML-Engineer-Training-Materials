"""
ONNX Model Export, Optimization, and Inference Runtime Engine.

This module provides:
1. export_to_onnx: Converts trained scikit-learn classifiers into portable ONNX computational graphs.
2. ONNXInferenceSession: High-performance inference runner using onnxruntime.
3. benchmark_inference: Benchmarks latency distributions (p50, p95, p99) between native Python/scikit-learn and ONNX Runtime.
"""

from pathlib import Path
import time
from typing import Any
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_to_onnx(
    model: Any,
    n_features: int,
    output_path: str | Path,
    target_opset: int = 15,
) -> Path:
    """
    Exports a trained scikit-learn model to an ONNX protobuf file.

    Args:
        model: Trained scikit-learn estimator.
        n_features: Input feature dimensionality.
        output_path: Destination path for .onnx file.
        target_opset: ONNX operator set version.

    Returns:
        Path to the saved ONNX model.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=target_opset,
    )

    onnx.checker.check_model(onnx_model)

    with open(output_file, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return output_file


class ONNXInferenceSession:
    """
    Production-grade inference runner utilizing ONNX Runtime.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        # Create execution session with CPU provider
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Runs inference and returns predicted class labels.
        """
        x_float = np.ascontiguousarray(X, dtype=np.float32)
        outputs = self.session.run([self.output_names[0]], {self.input_name: x_float})
        return np.array(outputs[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Runs inference and returns class probabilities.
        """
        x_float = np.ascontiguousarray(X, dtype=np.float32)
        if len(self.output_names) > 1:
            outputs = self.session.run([self.output_names[1]], {self.input_name: x_float})
            raw_probs = outputs[0]
            # ONNX probabilities may be represented as a list of dicts or a 2D float array
            if isinstance(raw_probs, list) and len(raw_probs) > 0 and isinstance(raw_probs[0], dict):
                probs = np.array([[row[k] for k in sorted(row.keys())] for row in raw_probs])
                return probs
            return np.array(raw_probs)

        # Fallback to single output
        outputs = self.session.run([self.output_names[0]], {self.input_name: x_float})
        return np.array(outputs[0])


def benchmark_inference(
    sklearn_model: Any,
    onnx_session: ONNXInferenceSession,
    sample: np.ndarray,
    n_iterations: int = 100,
) -> dict[str, dict[str, float]]:
    """
    Executes latency benchmarks comparing scikit-learn and ONNX Runtime.

    Args:
        sklearn_model: Original trained model.
        onnx_session: Instantiated ONNXInferenceSession.
        sample: Feature array of shape (1, n_features) or (N, n_features).
        n_iterations: Number of warmup and evaluation executions.

    Returns:
        Dictionary containing p50, p95, p99, and mean latency in milliseconds.
    """
    sample_float = np.ascontiguousarray(sample, dtype=np.float32)

    # Warmup
    for _ in range(10):
        _ = sklearn_model.predict(sample)
        _ = onnx_session.predict(sample_float)

    # Benchmark Sklearn
    sklearn_latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = sklearn_model.predict(sample)
        t1 = time.perf_counter()
        sklearn_latencies.append((t1 - t0) * 1000.0)

    # Benchmark ONNX
    onnx_latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = onnx_session.predict(sample_float)
        t1 = time.perf_counter()
        onnx_latencies.append((t1 - t0) * 1000.0)

    def _stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean_ms": float(np.round(np.mean(a), 4)),
            "p50_ms": float(np.round(np.percentile(a, 50), 4)),
            "p95_ms": float(np.round(np.percentile(a, 95), 4)),
            "p99_ms": float(np.round(np.percentile(a, 99), 4)),
        }

    return {
        "sklearn": _stats(sklearn_latencies),
        "onnx": _stats(onnx_latencies),
    }
