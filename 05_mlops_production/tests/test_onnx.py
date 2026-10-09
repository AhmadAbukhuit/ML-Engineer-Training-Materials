import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from onnx_converter import ONNXInferenceSession, benchmark_inference, export_to_onnx


def test_export_and_inference_onnx(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1.0, (100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Train sklearn model
    clf = LogisticRegression()
    clf.fit(X, y)

    onnx_file = tmp_path / "model.onnx"
    export_to_onnx(clf, n_features=5, output_path=onnx_file)
    assert onnx_file.exists()

    # Load session
    session = ONNXInferenceSession(onnx_file)
    sample = X[:10].astype(np.float32)

    # Compare predictions
    sklearn_preds = clf.predict(sample)
    onnx_preds = session.predict(sample)
    assert np.array_equal(sklearn_preds, onnx_preds)

    # Compare probabilities
    sklearn_probs = clf.predict_proba(sample)
    onnx_probs = session.predict_proba(sample)
    assert np.allclose(sklearn_probs, onnx_probs, atol=1e-5)


def test_benchmark_inference(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1.0, (50, 4))
    y = (X[:, 0] > 0).astype(int)

    clf = LogisticRegression()
    clf.fit(X, y)

    onnx_file = tmp_path / "bench_model.onnx"
    export_to_onnx(clf, n_features=4, output_path=onnx_file)
    session = ONNXInferenceSession(onnx_file)

    sample = X[:1].astype(np.float32)
    stats = benchmark_inference(clf, session, sample, n_iterations=20)

    assert "sklearn" in stats
    assert "onnx" in stats
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        assert key in stats["sklearn"]
        assert key in stats["onnx"]
        assert stats["sklearn"][key] >= 0.0
        assert stats["onnx"][key] >= 0.0
