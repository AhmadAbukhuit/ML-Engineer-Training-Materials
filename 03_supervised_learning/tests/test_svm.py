import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from svm import LinearSVM


def test_linear_svm_separable():
    # Linearly separable 2D clusters
    rng = np.random.default_rng(42)
    X_pos = rng.normal(loc=3.0, scale=0.6, size=(50, 2))
    X_neg = rng.normal(loc=-3.0, scale=0.6, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)

    svm = LinearSVM(lambda_param=0.01, lr=0.01, n_iters=500)
    svm.fit(X, y)

    preds = svm.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95, f"Expected SVM accuracy >= 95%, got {acc:.2f}"

    margins = svm.decision_function(X)
    assert margins.shape == (100,)
