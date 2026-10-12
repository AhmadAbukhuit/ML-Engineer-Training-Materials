import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lime_explainer import LimeTabularExplainer


def test_lime_local_linear_surrogate():
    # Ground truth model: f(x) = 4*x0 - 2*x1 + 0.1*(x0^2)
    def predict_fn(X):
        return 4.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * (X[:, 0] ** 2)

    x = np.array([1.0, 2.0])
    # Local gradient near x=[1, 2]:
    # df/dx0 = 4 + 0.2*1.0 = 4.2
    # df/dx1 = -2.0

    lime = LimeTabularExplainer(predict_fn, kernel_width=0.5, n_samples=1000, random_state=42)
    coefs, _ = lime.explain_instance(x)

    assert len(coefs) == 2
    # Local slope for feature 0 should be positive (~4.2)
    assert coefs[0] > 3.0
    # Local slope for feature 1 should be negative (~ -2.0)
    assert coefs[1] < -1.5
