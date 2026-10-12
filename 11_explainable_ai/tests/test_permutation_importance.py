import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from permutation_importance import permutation_feature_importance


def test_permutation_importance_identifies_key_features():
    rng = np.random.default_rng(42)
    # y depends strongly on feature 0, slightly on feature 1, and zero on feature 2 (noise)
    X = rng.normal(size=(200, 3))
    y = 5.0 * X[:, 0] + 0.5 * X[:, 1] + 0.0 * X[:, 2] + rng.normal(scale=0.1, size=200)

    model = LinearRegression().fit(X, y)

    def neg_mse(y_true, y_pred):
        return -float(np.mean((y_true - y_pred) ** 2))

    res = permutation_feature_importance(
        model, X, y, scoring_fn=neg_mse, n_repeats=5, random_state=42
    )

    means = res["importances_mean"]
    # Feature 0 should have the largest performance drop
    assert means[0] > means[1]
    # Noise feature 2 should have near-zero importance
    assert means[0] > means[2]
    assert means[2] < 0.2
