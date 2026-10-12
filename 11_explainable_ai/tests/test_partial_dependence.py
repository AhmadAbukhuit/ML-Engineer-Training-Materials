import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from partial_dependence import individual_conditional_expectation, partial_dependence


def test_partial_dependence_monotonic():
    # f(x) = 3*x0 + x1^2
    def predict_fn(X):
        return 3.0 * X[:, 0] + X[:, 1] ** 2

    X = np.random.randn(50, 2)
    grid_vals, pdp_vals = partial_dependence(
        predict_fn, X, feature_idx=0, grid_resolution=20, grid_range=(-2.0, 2.0)
    )

    assert len(grid_vals) == 20
    assert len(pdp_vals) == 20
    # For a positive linear coefficient 3*x0, PDP values should be monotonically increasing
    assert np.all(np.diff(pdp_vals) > 0)


def test_ice_curves_shape():
    def predict_fn(X):
        return X[:, 0] * 2.0

    X = np.random.randn(30, 3)
    grid_vals, ice = individual_conditional_expectation(predict_fn, X, feature_idx=0, grid_resolution=15)

    assert grid_vals.shape == (15,)
    assert ice.shape == (30, 15)
