import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shapley_values import ExactShapleyExplainer, SamplingShapleyExplainer


def test_exact_shapley_linear_model():
    # Linear function: f(x) = 2*x0 - 3*x1 + 5*x2
    weights = np.array([2.0, -3.0, 5.0])

    def predict_fn(X):
        return np.dot(X, weights)

    baseline = np.array([0.0, 0.0, 0.0])
    x = np.array([3.0, 2.0, 1.0])

    explainer = ExactShapleyExplainer(predict_fn, baseline)
    shap_vals = explainer.explain_instance(x)

    # For purely linear models, Shapley value of x_i is w_i * (x_i - baseline_i)
    expected_shap = weights * (x - baseline)
    np.testing.assert_allclose(shap_vals, expected_shap, rtol=1e-5)

    # Efficiency axiom: sum(phi_i) == f(x) - f(baseline)
    fx = float(predict_fn(x.reshape(1, -1))[0])
    f_base = float(predict_fn(baseline.reshape(1, -1))[0])
    assert np.sum(shap_vals) == pytest.approx(fx - f_base)


def test_sampling_shapley_approximation():
    weights = np.array([1.5, -2.0, 4.0])

    def predict_fn(X):
        return np.dot(X, weights)

    baseline = np.zeros(3)
    x = np.array([2.0, 1.0, 3.0])

    explainer = SamplingShapleyExplainer(predict_fn, baseline, n_samples=500, random_state=42)
    shap_vals = explainer.explain_instance(x)

    expected_shap = weights * x
    np.testing.assert_allclose(shap_vals, expected_shap, atol=0.25)
