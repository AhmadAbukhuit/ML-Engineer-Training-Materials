import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from linear_models import LinearRegression, LogisticRegression


def test_linear_regression_normal_equation():
    # Synthetic ground truth: y = 3*x1 - 2*x2 + 4.0
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 2))
    true_weights = np.array([3.0, -2.0])
    true_bias = 4.0
    y = X @ true_weights + true_bias + rng.normal(scale=0.01, size=200)

    model = LinearRegression(fit_intercept=True)
    model.fit_normal_equation(X, y)

    np.testing.assert_allclose(model.weights, true_weights, atol=0.05)
    assert np.isclose(model.bias, true_bias, atol=0.05)

    # Test predictions
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    np.testing.assert_allclose(preds, y[:5], atol=0.1)


def test_linear_regression_gradient_descent():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 2))
    true_weights = np.array([1.5, -1.0])
    true_bias = 2.0
    y = X @ true_weights + true_bias

    model = LinearRegression(fit_intercept=True)
    model.fit_gradient_descent(X, y, lr=0.1, epochs=1500)

    np.testing.assert_allclose(model.weights, true_weights, atol=0.05)
    assert np.isclose(model.bias, true_bias, atol=0.05)


def test_logistic_regression():
    # Synthetic linearly separable classification dataset
    rng = np.random.default_rng(42)
    X_pos = rng.normal(loc=2.0, scale=0.8, size=(100, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.8, size=(100, 2))

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 100 + [0] * 100)

    clf = LogisticRegression(lr=0.1, epochs=800)
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.95, f"Expected >95% accuracy, got {acc:.2f}"

    probs = clf.predict_proba(X)
    assert probs.shape == (200, 2)
    # Probabilities must sum to 1 across classes
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
