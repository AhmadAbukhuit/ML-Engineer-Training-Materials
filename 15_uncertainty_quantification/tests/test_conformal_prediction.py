"""Tests for Split Conformal Prediction."""

import sys
from pathlib import Path
import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conformal_prediction import ConformalClassifier, SplitConformalRegressor


class DummyRegressor:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] * 2.0


class DummyClassifier:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # 3 classes, returns deterministic softmax-like probabilities
        n = X.shape[0]
        p = np.zeros((n, 3))
        for i in range(n):
            if X[i, 0] > 0:
                p[i] = [0.8, 0.15, 0.05]
            else:
                p[i] = [0.1, 0.7, 0.2]
        return p


def test_split_conformal_regressor_coverage():
    rng = np.random.default_rng(42)
    # Generate linear dataset with Gaussian noise
    X_cal = rng.normal(size=(500, 2))
    y_cal = X_cal[:, 0] * 2.0 + rng.normal(scale=0.5, size=500)

    X_test = rng.normal(size=(200, 2))
    y_test = X_test[:, 0] * 2.0 + rng.normal(scale=0.5, size=200)

    base = DummyRegressor()
    conformal = SplitConformalRegressor(base)
    alpha = 0.1  # 90% target coverage
    conformal.calibrate(X_cal, y_cal, alpha=alpha)

    lower, upper = conformal.predict_interval(X_test)
    assert len(lower) == 200
    assert np.all(upper >= lower)

    # Calculate empirical test coverage
    covered = (y_test >= lower) & (y_test <= upper)
    empirical_coverage = np.mean(covered)

    # Finite sample coverage should be close to or above 1 - alpha (>= 85%)
    assert empirical_coverage >= 0.85


def test_conformal_classifier_prediction_sets():
    rng = np.random.default_rng(42)
    X_cal = rng.choice([-1.0, 1.0], size=(300, 1))
    y_cal = np.where(X_cal[:, 0] > 0, 0, 1)

    X_test = np.array([[1.0], [-1.0], [0.5]])
    base = DummyClassifier()
    clf = ConformalClassifier(base)
    clf.calibrate(X_cal, y_cal, alpha=0.1)

    sets = clf.predict_set(X_test)
    assert len(sets) == 3
    # Check that sets are non-empty lists of valid class indices
    for s in sets:
        assert isinstance(s, list)
        assert len(s) >= 1
        assert all(label in [0, 1, 2] for label in s)


def test_invalid_alpha_raises():
    base = DummyRegressor()
    conformal = SplitConformalRegressor(base)
    with pytest.raises(ValueError):
        conformal.calibrate(np.zeros((10, 1)), np.zeros(10), alpha=1.5)
