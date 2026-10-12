"""Tests for Gaussian Process Regression."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gaussian_process import GaussianProcessRegressor


def test_gaussian_process_interpolation():
    # In the limit of low noise, GP interpolates observations exactly
    X_train = np.array([[-2.0], [0.0], [2.0]])
    y_train = np.sin(X_train[:, 0])

    gp = GaussianProcessRegressor(length_scale=1.0, signal_variance=1.0, noise_variance=1e-6)
    gp.fit(X_train, y_train)

    mu, std = gp.predict(X_train, return_std=True)

    np.testing.assert_allclose(mu, y_train, atol=1e-3)
    # Uncertainty at known training points should be near noise level
    assert np.all(std < 0.05)


def test_gaussian_process_uncertainty_in_gap():
    # Observations at -2.0 and +2.0, leaving gap at 0.0
    X_train = np.array([[-2.0], [2.0]])
    y_train = np.array([1.0, 1.0])

    gp = GaussianProcessRegressor(length_scale=1.0, signal_variance=1.0, noise_variance=1e-4)
    gp.fit(X_train, y_train)

    X_test = np.array([[-2.0], [0.0], [10.0]])
    _, std = gp.predict(X_test, return_std=True)

    # std at 0.0 (in the gap) and 10.0 (extrapolating) should be higher than at -2.0
    assert std[1] > std[0]
    assert std[2] > std[0]
