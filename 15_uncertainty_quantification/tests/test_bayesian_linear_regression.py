"""Tests for Bayesian Linear Regression."""

import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bayesian_linear_regression import BayesianLinearRegression


def test_bayesian_linear_regression_fit_and_predict():
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-1, 1, size=(50, 1))
    y_train = 2.5 * X_train[:, 0] + rng.normal(scale=0.1, size=50)

    model = BayesianLinearRegression(alpha=1.0, beta=100.0)
    model.fit(X_train, y_train)

    assert model.m_N is not None
    assert model.S_N is not None
    # 2 parameters (bias + 1 feature)
    assert model.m_N.shape == (2,)
    assert model.S_N.shape == (2, 2)

    X_test = np.array([[-0.5], [0.0], [0.5]])
    mean, total_std, epistemic_std = model.predict(X_test, return_std=True)

    assert mean.shape == (3,)
    assert total_std.shape == (3,)
    assert epistemic_std.shape == (3,)

    # Variance decomposition verification: total_var == epistemic_var + 1/beta
    aleatoric_var = 1.0 / model.beta
    np.testing.assert_allclose(total_std**2, epistemic_std**2 + aleatoric_var, rtol=1e-5)


def test_epistemic_uncertainty_growth_out_of_distribution():
    # Training data concentrated in [-1, 1]
    X_train = np.linspace(-1, 1, 20).reshape(-1, 1)
    y_train = 1.5 * X_train[:, 0]

    model = BayesianLinearRegression(alpha=1.0, beta=50.0)
    model.fit(X_train, y_train)

    X_in_distribution = np.array([[0.0]])
    X_far_out_of_distribution = np.array([[10.0]])

    _, _, std_in = model.predict(X_in_distribution, return_std=True)
    _, _, std_ood = model.predict(X_far_out_of_distribution, return_std=True)

    # Epistemic uncertainty must be significantly higher far outside training domain
    assert std_ood[0] > std_in[0] * 5.0
