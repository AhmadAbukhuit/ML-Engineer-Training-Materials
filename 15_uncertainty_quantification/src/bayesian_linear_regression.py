"""Bayesian Linear Regression with Conjugate Gaussian Prior."""

import numpy as np


class BayesianLinearRegression:
    """Analytical Bayesian Linear Regression (Bishop Chapter 3).

    Prior: p(w) = N(0, alpha^-1 * I)
    Likelihood: p(y | X, w) = N(Xw, beta^-1 * I)

    Posterior: p(w | X, y) = N(m_N, S_N)
        S_N^-1 = alpha * I + beta * X^T X
        m_N = beta * S_N * X^T y

    Predictive Distribution: p(y_* | x_*, X, y) = N(mu_*, sigma_*^2)
        mu_* = x_*^T m_N
        sigma_*^2 = beta^-1 (aleatoric) + x_*^T S_N x_* (epistemic)

    Parameters
    ----------
    alpha : float
        Precision (inverse variance) of the isotropic Gaussian weight prior.
    beta : float
        Precision (inverse variance) of the observation noise likelihood.
    include_bias : bool
        Whether to prepend an intercept column of ones to X.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 25.0, include_bias: bool = True) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.include_bias = include_bias

        self.m_N: np.ndarray | None = None  # posterior mean
        self.S_N: np.ndarray | None = None  # posterior covariance

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.include_bias:
            bias = np.ones((X.shape[0], 1), dtype=np.float64)
            return np.hstack([bias, X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        """Compute closed-form posterior distribution over weight vector w."""
        Phi = self._prepare_X(X)
        y = np.asarray(y, dtype=np.float64)
        D = Phi.shape[1]

        # S_N^-1 = alpha * I + beta * Phi^T Phi
        precision_matrix = self.alpha * np.eye(D) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(precision_matrix)
        # m_N = beta * S_N * Phi^T y
        self.m_N = self.beta * (self.S_N @ Phi.T @ y)

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict posterior mean and decomposed uncertainties.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        return_std : bool
            If True, returns (mean, total_std, epistemic_std).

        Returns
        -------
        mean : np.ndarray of shape (n_samples,)
        total_std : np.ndarray (optional)
        epistemic_std : np.ndarray (optional)
        """
        if self.m_N is None or self.S_N is None:
            raise RuntimeError("Model must be fitted before predict.")

        Phi = self._prepare_X(X)
        mean = Phi @ self.m_N

        if not return_std:
            return mean

        # Epistemic variance: diag(Phi @ S_N @ Phi^T)
        epistemic_var = np.sum((Phi @ self.S_N) * Phi, axis=1)
        epistemic_var = np.maximum(epistemic_var, 0.0)
        epistemic_std = np.sqrt(epistemic_var)

        # Total variance = aleatoric (1 / beta) + epistemic
        aleatoric_var = 1.0 / self.beta
        total_std = np.sqrt(epistemic_var + aleatoric_var)

        return mean, total_std, epistemic_std
