"""Gaussian Process Regression (GPR) with RBF Kernel in NumPy."""

import numpy as np


class GaussianProcessRegressor:
    """Exact Gaussian Process Regression using Squared Exponential (RBF) Kernel.

    Prior: f(x) ~ GP(0, k(x, x'))
    Kernel: k(x, x') = sigma_f^2 * exp(- ||x - x'||^2 / (2 * length_scale^2))
    Likelihood: y = f(x) + eps, eps ~ N(0, noise_variance)

    Parameters
    ----------
    length_scale : float
        Characteristic length scale of the RBF kernel.
    signal_variance : float
        Output amplitude sigma_f^2.
    noise_variance : float
        Homoscedastic Gaussian observation noise sigma_n^2.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-4,
    ) -> None:
        self.length_scale = float(length_scale)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)

        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.L: np.ndarray | None = None  # Cholesky decomposition of K_y
        self.alpha: np.ndarray | None = None  # (K_y)^-1 * y

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Efficient pairwise squared Euclidean distances: ||x - x'||^2 = x^2 - 2xx' + x'^2
        dist_sq = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1, keepdims=True).T
            - 2.0 * np.dot(X1, X2.T)
        )
        dist_sq = np.maximum(dist_sq, 0.0)
        return self.signal_variance * np.exp(-dist_sq / (2.0 * self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """Fit GP by computing Cholesky factor of kernel covariance matrix."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        n = self.X_train.shape[0]

        K = self._rbf_kernel(self.X_train, self.X_train)
        K_y = K + (self.noise_variance + 1e-8) * np.eye(n)

        # Cholesky factor: K_y = L @ L.T
        self.L = np.linalg.cholesky(K_y)
        # Solve L @ L.T @ alpha = y
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Compute posterior predictive mean and epistemic uncertainty.

        Parameters
        ----------
        X : np.ndarray of shape (n_test, n_features)
        return_std : bool
            Whether to return standard deviation of the posterior latent function.
        """
        if self.X_train is None or self.alpha is None or self.L is None:
            raise RuntimeError("Must fit GP before predict.")

        X = np.asarray(X, dtype=np.float64)
        K_s = self._rbf_kernel(self.X_train, X)  # shape (n_train, n_test)

        # Posterior mean: mu_* = K_s^T @ alpha
        mu = np.dot(K_s.T, self.alpha)

        if not return_std:
            return mu

        # Posterior variance: v = L^-1 @ K_s; var = diag(K_ss) - sum(v^2, axis=0)
        v = np.linalg.solve(self.L, K_s)
        k_diag = self.signal_variance * np.ones(X.shape[0])
        var = k_diag - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)

        return mu, np.sqrt(var)
