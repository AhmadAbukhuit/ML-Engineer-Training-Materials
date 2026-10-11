"""Principal Component Analysis (PCA) from scratch using SVD and Covariance Eigendecomposition."""

from typing import Literal
import numpy as np


class PCA:
    """Principal Component Analysis (PCA) from first principles.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    method : {'svd', 'covariance'}
        Computational approach: Singular Value Decomposition or Eigendecomposition of Covariance.
    """

    def __init__(self, n_components: int = 2, method: Literal["svd", "covariance"] = "svd") -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        self.n_components = n_components
        self.method = method

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA on training data matrix X."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.n_components > min(n_samples, n_features):
            raise ValueError("n_components cannot exceed min(n_samples, n_features).")

        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if self.method == "svd":
            # X_centered = U * S * Vt
            # Eigenvalues of Covariance = S^2 / (n - 1)
            # Principal directions = rows of Vt (or columns of V)
            _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            explained_var = (S**2) / (n_samples - 1)
            total_var = np.sum(explained_var)

            self.components_ = Vt[: self.n_components]
            self.explained_variance_ = explained_var[: self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ / total_var

        elif self.method == "covariance":
            cov_matrix = np.cov(X_centered, rowvar=False)
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])

            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # eigh returns in ascending order, sort descending
            sort_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            total_var = np.sum(eigenvalues)
            self.components_ = eigenvectors[:, : self.n_components].T
            self.explained_variance_ = eigenvalues[: self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the principal component axes."""
        assert self.mean_ is not None and self.components_ is not None
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return projected coordinates."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """Reconstruct original feature space from principal components."""
        assert self.mean_ is not None and self.components_ is not None
        X_proj = np.asarray(X_proj, dtype=np.float64)
        return np.dot(X_proj, self.components_) + self.mean_
