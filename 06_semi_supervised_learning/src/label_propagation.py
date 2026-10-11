"""Graph-Based Label Propagation from First Principles (Zhu & Ghahramani)."""

import numpy as np


class LabelPropagation:
    """Label Propagation algorithm on graph affinity structures.

    Parameters
    ----------
    gamma : float
        RBF kernel bandwidth parameter: W_ij = exp(-gamma * ||x_i - x_j||^2).
    max_iter : int
        Maximum number of propagation steps.
    tol : float
        Convergence tolerance on Frobenius norm of probability shift.
    """

    def __init__(self, gamma: float = 20.0, max_iter: int = 200, tol: float = 1e-4) -> None:
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        self.classes_: np.ndarray | None = None
        self.label_distributions_: np.ndarray | None = None
        self.transduction_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LabelPropagation":
        """Run transductive label propagation.

        Unlabeled samples must be indicated by label -1 in y.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples = X.shape[0]
        labeled_mask = y != -1
        if not np.any(labeled_mask):
            raise ValueError("At least one sample must be labeled (y != -1).")

        self.classes_ = np.unique(y[labeled_mask])
        n_classes = len(self.classes_)

        # 1. Compute pairwise RBF affinity matrix W
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        W = np.exp(-self.gamma * sq_dist)
        np.fill_diagonal(W, 0.0)  # no self-loops

        # Row-normalize to construct transition probability matrix P = D^-1 W
        D = np.sum(W, axis=1, keepdims=True)
        D = np.maximum(D, 1e-12)
        P = W / D

        # 2. Initialize label distribution matrix Y: shape (n_samples, n_classes)
        Y = np.zeros((n_samples, n_classes), dtype=np.float64)
        for c_idx, c_val in enumerate(self.classes_):
            Y[y == c_val, c_idx] = 1.0

        # Preserve original clamp matrix for labeled nodes
        Y_clamp = Y[labeled_mask].copy()

        # 3. Iterative propagation
        for _ in range(self.max_iter):
            Y_new = np.dot(P, Y)

            # Clamp labeled nodes to ground truth
            Y_new[labeled_mask] = Y_clamp

            # Re-normalize rows to ensure valid probabilities
            row_sums = np.sum(Y_new, axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            Y_new = Y_new / row_sums

            # Check convergence
            diff_norm = np.linalg.norm(Y_new - Y, ord="fro")
            Y = Y_new

            if diff_norm < self.tol:
                break

        self.label_distributions_ = Y
        self.transduction_ = self.classes_[np.argmax(Y, axis=1)]
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        """Return transductive predictions."""
        assert self.transduction_ is not None
        return self.transduction_

    def predict_proba(self, X: np.ndarray | None = None) -> np.ndarray:
        """Return final label probability distributions."""
        assert self.label_distributions_ is not None
        return self.label_distributions_
