"""
Support Vector Machine (SVM) from First Principles.

This module implements a Linear SVM classifier using Soft-Margin Hinge Loss 
and Subgradient Descent optimization in pure NumPy.

Optimization Objective:
min_{w, b} (lambda/2) * ||w||^2 + (1/N) * sum_i max(0, 1 - y_i * (w^T x_i + b))
where y_i in {-1, +1}.
"""

import numpy as np


class LinearSVM:
    """
    Linear Support Vector Classifier using Subgradient Descent on Soft-Margin Hinge Loss.
    """

    def __init__(self, lambda_param: float = 0.01, lr: float = 0.001, n_iters: int = 1000):
        self.lambda_param = lambda_param
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        """
        Fits the SVM model using subgradient descent.
        Supports both binary {0, 1} and {-1, 1} target arrays.
        """
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Transform {0, 1} targets to {-1, +1} if needed
        y_transformed = np.where(y <= 0, -1.0, 1.0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_transformed[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1.0
                if condition:
                    # Point is strictly outside the margin: only apply L2 regularization shrinkage
                    self.weights -= self.lr * (2.0 * self.lambda_param * self.weights)
                else:
                    # Margin violation: subgradient includes both margin penalty and regularization
                    self.weights -= self.lr * (
                        2.0 * self.lambda_param * self.weights - np.dot(x_i, y_transformed[idx])
                    )
                    self.bias += self.lr * y_transformed[idx]

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the signed distance from samples to the separating hyperplane:
        f(x) = X @ w + b
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels in {0, 1}.
        """
        approx = self.decision_function(X)
        return np.where(approx >= 0, 1, 0)
