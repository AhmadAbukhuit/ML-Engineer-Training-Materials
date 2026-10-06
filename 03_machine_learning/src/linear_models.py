"""
Linear Models from First Principles.

This module implements:
1. LinearRegression with Ordinary Least Squares (Normal Equation) and Gradient Descent with L2 penalty.
2. LogisticRegression with Sigmoid hypothesis and Gradient Descent.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression for continuous target prediction.
    Models: y_hat = X @ w + b
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray, l2_reg: float = 0.0) -> "LinearRegression":
        """
        Calculates weights analytically using the closed-form Normal Equation:
        theta = (X_aug^T X_aug + lambda * I)^(-1) X_aug^T y
        """
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            reg_matrix = np.eye(n_features + 1) * l2_reg
            # Do not regularize the intercept term
            reg_matrix[0, 0] = 0.0
        else:
            X_aug = X
            reg_matrix = np.eye(n_features) * l2_reg

        # Solve (X^T X + lambda I) w = X^T y using np.linalg.solve (more stable than inv)
        A = X_aug.T @ X_aug + reg_matrix
        b = X_aug.T @ y
        theta = np.linalg.solve(A, b).flatten()

        if self.fit_intercept:
            self.bias = float(theta[0])
            self.weights = theta[1:]
        else:
            self.bias = 0.0
            self.weights = theta

        return self

    def fit_gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
        l2_reg: float = 0.0
    ) -> "LinearRegression":
        """
        Fits weights iteratively using batch gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y = y.flatten()

        for _ in range(epochs):
            # Forward prediction
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Analytical gradients
            dw = (1.0 / n_samples) * (X.T @ error) + (l2_reg * self.weights)
            db = (1.0 / n_samples) * np.sum(error)

            # Parameter updates
            self.weights -= lr * dw
            if self.fit_intercept:
                self.bias -= lr * db

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        return self.fit_normal_equation(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return X @ self.weights + self.bias


class LogisticRegression:
    """
    Logistic Regression for binary classification.
    Hypothesis: P(y=1|x) = sigmoid(X @ w + b)
    """

    def __init__(self, lr: float = 0.05, epochs: int = 1000, l2_reg: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Clip z to prevent numerical overflow in exp(-z)
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y = y.flatten()

        for _ in range(self.epochs):
            # Compute probabilities
            z = X @ self.weights + self.bias
            probs = self._sigmoid(z)
            error = probs - y

            # Analytical gradient of Binary Cross-Entropy
            dw = (1.0 / n_samples) * (X.T @ error) + (self.l2_reg * self.weights)
            db = (1.0 / n_samples) * np.sum(error)

            # Gradient descent step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        z = X @ self.weights + self.bias
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
