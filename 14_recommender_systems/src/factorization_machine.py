"""Factorization Machines (FM) for Click-Through Rate (CTR) Prediction."""

import numpy as np


class FactorizationMachine:
    """Factorization Machine (FM) Model for 2-Way Feature Interactions (Rendle, 2010).

    Computes pairwise second-order feature interactions in O(k * d) linear time
    using factorized latent interaction vectors.

    Parameters
    ----------
    n_factors : int
        Dimension of latent interaction vectors v_i.
    learning_rate : float
        SGD learning rate.
    reg_w : float
        L2 regularization on linear weights.
    reg_v : float
        L2 regularization on interaction vectors.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_factors: int = 8,
        learning_rate: float = 0.01,
        reg_w: float = 0.01,
        reg_v: float = 0.01,
        random_state: int | None = 42,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.random_state = random_state

        self.w0: float = 0.0
        self.w: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def _init_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = rng.normal(0, 0.1, size=(n_features, self.n_factors))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None and self.V is not None
        # Linear term: w0 + X @ w
        linear = self.w0 + np.dot(X, self.w)

        # Interaction term in O(k * d) time:
        # 0.5 * sum_f [ (sum_i v_if * x_i)^2 - sum_i v_if^2 * x_i^2 ]
        X_V = np.dot(X, self.V)  # (N, k)
        sum_sq = X_V**2

        X_sq = X**2
        V_sq = self.V**2
        sq_sum = np.dot(X_sq, V_sq)  # (N, k)

        interactions = 0.5 * np.sum(sum_sq - sq_sum, axis=1)
        return linear + interactions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict CTR click probability via sigmoid."""
        raw = self._predict_raw(X)
        clipped = np.clip(raw, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary interaction label (0 or 1)."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> "FactorizationMachine":
        """Train FM model using gradient descent on binary log loss."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.w is None or self.V is None:
            self._init_weights(n_features)

        assert self.w is not None and self.V is not None

        for _ in range(epochs):
            p = self.predict_proba(X)
            # Log loss derivative: (p - y) / N
            grad_loss = (p - y) / n_samples

            # Gradient wrt w0
            grad_w0 = float(np.sum(grad_loss))
            self.w0 -= self.learning_rate * grad_w0

            # Gradient wrt w: X.T @ grad_loss + reg * w
            grad_w = np.dot(X.T, grad_loss) + self.reg_w * self.w
            self.w -= self.learning_rate * grad_w

            # Gradient wrt V:
            # d/d V_if = grad_loss * [ x_i * (sum_j v_jf x_j) - v_if * x_i^2 ]
            X_V = np.dot(X, self.V)  # (N, k)
            # (N, d, 1) * (N, 1, k) -> (N, d, k)
            term1 = np.dot(X.T, grad_loss[:, np.newaxis] * X_V)
            term2 = np.dot((X**2).T, grad_loss)[:, np.newaxis] * self.V

            grad_V = (term1 - term2) + self.reg_v * self.V
            self.V -= self.learning_rate * np.clip(grad_V, -1.0, 1.0)

        return self
