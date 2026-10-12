"""Alternating Least Squares (ALS) for Implicit Feedback Collaborative Filtering."""

import numpy as np


class ImplicitALS:
    """Alternating Least Squares for Implicit Feedback (Hu, Koren, Volinsky).

    Parameters
    ----------
    n_factors : int
        Dimension of latent factor vectors.
    alpha : float
        Confidence scaling hyperparameter (c_ui = 1 + alpha * r_ui).
    regularization : float
        L2 regularization parameter lambda.
    n_iterations : int
        Number of alternating optimization epochs.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_factors: int = 16,
        alpha: float = 40.0,
        regularization: float = 0.01,
        n_iterations: int = 15,
        random_state: int | None = 42,
    ) -> None:
        self.n_factors = n_factors
        self.alpha = alpha
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, R: np.ndarray) -> "ImplicitALS":
        """Fit ALS factors on implicit feedback interaction matrix R.

        Parameters
        ----------
        R : np.ndarray of shape (n_users, n_items)
            Raw interaction counts (0 indicates no interaction).
        """
        R = np.asarray(R, dtype=np.float64)
        n_users, n_items = R.shape

        rng = np.random.default_rng(self.random_state)
        # Initialize factors with small positive values
        self.user_factors = rng.normal(0.0, 0.01, size=(n_users, self.n_factors))
        self.item_factors = rng.normal(0.0, 0.01, size=(n_items, self.n_factors))

        # Binary preference matrix P
        P = (R > 0).astype(np.float64)
        # Confidence matrix C
        C = 1.0 + self.alpha * R

        f = self.n_factors
        reg_I = self.regularization * np.eye(f)

        for _ in range(self.n_iterations):
            # 1. Update User Factors X given Item Factors Y
            # Precompute Y^T Y
            YTY = self.item_factors.T @ self.item_factors

            for u in range(n_users):
                cu = C[u]  # shape (n_items,)
                pu = P[u]  # shape (n_items,)

                # Only indices where user had interaction
                interacted = np.where(cu > 1.0)[0]
                if len(interacted) == 0:
                    continue

                # A_u = Y^T Y + Y_active^T (C_active - I) Y_active + lambda I
                Y_active = self.item_factors[interacted]  # shape (k, f)
                C_diff = cu[interacted] - 1.0  # shape (k,)

                A_u = YTY + (Y_active.T * C_diff) @ Y_active + reg_I
                # b_u = Y^T (C_u * p_u)
                b_u = self.item_factors.T @ (cu * pu)

                self.user_factors[u] = np.linalg.solve(A_u, b_u)

            # 2. Update Item Factors Y given User Factors X
            XTX = self.user_factors.T @ self.user_factors

            for i in range(n_items):
                ci = C[:, i]  # shape (n_users,)
                pi = P[:, i]  # shape (n_users,)

                interacted = np.where(ci > 1.0)[0]
                if len(interacted) == 0:
                    continue

                X_active = self.user_factors[interacted]
                C_diff = ci[interacted] - 1.0

                A_i = XTX + (X_active.T * C_diff) @ X_active + reg_I
                b_i = self.user_factors.T @ (ci * pi)

                self.item_factors[i] = np.linalg.solve(A_i, b_i)

        return self

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Predict user affinity score for a given item."""
        assert self.user_factors is not None and self.item_factors is not None
        return float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

    def recommend(self, user_idx: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Generate top-K recommended item indices with predicted affinity scores."""
        assert self.user_factors is not None and self.item_factors is not None
        scores = np.dot(self.item_factors, self.user_factors[user_idx])
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
