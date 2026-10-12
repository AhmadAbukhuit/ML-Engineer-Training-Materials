"""Local Interpretable Model-agnostic Explanations (LIME) in NumPy."""

from collections.abc import Callable
import numpy as np


class LimeTabularExplainer:
    """LIME explainer for tabular data from first principles.

    Generates a local surrogate linear model around instance x by:
    1. Perturbing x with Gaussian sampling.
    2. Weighting perturbed samples by exponential kernel distance pi_x(z).
    3. Fitting a locally weighted Ridge regression model.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Black-box model prediction function.
    kernel_width : float
        Exponential kernel bandwidth parameter sigma.
    n_samples : int
        Number of perturbed neighborhood samples to generate.
    alpha : float
        Ridge regularization penalty for local surrogate model.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        kernel_width: float = 1.0,
        n_samples: int = 500,
        alpha: float = 0.01,
        random_state: int | None = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        self.alpha = alpha
        self.rng = np.random.default_rng(random_state)

    def explain_instance(
        self, x: np.ndarray, feature_stds: np.ndarray | None = None
    ) -> tuple[np.ndarray, float]:
        """Explain instance x using a local linear surrogate.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
        feature_stds : np.ndarray or None
            Standard deviation per feature for perturbation scaling.

        Returns
        -------
        coefficients : np.ndarray of shape (n_features,)
            Local linear weights representing feature contributions.
        intercept : float
            Local surrogate model intercept.
        """
        x = np.asarray(x, dtype=np.float64)
        n_features = len(x)

        if feature_stds is None:
            scale = np.ones(n_features)
        else:
            scale = np.asarray(feature_stds, dtype=np.float64)
            scale = np.where(scale == 0, 1.0, scale)

        # 1. Generate neighborhood perturbations
        perturbations = self.rng.normal(0.0, 1.0, size=(self.n_samples, n_features)) * scale
        # Include original point x at index 0
        perturbations[0] = 0.0
        Z = x + perturbations

        # 2. Query black-box predictions
        y_perturbed = np.asarray(self.predict_fn(Z), dtype=np.float64).flatten()

        # 3. Compute exponential kernel proximity weights
        # Distance ||x - z||_2
        diff = Z - x
        distances_sq = np.sum(diff**2, axis=1)
        sigma_sq = (self.kernel_width**2) * float(n_features)
        weights = np.exp(-distances_sq / np.maximum(sigma_sq, 1e-12))
        W = np.diag(weights)

        # 4. Fit locally weighted Ridge regression: (Z_bias^T W Z_bias + alpha I)^(-1) Z_bias^T W y
        Z_bias = np.hstack([np.ones((self.n_samples, 1)), diff])  # relative coordinates
        d_dim = Z_bias.shape[1]

        reg_I = np.eye(d_dim) * self.alpha
        reg_I[0, 0] = 0.0  # do not penalize intercept

        A = Z_bias.T @ W @ Z_bias + reg_I
        b = Z_bias.T @ W @ y_perturbed

        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b

        intercept = float(theta[0])
        coefficients = theta[1:]

        return coefficients, intercept
