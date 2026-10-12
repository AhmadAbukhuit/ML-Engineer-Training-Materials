"""Differentially Private Stochastic Gradient Descent (DP-SGD)."""

import numpy as np


class DPSGDStep:
    """Differentially Private SGD Gradient Aggregation (Abadi et al., 2016).

    Protects individual training examples from reconstruction and membership inference:
    1. Per-sample gradient clipping with L2 norm bound C:
       g_clipped_i = g_i / max(1, ||g_i||_2 / C)
    2. Gaussian noise addition proportional to sensitivity C and noise multiplier sigma:
       g_private = (1 / B) * ( sum_i g_clipped_i + N(0, (sigma * C)^2 * I) )

    Parameters
    ----------
    clipping_norm : float
        Maximum allowed L2 norm C for each individual sample gradient.
    noise_multiplier : float
        Ratio of Gaussian noise standard deviation to clipping norm (sigma).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        random_state: int | None = 42,
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive.")
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier cannot be negative.")

        self.clipping_norm = float(clipping_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.rng = np.random.default_rng(random_state)

    def privatize_batch_gradients(
        self, per_sample_gradients: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Clip per-sample gradients and inject Gaussian mechanism noise.

        Parameters
        ----------
        per_sample_gradients : np.ndarray of shape (B, D)
            Raw gradient computed for each individual sample in the mini-batch.

        Returns
        -------
        privatized_gradient : np.ndarray of shape (D,)
            Averaged, differentially private gradient vector.
        clipping_ratio : float
            Fraction of samples whose gradients were scaled down by clipping.
        """
        grads = np.asarray(per_sample_gradients, dtype=np.float64)
        B, D = grads.shape

        # Compute per-sample L2 norms
        norms = np.linalg.norm(grads, axis=1)  # shape (B,)
        # Clipping factors: min(1, C / norm)
        clip_factors = np.minimum(1.0, self.clipping_norm / np.maximum(norms, 1e-12))
        clipped_grads = grads * clip_factors[:, np.newaxis]

        clipping_ratio = float(np.mean(norms > self.clipping_norm))

        # Sum of clipped gradients
        grad_sum = np.sum(clipped_grads, axis=0)

        # Calibrated Gaussian noise: std = sigma * C
        noise_std = self.noise_multiplier * self.clipping_norm
        noise = self.rng.normal(0.0, noise_std, size=D)

        privatized_grad = (grad_sum + noise) / B
        return privatized_grad, clipping_ratio
