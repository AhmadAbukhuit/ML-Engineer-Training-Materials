"""Consistency Regularization Utilities for Semi-Supervised Learning."""

import numpy as np


def sharpen(probabilities: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """Sharpen predicted probability distributions (used in MixMatch / FixMatch).

    Formula:
        q_i = p_i^(1/T) / sum_j p_j^(1/T)

    Parameters
    ----------
    probabilities : np.ndarray of shape (N, C)
        Predicted class probability distribution.
    temperature : float
        Temperature parameter T in (0, 1]. Lower values approach one-hot argmax.
    """
    if temperature <= 0.0 or temperature > 1.0:
        raise ValueError("temperature must be in (0, 1].")

    probs = np.asarray(probabilities, dtype=np.float64)
    # Exponentiate with inverse temperature
    powered = np.power(probs, 1.0 / temperature)
    row_sums = np.sum(powered, axis=1, keepdims=True)
    return powered / np.maximum(row_sums, 1e-12)


def rampup_weight(current_step: int, rampup_length: int = 40, max_weight: float = 1.0) -> float:
    """Gaussian ramp-up weight schedule for unsupervised consistency loss.

    Formula:
        w(t) = max_weight * exp(-5 * (1 - t / rampup_length)^2) for t < rampup_length
    """
    if current_step >= rampup_length:
        return float(max_weight)
    if rampup_length == 0:
        return float(max_weight)

    phase = 1.0 - (current_step / rampup_length)
    return float(max_weight * np.exp(-5.0 * phase * phase))


def consistency_loss(p1: np.ndarray, p2: np.ndarray, metric: str = "mse") -> float:
    """Compute consistency regularization loss between predictions of two perturbed views."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    if metric == "mse":
        return float(np.mean((p1 - p2) ** 2))
    if metric == "kl":
        # KL divergence: sum p1 * log(p1 / p2)
        eps = 1e-12
        kl = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)), axis=1)
        return float(np.mean(kl))

    raise ValueError(f"Unknown metric '{metric}'.")
