"""Contrastive Loss (NT-Xent / InfoNCE) from First Principles in NumPy.

Implements:
1. Vectorized pair-wise cosine similarity with temperature scaling tau.
2. Self-similarity masking on the diagonal.
3. NT-Xent / InfoNCE cross-entropy loss calculation with numerical stability.
4. Representation alignment and uniformity metrics.
"""

import numpy as np


class NTXentLoss:
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent / InfoNCE).

    Core loss function powering SimCLR, MoCo, and modern self-supervised learning.

    Parameters
    ----------
    temperature : float
        Temperature parameter tau (> 0) controlling distribution sharpness.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be strictly positive.")
        self.temperature = temperature

    def __call__(self, z_i: np.ndarray, z_j: np.ndarray) -> float:
        """Compute NT-Xent loss between two augmented views of a batch.

        Parameters
        ----------
        z_i : np.ndarray of shape (batch_size, dim)
            Representations of view 1.
        z_j : np.ndarray of shape (batch_size, dim)
            Representations of view 2.

        Returns
        -------
        loss : float
        """
        z_i = np.asarray(z_i, dtype=np.float64)
        z_j = np.asarray(z_j, dtype=np.float64)

        n_samples = z_i.shape[0]
        if n_samples < 2:
            raise ValueError("Batch size must be at least 2 for contrastive loss.")

        # L2-normalize feature representations
        norm_i = np.linalg.norm(z_i, axis=1, keepdims=True)
        norm_j = np.linalg.norm(z_j, axis=1, keepdims=True)
        eps = 1e-12
        z_i_norm = z_i / np.maximum(norm_i, eps)
        z_j_norm = z_j / np.maximum(norm_j, eps)

        # Full concatenated representation matrix: shape (2N, dim)
        representations = np.vstack([z_i_norm, z_j_norm])

        # Pairwise cosine similarity matrix: shape (2N, 2N)
        sim_matrix = np.dot(representations, representations.T) / self.temperature

        # Numerical stability: subtract row max (excluding diagonal)
        np.fill_diagonal(sim_matrix, -np.inf)
        sim_max = np.max(sim_matrix, axis=1, keepdims=True)
        exp_sim = np.exp(sim_matrix - sim_max)
        np.fill_diagonal(exp_sim, 0.0)

        # Denominator: sum of negative pairs + positive pair
        denom = np.sum(exp_sim, axis=1)

        # Numerator: positive pairs (i with N+i, and N+i with i)
        pos_sim = np.zeros(2 * n_samples)
        for idx in range(n_samples):
            pos_sim[idx] = exp_sim[idx, n_samples + idx]
            pos_sim[n_samples + idx] = exp_sim[n_samples + idx, idx]

        # InfoNCE loss: -log(pos / denom)
        losses = -np.log(pos_sim / (denom + eps) + eps)
        return float(np.mean(losses))


def compute_alignment_and_uniformity(
    z_i: np.ndarray, z_j: np.ndarray, t: float = 2.0
) -> tuple[float, float]:
    """Compute Wang & Isola (2020) Alignment and Uniformity metrics.

    Alignment: expected distance between positive pairs.
    Uniformity: logarithm of average pairwise Gaussian potential.
    """
    z_i = np.asarray(z_i, dtype=np.float64)
    z_j = np.asarray(z_j, dtype=np.float64)

    # Normalize
    eps = 1e-12
    z_i_norm = z_i / np.maximum(np.linalg.norm(z_i, axis=1, keepdims=True), eps)
    z_j_norm = z_j / np.maximum(np.linalg.norm(z_j, axis=1, keepdims=True), eps)

    # Alignment: E[||x - y||^2]
    alignment = float(np.mean(np.sum((z_i_norm - z_j_norm) ** 2, axis=1)))

    # Uniformity: log E[exp(-t ||u - v||^2)]
    all_reps = np.vstack([z_i_norm, z_j_norm])
    diff = all_reps[:, np.newaxis, :] - all_reps[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)

    n = all_reps.shape[0]
    # Mask out diagonal
    np.fill_diagonal(dist_sq, np.inf)
    uniformity = float(np.log(np.sum(np.exp(-t * dist_sq)) / (n * (n - 1)) + eps))

    return alignment, uniformity
