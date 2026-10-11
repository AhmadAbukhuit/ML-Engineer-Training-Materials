"""Domain Adaptation and Maximum Mean Discrepancy (MMD) from First Principles in NumPy."""

import numpy as np


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute Gaussian RBF kernel matrix between X and Y."""
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    return np.exp(-gamma * dist_sq)


class MaximumMeanDiscrepancy:
    """Maximum Mean Discrepancy (MMD) metric for Domain Adaptation.

    Measures statistical distance between distributions P(X_source) and Q(X_target)
    in a Reproducing Kernel Hilbert Space (RKHS).

    MMD^2(P, Q) = E[k(x, x')] - 2 E[k(x, y)] + E[k(y, y')]

    Parameters
    ----------
    gammas : list of float or None
        Bandwidth parameter values for multi-kernel Gaussian RBF.
    """

    def __init__(self, gammas: list[float] | None = None) -> None:
        self.gammas = gammas or [0.1, 1.0, 10.0]

    def compute_mmd(self, X_source: np.ndarray, X_target: np.ndarray) -> float:
        """Compute unbiased MMD^2 distance between source and target domain features.

        Parameters
        ----------
        X_source : np.ndarray of shape (N_s, D)
        X_target : np.ndarray of shape (N_t, D)

        Returns
        -------
        mmd_sq : float
        """
        X_s = np.asarray(X_source, dtype=np.float64)
        X_t = np.asarray(X_target, dtype=np.float64)

        n_s = X_s.shape[0]
        n_t = X_t.shape[0]

        if n_s < 2 or n_t < 2:
            raise ValueError("Both source and target samples must be at least 2.")

        total_mmd = 0.0

        for gamma in self.gammas:
            K_ss = rbf_kernel(X_s, X_s, gamma=gamma)
            K_tt = rbf_kernel(X_t, X_t, gamma=gamma)
            K_st = rbf_kernel(X_s, X_t, gamma=gamma)

            # Unbiased estimator: exclude self-similarity on diagonals for within-domain
            np.fill_diagonal(K_ss, 0.0)
            np.fill_diagonal(K_tt, 0.0)

            term_ss = np.sum(K_ss) / (n_s * (n_s - 1))
            term_tt = np.sum(K_tt) / (n_t * (n_t - 1))
            term_st = np.sum(K_st) / (n_s * n_t)

            total_mmd += float(term_ss + term_tt - 2.0 * term_st)

        return float(np.maximum(0.0, total_mmd / len(self.gammas)))
