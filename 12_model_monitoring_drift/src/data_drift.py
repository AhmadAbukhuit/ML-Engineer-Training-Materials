"""Data Drift and Covariate Shift Metrics from First Principles in NumPy."""

import numpy as np


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10, eps: float = 1e-6
) -> float:
    """Compute Population Stability Index (PSI) between baseline and production data.

    Formula:
        PSI = sum_{k=1}^K (Actual_k - Expected_k) * ln(Actual_k / Expected_k)

    Thresholds:
        PSI < 0.1: No significant distribution change.
        0.1 <= PSI < 0.2: Moderate shift; investigate.
        PSI >= 0.2: Significant drift; trigger model retraining.

    Parameters
    ----------
    expected : np.ndarray
        Baseline / training reference sample.
    actual : np.ndarray
        Current production / inference sample.
    num_bins : int
        Number of quantile-based buckets.
    eps : float
        Small constant to prevent log(0) and division by zero.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    # Define bin edges from quantiles of expected distribution
    quantiles = np.linspace(0.0, 100.0, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    # Ensure distinct bin edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute frequency counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Normalize into probability fractions
    p_expected = expected_counts / len(expected)
    p_actual = actual_counts / len(actual)

    # Regularize zero bins
    p_expected = np.where(p_expected == 0, eps, p_expected)
    p_actual = np.where(p_actual == 0, eps, p_actual)

    # Re-normalize
    p_expected = p_expected / np.sum(p_expected)
    p_actual = p_actual / np.sum(p_actual)

    # PSI calculation
    psi_value = np.sum((p_actual - p_expected) * np.log(p_actual / p_expected))
    return float(psi_value)


def ks_two_sample_test(data1: np.ndarray, data2: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test statistic.

    Computes the supremum distance between empirical cumulative distribution functions:
        D = sup_x |F_1(x) - F_2(x)|

    Returns
    -------
    d_stat : float
        Maximum ECDF vertical discrepancy in [0, 1].
    p_approx : float
        Asymptotic p-value approximation.
    """
    data1 = np.sort(np.asarray(data1, dtype=np.float64))
    data2 = np.sort(np.asarray(data2, dtype=np.float64))

    n1 = len(data1)
    n2 = len(data2)

    # Unified concatenated sorted values
    all_vals = np.concatenate([data1, data2])
    # Compute ECDFs at all observed points
    cdf1 = np.searchsorted(data1, all_vals, side="right") / n1
    cdf2 = np.searchsorted(data2, all_vals, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Kolmogorov distribution asymptotic p-value
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    # Approximation of Kolmogorov survival function 2 * sum_{k=1}^inf (-1)^(k-1) exp(-2 k^2 lambda^2)
    p_approx = 0.0
    for k in range(1, 100):
        term = 2.0 * ((-1.0) ** (k - 1)) * np.exp(-2.0 * (k**2) * (lambda_val**2))
        p_approx += term
        if np.abs(term) < 1e-10:
            break
    p_approx = float(np.clip(p_approx, 0.0, 1.0))

    return d_stat, p_approx


def wasserstein_distance_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """Compute 1D first Wasserstein Distance (Earth Mover's Distance).

    Formula:
        W_1(u, v) = integral |U(x) - V(x)| dx
    """
    u_values = np.sort(np.asarray(u_values, dtype=np.float64))
    v_values = np.sort(np.asarray(v_values, dtype=np.float64))

    n_u = len(u_values)
    n_v = len(v_values)

    all_vals = np.unique(np.concatenate([u_values, v_values]))
    cdf_u = np.searchsorted(u_values, all_vals, side="right") / n_u
    cdf_v = np.searchsorted(v_values, all_vals, side="right") / n_v

    # Trapezoidal integration along sorted steps
    deltas = np.diff(all_vals)
    abs_diff = np.abs(cdf_u[:-1] - cdf_v[:-1])
    return float(np.sum(deltas * abs_diff))


class DataDriftDetector:
    """Multi-feature Data Drift auditing engine for production batches."""

    def __init__(self, reference_data: np.ndarray, feature_names: list[str] | None = None) -> None:
        self.reference_data = np.asarray(reference_data, dtype=np.float64)
        self.n_features = self.reference_data.shape[1]
        self.feature_names = feature_names or [f"feat_{i}" for i in range(self.n_features)]

    def audit_batch(self, current_batch: np.ndarray, psi_threshold: float = 0.2) -> dict[str, dict]:
        """Audit production batch against reference distribution across all features."""
        current_batch = np.asarray(current_batch, dtype=np.float64)
        report: dict[str, dict] = {}

        for i, name in enumerate(self.feature_names):
            ref_col = self.reference_data[:, i]
            curr_col = current_batch[:, i]

            psi = population_stability_index(ref_col, curr_col)
            ks_stat, p_val = ks_two_sample_test(ref_col, curr_col)
            w_dist = wasserstein_distance_1d(ref_col, curr_col)

            report[name] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": p_val,
                "wasserstein_distance": w_dist,
                "drift_detected": psi >= psi_threshold,
            }

        return report
