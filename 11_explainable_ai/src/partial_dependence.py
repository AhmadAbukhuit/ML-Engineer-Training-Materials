"""Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) in NumPy."""

from collections.abc import Callable
import numpy as np


def partial_dependence(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_idx: int,
    grid_resolution: int = 50,
    grid_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Partial Dependence (PDP) for a single feature.

    Formula:
        f_S(x_S) = (1/N) * sum_{i=1}^N f(x_S, x_{i, C})

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Model prediction function.
    X : np.ndarray of shape (n_samples, n_features)
    feature_idx : int
        Index of the feature to evaluate.
    grid_resolution : int
        Number of evaluation points along the feature grid.
    grid_range : tuple of (float, float) or None
        Optional (min, max) range for the grid.

    Returns
    -------
    grid_values : np.ndarray of shape (grid_resolution,)
    pdp_values : np.ndarray of shape (grid_resolution,)
    """
    X = np.asarray(X, dtype=np.float64)

    feat_vals = X[:, feature_idx]
    if grid_range is None:
        grid_min, grid_max = float(np.min(feat_vals)), float(np.max(feat_vals))
    else:
        grid_min, grid_max = grid_range

    grid_values = np.linspace(grid_min, grid_max, grid_resolution)
    pdp_values = np.zeros(grid_resolution, dtype=np.float64)

    for idx, val in enumerate(grid_values):
        X_eval = X.copy()
        X_eval[:, feature_idx] = val
        preds = np.asarray(predict_fn(X_eval), dtype=np.float64).flatten()
        pdp_values[idx] = np.mean(preds)

    return grid_values, pdp_values


def individual_conditional_expectation(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_idx: int,
    grid_resolution: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Individual Conditional Expectation (ICE) curves for each sample.

    Returns
    -------
    grid_values : np.ndarray of shape (grid_resolution,)
    ice_curves : np.ndarray of shape (n_samples, grid_resolution)
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    grid_min, grid_max = float(np.min(X[:, feature_idx])), float(np.max(X[:, feature_idx]))
    grid_values = np.linspace(grid_min, grid_max, grid_resolution)
    ice_curves = np.zeros((n_samples, grid_resolution), dtype=np.float64)

    for idx, val in enumerate(grid_values):
        X_eval = X.copy()
        X_eval[:, feature_idx] = val
        preds = np.asarray(predict_fn(X_eval), dtype=np.float64).flatten()
        ice_curves[:, idx] = preds

    return grid_values, ice_curves
