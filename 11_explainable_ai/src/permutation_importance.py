"""Permutation Feature Importance from First Principles in NumPy."""

from collections.abc import Callable
import numpy as np


def permutation_feature_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 5,
    random_state: int | None = 42,
    greater_is_better: bool = True,
) -> dict[str, np.ndarray]:
    """Compute permutation feature importance on validation data.

    Breiman's permutation importance randomly shuffles the values of a single
    feature column, breaking its relationship with the target, and measures
    the resulting degradation in model evaluation score.

    Parameters
    ----------
    model : object
        Fitted model providing a `predict` method.
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    scoring_fn : Callable[[y_true, y_pred], float]
        Evaluation metric function (e.g. accuracy_score, r2_score).
    n_repeats : int
        Number of times each feature column is permuted.
    random_state : int or None
        Seed for pseudo-random number generator.
    greater_is_better : bool
        If True, importance is baseline - permuted score. If False, permuted - baseline.

    Returns
    -------
    dict with keys:
        'importances_mean' : np.ndarray of shape (n_features,)
        'importances_std' : np.ndarray of shape (n_features,)
        'importances' : np.ndarray of shape (n_features, n_repeats)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)
    predict_method = model.predict

    # Baseline evaluation score
    baseline_pred = predict_method(X)
    baseline_score = scoring_fn(y, baseline_pred)

    importances = np.zeros((n_features, n_repeats), dtype=np.float64)

    for col in range(n_features):
        for rep in range(n_repeats):
            X_permuted = X.copy()
            # Randomly permute column
            perm_indices = rng.permutation(n_samples)
            X_permuted[:, col] = X[perm_indices, col]

            perm_pred = predict_method(X_permuted)
            perm_score = scoring_fn(y, perm_pred)

            if greater_is_better:
                importances[col, rep] = baseline_score - perm_score
            else:
                importances[col, rep] = perm_score - baseline_score

    return {
        "importances_mean": np.mean(importances, axis=1),
        "importances_std": np.std(importances, axis=1),
        "importances": importances,
    }
