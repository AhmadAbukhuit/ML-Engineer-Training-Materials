"""Shapley Value Explainers from First Principles in NumPy.

Implements:
1. ExactShapleyExplainer: Exhaustive coalition powerset evaluation for exact Shapley values.
   Satisfies Efficiency, Symmetry, Dummy Player, and Additivity axioms.
2. SamplingShapleyExplainer: Monte Carlo permutation sampling for scalable Shapley approximation.
"""

from itertools import combinations
import math
from collections.abc import Callable
import numpy as np


class ExactShapleyExplainer:
    """Exact Shapley Value Explainer via Coalition Enumeration.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], np.ndarray]
        Model prediction function returning 1D array of scalar outputs.
    baseline : np.ndarray of shape (n_features,)
        Reference/background sample used to mask absent features.
    """

    def __init__(
        self, predict_fn: Callable[[np.ndarray], np.ndarray], baseline: np.ndarray
    ) -> None:
        self.predict_fn = predict_fn
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.n_features = len(self.baseline)

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        """Compute exact Shapley values phi_i for instance x.

        Returns
        -------
        shap_values : np.ndarray of shape (n_features,)
        """
        x = np.asarray(x, dtype=np.float64)
        n = self.n_features
        shap_values = np.zeros(n, dtype=np.float64)
        feature_indices = list(range(n))

        for i in range(n):
            other_features = [f for f in feature_indices if f != i]
            phi_i = 0.0

            # Iterate over all subset sizes of other features
            for s_len in range(n):
                weight = (
                    math.factorial(s_len)
                    * math.factorial(n - s_len - 1)
                    / math.factorial(n)
                )

                for subset in combinations(other_features, s_len):
                    # Mask S without feature i
                    sample_without_i = self.baseline.copy()
                    for f in subset:
                        sample_without_i[f] = x[f]

                    # Mask S with feature i
                    sample_with_i = sample_without_i.copy()
                    sample_with_i[i] = x[i]

                    v_without = float(self.predict_fn(sample_without_i.reshape(1, -1))[0])
                    v_with = float(self.predict_fn(sample_with_i.reshape(1, -1))[0])

                    marginal_contrib = v_with - v_without
                    phi_i += weight * marginal_contrib

            shap_values[i] = phi_i

        return shap_values


class SamplingShapleyExplainer:
    """Monte Carlo Permutation-Based Shapley Value Explainer.

    Approximates Shapley values by drawing random feature orderings.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        baseline: np.ndarray,
        n_samples: int = 200,
        random_state: int | None = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.n_features = len(self.baseline)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        """Approximate Shapley values via random permutation sampling."""
        x = np.asarray(x, dtype=np.float64)
        n = self.n_features
        shap_values = np.zeros(n, dtype=np.float64)
        feature_indices = np.arange(n)

        for _ in range(self.n_samples):
            # Draw random permutation of features
            perm = self.rng.permutation(feature_indices)

            sample_prev = self.baseline.copy()
            pred_prev = float(self.predict_fn(sample_prev.reshape(1, -1))[0])

            for f in perm:
                sample_curr = sample_prev.copy()
                sample_curr[f] = x[f]
                pred_curr = float(self.predict_fn(sample_curr.reshape(1, -1))[0])

                marginal_contrib = pred_curr - pred_prev
                shap_values[f] += marginal_contrib

                sample_prev = sample_curr
                pred_prev = pred_curr

        return shap_values / self.n_samples
