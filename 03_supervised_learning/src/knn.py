"""K-Nearest Neighbors (k-NN) implementation from scratch.

Covers:
- Distance metrics (Euclidean, Manhattan, Minkowski)
- KNNClassifier (majority voting, distance-weighted voting)
- KNNRegressor (mean prediction, inverse distance weighted prediction)
"""

from typing import Literal
import numpy as np


class KNNBase:
    """Base class for K-Nearest Neighbors."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
        p: float = 2.0,
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between query points X and stored X_train.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)

        Returns
        -------
        np.ndarray of shape (n_queries, n_train)
        """
        assert self.X_train is not None, "Model must be fitted before computing distances."
        X = np.asarray(X, dtype=np.float64)

        if self.metric == "euclidean":
            # Vectorized euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
            # For numerical stability, clip at 0 before sqrt
            dist_sq = (
                np.sum(X**2, axis=1, keepdims=True)
                + np.sum(self.X_train**2, axis=1, keepdims=True).T
                - 2 * np.dot(X, self.X_train.T)
            )
            return np.sqrt(np.maximum(dist_sq, 0.0))

        if self.metric == "manhattan":
            # |X_i - X_j| summed over features
            return np.sum(np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)

        if self.metric == "minkowski":
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.power(np.sum(np.power(diff, self.p), axis=2), 1.0 / self.p)

        raise ValueError(f"Unknown metric '{self.metric}'.")

    def _get_neighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors and their distances.

        Returns
        -------
        indices : np.ndarray of shape (n_queries, k)
        distances : np.ndarray of shape (n_queries, k)
        """
        distances = self._compute_distances(X)
        # argpartition is O(N) vs O(N log N) for full sort
        n_train = distances.shape[1]
        k = min(self.n_neighbors, n_train)

        partitioned_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        # Sort only the k nearest
        row_indices = np.arange(distances.shape[0])[:, np.newaxis]
        top_k_dists = distances[row_indices, partitioned_idx]
        sorted_order = np.argsort(top_k_dists, axis=1)

        final_indices = partitioned_idx[row_indices, sorted_order]
        final_distances = top_k_dists[row_indices, sorted_order]

        return final_indices, final_distances


class KNNClassifier(KNNBase):
    """K-Nearest Neighbors Classifier."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training data."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for query samples."""
        indices, distances = self._get_neighbors(X)
        assert self.y_train is not None

        neighbor_labels = self.y_train[indices]
        n_queries = X.shape[0]
        predictions = np.empty(n_queries, dtype=self.y_train.dtype)

        eps = 1e-10
        for i in range(n_queries):
            labels = neighbor_labels[i]
            if self.weights == "uniform":
                unique_vals, counts = np.unique(labels, return_counts=True)
                predictions[i] = unique_vals[np.argmax(counts)]
            elif self.weights == "distance":
                weights = 1.0 / (distances[i] + eps)
                unique_vals = np.unique(labels)
                scores = [np.sum(weights[labels == val]) for val in unique_vals]
                predictions[i] = unique_vals[np.argmax(scores)]

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities."""
        indices, distances = self._get_neighbors(X)
        assert self.y_train is not None

        neighbor_labels = self.y_train[indices]
        n_queries = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_queries, n_classes), dtype=np.float64)

        eps = 1e-10
        for i in range(n_queries):
            labels = neighbor_labels[i]
            if self.weights == "uniform":
                for c_idx, cls_val in enumerate(self.classes_):
                    probas[i, c_idx] = np.mean(labels == cls_val)
            elif self.weights == "distance":
                weights = 1.0 / (distances[i] + eps)
                total_w = np.sum(weights)
                for c_idx, cls_val in enumerate(self.classes_):
                    probas[i, c_idx] = np.sum(weights[labels == cls_val]) / total_w

        return probas


class KNNRegressor(KNNBase):
    """K-Nearest Neighbors Regressor."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """Store training data."""
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for query samples."""
        indices, distances = self._get_neighbors(X)
        assert self.y_train is not None

        neighbor_targets = self.y_train[indices]
        if self.weights == "uniform":
            return np.mean(neighbor_targets, axis=1)

        eps = 1e-10
        weights = 1.0 / (distances + eps)
        weights /= np.sum(weights, axis=1, keepdims=True)
        return np.sum(weights * neighbor_targets, axis=1)
