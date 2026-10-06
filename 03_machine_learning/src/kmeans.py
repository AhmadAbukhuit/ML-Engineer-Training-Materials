"""
K-Means Clustering and Silhouette Score from First Principles.

This module implements:
1. KMeans clustering with K-Means++ initialization and Expectation-Maximization updates.
2. Vectorized calculation of cluster inertia (Within-Cluster Sum of Squares).
3. silhouette_score calculation from scratch.
"""

import numpy as np


class KMeans:
    """
    K-Means clustering algorithm.
    Minimizes: Inertia = sum_k sum_{x in C_k} ||x - mu_k||^2
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = 42
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = 0.0

    def _init_centroids_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++ initialization algorithm:
        Probabilistically spreads initial centroids to reduce convergence time and avoid poor local minima.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        # 1. Choose the first center uniformly at random
        centroids = [X[rng.choice(n_samples)]]

        # 2. Choose the remaining k-1 centers
        for _ in range(1, self.n_clusters):
            # Compute squared Euclidean distance to closest centroid for all points
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            # Probability proportional to D(x)^2
            probs = dists / np.sum(dists)
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def fit(self, X: np.ndarray) -> "KMeans":
        n_samples, _ = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples cannot be less than number of clusters.")

        # Initialize centroids
        self.cluster_centers_ = self._init_centroids_kmeans_plus_plus(X)

        for _ in range(self.max_iter):
            # 1. Assignment Step (E-step): Distance from each point to each centroid
            # Shape: (n_samples, n_clusters)
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)

            # 2. Update Step (M-step): Recompute centroids
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                cluster_points = X[new_labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points, axis=0)
                else:
                    # If empty cluster, reinitialize randomly
                    rng = np.random.default_rng(self.random_state)
                    new_centers[k] = X[rng.choice(n_samples)]

            # Check convergence tolerance
            center_shift = np.sum(np.linalg.norm(new_centers - self.cluster_centers_, axis=1))
            self.cluster_centers_ = new_centers
            self.labels_ = new_labels

            if center_shift < self.tol:
                break

        # Calculate final inertia
        final_dists = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        min_dists = np.min(final_dists, axis=1)
        self.inertia_ = float(np.sum(min_dists ** 2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the Mean Silhouette Coefficient over all samples.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Where a(i) is mean intra-cluster distance, and b(i) is mean nearest-cluster distance.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        raise ValueError("Number of clusters must be between 2 and n_samples - 1.")

    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        point = X[i]
        label = labels[i]

        # a(i): mean distance to points in same cluster
        same_cluster_points = X[(labels == label) & (np.arange(n_samples) != i)]
        if len(same_cluster_points) == 0:
            a_i = 0.0
        else:
            a_i = np.mean(np.linalg.norm(same_cluster_points - point, axis=1))

        # b(i): mean distance to points in the closest other cluster
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster_points = X[labels == other_label]
            if len(other_cluster_points) > 0:
                dist = np.mean(np.linalg.norm(other_cluster_points - point, axis=1))
                b_i = min(b_i, dist)

        denom = max(a_i, b_i)
        silhouette_vals[i] = 0.0 if denom == 0 else (b_i - a_i) / denom

    return float(np.mean(silhouette_vals))
