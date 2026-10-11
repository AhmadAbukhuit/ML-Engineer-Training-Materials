"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN) from scratch."""

import numpy as np


class DBSCAN:
    """DBSCAN clustering algorithm from first principles.

    Identifies core points, expands clusters via density-reachability,
    and assigns -1 to noise/outlier points.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Compute clusters from feature matrix X."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # Compute pairwise Euclidean distance matrix
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # Find epsilon neighbors for all points
        neighbors_list = [np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)]

        # Identify core points
        is_core = np.array([len(nbrs) >= self.min_samples for nbrs in neighbors_list], dtype=bool)
        self.core_sample_indices_ = np.where(is_core)[0]

        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i] or not is_core[i]:
                continue

            # Start a new cluster
            labels[i] = cluster_id
            visited[i] = True

            # Queue for density reachability expansion
            queue = list(neighbors_list[i])

            idx = 0
            while idx < len(queue):
                neighbor = queue[idx]
                idx += 1

                if not visited[neighbor]:
                    visited[neighbor] = True
                    labels[neighbor] = cluster_id

                    # If this neighbor is also a core point, expand search
                    if is_core[neighbor]:
                        for n in neighbors_list[neighbor]:
                            if n not in queue:
                                queue.append(n)
                elif labels[neighbor] == -1:
                    # Point was previously marked as noise, now border point
                    labels[neighbor] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_
