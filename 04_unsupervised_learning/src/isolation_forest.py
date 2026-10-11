"""Isolation Forest for Unsupervised Anomaly Detection from first principles."""

import numpy as np


def _c(n: int) -> float:
    """Average path length of unsuccessful search in a Binary Search Tree (BST)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler_mascheroni = 0.5772156649
    return 2.0 * (np.log(n - 1) + euler_mascheroni) - (2.0 * (n - 1) / n)


class IsolationTreeNode:
    """Node in an Isolation Tree."""

    def __init__(
        self,
        left: "IsolationTreeNode | None" = None,
        right: "IsolationTreeNode | None" = None,
        split_feature: int | None = None,
        split_value: float | None = None,
        size: int = 0,
        is_leaf: bool = False,
    ) -> None:
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value
        self.size = size
        self.is_leaf = is_leaf


class IsolationTree:
    """Single isolation tree constructed by random recursive partitioning."""

    def __init__(self, max_depth: int, rng: np.random.Generator) -> None:
        self.max_depth = max_depth
        self.rng = rng
        self.root: IsolationTreeNode | None = None

    def fit(self, X: np.ndarray) -> "IsolationTree":
        self.root = self._build_tree(X, current_depth=0)
        return self

    def _build_tree(self, X: np.ndarray, current_depth: int) -> IsolationTreeNode:
        n_samples, n_features = X.shape

        if current_depth >= self.max_depth or n_samples <= 1:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        # Randomly select a feature
        feature_idx = self.rng.integers(0, n_features)
        feat_min = X[:, feature_idx].min()
        feat_max = X[:, feature_idx].max()

        if feat_min == feat_max:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        # Uniformly pick split value between min and max
        split_val = self.rng.uniform(feat_min, feat_max)

        left_mask = X[:, feature_idx] < split_val
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], current_depth + 1)
        right_node = self._build_tree(X[right_mask], current_depth + 1)

        return IsolationTreeNode(
            left=left_node,
            right=right_node,
            split_feature=feature_idx,
            split_value=split_val,
            size=n_samples,
            is_leaf=False,
        )

    def path_length(self, x: np.ndarray) -> float:
        """Calculate path length for a single point x."""
        node = self.root
        depth = 0
        while node is not None and not node.is_leaf:
            assert node.split_feature is not None and node.split_value is not None
            if x[node.split_feature] < node.split_value:
                node = node.left
            else:
                node = node.right
            depth += 1

        if node is not None and node.size > 1:
            return depth + _c(node.size)
        return float(depth)


class IsolationForest:
    """Isolation Forest ensemble for outlier and anomaly detection.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    max_samples : int or float
        Number of samples to draw from X to train each tree.
    contamination : float
        Proportion of outliers in the data set (determines threshold).
    random_state : int or None
        Seed for pseudo-random number generator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: float = 256,
        contamination: float = 0.1,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.trees: list[IsolationTree] = []
        self.subsample_size_: int = 256
        self.threshold_: float = 0.5

    def fit(self, X: np.ndarray) -> "IsolationForest":
        """Fit Isolation Forest on feature matrix X."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        if isinstance(self.max_samples, float):
            self.subsample_size_ = int(self.max_samples * n_samples)
        else:
            self.subsample_size_ = min(n_samples, self.max_samples)

        max_depth = int(np.ceil(np.log2(max(self.subsample_size_, 2))))
        self.trees = []

        for _ in range(self.n_estimators):
            subsample_idx = rng.choice(n_samples, size=self.subsample_size_, replace=False)
            tree = IsolationTree(max_depth=max_depth, rng=rng)
            tree.fit(X[subsample_idx])
            self.trees.append(tree)

        # Determine decision threshold based on contamination
        scores = self.score_samples(X)
        self.threshold_ = float(np.percentile(scores, 100.0 * (1.0 - self.contamination)))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score for each sample: s in [0, 1].

        Scores close to 1 are anomalies; scores < 0.5 are normal.
        """
        X = np.asarray(X, dtype=np.float64)
        c_sub = _c(self.subsample_size_)
        if c_sub == 0:
            c_sub = 1.0

        n_samples = X.shape[0]
        avg_path_lengths = np.zeros(n_samples)

        for i in range(n_samples):
            x = X[i]
            paths = [tree.path_length(x) for tree in self.trees]
            avg_path_lengths[i] = np.mean(paths)

        # Anomaly score: 2^(-E(h)/c)
        return 2.0 ** (-avg_path_lengths / c_sub)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels: -1 for anomaly/outlier, 1 for inlier."""
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, -1, 1)
