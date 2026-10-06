"""
Decision Tree Classifier from First Principles.

This module implements a binary decision tree classifier using the Gini Impurity criterion.
Features:
1. Recursive greedy binary splitting.
2. Vectorized calculation of candidate thresholds and Gini index.
3. Configurable stopping criteria: max_depth, min_samples_split.
"""

from typing import Optional
import numpy as np


class Node:
    """Represents a single decision node or terminal leaf in a tree."""

    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float | None = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        *,
        value: int | None = None,
        probs: np.ndarray | None = None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probs = probs

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    """
    Binary / Multi-Class Decision Tree Classifier.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 1e-7
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Node | None = None
        self.n_classes_: int = 0

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        current_gini = self._gini(y)
        best_gain = -1.0
        best_feat, best_thresh = None, None

        for feat_idx in range(n_features):
            values = X[:, feat_idx]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            # Candidate split thresholds: midpoints between sorted unique values
            sorted_vals = np.sort(unique_vals)
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])

                # Weighted Gini of the split
                split_gini = (n_l / n_samples) * gini_left + (n_r / n_samples) * gini_right
                gain = current_gini - split_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        if best_gain > self.min_impurity_decrease:
            return best_feat, best_thresh
        return None, None

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        # Class probability distribution at this node
        prob_dist = np.zeros(self.n_classes_)
        for c, count in zip(classes, counts):
            prob_dist[int(c)] = count / n_samples

        # Stopping criteria: pure node, reached max_depth, or cannot split further
        if len(classes) == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(value=int(majority_class), probs=prob_dist)

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return Node(value=int(majority_class), probs=prob_dist)

        # Split data
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature_idx=feat_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
            value=int(majority_class),
            probs=prob_dist
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.n_classes_ = int(np.max(y) + 1)
        self.root = self._build_tree(X, y.astype(int), depth=0)
        return self

    def _traverse_row(self, x: np.ndarray, node: Node) -> Node:
        if node.is_leaf or node.feature_idx is None or node.threshold is None:
            return node
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_row(x, node.left)
        return self._traverse_row(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("DecisionTree has not been fitted yet.")
        return np.array([self._traverse_row(row, self.root).value for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("DecisionTree has not been fitted yet.")
        return np.array([self._traverse_row(row, self.root).probs for row in X])
