"""Pseudo-Labeling / Self-Training Classifier from First Principles."""

from typing import Any
import numpy as np


class SelfTrainingClassifier:
    """Self-Training / Pseudo-Labeling Classifier.

    Iteratively trains a supervised base classifier on labeled data, predicts
    class probabilities on unlabeled data, and incorporates pseudo-labeled samples
    whose maximum predicted probability meets or exceeds the confidence threshold tau.

    Parameters
    ----------
    base_estimator : Any
        A supervised classifier instance supporting fit, predict, and predict_proba.
    threshold : float
        Confidence threshold tau in [0.0, 1.0].
    max_iter : int
        Maximum self-training expansion rounds.
    """

    def __init__(self, base_estimator: Any, threshold: float = 0.8, max_iter: int = 10) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be within [0.0, 1.0].")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter

        self.n_pseudo_labeled_: int = 0
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelfTrainingClassifier":
        """Fit self-training classifier.

        Unlabeled samples should be indicated by label -1 in y.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).copy()

        labeled_mask = y != -1
        if not np.any(labeled_mask):
            raise ValueError("Dataset must contain at least one labeled sample (y != -1).")

        self.classes_ = np.unique(y[labeled_mask])
        self.n_pseudo_labeled_ = 0

        for _ in range(self.max_iter):
            # Train base estimator on all currently labeled samples
            current_X_train = X[labeled_mask]
            current_y_train = y[labeled_mask]

            self.base_estimator.fit(current_X_train, current_y_train)

            # Check if any unlabeled samples remain
            unlabeled_indices = np.where(~labeled_mask)[0]
            if len(unlabeled_indices) == 0:
                break

            # Predict probabilities on unlabeled pool
            unlabeled_X = X[unlabeled_indices]
            probas = self.base_estimator.predict_proba(unlabeled_X)
            max_probs = np.max(probas, axis=1)
            pred_classes = self.classes_[np.argmax(probas, axis=1)]

            # Filter samples above threshold
            confident_mask = max_probs >= self.threshold
            if not np.any(confident_mask):
                break  # Convergence: no remaining unlabeled sample meets confidence threshold

            # Assign pseudo-labels
            selected_indices = unlabeled_indices[confident_mask]
            y[selected_indices] = pred_classes[confident_mask]
            labeled_mask[selected_indices] = True
            self.n_pseudo_labeled_ += int(np.sum(confident_mask))

        # Final fit on complete expanded dataset
        self.base_estimator.fit(X[labeled_mask], y[labeled_mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using the final trained base estimator."""
        return self.base_estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the final trained base estimator."""
        return self.base_estimator.predict_proba(X)
