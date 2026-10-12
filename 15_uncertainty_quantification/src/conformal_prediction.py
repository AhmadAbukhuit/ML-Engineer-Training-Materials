"""Conformal Prediction for Distribution-Free Uncertainty Quantification."""

from typing import Any, Protocol
import numpy as np


class RegressorProtocol(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class ClassifierProtocol(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class SplitConformalRegressor:
    """Inductive (Split) Conformal Prediction for Regression.

    Guarantees distribution-free finite-sample marginal coverage:
    P(Y_new in C(X_new)) >= 1 - alpha,
    provided calibration and test points are exchangeable.

    Parameters
    ----------
    base_model : RegressorProtocol
        Fitted regression model with a .predict() method.
    """

    def __init__(self, base_model: Any) -> None:
        self.base_model = base_model
        self.q_hat: float | None = None
        self.alpha: float | None = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> "SplitConformalRegressor":
        """Compute conformal quantile on held-out calibration dataset.

        Parameters
        ----------
        X_cal : np.ndarray of shape (n_cal, n_features)
            Calibration feature matrix.
        y_cal : np.ndarray of shape (n_cal,)
            Ground truth calibration targets.
        alpha : float
            Significance level (error rate), yielding 1 - alpha coverage.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)
        n = len(y_cal)

        # Base predictions
        y_pred = np.asarray(self.base_model.predict(X_cal), dtype=np.float64)
        # Absolute residuals as non-conformity scores
        scores = np.abs(y_cal - y_pred)

        # Finite-sample adjusted quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        q_level = float(np.clip(q_level, 0.0, 1.0))

        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.alpha = alpha
        return self

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict lower and upper bounds of 1 - alpha prediction intervals.

        Returns
        -------
        lower_bounds : np.ndarray
        upper_bounds : np.ndarray
        """
        if self.q_hat is None:
            raise RuntimeError("Must calibrate model with .calibrate() before prediction.")

        X = np.asarray(X, dtype=np.float64)
        y_pred = np.asarray(self.base_model.predict(X), dtype=np.float64)

        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper


class ConformalClassifier:
    """Conformal Prediction for Multi-Class Classification Sets.

    Produces prediction sets C(x) subset of {1, ..., K} guaranteeing:
    P(Y_new in C(X_new)) >= 1 - alpha.

    Parameters
    ----------
    base_model : ClassifierProtocol
        Fitted classification model with a .predict_proba() method.
    """

    def __init__(self, base_model: Any) -> None:
        self.base_model = base_model
        self.q_hat: float | None = None
        self.alpha: float | None = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> "ConformalClassifier":
        """Calibrate on held-out dataset using softmax non-conformity scores."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=int)
        n = len(y_cal)

        probs = np.asarray(self.base_model.predict_proba(X_cal), dtype=np.float64)
        # Non-conformity score: 1 - P(Y = y_true | X)
        true_class_probs = probs[np.arange(n), y_cal]
        scores = 1.0 - true_class_probs

        q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        q_level = float(np.clip(q_level, 0.0, 1.0))

        self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        self.alpha = alpha
        return self

    def predict_set(self, X: np.ndarray) -> list[list[int]]:
        """Return list of candidate class labels for each sample."""
        if self.q_hat is None:
            raise RuntimeError("Must calibrate model with .calibrate() before prediction.")

        X = np.asarray(X, dtype=np.float64)
        probs = np.asarray(self.base_model.predict_proba(X), dtype=np.float64)

        prediction_sets: list[list[int]] = []
        # Include class k if 1 - prob[k] <= q_hat <=> prob[k] >= 1 - q_hat
        threshold = 1.0 - self.q_hat

        for row in probs:
            candidate_classes = [int(k) for k in np.where(row >= threshold)[0]]
            # Avoid empty set degenerate case by including argmax if none meet threshold
            if len(candidate_classes) == 0:
                candidate_classes = [int(np.argmax(row))]
            prediction_sets.append(candidate_classes)

        return prediction_sets
