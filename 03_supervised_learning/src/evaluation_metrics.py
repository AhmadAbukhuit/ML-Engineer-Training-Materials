"""Comprehensive Evaluation Metrics from First Principles in NumPy.

Implements:
1. Classification:
   - confusion_matrix
   - accuracy_score
   - precision_score (binary, macro, micro, weighted)
   - recall_score (binary, macro, micro, weighted)
   - f1_score (binary, macro, micro, weighted)
   - fbeta_score
   - specificity_score
   - balanced_accuracy_score
   - matthews_corrcoef (MCC)
   - brier_score_loss
   - roc_curve & roc_auc_score
   - precision_recall_curve & average_precision_score
   - classification_report
2. Regression:
   - mae_score
   - mse_score
   - rmse_score
   - r2_score
   - mape_score
"""

from typing import Any, Literal
import numpy as np


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[Any] | np.ndarray | None = None
) -> np.ndarray:
    """Compute confusion matrix evaluating classification accuracy.

    Returns matrix of shape (n_classes, n_classes) where entry (i, j)
    is the number of samples with true class i and predicted class j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    else:
        classes = np.asarray(labels)

    n_classes = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1

    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples: (TP + TN) / Total."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _compute_class_tp_fp_fn_support(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to compute TP, FP, FN, and support for each unique class."""
    n_classes = len(classes)
    tp = np.zeros(n_classes, dtype=np.float64)
    fp = np.zeros(n_classes, dtype=np.float64)
    fn = np.zeros(n_classes, dtype=np.float64)
    support = np.zeros(n_classes, dtype=np.float64)

    for i, c in enumerate(classes):
        actual_pos = y_true == c
        pred_pos = y_pred == c

        tp[i] = np.sum(actual_pos & pred_pos)
        fp[i] = np.sum((~actual_pos) & pred_pos)
        fn[i] = np.sum(actual_pos & (~pred_pos))
        support[i] = np.sum(actual_pos)

    return tp, fp, fn, support


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    """Compute precision: TP / (TP + FP)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(zero_division)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    tp, fp, _, support = _compute_class_tp_fp_fn_support(y_true, y_pred, classes)

    if average == "micro":
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        denom = total_tp + total_fp
        return float(total_tp / denom) if denom > 0 else float(zero_division)

    denoms = tp + fp
    class_precisions = np.where(denoms > 0, tp / np.maximum(denoms, 1e-12), zero_division)

    if average == "macro":
        return float(np.mean(class_precisions))
    if average == "weighted":
        total_supp = np.sum(support)
        return (
            float(np.sum(class_precisions * support) / total_supp)
            if total_supp > 0
            else float(zero_division)
        )

    raise ValueError(f"Unknown average '{average}'.")


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    """Compute recall (sensitivity / TPR): TP / (TP + FN)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(zero_division)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    tp, _, fn, support = _compute_class_tp_fp_fn_support(y_true, y_pred, classes)

    if average == "micro":
        total_tp = np.sum(tp)
        total_fn = np.sum(fn)
        denom = total_tp + total_fn
        return float(total_tp / denom) if denom > 0 else float(zero_division)

    denoms = tp + fn
    class_recalls = np.where(denoms > 0, tp / np.maximum(denoms, 1e-12), zero_division)

    if average == "macro":
        return float(np.mean(class_recalls))
    if average == "weighted":
        total_supp = np.sum(support)
        return (
            float(np.sum(class_recalls * support) / total_supp)
            if total_supp > 0
            else float(zero_division)
        )

    raise ValueError(f"Unknown average '{average}'.")


def fbeta_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    """Compute F-beta score: (1 + beta^2) * (P * R) / (beta^2 * P + R)."""
    if beta <= 0:
        raise ValueError("beta must be strictly positive.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    beta_sq = beta**2

    if average == "binary":
        p = precision_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0.0)
        r = recall_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0.0)
        denom = beta_sq * p + r
        if denom == 0:
            return float(zero_division)
        return float((1.0 + beta_sq) * (p * r) / denom)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    tp, fp, fn, support = _compute_class_tp_fp_fn_support(y_true, y_pred, classes)

    if average == "micro":
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)
        denom = (1.0 + beta_sq) * total_tp + beta_sq * total_fn + total_fp
        return float((1.0 + beta_sq) * total_tp / denom) if denom > 0 else float(zero_division)

    class_fbeta = np.zeros(len(classes))
    for i in range(len(classes)):
        num = (1.0 + beta_sq) * tp[i]
        denom = (1.0 + beta_sq) * tp[i] + beta_sq * fn[i] + fp[i]
        class_fbeta[i] = num / denom if denom > 0 else zero_division

    if average == "macro":
        return float(np.mean(class_fbeta))
    if average == "weighted":
        total_supp = np.sum(support)
        return (
            float(np.sum(class_fbeta * support) / total_supp)
            if total_supp > 0
            else float(zero_division)
        )

    raise ValueError(f"Unknown average '{average}'.")


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro", "micro", "weighted"] = "binary",
    pos_label: Any = 1,
    zero_division: float = 0.0,
) -> float:
    """Compute F1 score: harmonic mean of precision and recall."""
    return fbeta_score(
        y_true,
        y_pred,
        beta=1.0,
        average=average,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: Any = 1) -> float:
    """Compute specificity (True Negative Rate): TN / (TN + FP)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy: average recall across all classes."""
    return recall_score(y_true, y_pred, average="macro", zero_division=0.0)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Matthews Correlation Coefficient (MCC) for binary classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score loss: mean squared error on probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_true - y_prob) ** 2))


def roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve points.

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    thresholds : np.ndarray
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort descending by score
    desc_order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_order]
    y_score_sorted = y_score[desc_order]

    # Distinct threshold indices
    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idx = np.concatenate([distinct_idx, [len(y_score_sorted) - 1]])

    tps = np.cumsum(y_true_sorted == 1)[threshold_idx]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idx]

    total_p = np.sum(y_true == 1)
    total_n = np.sum(y_true == 0)

    tpr = tps / total_p if total_p > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / total_n if total_n > 0 else np.zeros_like(fps, dtype=float)
    thresholds = y_score_sorted[threshold_idx]

    # Prepend (0, 0) point
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1.0], thresholds])

    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (ROC-AUC) via trapezoidal rule."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Trapezoidal integration along fpr: sum 0.5 * (tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1])
    return float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * np.diff(fpr)))


def precision_recall_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve points.

    Returns
    -------
    precision : np.ndarray
    recall : np.ndarray
    thresholds : np.ndarray
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    asc_order = np.argsort(y_score)
    y_true_sorted = y_true[asc_order]
    y_score_sorted = y_score[asc_order]

    distinct_idx = np.where(np.diff(y_score_sorted))[0]
    threshold_idx = np.concatenate([[0], distinct_idx + 1])

    cum_tp = np.cumsum((y_true_sorted == 1)[::-1])[::-1]
    cum_fp = np.cumsum((y_true_sorted == 0)[::-1])[::-1]

    tps = cum_tp[threshold_idx]
    fps = cum_fp[threshold_idx]
    total_p = np.sum(y_true == 1)

    precision = tps / (tps + fps)
    recall = tps / total_p if total_p > 0 else np.zeros_like(tps, dtype=float)
    thresholds = y_score_sorted[threshold_idx]

    # Append (P=1, R=0) terminal point per scikit-learn convention
    precision = np.concatenate([precision, [1.0]])
    recall = np.concatenate([recall, [0.0]])

    return precision, recall, thresholds


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall curve (PR-AUC / Average Precision)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # AP = sum (R_n - R_{n-1}) * P_n where recall decreases from 1 to 0
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    """Generate comprehensive per-class and aggregated metric dictionary."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    report: dict[str, dict[str, float]] = {}
    for c in classes:
        p = precision_score(y_true, y_pred, average="binary", pos_label=c, zero_division=0.0)
        r = recall_score(y_true, y_pred, average="binary", pos_label=c, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=c, zero_division=0.0)
        supp = float(np.sum(y_true == c))
        report[str(c)] = {"precision": p, "recall": r, "f1-score": f1, "support": supp}

    report["accuracy"] = {"accuracy": accuracy_score(y_true, y_pred)}
    report["macro avg"] = {
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1-score": f1_score(y_true, y_pred, average="macro"),
        "support": float(len(y_true)),
    }
    report["weighted avg"] = {
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1-score": f1_score(y_true, y_pred, average="weighted"),
        "support": float(len(y_true)),
    }
    return report


# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------

def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/N) * sum |y - y_hat|."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error: (1/N) * sum (y - y_hat)^2."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt(MSE)."""
    return float(np.sqrt(mse_score(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2): 1 - (SS_res / SS_tot)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def mape_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Mean Absolute Percentage Error in percentage (0 to 100%)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # Avoid zero division
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
