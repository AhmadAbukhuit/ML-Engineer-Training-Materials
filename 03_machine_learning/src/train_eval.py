"""
Production Model Training, Cross-Validation, and Serialization Pipeline.

This script executes:
1. Feature extraction and train/test split.
2. Stratified 5-Fold Cross Validation.
3. Metric computation: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
4. Model artifact persistence to models/baseline_classifier.joblib.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold


def evaluate_model_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> dict[str, float]:
    """
    Evaluates a classification model using Stratified K-Fold cross validation.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_val, y_proba))

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "f1_mean": float(np.mean(f1s)),
        "roc_auc_mean": float(np.mean(aucs)),
    }


def train_and_export_baseline(
    output_model_path: Path,
    n_samples: int = 1000,
    n_features: int = 10
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """
    Trains a production Random Forest classifier and exports its serialized weights.
    """
    # Generate reproducible synthetic telemetry failure dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        weights=[0.8, 0.2],  # Realistic imbalanced failure rate (20% positive)
        random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    metrics = evaluate_model_cross_validation(clf, X, y)

    # Final fit on entire dataset
    clf.fit(X, y)

    # Serialize
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_model_path)

    return clf, metrics


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir.parent / "models" / "baseline_classifier.joblib"
    _, results = train_and_export_baseline(model_path)
    print("Baseline Model Training Complete.")
    print("Cross-Validation Metrics:")
    for metric_name, val in results.items():
        print(f"  {metric_name}: {val:.4f}")
    print(f"Serialized to: {model_path}")
