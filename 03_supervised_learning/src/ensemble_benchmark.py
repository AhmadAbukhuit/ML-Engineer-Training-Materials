"""
Ensemble Algorithms Benchmark: Random Forest vs Gradient Boosting vs XGBoost.

This module benchmarks tree ensemble architectures:
1. RandomForestClassifier (Bagging & random feature subspace sampling)
2. GradientBoostingClassifier (Sequential pseudo-residual descent)
3. XGBClassifier (2nd-order Taylor expansion & leaf regularization)
"""

from typing import Any
import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def run_ensemble_benchmark(
    n_samples: int = 1500,
    n_features: int = 20,
    random_state: int = 42
) -> list[dict[str, Any]]:
    """
    Trains and compares multiple ensemble classifiers on a synthetic dataset.
    """
    n_informative = max(2, int(n_features * 0.6))
    n_redundant = max(1, n_features - n_informative - 1)
    if n_informative + n_redundant > n_features:
        n_redundant = max(0, n_features - n_informative)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        weights=[0.75, 0.25],
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=random_state)
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, eval_metric="logloss", random_state=random_state
        )

    results = []

    for name, model in models.items():
        # Measure training latency
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0

        # Measure inference latency
        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        infer_time_ms = (time.perf_counter() - t1) * 1000.0

        results.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "train_latency_ms": round(train_time_ms, 2),
            "infer_latency_ms": round(infer_time_ms, 2)
        })

    return results
