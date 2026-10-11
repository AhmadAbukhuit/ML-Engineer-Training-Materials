import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from isolation_forest import IsolationForest


def test_isolation_forest_anomaly_detection():
    rng = np.random.default_rng(42)
    # 200 normal points concentrated near origin
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    # 5 extreme outliers far away
    X_outliers = rng.uniform(low=15.0, high=25.0, size=(5, 2))
    X = np.vstack([X_normal, X_outliers])

    iso = IsolationForest(n_estimators=100, max_samples=128, contamination=0.03, random_state=42)
    iso.fit(X)

    scores = iso.score_samples(X)
    preds = iso.predict(X)

    # Anomaly scores for outliers should be significantly higher than normal points
    normal_mean_score = np.mean(scores[:200])
    outlier_mean_score = np.mean(scores[200:])
    assert outlier_mean_score > normal_mean_score

    # Outliers should be predominantly classified as -1
    outlier_preds = preds[200:]
    assert np.sum(outlier_preds == -1) >= 4
