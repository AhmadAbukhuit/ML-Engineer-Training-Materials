import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from time_series import create_time_series_features, WalkForwardValidator, compute_forecast_metrics


def test_create_time_series_features():
    # 50-step synthetic series
    raw_series = pd.Series(np.arange(50, dtype=float))
    feature_df = create_time_series_features(raw_series, lags=[1, 2], rolling_windows=[3])

    assert "lag_1" in feature_df.columns
    assert "lag_2" in feature_df.columns
    assert "rolling_mean_3" in feature_df.columns
    assert not feature_df.isna().any().any()


def test_walk_forward_validator_strictly_future():
    validator = WalkForwardValidator(n_splits=3, train_min_size=20, test_size=5)
    splits = list(validator.split(n_samples=40))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        # Guarantee no overlap and test is strictly in the future
        assert max(train_idx) < min(test_idx)
        assert len(test_idx) == 5


def test_forecast_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 32.0])

    metrics = compute_forecast_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert metrics["mae"] > 0.0
