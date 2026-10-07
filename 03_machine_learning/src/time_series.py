"""
Time Series Feature Engineering & Walk-Forward Validation.

In time series forecasting, random shuffling causes catastrophic lookahead data leakage.
This module provides:
1. create_time_series_features: Generates lag features, rolling statistics, and cyclical calendar encodings.
2. WalkForwardValidator: Rolling-origin / expanding-window temporal cross-validation.
3. Temporal evaluation metrics: MAE, RMSE, and MAPE.
"""

from collections.abc import Generator
import numpy as np
import pandas as pd


def create_time_series_features(
    series: pd.Series,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Transforms a 1D temporal series into an ML feature matrix with lag and rolling statistics.

    Args:
        series: Continuous time series index-aligned with timestamps.
        lags: List of time offsets (e.g., [1, 2, 3, 7]) for historical lag values.
        rolling_windows: Window sizes for rolling mean and standard deviation.

    Returns:
        DataFrame with feature columns and target 'y', with initial NaN rows dropped.
    """
    if lags is None:
        lags = [1, 2, 3, 7]
    if rolling_windows is None:
        rolling_windows = [3, 7]

    df = pd.DataFrame({"y": series})

    # 1. Historical lag features (strictly past values: y_{t-k})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # 2. Rolling window features computed strictly over past observations
    for w in rolling_windows:
        # Shift by 1 first so current row's value is not included in its own rolling feature
        shifted = df["y"].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w).std()

    # Drop warm-up rows containing NaNs
    return df.dropna().copy()


class WalkForwardValidator:
    """
    Expanding-window (Walk-Forward) temporal cross-validator.
    Guarantees test indices are always chronologically after training indices.
    """

    def __init__(self, n_splits: int = 5, train_min_size: int = 30, test_size: int = 10):
        self.n_splits = n_splits
        self.train_min_size = train_min_size
        self.test_size = test_size

    def split(self, n_samples: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) pairs expanding forward in time.
        """
        required_samples = self.train_min_size + (self.n_splits * self.test_size)
        if n_samples < required_samples:
            raise ValueError(
                f"n_samples ({n_samples}) is too small for {self.n_splits} splits "
                f"with train_min={self.train_min_size} and test_size={self.test_size}."
            )

        for i in range(self.n_splits):
            train_end = self.train_min_size + (i * self.test_size)
            test_end = train_end + self.test_size

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)

            yield train_idx, test_idx


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Computes standard regression metrics for time series evaluation.
    """
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    # Avoid zero division in MAPE
    eps = 1e-8
    mape = float(np.mean(np.abs(error / np.maximum(np.abs(y_true), eps)))) * 100.0

    return {"mae": mae, "rmse": rmse, "mape": mape}
