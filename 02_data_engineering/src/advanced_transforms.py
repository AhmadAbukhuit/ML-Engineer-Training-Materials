"""
Advanced Feature Engineering: Out-of-Fold Target Encoding & Power Transforms.

This module implements:
1. OutOfFoldTargetEncoder: High-cardinality categorical encoding with m-estimate smoothing 
   and K-Fold splitting to strictly prevent target leakage.
2. YeoJohnsonTransformer: Non-linear power transformations to normalize skewed distributions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer


class OutOfFoldTargetEncoder:
    """
    Smoothed Target Encoder using Out-of-Fold (OOF) cross-validation to prevent target leakage.
    Formula with m-estimate smoothing:
    S_c = (n_c * mean_c + m * global_mean) / (n_c + m)
    """

    def __init__(self, categorical_cols: list[str], m_smoothing: float = 10.0, n_splits: int = 5):
        self.categorical_cols = categorical_cols
        self.m_smoothing = m_smoothing
        self.n_splits = n_splits
        self.global_mean: float = 0.0
        self.mapping_: dict[str, dict[str, float]] = {}

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """
        Fits the encoder and transforms the training DataFrame using K-Fold out-of-fold splits.
        """
        df_out = df.copy()
        y_arr = np.array(y)
        self.global_mean = float(np.mean(y_arr))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Initialize output columns for OOF values
        for col in self.categorical_cols:
            df_out[f"{col}_encoded"] = np.nan

        # 1. Out-of-fold assignment for training set
        for train_idx, val_idx in kf.split(df_out):
            df_tr = df_out.iloc[train_idx].copy()
            df_tr["_target"] = y_arr[train_idx]
            df_val = df_out.iloc[val_idx]

            for col in self.categorical_cols:
                grouped = df_tr.groupby(col)["_target"].agg(["count", "mean"])
                smoothed_map = (
                    (grouped["count"] * grouped["mean"] + self.m_smoothing * self.global_mean)
                    / (grouped["count"] + self.m_smoothing)
                ).to_dict()

                df_out.iloc[val_idx, df_out.columns.get_loc(f"{col}_encoded")] = (
                    df_val[col].map(smoothed_map).fillna(self.global_mean)
                )

        # 2. Compute full-dataset mapping for future test sets
        df_full = df.copy()
        df_full["_target"] = y_arr
        self.mapping_ = {}
        for col in self.categorical_cols:
            grouped = df_full.groupby(col)["_target"].agg(["count", "mean"])
            self.mapping_[col] = (
                (grouped["count"] * grouped["mean"] + self.m_smoothing * self.global_mean)
                / (grouped["count"] + self.m_smoothing)
            ).to_dict()

        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms unseen test data using full-dataset smoothed mappings.
        """
        df_out = df.copy()
        for col in self.categorical_cols:
            col_map = self.mapping_.get(col, {})
            df_out[f"{col}_encoded"] = df_out[col].map(col_map).fillna(self.global_mean)
        return df_out


class YeoJohnsonTransformer:
    """
    Applies Yeo-Johnson power transformation to normalize positive, negative, and zero values.
    """

    def __init__(self):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.pt.fit_transform(X)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.pt.transform(X)
