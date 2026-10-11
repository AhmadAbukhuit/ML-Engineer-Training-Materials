"""
Data Analysis, Profiling, and Preprocessing Utilities.

This module provides reusable, vectorized routines leveraging NumPy, Pandas,
and Scikit-Learn for:
1. Missingness auditing and profiling.
2. Outlier detection using the Interquartile Range (IQR) rule.
3. Comprehensive statistical summarization of numerical distributions.
4. Pairwise feature correlation matrix calculation.
5. Composable, zero-leakage scikit-learn preprocessing pipelines.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler


def generate_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes missing value frequencies and percentages across all DataFrame columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame sorted by missing percentage in descending order with columns:
        ['column', 'missing_count', 'missing_percentage', 'non_null_count', 'dtype']
    """
    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame(columns=["column", "missing_count", "missing_percentage", "non_null_count", "dtype"])

    missing_counts = df.isna().sum()
    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_counts.values,
            "missing_percentage": np.round((missing_counts.values / total_rows) * 100.0, 2),
            "non_null_count": df.notna().sum().values,
            "dtype": [str(t) for t in df.dtypes.values],
        }
    )

    return report.sort_values(by="missing_percentage", ascending=False).reset_index(drop=True)


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5,
) -> dict[str, Any]:
    """
    Detects distribution outliers in a numerical column using the Interquartile Range rule:
    Lower Bound = Q1 - (factor * IQR)
    Upper Bound = Q3 + (factor * IQR)

    Args:
        df: Input DataFrame.
        column: Numerical column name to evaluate.
        factor: Multiplier for IQR (standard default is 1.5, extreme outlier threshold is 3.0).

    Returns:
        Dictionary containing quartile metrics, bounds, outlier count, and outlier row indices.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    series = df[column].dropna()
    if len(series) == 0:
        return {
            "column": column,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "outlier_count": 0,
            "outlier_indices": [],
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)

    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_indices = df.index[outlier_mask].tolist()

    return {
        "column": column,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices,
    }


def summarize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an extended summary profile for all numerical columns including
    mean, median, standard deviation, IQR, and skewness.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame indexed by column names with calculated summary metrics.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    stats = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        mean_val = float(s.mean())
        std_val = float(s.std())
        median_val = float(s.median())
        skew_val = float(s.skew()) if len(s) > 2 else 0.0

        stats.append(
            {
                "feature": col,
                "count": len(s),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(float(s.min()), 4),
                "q25": round(q1, 4),
                "median": round(median_val, 4),
                "q75": round(q3, 4),
                "max": round(float(s.max()), 4),
                "iqr": round(iqr, 4),
                "skewness": round(skew_val, 4),
            }
        )

    return pd.DataFrame(stats).set_index("feature")


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Computes pairwise feature correlation matrix for numerical features.

    Args:
        df: Input DataFrame.
        method: Correlation algorithm ('pearson', 'spearman', or 'kendall').

    Returns:
        Square correlation DataFrame of shape (n_features, n_features).
    """
    num_df = df.select_dtypes(include=[np.number])
    return num_df.corr(method=method)


def build_analysis_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scaler_type: str = "robust",
) -> ColumnTransformer:
    """
    Constructs an isolated, zero-leakage Scikit-Learn preprocessing pipeline.

    Args:
        numeric_features: List of continuous numerical column names.
        categorical_features: List of discrete categorical column names.
        scaler_type: Scaling algorithm ('robust', 'standard', or 'minmax').

    Returns:
        Unfitted ColumnTransformer ready for .fit_transform().
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust', 'standard', or 'minmax'.")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor
