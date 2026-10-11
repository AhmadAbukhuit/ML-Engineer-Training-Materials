import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_analysis import (
    build_analysis_pipeline,
    compute_correlation_matrix,
    detect_outliers_iqr,
    generate_missing_report,
    summarize_numeric_features,
)


def test_generate_missing_report():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [np.nan, np.nan, np.nan, "x", "y"],
            "c": [10, 20, 30, 40, 50],
        }
    )

    report = generate_missing_report(df)
    assert len(report) == 3
    # Column 'b' should be first (60% missing)
    assert report.iloc[0]["column"] == "b"
    assert report.iloc[0]["missing_count"] == 3
    assert report.iloc[0]["missing_percentage"] == 60.0

    # Column 'c' should have 0% missing
    c_row = report[report["column"] == "c"].iloc[0]
    assert c_row["missing_count"] == 0
    assert c_row["missing_percentage"] == 0.0


def test_detect_outliers_iqr():
    # Normal cluster with one prominent outlier
    values = [10.0, 11.0, 12.0, 10.5, 11.5, 9.5, 10.8, 100.0]
    df = pd.DataFrame({"temperature": values})

    result = detect_outliers_iqr(df, column="temperature", factor=1.5)
    assert result["column"] == "temperature"
    assert result["outlier_count"] == 1
    assert 7 in result["outlier_indices"]  # Index of 100.0 is an outlier
    assert result["upper_bound"] < 100.0


def test_summarize_numeric_features():
    df = pd.DataFrame(
        {
            "val1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "val2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )

    summary = summarize_numeric_features(df)
    assert len(summary) == 2  # Only numeric columns
    assert "val1" in summary.index
    assert "val2" in summary.index

    assert summary.loc["val1", "mean"] == 3.0
    assert summary.loc["val1", "median"] == 3.0
    assert summary.loc["val1", "min"] == 1.0
    assert summary.loc["val1", "max"] == 5.0
    assert summary.loc["val1", "iqr"] == 2.0


def test_compute_correlation_matrix():
    x = np.linspace(0, 10, 50)
    df = pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + 1,  # Perfectly correlated
            "z": -x,  # Perfectly negatively correlated
        }
    )

    corr = compute_correlation_matrix(df, method="pearson")
    assert corr.shape == (3, 3)
    assert np.isclose(corr.loc["x", "y"], 1.0)
    assert np.isclose(corr.loc["x", "z"], -1.0)


def test_build_analysis_pipeline_fit_transform():
    df_train = pd.DataFrame(
        {
            "numeric_a": [1.0, 2.0, np.nan, 4.0, 100.0],
            "numeric_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat_feature": ["apple", "banana", "apple", np.nan, "orange"],
        }
    )

    df_test = pd.DataFrame(
        {
            "numeric_a": [3.0, np.nan],
            "numeric_b": [25.0, 35.0],
            "cat_feature": ["banana", "unknown_fruit"],  # Unseen category
        }
    )

    preprocessor = build_analysis_pipeline(
        numeric_features=["numeric_a", "numeric_b"],
        categorical_features=["cat_feature"],
        scaler_type="robust",
    )

    # Fit and transform train
    train_transformed = preprocessor.fit_transform(df_train)
    assert train_transformed.shape[0] == 5
    assert not np.isnan(train_transformed).any()

    # Transform test without crashing on unseen category
    test_transformed = preprocessor.transform(df_test)
    assert test_transformed.shape[0] == 2
    assert not np.isnan(test_transformed).any()


def test_build_analysis_pipeline_invalid_scaler():
    with pytest.raises(ValueError, match="Unsupported scaler_type"):
        build_analysis_pipeline(
            numeric_features=["a"],
            categorical_features=["b"],
            scaler_type="non_existent_scaler",
        )
