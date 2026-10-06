"""
Zero-Leakage Feature Engineering Pipeline.

In production Machine Learning, data leakage happens when statistical properties 
from the evaluation/test set (like means, variances, or category vocabularies) 
leak into the training phase.

This module provides a production-grade ColumnTransformer that:
1. Imputes missing numerical values using median and scales them with StandardScaler.
2. Imputes missing categorical values using most frequent and encodes them with OneHotEncoder.
3. Fits exclusively on training data and transforms test data safely.
4. Serializes the fitted pipeline with joblib for production deployment.
"""

from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str]
) -> ColumnTransformer:
    """
    Constructs an isolated, composable scikit-learn ColumnTransformer.

    Args:
        numeric_features: List of column names corresponding to continuous numerical variables.
        categorical_features: List of column names corresponding to discrete categories.

    Returns:
        A ColumnTransformer ready to fit on training data.
    """
    # 1. Numeric pipeline: Median imputation -> Standard Scaling (zero mean, unit variance)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Categorical pipeline: Most frequent imputation -> One-Hot Encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 3. Combine both branches
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"  # Drop any unselected columns to avoid unintentional leakage
    )

    return preprocessor


def fit_and_export_pipeline(
    preprocessor: ColumnTransformer,
    X_train,
    output_path: Path
) -> ColumnTransformer:
    """
    Fits the preprocessing pipeline strictly on X_train and serializes it to disk.

    Args:
        preprocessor: The unfitted ColumnTransformer.
        X_train: The training feature set (DataFrame or array-like).
        output_path: Destination path for the .joblib artifact.

    Returns:
        The fitted preprocessor.
    """
    fitted_preprocessor = preprocessor.fit(X_train)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_preprocessor, output_path)
    return fitted_preprocessor


def load_pipeline(artifact_path: Path) -> ColumnTransformer:
    """
    Loads a serialized preprocessing pipeline from disk for inference.
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {artifact_path}")
    return joblib.load(artifact_path)
