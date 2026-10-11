"""
Data Engineering, ETL, Validation, and Feature Pipeline Modules.
"""

from .advanced_transforms import OutOfFoldTargetEncoder, YeoJohnsonTransformer
from .data_analysis import (
    build_analysis_pipeline,
    compute_correlation_matrix,
    detect_outliers_iqr,
    generate_missing_report,
    summarize_numeric_features,
)
from .data_validation import TelemetryDataContract, validate_telemetry_dataframe
from .duckdb_parquet import compute_rolling_features_duckdb, export_to_parquet, query_parquet_with_duckdb
from .feature_engineering import build_preprocessor, fit_and_export_pipeline, load_pipeline
from .image_pipeline import letterbox_image, normalize_pixels, voc_box_to_yolo
from .telemetry_parser import process_telemetry_directory
from .temporal_joins import point_in_time_join

__all__ = [
    "OutOfFoldTargetEncoder",
    "TelemetryDataContract",
    "YeoJohnsonTransformer",
    "build_analysis_pipeline",
    "build_preprocessor",
    "compute_correlation_matrix",
    "compute_rolling_features_duckdb",
    "detect_outliers_iqr",
    "export_to_parquet",
    "fit_and_export_pipeline",
    "generate_missing_report",
    "letterbox_image",
    "load_pipeline",
    "normalize_pixels",
    "point_in_time_join",
    "process_telemetry_directory",
    "query_parquet_with_duckdb",
    "summarize_numeric_features",
    "validate_telemetry_dataframe",
    "voc_box_to_yolo",
]
