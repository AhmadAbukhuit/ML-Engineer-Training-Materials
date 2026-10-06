"""
Data Engineering, ETL, Validation, and Feature Pipeline Modules.
"""

from .telemetry_parser import process_telemetry_directory
from .image_pipeline import letterbox_image, voc_box_to_yolo, normalize_pixels
from .feature_engineering import build_preprocessor, fit_and_export_pipeline, load_pipeline
from .duckdb_parquet import export_to_parquet, query_parquet_with_duckdb, compute_rolling_features_duckdb
from .data_validation import validate_telemetry_dataframe, TelemetryDataContract
from .advanced_transforms import OutOfFoldTargetEncoder, YeoJohnsonTransformer
from .temporal_joins import point_in_time_join

__all__ = [
    "OutOfFoldTargetEncoder",
    "TelemetryDataContract",
    "YeoJohnsonTransformer",
    "build_preprocessor",
    "compute_rolling_features_duckdb",
    "export_to_parquet",
    "fit_and_export_pipeline",
    "letterbox_image",
    "load_pipeline",
    "normalize_pixels",
    "point_in_time_join",
    "process_telemetry_directory",
    "query_parquet_with_duckdb",
    "validate_telemetry_dataframe",
    "voc_box_to_yolo",
]
