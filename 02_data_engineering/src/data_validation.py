"""
Data Quality Contracts and Runtime Schema Validation via Pandera.

This module guarantees:
1. Strict data contracts for production datasets before model ingestion.
2. Runtime enforcement of allowed ranges, types, and null constraints.
3. Diagnostic error reporting for poisoned or malformed data batches.
"""

import pandas as pd
try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pandera.errors import SchemaError, SchemaErrors


# Define the production telemetry contract
TelemetryDataContract = DataFrameSchema(
    columns={
        "device_id": Column(str, nullable=False),
        "temperature": Column(
            float,
            Check.in_range(-50.0, 150.0, error="Temperature must be between -50°C and 150°C"),
            nullable=True
        ),
        "lat": Column(
            float,
            Check.in_range(-90.0, 90.0, error="Latitude must be between -90 and 90"),
            nullable=False
        ),
        "lon": Column(
            float,
            Check.in_range(-180.0, 180.0, error="Longitude must be between -180 and 180"),
            nullable=False
        ),
        "timestamp": Column(pa.DateTime, nullable=False)
    },
    coerce=True,
    strict=False  # Allows extra columns if present
)


def validate_telemetry_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates an incoming DataFrame against the telemetry data contract.

    Returns:
        (is_valid, message): True and 'Success' if valid, False and error description otherwise.
    """
    try:
        TelemetryDataContract.validate(df, lazy=True)
        return True, "Dataset successfully verified against data contract."
    except (SchemaError, SchemaErrors) as e:
        return False, f"Contract Violation: {e!s}"
