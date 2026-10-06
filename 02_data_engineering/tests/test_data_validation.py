import sys
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_validation import validate_telemetry_dataframe


def test_valid_telemetry_data():
    valid_df = pd.DataFrame({
        "device_id": ["sensor_alpha", "sensor_beta"],
        "temperature": [22.5, 45.0],
        "lat": [37.7749, 40.7128],
        "lon": [-122.4194, -74.0060],
        "timestamp": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-01 12:01:00"])
    })

    is_valid, msg = validate_telemetry_dataframe(valid_df)
    assert is_valid is True
    assert "Success" in msg or "verified" in msg


def test_invalid_telemetry_data_temperature_out_of_range():
    # Temperature 999.0 violates the [-50, 150] contract
    invalid_df = pd.DataFrame({
        "device_id": ["sensor_broken"],
        "temperature": [999.0],
        "lat": [37.7749],
        "lon": [-122.4194],
        "timestamp": pd.to_datetime(["2024-01-01 12:00:00"])
    })

    is_valid, msg = validate_telemetry_dataframe(invalid_df)
    assert is_valid is False
    assert "Contract Violation" in msg
