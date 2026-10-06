import sys
from pathlib import Path
import polars as pl
from datetime import datetime

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from temporal_joins import point_in_time_join


def test_point_in_time_join_prevents_leakage():
    # Prediction events: Device A has an event at 10:15
    events_df = pl.DataFrame({
        "device_id": ["sensor_A"],
        "timestamp": [datetime(2024, 1, 1, 10, 15, 0)],
        "target_anomaly": [1]
    })

    # Feature updates for Device A:
    # Reading 1 at 10:00 (temp=20)
    # Reading 2 at 10:10 (temp=25) -> THIS should be matched (latest prior to 10:15)
    # Reading 3 at 10:20 (temp=80) -> FUTURE event, must NOT be leaked!
    features_df = pl.DataFrame({
        "device_id": ["sensor_A", "sensor_A", "sensor_A"],
        "timestamp": [
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 1, 1, 10, 10, 0),
            datetime(2024, 1, 1, 10, 20, 0)
        ],
        "temperature": [20.0, 25.0, 80.0]
    })

    merged = point_in_time_join(events_df, features_df)

    assert len(merged) == 1
    # Matched temperature must be 25.0 (from 10:10), NEVER 80.0 (from 10:20)
    assert merged["temperature"][0] == 25.0
