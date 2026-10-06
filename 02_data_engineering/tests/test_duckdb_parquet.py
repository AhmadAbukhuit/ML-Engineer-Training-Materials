import sys
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from duckdb_parquet import export_to_parquet, query_parquet_with_duckdb, compute_rolling_features_duckdb


def test_parquet_export_and_duckdb_query(tmp_path):
    df = pd.DataFrame({
        "device_id": ["dev_1", "dev_1", "dev_2", "dev_2"],
        "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:00", "2024-01-01 10:05"]),
        "temperature": [20.0, 85.0, 15.0, 90.0]
    })

    parquet_file = tmp_path / "test_telemetry.parquet"
    export_to_parquet(df, parquet_file)
    assert parquet_file.exists()

    # Query with predicate pushdown (temp >= 80.0)
    result = query_parquet_with_duckdb(parquet_file, min_temperature=80.0)
    assert len(result) == 2
    assert (result["temperature"] >= 80.0).all()


def test_rolling_window_duckdb(tmp_path):
    df = pd.DataFrame({
        "device_id": ["dev_1", "dev_1", "dev_1"],
        "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:10"]),
        "temperature": [10.0, 20.0, 30.0]
    })

    parquet_file = tmp_path / "rolling_test.parquet"
    export_to_parquet(df, parquet_file)

    rolling_res = compute_rolling_features_duckdb(parquet_file)
    assert "rolling_avg_temp" in rolling_res.columns
    assert "prev_temperature" in rolling_res.columns

    # 3-reading moving average for 3rd reading: (10 + 20 + 30) / 3 = 20.0
    assert rolling_res["rolling_avg_temp"].iloc[2] == 20.0
    # prev_temperature for 2nd reading should be 10.0
    assert rolling_res["prev_temperature"].iloc[1] == 10.0
