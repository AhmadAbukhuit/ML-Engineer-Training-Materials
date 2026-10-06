"""
High-Performance Columnar Storage & DuckDB Analytical Engine.

This module demonstrates:
1. Converting row-oriented CSV/DataFrames to compressed Apache Parquet.
2. Executing fast SQL queries with Predicate and Projection Pushdown via DuckDB.
3. Computing SQL window functions directly over Parquet files.
"""

from pathlib import Path
import duckdb
import polars as pl
import pandas as pd


def export_to_parquet(
    df: pd.DataFrame | pl.DataFrame,
    output_path: Path,
    compression: str = "snappy"
) -> Path:
    """
    Exports a tabular DataFrame to a compressed Apache Parquet file
    using DuckDB's native high-performance engine (zero pyarrow dependency).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pl.DataFrame):
        df.write_parquet(output_path, compression=compression)
    else:
        conn = duckdb.connect(database=":memory:")
        conn.register("df_temp", df)
        conn.execute(f"COPY df_temp TO '{output_path!s}' (FORMAT PARQUET, CODEC '{compression}')")
        conn.close()
    return output_path


def query_parquet_with_duckdb(
    parquet_path: Path,
    min_temperature: float | None = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    Executes a SQL query directly on a Parquet file using DuckDB,
    leveraging projection pushdown and predicate pushdown.
    """
    conn = duckdb.connect(database=":memory:")
    
    query = f"SELECT * FROM read_parquet('{parquet_path!s}')"
    if min_temperature is not None:
        query += f" WHERE temperature >= {min_temperature}"
    query += f" LIMIT {limit}"

    result_df = conn.execute(query).df()
    conn.close()
    return result_df


def compute_rolling_features_duckdb(parquet_path: Path) -> pd.DataFrame:
    """
    Uses DuckDB SQL window functions to compute rolling statistics over time per device.
    """
    conn = duckdb.connect(database=":memory:")
    sql = f"""
    SELECT
        device_id,
        timestamp,
        temperature,
        -- 3-reading moving average per device
        AVG(temperature) OVER (
            PARTITION BY device_id
            ORDER BY timestamp
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_temp,
        -- Previous temperature for delta calculation
        LAG(temperature, 1) OVER (
            PARTITION BY device_id
            ORDER BY timestamp
        ) AS prev_temperature
    FROM read_parquet('{parquet_path!s}')
    ORDER BY device_id, timestamp;
    """
    result_df = conn.execute(sql).df()
    conn.close()
    return result_df
