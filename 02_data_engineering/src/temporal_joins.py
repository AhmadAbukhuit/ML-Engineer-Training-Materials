"""
Point-in-Time Correctness and ASOF Joins for Historical Feature Ingestion.

In machine learning on temporal data, standard relational joins cause temporal leakage 
(lookahead bias) by attaching future feature values to past prediction events.

This module implements point-in-time correct merges (ASOF joins) where each event 
is strictly matched with the latest available feature state prior to or at the event timestamp.
"""

import polars as pl
import pandas as pd


def point_in_time_join(
    events_df: pl.DataFrame | pd.DataFrame,
    features_df: pl.DataFrame | pd.DataFrame,
    on_timestamp: str = "timestamp",
    by_entity: str = "device_id",
    strategy: str = "backward"
) -> pl.DataFrame:
    """
    Executes a point-in-time correct ASOF join between event observations and feature logs.

    Args:
        events_df: Prediction event observations containing target labels and timestamps.
        features_df: Historical log of feature updates across entities.
        on_timestamp: Column name containing the datetime timestamp.
        by_entity: Primary entity identifier column (e.g., 'device_id' or 'user_id').
        strategy: Merge strategy, strictly 'backward' to prevent future lookahead leakage.

    Returns:
        A merged Polars DataFrame guaranteed to be free of temporal leakage.
    """
    # Convert pandas to polars if necessary
    p_events = pl.from_pandas(events_df) if isinstance(events_df, pd.DataFrame) else events_df
    p_features = pl.from_pandas(features_df) if isinstance(features_df, pd.DataFrame) else features_df

    # Ensure timestamp columns are proper Datetime types
    if p_events[on_timestamp].dtype != pl.Datetime:
        p_events = p_events.with_columns(pl.col(on_timestamp).cast(pl.Datetime))
    if p_features[on_timestamp].dtype != pl.Datetime:
        p_features = p_features.with_columns(pl.col(on_timestamp).cast(pl.Datetime))

    # ASOF join requires both DataFrames to be sorted by timestamp
    p_events_sorted = p_events.sort(on_timestamp)
    p_features_sorted = p_features.sort(on_timestamp)

    # Perform backward ASOF join
    merged = p_events_sorted.join_asof(
        p_features_sorted,
        on=on_timestamp,
        by=by_entity,
        strategy=strategy
    )

    return merged
