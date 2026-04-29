"""Data loading and validation utilities."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "created_at",
    "actual_delivery_time",
    "store_latitude",
    "store_longitude",
    "delivery_latitude",
    "delivery_longitude",
    "total_items",
    "subtotal",
    "num_distinct_items",
    "total_onshift_riders",
    "total_busy_riders",
    "total_outstanding_orders",
]


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV file and parse datetime columns."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows from %s", len(df), filepath.name)

    # Auto-detect and parse datetime columns
    for col in ["created_at", "actual_delivery_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the dataframe has required columns and reasonable values."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_before = len(df)

    # Drop rows with null target
    if "actual_delivery_time" in df.columns and "created_at" in df.columns:
        df = df.dropna(subset=["actual_delivery_time", "created_at"])

    # Drop rows with invalid coordinates
    coord_cols = [
        "store_latitude",
        "store_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]
    for col in coord_cols:
        if col in df.columns:
            df = df[df[col].between(-180, 180)]

    # Drop rows with non-positive item count
    if "total_items" in df.columns:
        df = df[df["total_items"] > 0]

    n_after = len(df)
    if n_before != n_after:
        logger.warning("Dropped %d invalid rows during validation", n_before - n_after)

    return df.reset_index(drop=True)


def compute_target(df: pd.DataFrame, target_col: str = "delivery_duration_minutes") -> pd.DataFrame:
    """Compute target variable: delivery duration in minutes."""
    if "actual_delivery_time" in df.columns and "created_at" in df.columns:
        df[target_col] = (
            df["actual_delivery_time"] - df["created_at"]
        ).dt.total_seconds() / 60.0

        # Filter unreasonable durations (< 5 min or > 120 min)
        mask = df[target_col].between(5, 120)
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            logger.warning(
                "Dropped %d rows with unreasonable delivery duration", n_dropped
            )
        df = df[mask].reset_index(drop=True)

    return df


def load_and_prepare(
    filepath: str | Path,
    target_col: str = "delivery_duration_minutes",
) -> pd.DataFrame:
    """End-to-end data loading: load, validate, compute target."""
    df = load_csv(filepath)
    df = validate_dataframe(df)
    df = compute_target(df, target_col)
    logger.info("Data ready: %d rows, %d columns", len(df), len(df.columns))
    return df
