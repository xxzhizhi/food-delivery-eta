"""Train/validation/test data splitting strategies."""

import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_by_random(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random stratified split into train/val/test."""
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, random_state=random_state
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_by_time(
    df: pd.DataFrame,
    time_col: str = "created_at",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — prevents data leakage from future orders."""
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    logger.info(
        "Time-based split — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
