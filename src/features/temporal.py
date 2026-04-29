"""Temporal feature engineering."""

import numpy as np
import pandas as pd

# Peak hours for food delivery (lunch and dinner)
LUNCH_PEAK = (11, 13)  # 11:00 - 13:59
DINNER_PEAK = (17, 20)  # 17:00 - 20:59


def add_temporal_features(df: pd.DataFrame, time_col: str = "created_at") -> pd.DataFrame:
    """Extract time-based features from order timestamp."""
    if time_col not in df.columns:
        return df

    ts = df[time_col]

    df["hour"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month

    # Peak hour flags
    df["is_lunch_peak"] = df["hour"].between(*LUNCH_PEAK).astype(np.int8)
    df["is_dinner_peak"] = df["hour"].between(*DINNER_PEAK).astype(np.int8)
    df["is_peak_hour"] = (df["is_lunch_peak"] | df["is_dinner_peak"]).astype(np.int8)

    # Cyclical encoding for hour (captures 23→0 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclical encoding for day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df
