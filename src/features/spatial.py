"""Spatial and geographic feature engineering."""

import numpy as np
import pandas as pd

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute great-circle distance between two points on Earth (km)."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return EARTH_RADIUS_KM * c


def manhattan_distance_approx(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Approximate Manhattan distance using lat/lon differences (km)."""
    dlat_km = np.abs(lat2 - lat1) * 111.0
    avg_lat = np.radians((lat1 + lat2) / 2)
    dlon_km = np.abs(lon2 - lon1) * 111.0 * np.cos(avg_lat)
    return dlat_km + dlon_km


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance and geographic features."""
    lat1 = df["store_latitude"].values
    lon1 = df["store_longitude"].values
    lat2 = df["delivery_latitude"].values
    lon2 = df["delivery_longitude"].values

    df["haversine_distance_km"] = haversine_distance(lat1, lon1, lat2, lon2)
    df["manhattan_distance_km"] = manhattan_distance_approx(lat1, lon1, lat2, lon2)

    # Distance buckets for scenario evaluation
    df["distance_bucket"] = pd.cut(
        df["haversine_distance_km"],
        bins=[0, 2, 5, 10, np.inf],
        labels=["short", "medium", "long", "very_long"],
    )

    # Bearing (direction from store to delivery)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(np.radians(lat2))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - np.sin(
        np.radians(lat1)
    ) * np.cos(np.radians(lat2)) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(x, y)) % 360

    return df
