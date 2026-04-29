"""Download sample datasets for development and testing.

Usage:
    python scripts/download_data.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw")


def generate_synthetic_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic food delivery data for development.

    This creates data with realistic patterns: peak hour effects,
    distance-duration correlation, supply-demand dynamics, etc.
    """
    rng = np.random.default_rng(seed)

    # Base coordinates (Shanghai area)
    base_lat, base_lon = 31.23, 121.47

    store_lat = base_lat + rng.normal(0, 0.05, n_samples)
    store_lon = base_lon + rng.normal(0, 0.05, n_samples)
    delivery_lat = store_lat + rng.normal(0, 0.02, n_samples)
    delivery_lon = store_lon + rng.normal(0, 0.02, n_samples)

    # Timestamps over 30 days
    start = pd.Timestamp("2026-03-01")
    random_seconds = rng.integers(0, 30 * 24 * 3600, n_samples)
    created_at = start + pd.to_timedelta(random_seconds, unit="s")

    hours = created_at.hour

    # Distance in km (approximate)
    distance_km = np.sqrt(
        ((delivery_lat - store_lat) * 111) ** 2
        + ((delivery_lon - store_lon) * 111 * np.cos(np.radians(base_lat))) ** 2
    )

    # Rider supply-demand
    total_onshift = rng.integers(5, 30, n_samples)
    peak_mask = ((hours >= 11) & (hours <= 13)) | ((hours >= 17) & (hours <= 20))
    total_busy = np.where(
        peak_mask,
        (total_onshift * rng.uniform(0.7, 0.95, n_samples)).astype(int),
        (total_onshift * rng.uniform(0.3, 0.6, n_samples)).astype(int),
    )
    total_outstanding = np.where(
        peak_mask,
        rng.integers(15, 50, n_samples),
        rng.integers(5, 20, n_samples),
    )

    # Order details
    total_items = rng.integers(1, 8, n_samples)
    num_distinct_items = np.minimum(total_items, rng.integers(1, 5, n_samples))
    subtotal = total_items * rng.uniform(8, 25, n_samples)

    # Target: delivery duration (minutes) with realistic patterns
    base_time = 15.0  # base preparation + pickup time
    distance_time = distance_km * 4.0  # ~4 min per km
    peak_delay = np.where(peak_mask, rng.uniform(3, 8, n_samples), 0)
    demand_delay = np.maximum(0, (total_busy / total_onshift - 0.5) * 10)
    item_delay = (total_items - 1) * 0.5
    noise = rng.normal(0, 3, n_samples)

    duration_minutes = (
        base_time + distance_time + peak_delay + demand_delay + item_delay + noise
    )
    duration_minutes = np.clip(duration_minutes, 8, 90)

    actual_delivery_time = created_at + pd.to_timedelta(duration_minutes, unit="m")

    df = pd.DataFrame(
        {
            "order_id": [f"ORD_{i:06d}" for i in range(n_samples)],
            "created_at": created_at,
            "actual_delivery_time": actual_delivery_time,
            "store_latitude": store_lat,
            "store_longitude": store_lon,
            "delivery_latitude": delivery_lat,
            "delivery_longitude": delivery_lon,
            "total_items": total_items,
            "subtotal": np.round(subtotal, 2),
            "num_distinct_items": num_distinct_items,
            "total_onshift_riders": total_onshift,
            "total_busy_riders": total_busy,
            "total_outstanding_orders": total_outstanding,
        }
    )

    return df


def main() -> None:
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic delivery data...")
    df = generate_synthetic_data(n_samples=50000)

    output_path = OUTPUT_DIR / "delivery_data.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved %d records to %s", len(df), output_path)

    # Print summary statistics
    duration = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds() / 60
    logger.info(
        "Duration stats — mean: %.1f min, median: %.1f min, std: %.1f min",
        duration.mean(),
        duration.median(),
        duration.std(),
    )


if __name__ == "__main__":
    main()
