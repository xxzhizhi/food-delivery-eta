"""Contextual features: supply-demand, order complexity, interactions."""

import numpy as np
import pandas as pd


def add_supply_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rider supply-demand balance features."""
    riders_on = df.get("total_onshift_riders")
    riders_busy = df.get("total_busy_riders")
    outstanding = df.get("total_outstanding_orders")

    if riders_on is not None and riders_busy is not None:
        available = (riders_on - riders_busy).clip(lower=0)
        df["available_riders"] = available
        df["rider_utilization"] = (
            riders_busy / riders_on.replace(0, np.nan)
        ).fillna(1.0)

    if outstanding is not None and riders_on is not None:
        df["orders_per_rider"] = (
            outstanding / riders_on.replace(0, np.nan)
        ).fillna(0.0)

    if riders_on is not None and riders_busy is not None and outstanding is not None:
        available = (riders_on - riders_busy).clip(lower=0)
        df["supply_demand_ratio"] = (
            available / outstanding.replace(0, np.nan)
        ).fillna(1.0)

        # Demand pressure buckets
        df["demand_pressure"] = pd.cut(
            df["supply_demand_ratio"],
            bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=["high_pressure", "medium_pressure", "balanced", "oversupply"],
        )

    return df


def add_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from order details."""
    if "total_items" in df.columns:
        df["order_size_bucket"] = pd.cut(
            df["total_items"],
            bins=[0, 2, 5, np.inf],
            labels=["small", "medium", "large"],
        )

    if "subtotal" in df.columns and "total_items" in df.columns:
        df["avg_item_price"] = (
            df["subtotal"] / df["total_items"].replace(0, np.nan)
        ).fillna(0.0)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature interactions that capture combined effects."""
    if "haversine_distance_km" in df.columns:
        if "is_peak_hour" in df.columns:
            df["distance_x_peak"] = df["haversine_distance_km"] * df["is_peak_hour"]

        if "rider_utilization" in df.columns:
            df["distance_x_utilization"] = (
                df["haversine_distance_km"] * df["rider_utilization"]
            )

    if "total_items" in df.columns and "rider_utilization" in df.columns:
        df["items_x_utilization"] = df["total_items"] * df["rider_utilization"]

    return df
