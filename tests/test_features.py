"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from src.features.contextual import (
    add_interaction_features,
    add_order_features,
    add_supply_demand_features,
)
from src.features.pipeline import FeaturePipeline
from src.features.spatial import add_spatial_features, haversine_distance
from src.features.temporal import add_temporal_features


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a small sample dataframe for testing."""
    rng = np.random.default_rng(42)
    n = 100
    base_time = pd.Timestamp("2026-03-15 12:00:00")

    df = pd.DataFrame(
        {
            "created_at": pd.date_range(base_time, periods=n, freq="15min"),
            "store_latitude": 31.23 + rng.normal(0, 0.01, n),
            "store_longitude": 121.47 + rng.normal(0, 0.01, n),
            "delivery_latitude": 31.23 + rng.normal(0, 0.02, n),
            "delivery_longitude": 121.47 + rng.normal(0, 0.02, n),
            "total_items": rng.integers(1, 6, n),
            "subtotal": rng.uniform(10, 80, n).round(2),
            "num_distinct_items": rng.integers(1, 4, n),
            "total_onshift_riders": rng.integers(5, 25, n),
            "total_busy_riders": rng.integers(3, 20, n),
            "total_outstanding_orders": rng.integers(5, 40, n),
            "delivery_duration_minutes": rng.uniform(10, 50, n).round(1),
        }
    )
    return df


class TestTemporalFeatures:
    def test_adds_expected_columns(self, sample_df: pd.DataFrame) -> None:
        result = add_temporal_features(sample_df.copy())
        expected_cols = ["hour", "day_of_week", "is_weekend", "is_peak_lunch", "is_peak_dinner"]
        for col in expected_cols:
            assert col in result.columns

    def test_hour_range(self, sample_df: pd.DataFrame) -> None:
        result = add_temporal_features(sample_df.copy())
        assert result["hour"].between(0, 23).all()

    def test_cyclical_sin_cos(self, sample_df: pd.DataFrame) -> None:
        result = add_temporal_features(sample_df.copy())
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()


class TestSpatialFeatures:
    def test_haversine_known_distance(self) -> None:
        # Shanghai Bund to Pudong ~5 km
        d = haversine_distance(31.24, 121.49, 31.24, 121.54)
        assert 4.0 < d < 6.0

    def test_haversine_zero(self) -> None:
        d = haversine_distance(31.0, 121.0, 31.0, 121.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_adds_distance_columns(self, sample_df: pd.DataFrame) -> None:
        result = add_spatial_features(sample_df.copy())
        assert "distance_km" in result.columns
        assert "manhattan_distance_km" in result.columns
        assert result["distance_km"].ge(0).all()


class TestContextualFeatures:
    def test_supply_demand(self, sample_df: pd.DataFrame) -> None:
        result = add_supply_demand_features(sample_df.copy())
        assert "rider_utilization" in result.columns
        assert result["rider_utilization"].between(0, 1).all()

    def test_order_features(self, sample_df: pd.DataFrame) -> None:
        result = add_order_features(sample_df.copy())
        assert "avg_item_price" in result.columns
        assert result["avg_item_price"].gt(0).all()

    def test_interaction_features(self, sample_df: pd.DataFrame) -> None:
        df = add_spatial_features(sample_df.copy())
        df = add_supply_demand_features(df)
        df = add_temporal_features(df)
        result = add_interaction_features(df)
        assert "distance_x_peak" in result.columns


class TestFeaturePipeline:
    def test_fit_transform_returns_numeric(self, sample_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        X, y = pipeline.fit_transform(sample_df.copy())
        assert X.select_dtypes(include=np.number).shape == X.shape
        assert len(y) == len(X)

    def test_transform_same_columns(self, sample_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        X_train, _ = pipeline.fit_transform(sample_df.copy())
        X_test, _ = pipeline.transform(sample_df.copy())
        assert list(X_train.columns) == list(X_test.columns)
