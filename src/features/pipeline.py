"""Feature pipeline: orchestrates all feature engineering steps."""

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.features.contextual import (
    add_interaction_features,
    add_order_features,
    add_supply_demand_features,
)
from src.features.spatial import add_spatial_features
from src.features.temporal import add_temporal_features

logger = logging.getLogger(__name__)

# Columns to exclude from model input
NON_FEATURE_COLS = [
    "order_id",
    "created_at",
    "actual_delivery_time",
    "delivery_duration_minutes",
    "distance_bucket",
    "demand_pressure",
    "order_size_bucket",
]


@dataclass
class FeaturePipeline:
    """End-to-end feature engineering pipeline.

    Usage:
        pipeline = FeaturePipeline()
        X_train, y_train = pipeline.fit_transform(train_df)
        X_test, y_test = pipeline.transform(test_df)
    """

    target_col: str = "delivery_duration_minutes"
    time_col: str = "created_at"
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = add_temporal_features(df, self.time_col)
        df = add_spatial_features(df)
        df = add_supply_demand_features(df)
        df = add_order_features(df)
        df = add_interaction_features(df)
        return df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select numeric feature columns, excluding metadata and target."""
        exclude = set(NON_FEATURE_COLS)
        feature_cols = [
            c
            for c in df.columns
            if c not in exclude and df[c].dtype in ("int8", "int64", "float64")
        ]
        return df[feature_cols]

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit the pipeline on training data and return features + target."""
        df = self._engineer_features(df.copy())
        X = self._select_features(df)
        self.feature_names_ = list(X.columns)

        y = df[self.target_col]
        logger.info(
            "Feature pipeline fitted: %d features, %d samples",
            len(self.feature_names_),
            len(X),
        )
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transform new data using the fitted pipeline."""
        df = self._engineer_features(df.copy())
        X = df[self.feature_names_]
        y = df[self.target_col]
        return X, y

    def transform_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data without requiring target (for inference)."""
        df = self._engineer_features(df.copy())
        return df[self.feature_names_]
