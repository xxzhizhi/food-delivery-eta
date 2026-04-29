"""Simple baseline models for ETA prediction."""

import numpy as np
import pandas as pd


class MeanBaseline:
    """Predicts the global mean delivery time."""

    def __init__(self):
        self.mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanBaseline":
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class MedianBaseline:
    """Predicts the global median delivery time."""

    def __init__(self):
        self.median_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MedianBaseline":
        self.median_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.median_)


class BucketMedianBaseline:
    """Predicts the median delivery time per hour-of-day bucket."""

    def __init__(self, bucket_col: str = "hour"):
        self.bucket_col = bucket_col
        self.bucket_medians_: dict[int, float] = {}
        self.global_median_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BucketMedianBaseline":
        self.global_median_ = float(y.median())
        if self.bucket_col in X.columns:
            combined = X[[self.bucket_col]].copy()
            combined["_target"] = y.values
            self.bucket_medians_ = (
                combined.groupby(self.bucket_col)["_target"].median().to_dict()
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.bucket_col in X.columns:
            return X[self.bucket_col].map(self.bucket_medians_).fillna(
                self.global_median_
            ).values
        return np.full(len(X), self.global_median_)
