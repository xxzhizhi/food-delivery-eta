"""Tests for model training and prediction."""

import numpy as np
import pytest

from src.models.baseline import BucketMedianBaseline, MeanBaseline, MedianBaseline
from src.models.trainer import train_and_evaluate


@pytest.fixture()
def training_data() -> tuple:
    rng = np.random.default_rng(42)
    n = 200
    import pandas as pd

    X = pd.DataFrame(
        {
            "distance_km": rng.uniform(0.5, 10, n),
            "hour": rng.integers(0, 24, n),
            "rider_utilization": rng.uniform(0, 1, n),
        }
    )
    y = pd.Series(15 + X["distance_km"] * 3 + rng.normal(0, 2, n), name="target")
    return X, y


class TestBaselines:
    def test_mean_baseline(self, training_data: tuple) -> None:
        X, y = training_data
        model = MeanBaseline()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.all(preds == pytest.approx(y.mean(), abs=0.01))

    def test_median_baseline(self, training_data: tuple) -> None:
        X, y = training_data
        model = MedianBaseline()
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(preds == pytest.approx(y.median(), abs=0.01))

    def test_bucket_baseline(self, training_data: tuple) -> None:
        X, y = training_data
        model = BucketMedianBaseline(bucket_col="hour")
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestTrainer:
    def test_train_and_evaluate(self, training_data: tuple) -> None:
        X, y = training_data
        model = MeanBaseline()
        result = train_and_evaluate(
            model, "mean_test", X, y, X, y, X, y
        )
        assert result.model_name == "mean_test"
        assert "mae" in result.metrics
        assert result.predictions is not None
        assert len(result.predictions) == len(X)
