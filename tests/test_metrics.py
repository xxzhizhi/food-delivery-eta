"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mae,
    mape,
    mean_bias,
    on_time_rate,
    over_prediction_rate,
    percentile_error,
    rmse,
)


@pytest.fixture()
def predictions() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([20.0, 30.0, 25.0, 40.0, 15.0])
    y_pred = np.array([22.0, 28.0, 27.0, 38.0, 18.0])
    return y_true, y_pred


class TestMetrics:
    def test_mae(self, predictions: tuple) -> None:
        y_true, y_pred = predictions
        assert mae(y_true, y_pred) == pytest.approx(2.4, abs=0.01)

    def test_rmse_ge_mae(self, predictions: tuple) -> None:
        y_true, y_pred = predictions
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_mape_positive(self, predictions: tuple) -> None:
        y_true, y_pred = predictions
        assert mape(y_true, y_pred) > 0

    def test_percentile_error(self, predictions: tuple) -> None:
        y_true, y_pred = predictions
        p50 = percentile_error(y_true, y_pred, 50)
        p90 = percentile_error(y_true, y_pred, 90)
        assert p90 >= p50

    def test_on_time_rate(self, predictions: tuple) -> None:
        y_true, y_pred = predictions
        rate = on_time_rate(y_true, y_pred, tolerance=5.0)
        assert 0 <= rate <= 1.0

    def test_perfect_predictions(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        assert mae(y, y) == pytest.approx(0.0)
        assert rmse(y, y) == pytest.approx(0.0)
        assert on_time_rate(y, y, tolerance=1.0) == pytest.approx(1.0)

    def test_mean_bias_direction(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_over = np.array([15.0, 25.0, 35.0])
        y_under = np.array([5.0, 15.0, 25.0])
        assert mean_bias(y_true, y_over) > 0
        assert mean_bias(y_true, y_under) < 0

    def test_over_prediction_rate(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 18.0, 35.0])
        rate = over_prediction_rate(y_true, y_pred)
        assert 0 <= rate <= 1.0

    def test_compute_all_returns_dict(self, predictions: tuple) -> None:
        y_true, y_pred = predictions
        metrics = compute_all_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "on_time_rate" in metrics
