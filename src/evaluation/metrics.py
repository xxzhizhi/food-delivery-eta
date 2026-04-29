"""Core evaluation metrics for ETA prediction."""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Filters zero targets."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def percentile_error(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """P-th percentile of absolute errors (minutes)."""
    abs_errors = np.abs(y_true - y_pred)
    return float(np.percentile(abs_errors, p))


def on_time_rate(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance_min: float = 5.0
) -> float:
    """Percentage of predictions within ±tolerance of actual (%)."""
    within = np.abs(y_true - y_pred) <= tolerance_min
    return float(np.mean(within) * 100)


def over_prediction_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of predictions that overestimate delivery time (%)."""
    return float(np.mean(y_pred > y_true) * 100)


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (positive = over-prediction bias)."""
    return float(np.mean(y_pred - y_true))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance_min: float = 5.0,
) -> dict[str, float]:
    """Compute all standard ETA evaluation metrics."""
    return {
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "p50_error": percentile_error(y_true, y_pred, 50),
        "p80_error": percentile_error(y_true, y_pred, 80),
        "p90_error": percentile_error(y_true, y_pred, 90),
        "p95_error": percentile_error(y_true, y_pred, 95),
        "on_time_rate": on_time_rate(y_true, y_pred, tolerance_min),
        "over_prediction_rate": over_prediction_rate(y_true, y_pred),
        "mean_bias": mean_bias(y_true, y_pred),
        "n_samples": len(y_true),
    }
