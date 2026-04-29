"""Training orchestration: run experiments, compare models, save artifacts."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Protocol for any model that can fit and predict."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ExperimentResult:
    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray


def train_and_evaluate(
    model: Predictor,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ExperimentResult:
    """Train a model and evaluate on test set."""
    logger.info("Training model: %s", model_name)

    # Train with validation for early stopping (if supported)
    try:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    except TypeError:
        model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = compute_all_metrics(y_test.values, y_pred)

    logger.info("Model %s — MAE: %.3f, MAPE: %.2f%%", model_name, metrics["mae"], metrics["mape"])

    return ExperimentResult(
        model_name=model_name,
        metrics=metrics,
        predictions=y_pred,
    )


def save_experiment(
    result: ExperimentResult,
    model: Any,
    output_dir: str | Path,
) -> None:
    """Save model artifact and experiment metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / f"{result.model_name}_model.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    # Save metrics
    metrics_path = output_dir / f"{result.model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)
