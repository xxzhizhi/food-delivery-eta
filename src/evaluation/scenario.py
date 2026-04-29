"""Multi-scenario evaluation: slice model performance by real-world conditions.

This is the most distinctive part of this project — informed by 6 years of
evaluating ETA models at Alibaba's Ele.me platform. Single aggregate metrics
(e.g. overall MAE) can hide critical failure modes. This module evaluates
model performance across 30+ scenario slices to expose weaknesses.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario_name: str
    slice_value: str
    metrics: dict[str, float]


@dataclass
class ScenarioEvaluator:
    """Evaluates model predictions across multiple scenario slices."""

    tolerance_min: float = 5.0

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features_df: pd.DataFrame,
    ) -> list[ScenarioResult]:
        """Run full scenario-based evaluation."""
        results: list[ScenarioResult] = []

        # Overall
        overall = compute_all_metrics(y_true, y_pred, self.tolerance_min)
        results.append(ScenarioResult("overall", "all", overall))

        # Distance slices
        results.extend(
            self._evaluate_by_column(
                y_true, y_pred, features_df, "distance_bucket", "distance"
            )
        )

        # Time slices
        if "hour" in features_df.columns:
            time_buckets = self._create_time_buckets(features_df["hour"])
            results.extend(
                self._evaluate_by_series(y_true, y_pred, time_buckets, "time_period")
            )

        # Weekend vs weekday
        if "is_weekend" in features_df.columns:
            labels = features_df["is_weekend"].map({0: "weekday", 1: "weekend"})
            results.extend(
                self._evaluate_by_series(y_true, y_pred, labels, "day_type")
            )

        # Peak vs off-peak
        if "is_peak_hour" in features_df.columns:
            labels = features_df["is_peak_hour"].map({0: "off_peak", 1: "peak"})
            results.extend(
                self._evaluate_by_series(y_true, y_pred, labels, "peak_status")
            )

        # Supply-demand pressure
        results.extend(
            self._evaluate_by_column(
                y_true, y_pred, features_df, "demand_pressure", "demand"
            )
        )

        # Order size
        results.extend(
            self._evaluate_by_column(
                y_true, y_pred, features_df, "order_size_bucket", "order_size"
            )
        )

        # Rider utilization quartiles
        if "rider_utilization" in features_df.columns:
            quartiles = pd.qcut(
                features_df["rider_utilization"],
                q=4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            )
            results.extend(
                self._evaluate_by_series(y_true, y_pred, quartiles, "utilization")
            )

        logger.info("Scenario evaluation complete: %d slices", len(results))
        return results

    def _evaluate_by_column(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: pd.DataFrame,
        col: str,
        scenario_name: str,
    ) -> list[ScenarioResult]:
        """Evaluate by an existing categorical column."""
        results = []
        if col not in df.columns:
            return results

        for value in df[col].dropna().unique():
            mask = df[col] == value
            if mask.sum() < 10:
                continue
            metrics = compute_all_metrics(
                y_true[mask.values], y_pred[mask.values], self.tolerance_min
            )
            results.append(ScenarioResult(scenario_name, str(value), metrics))
        return results

    def _evaluate_by_series(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: pd.Series,
        scenario_name: str,
    ) -> list[ScenarioResult]:
        """Evaluate by a label series."""
        results = []
        for value in labels.dropna().unique():
            mask = labels == value
            if mask.sum() < 10:
                continue
            metrics = compute_all_metrics(
                y_true[mask.values], y_pred[mask.values], self.tolerance_min
            )
            results.append(ScenarioResult(scenario_name, str(value), metrics))
        return results

    @staticmethod
    def _create_time_buckets(hour_series: pd.Series) -> pd.Series:
        """Map hours to semantic time periods."""
        def _bucket(h: int) -> str:
            if 6 <= h < 11:
                return "morning"
            elif 11 <= h < 14:
                return "lunch_peak"
            elif 14 <= h < 17:
                return "afternoon"
            elif 17 <= h < 21:
                return "dinner_peak"
            else:
                return "late_night"

        return hour_series.map(_bucket)
