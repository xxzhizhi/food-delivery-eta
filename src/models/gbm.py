"""Gradient Boosting models (LightGBM / XGBoost) for ETA prediction."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LightGBMModel:
    """LightGBM regression model wrapper."""

    params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    })
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    model_: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMModel":
        import lightgbm as lgb

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_set]

        callbacks = [lgb.log_evaluation(period=100)]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds)
            )

        self.model_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

        logger.info(
            "LightGBM trained: %d iterations, best MAE: %.4f",
            self.model_.best_iteration,
            self.model_.best_score.get("valid_1", {}).get("l1", float("nan")),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model_.predict(X, num_iteration=self.model_.best_iteration)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet.")
        return (
            pd.DataFrame(
                {
                    "feature": self.model_.feature_name(),
                    "importance": self.model_.feature_importance(importance_type="gain"),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, filepath: str) -> None:
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet.")
        self.model_.save_model(filepath)
        logger.info("Model saved to %s", filepath)

    def load(self, filepath: str) -> "LightGBMModel":
        import lightgbm as lgb
        self.model_ = lgb.Booster(model_file=filepath)
        logger.info("Model loaded from %s", filepath)
        return self


@dataclass
class XGBoostModel:
    """XGBoost regression model wrapper."""

    params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    })
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    model_: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostModel":
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        self.model_ = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
        )

        logger.info("XGBoost trained: %d iterations", self.model_.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model_.predict(xgb.DMatrix(X))

    def save(self, filepath: str) -> None:
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet.")
        self.model_.save_model(filepath)

    def load(self, filepath: str) -> "XGBoostModel":
        import xgboost as xgb
        self.model_ = xgb.Booster()
        self.model_.load_model(filepath)
        return self
