"""End-to-end experiment runner.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
"""

import argparse
import logging
from pathlib import Path

import joblib

from src.data.loader import load_and_prepare
from src.data.splitter import split_by_random, split_by_time
from src.evaluation.report import print_report
from src.evaluation.scenario import ScenarioEvaluator
from src.features.pipeline import FeaturePipeline
from src.models.baseline import MeanBaseline, MedianBaseline
from src.models.gbm import LightGBMModel
from src.models.trainer import save_experiment, train_and_evaluate
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    setup_logging()
    config = load_config(config_path)

    # ---- Load Data ----
    logger.info("Loading data...")
    df = load_and_prepare(
        config["data"]["raw_path"],
        target_col=config["data"]["target_col"],
    )

    # ---- Split Data ----
    if config["data"]["split_strategy"] == "time":
        train_df, val_df, test_df = split_by_time(df)
    else:
        train_df, val_df, test_df = split_by_random(
            df,
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            random_state=config["data"]["random_state"],
        )

    # ---- Feature Engineering ----
    logger.info("Engineering features...")
    pipeline = FeaturePipeline(
        target_col=config["data"]["target_col"],
        time_col=config["features"]["time_col"],
    )
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val, y_val = pipeline.transform(val_df)
    X_test, y_test = pipeline.transform(test_df)

    # Save fitted pipeline for API deployment
    output_dir = Path(config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "feature_pipeline.pkl")

    # ---- Train Models ----
    results = []

    # Baselines
    for ModelClass, name in [
        (MeanBaseline, "mean_baseline"),
        (MedianBaseline, "median_baseline"),
    ]:
        model = ModelClass()
        result = train_and_evaluate(
            model, name, X_train, y_train, X_val, y_val, X_test, y_test
        )
        results.append((result, model))

    # LightGBM
    lgbm = LightGBMModel(
        params=config["model"]["params"],
        num_boost_round=config["model"]["num_boost_round"],
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
    )
    lgbm_result = train_and_evaluate(
        lgbm, "lightgbm", X_train, y_train, X_val, y_val, X_test, y_test
    )
    results.append((lgbm_result, lgbm))

    # ---- Save Best Model ----
    best_result, best_model = min(results, key=lambda x: x[0].metrics["mae"])
    save_experiment(best_result, best_model, output_dir)
    logger.info("Best model: %s (MAE=%.3f)", best_result.model_name, best_result.metrics["mae"])

    # ---- Scenario Evaluation (on best model) ----
    logger.info("Running scenario evaluation...")

    # Reconstruct features for test set (with categorical columns for slicing)
    evaluator = ScenarioEvaluator(
        tolerance_min=config["evaluation"]["tolerance_min"]
    )
    scenario_results = evaluator.evaluate(
        y_test.values, best_result.predictions, test_df
    )
    print_report(scenario_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETA prediction experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment config file",
    )
    args = parser.parse_args()
    main(args.config)
