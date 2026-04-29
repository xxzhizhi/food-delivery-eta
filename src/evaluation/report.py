"""Generate formatted evaluation reports."""

import pandas as pd

from src.evaluation.scenario import ScenarioResult


def results_to_dataframe(results: list[ScenarioResult]) -> pd.DataFrame:
    """Convert scenario results to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "scenario": r.scenario_name,
                "slice": r.slice_value,
                "n_samples": int(r.metrics.get("n_samples", 0)),
                "MAE": round(r.metrics.get("mae", 0), 2),
                "MAPE(%)": round(r.metrics.get("mape", 0), 1),
                "RMSE": round(r.metrics.get("rmse", 0), 2),
                "P80_err": round(r.metrics.get("p80_error", 0), 2),
                "P90_err": round(r.metrics.get("p90_error", 0), 2),
                "OnTime(%)": round(r.metrics.get("on_time_rate", 0), 1),
                "Bias": round(r.metrics.get("mean_bias", 0), 2),
            }
        )
    return pd.DataFrame(rows)


def print_report(results: list[ScenarioResult]) -> None:
    """Print a formatted evaluation report to stdout."""
    df = results_to_dataframe(results)
    print("\n" + "=" * 90)
    print("ETA MODEL EVALUATION REPORT")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")
