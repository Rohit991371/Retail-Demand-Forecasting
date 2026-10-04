import pandas as pd
from src.evaluation.evaluate import evaluate_predictions, print_metrics


def evaluate_baselines(df, test_start_date):
    """
    Evaluate simple forecasting strategies on the test period.

    Baselines:
    - Previous observed demand
    - Same day previous week
    - Previous 7-day average
    """

    test_df = df[
        df["Date"] >= test_start_date
    ].copy()

    # -------------------------------------------
    # Baseline 1: Tomorrow's demand = today's demand
    # -------------------------------------------

    test_df["prediction_lag_1"] = (
        test_df["Demand"]
    )

    # -------------------------------------------------
    # Baseline 2: Tomorrow's demand = demand 7 days ago
    # -------------------------------------------------

    test_df["prediction_lag_7"] = (
        test_df["demand_lag_7"]
    )

    # --------------------------------------------------------------
    # Baseline 3: Tomorrow's demand = historical 7-day average
    # --------------------------------------------------------------
    test_df["prediction_rolling_7"] = (
        test_df["demand_rolling_mean_7"]
    )

    results = {}

    for name, prediction_column in [
        ("Naive - Previous Day", "prediction_lag_1"),
        ("Naive - Previous Week", "prediction_lag_7"),
        ("Naive - 7 Day Mean", "prediction_rolling_7"),
    ]:
        valid = test_df[["target", prediction_column]].dropna()

        metrics = evaluate_predictions(
            valid["target"],
            valid[prediction_column]
        )

        results[name] = metrics

        print_metrics(name, metrics)

    return results
