import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (
        np.abs(y_true) + np.abs(y_pred)
    )

    mask = denominator != 0

    return (
        100 * np.mean(2 * np.abs(y_pred[mask] -
                      y_true[mask]) / denominator[mask])
    )


def evaluate_predictions(y_true, y_pred):
    """Return the main regression evaluation metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    smape_value = smape(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_value,
    }


def print_metrics(name, metrics):
    """Print metrics in a consistent format."""

    print(f"\n{name}")
    print("-" * len(name))

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
