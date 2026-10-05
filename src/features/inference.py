import pandas as pd


def build_prediction_features(
    data: pd.DataFrame,
    store_id: str,
    product_id: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:

    df = data.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    history = df[
        (df["Store ID"] == store_id)
        & (df["Product ID"] == product_id)
    ].copy()

    if history.empty:
        raise ValueError(
            f"No historical data found for "
            f"store={store_id}, product={product_id}"
        )

    history = (
        history
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(history) < 28:
        raise ValueError(
            f"Not enough historical data for "
            f"store={store_id}, product={product_id}. "
            f"At least 28 observations are required."
        )

    latest = history.iloc[-1]

    latest_date = latest["Date"]

    # Start with the latest raw observation.
    row = latest.to_dict()

    # --------------------------------------------------
    # Demand lag features
    # --------------------------------------------------

    for lag in [1, 7, 14, 28]:
        row[f"demand_lag_{lag}"] = (
            history["Demand"].iloc[-lag]
        )

    # --------------------------------------------------
    # Units sold lag features
    # --------------------------------------------------

    for lag in [1, 7, 14]:
        row[f"units_sold_lag_{lag}"] = (
            history["Units Sold"].iloc[-lag]
        )

    # --------------------------------------------------
    # Inventory lag features
    # --------------------------------------------------

    for lag in [1, 7]:
        row[f"inventory_lag_{lag}"] = (
            history["Inventory Level"].iloc[-lag]
        )

    # --------------------------------------------------
    # Orders lag features
    #
    # IMPORTANT:
    # These names must match build_features.py exactly.
    # --------------------------------------------------

    for lag in [1, 7]:
        row[f"orders_lag_{lag}"] = (
            history["Units Ordered"].iloc[-lag]
        )

    # --------------------------------------------------
    # Rolling demand features
    # --------------------------------------------------

    demand_history = history["Demand"].shift(1)

    row["demand_rolling_mean_7"] = (
        demand_history.tail(7).mean()
    )

    row["demand_rolling_std_7"] = (
        demand_history.tail(7).std()
    )

    row["demand_rolling_mean_14"] = (
        demand_history.tail(14).mean()
    )

    row["demand_rolling_mean_28"] = (
        demand_history.tail(28).mean()
    )

    # --------------------------------------------------
    # Prediction date
    # --------------------------------------------------

    prediction_date = (
        latest_date + pd.Timedelta(days=1)
    )

    row["Date"] = prediction_date

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    row["day_of_week"] = prediction_date.dayofweek
    row["day_of_month"] = prediction_date.day
    row["week_of_year"] = (
        prediction_date.isocalendar().week
    )
    row["month"] = prediction_date.month
    row["quarter"] = prediction_date.quarter
    row["is_weekend"] = int(
        prediction_date.dayofweek >= 5
    )

    # --------------------------------------------------
    # Price features
    # --------------------------------------------------

    row["price_difference"] = (
        row["Price"]
        - row["Competitor Pricing"]
    )

    if row["Competitor Pricing"] != 0:
        row["price_ratio"] = (
            row["Price"]
            / row["Competitor Pricing"]
        )
    else:
        row["price_ratio"] = 0.0

    # --------------------------------------------------
    # DO NOT remove Demand.
    #
    # The trained model expects Demand as an input feature.
    #
    # Only target must be removed because target is what
    # we are trying to predict.
    # --------------------------------------------------

    row.pop("target", None)

    features = pd.DataFrame([row])

    return features, prediction_date