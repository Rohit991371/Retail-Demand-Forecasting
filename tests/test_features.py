import pandas as pd

from src.features.build_features import build_features


def test_target_is_next_day_demand():

    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-11",
            "2025-01-12",
            "2025-01-13",
            "2025-01-14",
            "2025-01-15",
            "2025-01-16",
            "2025-01-17",
            "2025-01-18",
            "2025-01-19",
            "2025-01-20",
            "2025-01-21",
            "2025-01-22",
            "2025-01-23",
            "2025-01-24",
            "2025-01-25",
            "2025-01-26",
            "2025-01-27",
            "2025-01-28",
            "2025-01-29",
            "2025-01-30",
        ]),
        "Store ID": ["S1"] * 30,
        "Product ID": ["P1"] * 30,
        "Category": ["A"] * 30,
        "Region": ["North"] * 30,
        "Inventory Level": [100] * 30,
        "Units Sold": [50] * 30,
        "Units Ordered": [50] * 30,
        "Price": [10.0] * 30,
        "Discount": [0] * 30,
        "Weather Condition": ["Clear"] * 30,
        "Promotion": [0] * 30,
        "Competitor Pricing": [10.0] * 30,
        "Seasonality": ["Normal"] * 30,
        "Epidemic": [0] * 30,
        "Demand": list(range(30)),
    })

    result = build_features(df)

    assert "target" in result.columns
    assert "demand_lag_1" in result.columns
    assert "demand_rolling_mean_7" in result.columns

    # After enough history exists:
    # today's target should equal tomorrow's demand.
    row = result.iloc[0]

    original_date = row["Date"]

    tomorrow_demand = df[
        df["Date"] == original_date + pd.Timedelta(days=1)
    ]["Demand"].iloc[0]

    assert row["target"] == tomorrow_demand
