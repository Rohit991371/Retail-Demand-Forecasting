from pathlib import Path
import pandas as pd
from src.data.load_data import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "model_data.csv"

GROUP_COLUMNS = ["Store ID", "Product ID"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe features for next-day demand forecasting.
    
    Each row represents information available at the end of day t.
    The target is Demand at day t+1.
    """
    
    df = df.copy()
    
    ## 1. Basic Preparation
    df["Date"] = pd.to_datetime(df["Date"])
    
    df = df.sort_values(
        GROUP_COLUMNS + ["Date"]
    ).reset_index(drop=True)
    
    
    ## 2. Validate the forcasting grain
    duplicate_count = df.duplicated(
        subset=GROUP_COLUMNS + ["Date"]
    ).sum()
    
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate Store/Product/Date rows"
        )
        
    
    ## 3. Check whether observations are daily
    date_diff = (
        df.groupby(GROUP_COLUMNS)["Date"]
        .diff()
        .dt.days
    )
    
    invalid_intervals = date_diff.dropna()[date_diff.dropna() != 1]
    
    if not invalid_intervals.empty:
        print(
            "Warning: some Store/Product series contain missing dates or irregular intervals."
        )
    
    
    ## 4. Create next-day target
    df["target"] = (
        df.groupby(GROUP_COLUMNS)["Demand"]
        .shift(-1)
    )
    
    ## 5. Demand History
    for lag in [1, 7, 14, 28]:
        df[f"demand_lag_{lag}"] = (
            df.groupby(GROUP_COLUMNS)["Demand"]
            .shift(lag)
        )
        
    ## 6. Sales history
    for lag in [1, 7, 14]:
        df[f"units_sold_lag_{lag}"] = (
            df.groupby(GROUP_COLUMNS)["Units Sold"]
            .shift(lag)
        )
        
    
    ## 7. Inventory history
    for lag in [1, 7]:
        df[f"inventory_lag_{lag}"] = (
            df.groupby(GROUP_COLUMNS)["Inventory Level"]
            .shift(lag)
        )
        
    
    ## 8. Order history
    for lag in [1, 7]:
        df[f"orders_lag_{lag}"] = (
            df.groupby(GROUP_COLUMNS)["Units Ordered"]
            .shift(lag)
        )
        
        
    ## 9. Rolling demand statistics
    # We shift first so that the rolling windows contain only previously observed demand values.
    
    grouped_demand = df.groupby(GROUP_COLUMNS)["Demand"]
    
    df["demand_rolling_mean_7"] = (
        grouped_demand
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    
    df["demand_rolling_std_7"] = (
        grouped_demand
        .transform(lambda x: x.shift(1).rolling(7).std())
    )
    
    df["demand_rolling_mean_14"] = (
        grouped_demand
        .transform(lambda x: x.shift(1).rolling(14).mean())
    )
    
    df["demand_rolling_mean_28"] = (
        grouped_demand
        .transform(lambda x: x.shift(1).rolling(28).std())
    )
    
    
    ## 10. Calender features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_month"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    
    
    ## 11. Price relationship features
    
    df["price_difference"] = (
        df["Price"] - df["Competitor Pricing"]
    )
    
    df["price_ratio"] = (
        df["Price"] / df["Competitor Pricing"].replace(0, pd.NA)
    )
    
    
    ## 12. Drop rows that cannot have complete historical features
    required_history_columns = [
        "target",
        "demand_lag_1",
        "demand_lag_7",
        "demand_lag_14",
        "demand_lag_28",
        "units_sold_lag_1",
        "units_sold_lag_7",
        "units_sold_lag_14",
        "inventory_lag_1",
        "inventory_lag_7",
        "orders_lag_1",
        "orders_lag_7",
        "demand_rolling_mean_7",
        "demand_rolling_std_7",
        "demand_rolling_mean_14",
        "demand_rolling_mean_28",
    ]
    
    before_drop = len(df)
    
    df = df.dropna(
        subset = required_history_columns
    ).reset_index(drop=True)

    rows_removed = before_drop - len(df)
    
    print(f"Rows before feature cleanup: {before_drop}")
    print(f"Rows removed: {rows_removed}")
    print(f"Rows after feature cleanup: {len(df)}")

    return df


def save_features(df: pd.DataFrame) -> None:
    """ Save the processed modelling dataset."""
    PROCESSED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    df.to_csv(
        PROCESSED_PATH,
        index=False
    )
    
    print(f"Processed dataset saved to: {PROCESSED_PATH}")


if __name__ == "__main__":
    raw_df = load_data()

    model_df = build_features(raw_df)

    print("\nFinal feature dataset:")
    print(model_df.shape)

    print("\nFeature columns:")
    for column in model_df.columns:
        print(f"  - {column}")

    save_features(model_df)