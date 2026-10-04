from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales_data.csv"

REQUIRED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Weather Condition",
    "Promotion",
    "Competitor Pricing",
    "Seasonality",
    "Epidemic",
    "Demand",
]

def load_data() -> pd.DataFrame:
    """Load and perform basic validation on the raw dataset."""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )
        
    df = pd.read_csv(DATA_PATH)
    
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    return df


if __name__ == "__main__":
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"Unique stores: {df['Store ID'].nunique()}")
    print(f"Unique products: {df['Product ID'].nunique()}")