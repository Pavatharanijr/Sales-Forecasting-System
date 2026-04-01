"""
Generate synthetic Superstore-like sales data and preprocess for forecasting.
Run: python data/prepare_data.py
"""
import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/superstore.csv"


def generate_dataset():
    os.makedirs("data/raw", exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"Dataset already exists at {RAW_PATH}")
        return

    print("Generating synthetic Superstore sales data...")
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(dates)

    # Realistic sales with trend + seasonality + noise
    trend = np.linspace(1000, 3000, n)
    seasonality = 500 * np.sin(2 * np.pi * np.arange(n) / 365)
    weekly = 200 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 150, n)
    sales = np.abs(trend + seasonality + weekly + noise)

    df = pd.DataFrame({"Order Date": dates, "Sales": sales})
    df.to_csv(RAW_PATH, index=False)
    print(f"Synthetic dataset saved at {RAW_PATH} ({n} rows)")


def engineer_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.groupby("Order Date")["Sales"].sum().reset_index()
    df = df.set_index("Order Date").resample(freq).sum().reset_index()
    df.columns = ["ds", "y"]
    df = df[df["y"] > 0].copy()

    df["lag_1"] = df["y"].shift(1)
    df["lag_2"] = df["y"].shift(2)
    df["lag_4"] = df["y"].shift(4)
    df["rolling_mean_4"] = df["y"].shift(1).rolling(4).mean()
    df["rolling_std_4"] = df["y"].shift(1).rolling(4).std()
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["year"] = df["ds"].dt.year
    df["trend"] = np.arange(len(df))
    return df.dropna()


def prepare(freq: str = "W"):
    os.makedirs("data/processed", exist_ok=True)
    raw = pd.read_csv(RAW_PATH, encoding="latin-1")
    processed = engineer_features(raw, freq)
    out = f"data/processed/sales_{freq}.csv"
    processed.to_csv(out, index=False)
    print(f"Saved {len(processed)} rows -> {out}")
    return processed


if __name__ == "__main__":
    generate_dataset()
    for f in ["W", "MS", "QS"]:
        prepare(f)
