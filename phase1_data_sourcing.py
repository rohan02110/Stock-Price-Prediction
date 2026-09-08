"""
=============================================================================
Phase 1: Setup, Data Sourcing & Problem Confirmation
Project: Machine Learning Techniques for Financial Data
Domain: Stock Market Next-Day Closing Price Prediction
=============================================================================
"""

import os
import sys

# Ensure portable site-packages is in sys.path if not automatically loaded
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import yfinance as yf  # pyrefly: ignore [missing-import] # type: ignore
import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore

# Set random seed for reproducibility
np.random.seed(42)

# 1. Create Directory Hierarchy
dirs = [
    os.path.join("outputs", "charts"),
    os.path.join("outputs", "tables"),
    os.path.join("outputs", "models")
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
print("[OK] Output directories initialized: /outputs/charts, /outputs/tables, /outputs/models")

# 2. Define Parameters
TICKER = "RELIANCE.NS"
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

print(f"\nFetching historical OHLCV data for {TICKER} from {START_DATE} to {END_DATE} via yfinance...")

# 3. Download Raw OHLCV Data with Offline Fallback
raw_csv_path = os.path.join("outputs", "tables", "raw_data.csv")
raw_df = pd.DataFrame()
try:
    downloaded = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    if downloaded is not None and isinstance(downloaded, pd.DataFrame) and not downloaded.empty:
        ticker_data = downloaded.copy()
        if isinstance(ticker_data.columns, pd.MultiIndex):
            ticker_data.columns = [col[0] for col in ticker_data.columns]
        temp_df = ticker_data.reset_index()
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        temp_df = temp_df[[c for c in required_cols if c in temp_df.columns]]
        date_series = pd.Series(pd.to_datetime(temp_df['Date']))
        temp_df['Date'] = date_series.dt.strftime('%Y-%m-%d')
        if len(temp_df) > 50:
            raw_df = temp_df
            raw_df.to_csv(raw_csv_path, index=False)
            print(f"[OK] Raw dataset downloaded via yfinance and saved to {raw_csv_path}")
        elif os.path.exists(raw_csv_path):
            raw_df = pd.read_csv(raw_csv_path)
            print(f"[OK] Loaded existing raw dataset from {raw_csv_path}")
    elif os.path.exists(raw_csv_path):
        raw_df = pd.read_csv(raw_csv_path)
        print(f"[OK] Loaded existing raw dataset from {raw_csv_path}")
except Exception as e:
    if os.path.exists(raw_csv_path):
        raw_df = pd.read_csv(raw_csv_path)
        print(f"[NOTE] Network/yfinance notice ({e}). Loaded existing dataset from {raw_csv_path}")
    else:
        raise e

# Display Summary Information
print("\n" + "="*80)
print("RAW DATASET PREVIEW & METADATA")
print("="*80)
print(f"Dataset Shape: {raw_df.shape[0]} trading days, {raw_df.shape[1]} columns")
print(f"Date Range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")
print("\nData Types:")
print(pd.Series(raw_df.dtypes).to_string())

print("\n--- FIRST 5 ROWS ---")
print(raw_df.head().to_string(index=False))

print("\n--- LAST 5 ROWS ---")
print(raw_df.tail().to_string(index=False))

print("\n--- DESCRIPTIVE STATISTICS (.describe()) ---")
stats_df = pd.DataFrame(raw_df[['Open', 'High', 'Low', 'Close', 'Volume']])
print(stats_df.describe().round(2).to_string())
