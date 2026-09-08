"""
=============================================================================
Phase 2: Data Cleaning, Feature Engineering & Target Variable Creation
Project: Machine Learning Techniques for Financial Data
Domain: Stock Market Next-Day Opening Price Prediction (Open_{t+1})
=============================================================================
Problem Statement: To design and implement a Machine Learning model that 
predicts the next-day opening price (Open_{t+1}) of a target equity asset 
by analyzing historical OHLCV data, inter-day momentum indicators, 
volatility metrics, and overnight market sentiment.
=============================================================================
"""

import os
import sys

# Ensure portable site-packages is in sys.path
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore
import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import] # type: ignore
import seaborn as sns  # pyrefly: ignore [missing-import] # type: ignore

# Set style for academic, publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load Raw Dataset from Phase 1
raw_csv_path = os.path.join("outputs", "tables", "raw_data.csv")
df = pd.read_csv(raw_csv_path)
print(f"[OK] Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Data Cleaning & Validation
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
df = df.ffill().bfill()
print(f"[OK] Chronological sorting verified and missing values checked.")

# 3. Feature Engineering aligned with Problem Statement

# A. Base OHLCV Features: Open, High, Low, Close, Volume
# (Retained as primary price anchor points)

# B. Inter-Day Momentum & Trend Indicators:
# 1. 5-Day Simple Moving Average of Opening Price
df['SMA_5_Open'] = df['Open'].rolling(window=5).mean()
# 2. 10-Day Simple Moving Average of Opening Price
df['SMA_10_Open'] = df['Open'].rolling(window=10).mean()
# 3. Daily Percentage Return (Inter-day closing price momentum)
df['Daily_Return'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

# 4. Multi-Day Return Momentum (1d, 2d, 3d, 5d)
for lag in [1, 2, 3, 5]:
    df[f'Return_{lag}d'] = df['Close'].pct_change(lag) * 100

# 5. Exponential Moving Averages & Trend Divergence (EMA 9, EMA 21)
df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['EMA_Spread'] = df['EMA_9'] - df['EMA_21']

# 6. Relative Strength Index (RSI 14)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-6)
df['RSI_14'] = 100 - (100 / (1 + rs))

# 7. Moving Average Convergence Divergence (MACD 12, 26, 9)
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# C. Volatility & Spread Metrics:
# 8. Daily Intraday Volatility Spread (High - Low)
df['Daily_Volatility'] = df['High'] - df['Low']
# 9. 10-Day Rolling Average Volatility Spread
df['Rolling_Volatility_10'] = df['Daily_Volatility'].rolling(window=10).mean()
# 10. Average True Range (ATR 14)
hl = df['High'] - df['Low']
hc = (df['High'] - df['Close'].shift(1)).abs()
lc = (df['Low'] - df['Close'].shift(1)).abs()
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df['ATR_14'] = tr.rolling(window=14).mean()

# D. Overnight & Intraday Market Sentiment Signals:
# 11. Intraday Closing Sentiment (Close - Open) -> Bullish/Bearish session pressure
df['Intraday_Sentiment'] = df['Close'] - df['Open']
# 12. Previous Session Overnight Gap (Open_t - Close_{t-1}) -> Pre-market gap persistence
df['Overnight_Gap'] = df['Open'] - df['Close'].shift(1)

# 4. Target Variable Creation: Next-Day Opening Price (Open_{t+1}) & Overnight Gap
# Shift Open column by -1 day (lead of 1)
df['Next_Day_Open'] = df['Open'].shift(-1)
df['Target_Gap'] = df['Next_Day_Open'] - df['Close']
df['Target_Direction'] = (df['Next_Day_Open'] > df['Close']).astype(int)

# 5. Handle Boundary NaNs (drop initial rolling window rows & final lead row)
pre_drop_len = len(df)
df = df.replace([np.inf, -np.inf], np.nan)
df_clean = df.dropna().reset_index(drop=True)
dropped_rows = pre_drop_len - len(df_clean)
print(f"\n[OK] Dropped {dropped_rows} boundary rows containing NaNs from lag/lead/rolling operations.")
print(f"Final Cleaned & Engineered Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

# Format Date back to YYYY-MM-DD string
df_clean['Date'] = df_clean['Date'].dt.strftime('%Y-%m-%d')

# 6. Save Processed Dataset
processed_csv_path = os.path.join("outputs", "tables", "processed_data.csv")
df_clean.to_csv(processed_csv_path, index=False)
print(f"[OK] Processed dataset saved to: {processed_csv_path}")

# 7. Display Sample Target Alignment Table
print("\n" + "="*80)
print("SAMPLE TARGET ALIGNMENT TABLE: NEXT-DAY OPENING PRICE (First 10 Rows)")
print("="*80)
preview_df = pd.DataFrame({
    'Date': df_clean['Date'].head(10).values,
    "Today's Open": [f"INR {float(x):,.2f}" for x in df_clean['Open'].head(10)],
    "Today's Close": [f"INR {float(x):,.2f}" for x in df_clean['Close'].head(10)],
    'Next-Day Open (Target)': [f"INR {float(x):,.2f}" for x in df_clean['Next_Day_Open'].head(10)],
    'Target Direction (1=Up)': df_clean['Target_Direction'].head(10).values
})
print(preview_df.to_string(index=False))

# 8. Feature Matrix (X) and Target Vector (y)
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5_Open', 'SMA_10_Open', 'Daily_Return',
    'Daily_Volatility', 'Rolling_Volatility_10',
    'Intraday_Sentiment', 'Overnight_Gap',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'EMA_Spread', 'Return_1d', 'Return_2d', 'Return_3d', 'Return_5d',
    'ATR_14'
]
X = df_clean[feature_cols]
y = df_clean['Next_Day_Open']

print("\n" + "="*80)
print("FEATURE MATRIX (X) & TARGET VECTOR (y) OVERVIEW")
print("="*80)
print(f"Feature Matrix X Shape: {X.shape} ({X.shape[1]} engineered features)")
print(f"Features: {list(X.columns)}")
print(f"Target Vector y Shape:  {y.shape} (Next-Day Open Price: Open_{{t+1}})")
print("\nFirst 5 Rows of Feature Matrix (X):")
print(X.head().round(2).to_string())

# 9. Correlation Heatmap Generation
plt.figure(figsize=(14, 11))
analysis_cols = feature_cols + ['Next_Day_Open']
corr_matrix = pd.DataFrame(
    np.corrcoef(df_clean[analysis_cols].values.astype(float), rowvar=False),
    index=analysis_cols,
    columns=analysis_cols
)

# Create Heatmap
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=1,
    linecolor='white',
    square=True,
    annot_kws={"size": 7, "weight": "bold"}
)

plt.title("Feature Correlation Heatmap with Target Variable (Next-Day Open)", fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=8, fontweight='bold')
plt.yticks(rotation=0, fontsize=8, fontweight='bold')
plt.tight_layout()

heatmap_path = os.path.join("outputs", "charts", "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[OK] Correlation heatmap generated and saved to: {heatmap_path}")

# Display Target Correlation Vector
print("\n" + "="*80)
print("PEARSON CORRELATION WITH TARGET VARIABLE (Next_Day_Open)")
print("="*80)
corr_with_target = pd.Series(corr_matrix['Next_Day_Open']).sort_values(ascending=False)
for feat, score in corr_with_target.items():
    print(f"{feat:<25}: {score:+.4f}")
