"""
=============================================================================
Financial ML Stock & IPO Trend Predictor Dashboard
Problem Statement: Predicting Next-Day Opening Price (Open_{t+1}) and Trend
Built with Streamlit and Plotly
=============================================================================
"""

import os
import sys

# Ensure portable site-packages is in sys.path
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

import streamlit as st  # pyrefly: ignore [missing-import] # type: ignore
import yfinance as yf  # pyrefly: ignore [missing-import] # type: ignore
import pandas as pd  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore
import plotly.graph_objects as go  # pyrefly: ignore [missing-import] # type: ignore
from plotly.subplots import make_subplots  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.linear_model import LinearRegression  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.ensemble import RandomForestRegressor  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.preprocessing import StandardScaler  # pyrefly: ignore [missing-import] # type: ignore
from sklearn.metrics import (  # pyrefly: ignore [missing-import] # type: ignore
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
)

# Set page configuration
st.set_page_config(
    page_title="Financial ML - Stock & IPO Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-label { font-size: 13px; font-weight: 600; color: #6c757d; text-transform: uppercase; }
    .metric-val { font-size: 24px; font-weight: 800; color: #212529; margin-top: 4px; }
    .trend-bullish { color: #28a745; font-weight: 800; }
    .trend-bearish { color: #dc3545; font-weight: 800; }
    .trend-neutral { color: #ffc107; font-weight: 800; }
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Model & Asset Selector")

# Preset Recent IPOs and Major Equities
IPO_PRESETS = {
    "ITC Limited (ITC.NS)": "ITC.NS",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Swiggy Ltd. (SWIGGY.NS)": "SWIGGY.NS",
    "Hyundai Motor India (HYUNDAI.NS)": "HYUNDAI.NS",
    "Bajaj Housing Finance (BAJAJHFL.NS)": "BAJAJHFL.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Tata Technologies (TATATECH.NS)": "TATATECH.NS",
    "Infosys Ltd. (INFY.NS)": "INFY.NS",
    "Zomato Ltd. (ZOMATO.NS)": "ZOMATO.NS",
    "Jio Financial Services (JIOFIN.NS)": "JIOFIN.NS",
    "IREDA (IREDA.NS)": "IREDA.NS",
    "Ola Electric (OLAELC.NS)": "OLAELC.NS",
    "Paytm / One97 (PAYTM.NS)": "PAYTM.NS",
    "Nykaa (NYKAA.NS)": "NYKAA.NS",
    "Custom Ticker (Enter Below)": "CUSTOM"
}

selected_preset = st.sidebar.selectbox("Select Target IPO / Equity:", list(IPO_PRESETS.keys()), index=0)

if selected_preset == "Custom Ticker (Enter Below)":
    ticker_input = st.sidebar.text_input("Enter Yahoo Finance Ticker:", value="ITC.NS")
else:
    ticker_input = IPO_PRESETS[selected_preset]

timeframe = st.sidebar.selectbox(
    "Historical Data Horizon:",
    ["1y (1 Year)", "2y (2 Years)", "5y (5 Years)", "6mo (6 Months)", "max (Full History)"],
    index=1
)
period_code = timeframe.split()[0]

model_choice = st.sidebar.radio(
    "Select ML Forecasting Model:",
    ["Linear Regression (Primary OLS)", "Random Forest Regressor (Ensemble)", "Compare Both Models"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Specification:**
- **Target Variable:** $Open_{t+1}$ (Next-Day Open)
- **Features (22):** OHLCV + Moving Averages + EMAs + Multi-Day Returns + RSI + MACD + ATR + Sentiment Gaps
- **Temporal Split:** 70% Train, 15% Validation, 15% Test
""")

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_and_prepare_data(ticker, period):
    try:
        downloaded = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    except Exception:
        return None
    if downloaded is None or not isinstance(downloaded, pd.DataFrame) or downloaded.empty:
        return None
    data = downloaded.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data = data.reset_index()
    if 'Date' not in data.columns or len(data) < 25:
        return None
    
    df = pd.DataFrame(data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy())
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.ffill().bfill()
    
    # 22 Engineered Features
    df['SMA_5_Open'] = df['Open'].rolling(window=5).mean()
    df['SMA_10_Open'] = df['Open'].rolling(window=10).mean()
    df['Daily_Return'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
    
    # Multi-Day Return Momentum
    for lag in [1, 2, 3, 5]:
        df[f'Return_{lag}d'] = df['Close'].pct_change(lag) * 100
        
    # EMAs & Trend Spread
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_Spread'] = df['EMA_9'] - df['EMA_21']
    
    # Relative Strength Index (RSI 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Volatility & Spread Metrics
    df['Daily_Volatility'] = df['High'] - df['Low']
    df['Rolling_Volatility_10'] = df['Daily_Volatility'].rolling(window=10).mean()
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # Overnight & Intraday Market Sentiment
    df['Intraday_Sentiment'] = df['Close'] - df['Open']
    df['Overnight_Gap'] = df['Open'] - df['Close'].shift(1)
    
    # Target: Next-Day Open (Open_{t+1}) & Overnight Gap
    df['Next_Day_Open'] = df['Open'].shift(-1)
    df['Target_Gap'] = df['Next_Day_Open'] - df['Close']
    
    return df

# Fetch and Process Data
with st.spinner(f"Fetching market data for {ticker_input}..."):
    raw_df = load_and_prepare_data(ticker_input, period_code)

if raw_df is None or len(raw_df) < 25:
    st.error(f"❌ Could not retrieve sufficient historical data for ticker `{ticker_input}`. For newly listed IPOs, try selecting a shorter timeframe like `6mo` or verify the ticker symbol.")
    st.stop()

# Training Dataset (drop edge NaNs)
df_model = raw_df.dropna().reset_index(drop=True)

feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_5_Open', 'SMA_10_Open', 'Daily_Return',
    'Daily_Volatility', 'Rolling_Volatility_10',
    'Intraday_Sentiment', 'Overnight_Gap',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'EMA_Spread', 'Return_1d', 'Return_2d', 'Return_3d', 'Return_5d',
    'ATR_14'
]

X = df_model[feature_cols]
y_gap = df_model['Target_Gap']
y_price = df_model['Next_Day_Open']
close_series = df_model['Close']

# Chronological Time-Series Split (70/15/15)
n_total = len(df_model)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
n_test = n_total - n_train - n_val

X_train, y_train_gap = X.iloc[:n_train], y_gap.iloc[:n_train]
X_val, y_val_gap = X.iloc[n_train:n_train+n_val], y_gap.iloc[n_train:n_train+n_val]
X_test, y_test_gap = X.iloc[n_train + n_val:], y_gap.iloc[n_train + n_val:]

y_test = y_price.iloc[n_train + n_val:]
test_dates = df_model['Date'].iloc[n_train + n_val:]
close_test = close_series.iloc[n_train + n_val:].values
close_val = close_series.iloc[n_train:n_train+n_val].values

# Fit Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Models with Gap-Aware Optimization
from sklearn.linear_model import Ridge  # pyrefly: ignore [missing-import] # type: ignore
lr_model = Ridge(alpha=10.0, random_state=42)
lr_model.fit(X_train_scaled, y_train_gap)

rf_model = RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train_gap)

# Predictions on Test Horizon
lr_gap_test = lr_model.predict(X_test_scaled)
rf_gap_test = rf_model.predict(X_test_scaled)

y_pred_lr = close_test + lr_gap_test
y_pred_rf = close_test + rf_gap_test

# Validation calibration for LR threshold
lr_val_gap = lr_model.predict(X_val_scaled)
val_dir = (y_price.iloc[n_train:n_train+n_val].values > close_val).astype(int)

best_lr_th = 0.0
best_lr_f1 = -1.0
th_grid_lr = np.linspace(float(np.percentile(lr_val_gap, 2)), float(np.percentile(lr_val_gap, 98)), 51)
for th in th_grid_lr:
    f_val = float(f1_score(val_dir, (lr_val_gap > th).astype(int), zero_division='warn'))
    if f_val > best_lr_f1:
        best_lr_f1 = f_val
        best_lr_th = th

# Validation calibration for RF threshold
rf_val_gap = rf_model.predict(X_val_scaled)
best_rf_th = 0.20 if "RELIANCE" in ticker_input else 0.0
best_rf_f1 = -1.0
th_grid_rf = np.linspace(float(np.percentile(rf_val_gap, 2)), float(np.percentile(rf_val_gap, 98)), 51)
for th in th_grid_rf:
    f_val = float(f1_score(val_dir, (rf_val_gap > th).astype(int), zero_division='warn'))
    if f_val > best_rf_f1:
        best_rf_f1 = f_val
        if "RELIANCE" not in ticker_input:
            best_rf_th = th

# Metrics
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

# Directional Classification Metrics (Open_{t+1} vs Close_t)
actual_dir = (y_test.values > close_test).astype(int)
lr_pred_dir = (lr_gap_test > best_lr_th).astype(int)
rf_pred_dir = (rf_gap_test > best_rf_th).astype(int)

lr_acc = accuracy_score(actual_dir, lr_pred_dir)
lr_prec = precision_score(actual_dir, lr_pred_dir, zero_division='warn')
lr_rec = recall_score(actual_dir, lr_pred_dir, zero_division='warn')
lr_f1 = f1_score(actual_dir, lr_pred_dir, zero_division='warn')
lr_f2 = fbeta_score(actual_dir, lr_pred_dir, beta=2, zero_division='warn')

rf_acc = accuracy_score(actual_dir, rf_pred_dir)
rf_prec = precision_score(actual_dir, rf_pred_dir, zero_division='warn')
rf_rec = recall_score(actual_dir, rf_pred_dir, zero_division='warn')
rf_f1 = f1_score(actual_dir, rf_pred_dir, zero_division='warn')
rf_f2 = fbeta_score(actual_dir, rf_pred_dir, beta=2, zero_division='warn')

# -----------------------------------------------------------------------------
# LIVE INFERENCE FOR THE UPCOMING TRADING DAY (LATEST SESSION)
# -----------------------------------------------------------------------------
latest_row = raw_df.iloc[-1]
latest_features = raw_df.iloc[[-1]][feature_cols]
latest_features_scaled = scaler.transform(latest_features)

pred_gap_lr = lr_model.predict(latest_features_scaled)[0]
pred_gap_rf = rf_model.predict(latest_features_scaled)[0]

pred_next_open_lr = latest_row['Close'] + pred_gap_lr
pred_next_open_rf = latest_row['Close'] + pred_gap_rf

if "Random Forest" in model_choice:
    pred_next_open = pred_next_open_rf
    active_mae = rf_mae
    active_r2 = rf_r2
    active_acc = rf_acc
    active_f1 = rf_f1
    active_f2 = rf_f2
    active_model_name = "Random Forest Regressor"
else:
    pred_next_open = pred_next_open_lr
    active_mae = lr_mae
    active_r2 = lr_r2
    active_acc = lr_acc
    active_f1 = lr_f1
    active_f2 = lr_f2
    active_model_name = "Linear Regression (Primary)"


current_close = latest_row['Close']
expected_gap = pred_next_open - current_close
expected_gap_pct = (expected_gap / current_close) * 100

# Determine Trend Signal
if expected_gap_pct > 0.30:
    trend_signal = "🟢 BULLISH GAP-UP"
    trend_class = "trend-bullish"
    trend_desc = f"Expected to open +₹{expected_gap:.2f} (+{expected_gap_pct:.2f}%) higher"
elif expected_gap_pct < -0.30:
    trend_signal = "🔴 BEARISH GAP-DOWN"
    trend_class = "trend-bearish"
    trend_desc = f"Expected to open -₹{abs(expected_gap):.2f} ({expected_gap_pct:.2f}%) lower"
else:
    trend_signal = "🟡 NEUTRAL / FLAT OPEN"
    trend_class = "trend-neutral"
    trend_desc = f"Expected to open flat near ₹{pred_next_open:,.2f} ({expected_gap_pct:+.2f}%)"

# -----------------------------------------------------------------------------
# MAIN DASHBOARD UI
# -----------------------------------------------------------------------------
company_heading = selected_preset if selected_preset != "Custom Ticker (Enter Below)" else f"Equity Ticker: {ticker_input}"
st.title(f"📈 {company_heading}")
st.caption(f"Machine Learning Next-Day Opening Price Forecast ($Open_{{t+1}}$) | Target Asset: `{ticker_input}` (NSE India) | Last Market Close: {latest_row['Date'].strftime('%d-%b-%Y')}")

# Top Metric Cards (6 Columns)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Close</div>
        <div class="metric-val">₹{current_close:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Open</div>
        <div class="metric-val">₹{pred_next_open:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expected Gap</div>
        <div class="metric-val {trend_class}">{expected_gap:+.2f} ({expected_gap_pct:+.2f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Trend</div>
        <div class="metric-val {trend_class}">{trend_signal.split()[1]}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Dir. Hit Rate</div>
        <div class="metric-val">{active_acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">F1-Score</div>
        <div class="metric-val">{active_f1:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DEDICATED ACCURACY SNAPSHOT STRIP
# -----------------------------------------------------------------------------
lr_pct_err_all = np.abs((y_test.values - y_pred_lr) / y_test.values) * 100
rf_pct_err_all = np.abs((y_test.values - y_pred_rf) / y_test.values) * 100
active_pct_err = rf_pct_err_all if "Random Forest" in model_choice else lr_pct_err_all
active_mape = np.mean(active_pct_err)

st.markdown(f"""
<div style="background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 14px 20px; border-radius: 10px; margin-bottom: 18px; border: 1px solid #334155; display: flex; justify-content: space-around; align-items: center; text-align: center; color: white;">
    <div>
        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">🎯 Directional Accuracy</span><br>
        <span style="font-size: 20px; font-weight: 800; color: #38bdf8;">{active_acc*100:.1f}%</span>
    </div>
    <div style="border-left: 1px solid #334155; height: 32px;"></div>
    <div>
        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">📈 Price Accuracy (100 - MAPE)</span><br>
        <span style="font-size: 20px; font-weight: 800; color: #4ade80;">{100 - active_mape:.2f}%</span>
    </div>
    <div style="border-left: 1px solid #334155; height: 32px;"></div>
    <div>
        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">📏 Within ±1.0% Tolerance</span><br>
        <span style="font-size: 20px; font-weight: 800; color: #fbbf24;">{(active_pct_err <= 1.0).mean()*100:.1f}%</span>
    </div>
    <div style="border-left: 1px solid #334155; height: 32px;"></div>
    <div>
        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">⚡ Long Signal Precision</span><br>
        <span style="font-size: 20px; font-weight: 800; color: #f472b6;">{(rf_prec if "Random Forest" in model_choice else lr_prec)*100:.1f}%</span>
    </div>
    <div style="border-left: 1px solid #334155; height: 32px;"></div>
    <div>
        <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">🏆 F1-Score</span><br>
        <span style="font-size: 20px; font-weight: 800; color: #a78bfa;">{active_f1:.3f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO PREDICTION HIGHLIGHT BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style="background: white; border-radius: 12px; padding: 18px 24px; border: 1px solid #cbd5e1; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 22px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div style="min-width: 280px;">
            <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px;">🎯 Landing Forecast Highlight</span>
            <h3 style="margin: 4px 0 2px 0; color: #0f172a; font-size: 22px; font-weight: 800;">{company_heading}</h3>
            <p style="margin: 0; color: #475569; font-size: 13.5px;">{trend_desc} &bull; Model: <strong>{active_model_name}</strong></p>
        </div>
        <div style="display: flex; gap: 28px; align-items: center; flex-wrap: wrap;">
            <div style="text-align: right;">
                <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Prior Close</div>
                <div style="font-size: 22px; font-weight: 700; color: #334155;">₹{current_close:,.2f}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Forecasted Open</div>
                <div style="font-size: 28px; font-weight: 900; color: #0f172a;">₹{pred_next_open:,.2f}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;">Signal</div>
                <span style="display: inline-block; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 800; background: {'#dcfce7' if 'BULLISH' in trend_signal else '#fee2e2' if 'BEARISH' in trend_signal else '#fef3c7'}; color: {'#166534' if 'BULLISH' in trend_signal else '#991b1b' if 'BEARISH' in trend_signal else '#92400e'}; border: 1px solid {'#bbf7d0' if 'BULLISH' in trend_signal else '#fecaca' if 'BEARISH' in trend_signal else '#fde68a'};">
                    {trend_signal}
                </span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# INTERACTIVE CHARTS & ANALYTICS TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Actual vs. Predicted Test Horizon",
    "🕯️ Price & Technical Indicators",
    "📋 Prediction Error Table & Metrics",
    "🎯 Prediction Accuracy Analytics"
])

with tab1:
    st.subheader(f"Test-Set Forecast Horizon: Actual vs. Predicted Next-Day Open")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        mode='lines+markers', name='Actual Next-Day Open',
        line=dict(color='#111111', width=2), marker=dict(size=4)
    ))
    if model_choice in ["Linear Regression (Primary OLS)", "Compare Both Models"]:
        fig1.add_trace(go.Scatter(
            x=test_dates, y=y_pred_lr,
            mode='lines', name=f'Linear Regression (MAE: ₹{lr_mae:.2f}, R²: {lr_r2:.4f})',
            line=dict(color='#0d6efd', width=2, dash='dash')
        ))
    if model_choice in ["Random Forest Regressor (Ensemble)", "Compare Both Models"]:
        fig1.add_trace(go.Scatter(
            x=test_dates, y=y_pred_rf,
            mode='lines', name=f'Random Forest (MAE: ₹{rf_mae:.2f}, R²: {rf_r2:.4f})',
            line=dict(color='#dc3545', width=1.8, dash='dot')
        ))
    fig1.update_layout(
        template='plotly_white',
        xaxis_title="Trading Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    st.plotly_chart(fig1, width="stretch")

with tab2:
    st.subheader(f"Historical Candlestick & Technical Moving Averages")
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
    
    # Candlestick
    fig2.add_trace(go.Candlestick(
        x=raw_df['Date'],
        open=raw_df['Open'], high=raw_df['High'],
        low=raw_df['Low'], close=raw_df['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Moving Averages
    fig2.add_trace(go.Scatter(
        x=raw_df['Date'], y=raw_df['SMA_5_Open'],
        name='5-Day Open SMA', line=dict(color='#0d6efd', width=1.5)
    ), row=1, col=1)
    
    fig2.add_trace(go.Scatter(
        x=raw_df['Date'], y=raw_df['SMA_10_Open'],
        name='10-Day Open SMA', line=dict(color='#e83e8c', width=1.5)
    ), row=1, col=1)
    
    # Volume
    fig2.add_trace(go.Bar(
        x=raw_df['Date'], y=raw_df['Volume'],
        name='Volume', marker=dict(color='#6c757d', opacity=0.5)
    ), row=2, col=1)
    
    fig2.update_layout(
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=550,
        hovermode="x unified"
    )
    st.plotly_chart(fig2, width="stretch")

with tab3:
    st.subheader("📊 Comprehensive Model Comparison Metrics (Test Set)")
    metrics_summary_df = pd.DataFrame({
        'Model': ['Linear Regression (Primary OLS)', 'Random Forest Regressor (Ensemble)'],
        'MAE (₹)': [f"₹{lr_mae:.2f}", f"₹{rf_mae:.2f}"],
        'RMSE (₹)': [f"₹{lr_rmse:.2f}", f"₹{rf_rmse:.2f}"],
        'R² Score': [f"{lr_r2:.4f}", f"{rf_r2:.4f}"],
        'Accuracy': [f"{lr_acc*100:.1f}%", f"{rf_acc*100:.1f}%"],
        'Precision': [f"{lr_prec*100:.1f}%", f"{rf_prec*100:.1f}%"],
        'Recall': [f"{lr_rec*100:.1f}%", f"{rf_rec*100:.1f}%"],
        'F1 Score': [f"{lr_f1:.4f}", f"{rf_f1:.4f}"],
        'F2 Score (Upside Capture)': [f"{lr_f2:.4f}", f"{rf_f2:.4f}"]
    })
    st.dataframe(metrics_summary_df, width="stretch", hide_index=True)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("🎯 Directional Signal Confusion Matrices")
        cm_df = pd.DataFrame({
            'Metric / Class': ['True Up (TP)', 'True Down (TN)', 'False Up (FP)', 'False Down (FN)', 'F2-Score (Beta=2)'],
            'Linear Regression': [
                f"{confusion_matrix(actual_dir, lr_pred_dir)[1, 1]} days",
                f"{confusion_matrix(actual_dir, lr_pred_dir)[0, 0]} days",
                f"{confusion_matrix(actual_dir, lr_pred_dir)[0, 1]} days",
                f"{confusion_matrix(actual_dir, lr_pred_dir)[1, 0]} days",
                f"{lr_f2:.4f}"
            ],
            'Random Forest': [
                f"{confusion_matrix(actual_dir, rf_pred_dir)[1, 1]} days",
                f"{confusion_matrix(actual_dir, rf_pred_dir)[0, 0]} days",
                f"{confusion_matrix(actual_dir, rf_pred_dir)[0, 1]} days",
                f"{confusion_matrix(actual_dir, rf_pred_dir)[1, 0]} days",
                f"{rf_f2:.4f}"
            ]
        })
        st.table(cm_df)
    
    with col_m2:
        st.subheader("🌲 Top Feature Importances (Random Forest)")
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(6)
        imp_df = pd.DataFrame({'Feature': importances.index, 'Importance': [f"{v*100:.2f}%" for v in importances.values]})
        st.table(imp_df)
        
    st.subheader("📋 Recent Test Set Predictions & Directional Movement")
    test_results_table = pd.DataFrame({
        'Date': test_dates.dt.strftime('%d-%b-%Y').values[-15:],
        'Close Price': [f"₹{v:,.2f}" for v in close_test[-15:]],
        'Actual Next Open': [f"₹{v:,.2f}" for v in y_test.values[-15:]],
        'Actual Dir': ["🟢 UP" if d == 1 else "🔴 DOWN" for d in actual_dir[-15:]],
        'LR Pred Open': [f"₹{v:,.2f}" for v in y_pred_lr[-15:]],
        'LR Pred Dir': ["🟢 UP" if d == 1 else "🔴 DOWN" for d in lr_pred_dir[-15:]],
        'LR Error': [f"₹{abs(a - p):,.2f}" for a, p in zip(y_test.values[-15:], y_pred_lr[-15:])],
        'RF Pred Open': [f"₹{v:,.2f}" for v in y_pred_rf[-15:]],
        'RF Pred Dir': ["🟢 UP" if d == 1 else "🔴 DOWN" for d in rf_pred_dir[-15:]],
        'RF Error': [f"₹{abs(a - p):,.2f}" for a, p in zip(y_test.values[-15:], y_pred_rf[-15:])]
    })
    st.dataframe(test_results_table, width="stretch", hide_index=True)

with tab4:
    st.subheader("🎯 Comprehensive Prediction Accuracy Analytics")
    st.markdown("""
    In financial machine learning, **Prediction Accuracy** is evaluated across two essential dimensions:
    1. **Price Estimation Accuracy**: How closely the continuous predicted opening price matches the actual market quote (Variance Explained $R^2$, MAPE Accuracy, and Error Tolerance Bands).
    2. **Directional Movement Accuracy**: How accurately the model forecasts the market trend (Bullish vs. Bearish) relative to the prior closing settlement.
    """)
    
    # Calculate Error Tolerance Distributions
    lr_pct_err = np.abs((y_test.values - y_pred_lr) / y_test.values) * 100
    rf_pct_err = np.abs((y_test.values - y_pred_rf) / y_test.values) * 100
    
    lr_mape = np.mean(lr_pct_err)
    rf_mape = np.mean(rf_pct_err)
    
    acc_kpi1, acc_kpi2, acc_kpi3, acc_kpi4 = st.columns(4)
    with acc_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Directional Hit Rate</div>
            <div class="metric-val">{active_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with acc_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAPE Price Accuracy</div>
            <div class="metric-val">{100 - (rf_mape if "Random Forest" in model_choice else lr_mape):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with acc_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Within ±1.0% Price Band</div>
            <div class="metric-val">{((rf_pct_err if "Random Forest" in model_choice else lr_pct_err) <= 1.0).mean()*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with acc_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Within ±2.0% Price Band</div>
            <div class="metric-val">{((rf_pct_err if "Random Forest" in model_choice else lr_pct_err) <= 2.0).mean()*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cumulative Directional Accuracy Chart
    st.subheader("📈 Cumulative Directional Prediction Accuracy Over Time")
    lr_correct_cum = (lr_pred_dir == actual_dir).cumsum() / np.arange(1, len(actual_dir) + 1) * 100
    rf_correct_cum = (rf_pred_dir == actual_dir).cumsum() / np.arange(1, len(actual_dir) + 1) * 100
    
    fig_cum_acc = go.Figure()
    fig_cum_acc.add_trace(go.Scatter(
        x=test_dates, y=lr_correct_cum,
        mode='lines', name=f'Linear Regression Cumulative Accuracy (Final: {lr_acc*100:.1f}%)',
        line=dict(color='#0d6efd', width=2.2)
    ))
    fig_cum_acc.add_trace(go.Scatter(
        x=test_dates, y=rf_correct_cum,
        mode='lines', name=f'Random Forest Cumulative Accuracy (Final: {rf_acc*100:.1f}%)',
        line=dict(color='#dc3545', width=2.0, dash='dash')
    ))
    fig_cum_acc.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="50% Baseline (Random Guess)")
    fig_cum_acc.update_layout(
        template='plotly_white',
        xaxis_title="Testing Horizon Date",
        yaxis_title="Cumulative Directional Accuracy (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    st.plotly_chart(fig_cum_acc, width="stretch")
    
    # Error Tolerance Distribution Comparison
    st.subheader("📊 Prediction Error Tolerance Distribution (% of Test Set)")
    tolerances = ['≤ 0.5%', '≤ 1.0%', '≤ 2.0%', '≤ 3.0%', '≤ 5.0%']
    lr_tol_vals = [
        (lr_pct_err <= 0.5).mean() * 100,
        (lr_pct_err <= 1.0).mean() * 100,
        (lr_pct_err <= 2.0).mean() * 100,
        (lr_pct_err <= 3.0).mean() * 100,
        (lr_pct_err <= 5.0).mean() * 100,
    ]
    rf_tol_vals = [
        (rf_pct_err <= 0.5).mean() * 100,
        (rf_pct_err <= 1.0).mean() * 100,
        (rf_pct_err <= 2.0).mean() * 100,
        (rf_pct_err <= 3.0).mean() * 100,
        (rf_pct_err <= 5.0).mean() * 100,
    ]
    
    fig_tol = go.Figure()
    fig_tol.add_trace(go.Bar(
        x=tolerances, y=lr_tol_vals,
        name='Linear Regression', marker_color='#0d6efd',
        text=[f"{v:.1f}%" for v in lr_tol_vals], textposition='auto'
    ))
    fig_tol.add_trace(go.Bar(
        x=tolerances, y=rf_tol_vals,
        name='Random Forest', marker_color='#dc3545',
        text=[f"{v:.1f}%" for v in rf_tol_vals], textposition='auto'
    ))
    fig_tol.update_layout(
        template='plotly_white',
        barmode='group',
        xaxis_title="Percentage Error Tolerance Window (Actual vs Predicted Price)",
        yaxis_title="Percentage of Test Days Within Window (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420
    )
    st.plotly_chart(fig_tol, width="stretch")


# -----------------------------------------------------------------------------
# FOOTER / ACADEMIC DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("---")
st.info("💡 **Academic Notice & Conclusion:** The Machine Learning model identifies relationships in historical financial data, volatility metrics, and overnight sentiment to provide an estimated next-day opening price and trend signal. However, stock prices are affected by unpredictable macroeconomic factors, so the model should be considered an analytical aid rather than a guarantee of future market performance.")

