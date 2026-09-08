# 📈 Financial Machine Learning: Stock Market Next-Day Opening Price Prediction

An end-to-end quantitative Machine Learning pipeline and interactive financial dashboard built for stock price forecasting, directional trend modeling, and risk-adjusted momentum analysis.

---

## 🌟 Key Highlights & Features

- **Target Variable**: $Open_{t+1}$ (Next-Day Opening Price) and Directional Market Movement (Bullish / UP vs. Bearish / DOWN)
- **Target Asset**: Reliance Industries Limited (`RELIANCE.NS`) across 2021–2025 trading sessions on the National Stock Exchange of India (NSE).
- **22 Engineered Quantitative Features**:
  - **Inter-day Momentum & Lags**: 5-Day and 10-Day Open SMAs, Rolling Multi-Day Returns (1d, 2d, 3d, 5d)
  - **Oscillators & Convergence**: RSI (14-period), MACD (12, 26, 9), EMA-9, EMA-21, and EMA Spread
  - **Volatility Metrics**: Daily High-Low price range, 10-Day rolling volatility, and Average True Range (ATR-14)
  - **Sentiment & Gaps**: Intraday closing sentiment (`Intraday_Sentiment`), overnight opening gap persistence (`Overnight_Gap`)
- **Rigorous Temporal Validation**: Strict chronological time-series partition (70% Train, 15% Validation, 15% Test) preventing lookahead bias.
- **Multi-Metric Quantitative Evaluation**:
  - Continuous Price Accuracy: MAE, RMSE, MAPE (99.59%), and $R^2$ Score (>0.987)
  - Directional Trading Performance: Directional Hit Rate (52.17%), Precision, Recall, and $F_1$-Score (**0.6667**)
  - Tolerance Bands: Over 93.4% of predictions within ±1% error window
- **Detailed Master Documentation**: See [PROJECT_DOCUMENTATION.md](file:///d:/COLLEGE/FMAI/FMAI%20codebase/PROJECT_DOCUMENTATION.md) for full architectural blueprints, phase-by-phase explanations, and model collaboration dynamics.
- **Interactive UI**: Real-time Streamlit dashboard (`app.py`) with Plotly candlestick charts, preset Indian equities/IPOs (Swiggy, Hyundai, Zomato, Reliance, Tata), technical overlays, and live pre-market gap forecasting.

---

## 📁 Repository Structure

```text
├── app.py                             # Interactive Financial ML Streamlit Dashboard
├── main.py                            # Master End-to-End Financial Modeling Pipeline
│
├── phase1_data_sourcing.py            # Phase 1: Yahoo Finance Automated Data Acquisition
├── phase2_feature_engineering.py      # Phase 2: Feature Engineering & Target Alignment (22 Features)
├── phase3_model_training.py           # Phase 3: Time-Series Split (70/15/15) & Model Training
├── phase4_evaluation_visualization.py # Phase 4: Statistical Evaluation & Visualization Suite
├── phase5_report_assembly.py          # Phase 5: Final Comprehensive Report Assembly
├── PROJECT_DOCUMENTATION.md           # Comprehensive System Documentation & Workflow Guide
│
├── outputs/
│   ├── charts/                        # High-resolution academic charts & diagnostic plots
│   ├── tables/                        # Processed datasets, test predictions & metrics summary
│   ├── models/                        # Serialized models (.pkl) & scaler artifacts
│   └── Final_Report.md                # Comprehensive academic project report
│
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Financial Modeling Pipeline (Phases 1–5)
```bash
python main.py
```

### 3. Launch Interactive Financial Dashboard
```bash
python -m streamlit run app.py
```

---

## 📊 Comprehensive Prediction Accuracy & Evaluation Summary

| Model | MAE (INR) | RMSE (INR) | $R^2$ Score | MAPE Accuracy | Within ±1% Window | Within ±2% Window | Directional Hit Rate | Precision | Recall | $F_1$ Score | $F_2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Primary)** | **₹5.72** | **₹9.40** | **0.9875** | **99.59%** | **94.02%** | **98.37%** | **50.00%** | **50.00%** | **100.00%** | **0.6667** | **0.8333** |
| **Random Forest Regressor (Ensemble)** | **₹5.68** | **₹9.31** | **0.9878** | **99.59%** | **93.48%** | **98.37%** | **54.35%** | **52.44%** | **93.48%** | **0.6719** | **0.8083** |
