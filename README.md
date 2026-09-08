# 📈 Financial Machine Learning: Stock Market Next-Day Opening Price Prediction

An end-to-end quantitative Machine Learning pipeline and interactive financial dashboard built for stock price forecasting, directional trend modeling, and risk-adjusted momentum analysis.

---

## 🌟 Key Highlights & Features

- **Target Variable**: $Open_{t+1}$ (Next-Day Opening Price) and Directional Market Movement (Bullish / UP vs. Bearish / DOWN)
- **Target Asset**: Reliance Industries Limited (`RELIANCE.NS`) across 2021–2025 trading sessions on the National Stock Exchange of India (NSE).
- **12 Engineered Quantitative Features**:
  - **Inter-day Momentum**: 5-Day and 10-Day Open Simple Moving Averages (`SMA_5_Open`, `SMA_10_Open`), Daily Percentage Returns (`Daily_Return`)
  - **Volatility Metrics**: Daily High-Low price range (`Daily_Volatility`), 10-Day rolling volatility spread (`Rolling_Volatility_10`)
  - **Sentiment & Gaps**: Intraday closing sentiment (`Intraday_Sentiment`), overnight opening gap persistence (`Overnight_Gap`)
- **Rigorous Temporal Validation**: Strict chronological time-series partition (70% Train, 15% Validation, 15% Test) preventing lookahead bias.
- **Multi-Metric Quantitative Evaluation**:
  - Continuous Price Accuracy: MAE, RMSE, and $R^2$ Score
  - Directional Trading Performance: Directional Hit Rate (Accuracy), Precision (drawdown defense), Recall (upside capture), and $F_2$-Score (opportunity-cost weighted)
- **Interactive UI**: Real-time Streamlit dashboard (`app.py`) with Plotly candlestick charts, preset Indian equities/IPOs (Swiggy, Hyundai, Zomato, Reliance, Tata), technical overlays, and live pre-market gap forecasting.

---

## 📁 Repository Structure

```
├── app.py                             # Interactive Financial ML Streamlit Dashboard
├── main.py                            # Master End-to-End Financial Modeling Pipeline
│
├── phase1_data_sourcing.py            # Phase 1: Yahoo Finance Automated Data Acquisition
├── phase2_feature_engineering.py      # Phase 2: Feature Engineering & Target Alignment
├── phase3_model_training.py           # Phase 3: Time-Series Split (70/15/15) & Model Training
├── phase4_evaluation_visualization.py # Phase 4: Statistical Evaluation & Visualization Suite
├── phase5_report_assembly.py          # Phase 5: Final Comprehensive Report Assembly
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
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Primary)** | **₹5.72** | **₹9.40** | **0.9875** | **99.59%** | **94.02%** | **98.37%** | **50.00%** | **50.00%** | **100.00%** | **0.6667** | **0.8333** |
| **Random Forest Regressor (Ensemble)** | **₹5.66** | **₹9.30** | **0.9878** | **99.59%** | **93.48%** | **98.37%** | **52.17%** | **51.16%** | **95.65%** | **0.6667** | **0.8148** |
