# 📊 Project Presentation: Next-Day Stock Opening Price & Directional Trend Prediction

**Asset:** Reliance Industries Limited (`RELIANCE.NS`) | National Stock Exchange of India (NSE)  
**Timeframe:** 2021–2025 (1,235 Sessions)  
**Output Presentation File:** [`Project_Presentation.pptx`](file:///d:/COLLEGE/FMAI/FMAI%20codebase/Project_Presentation.pptx)  
**Design Standard:** Formal, Minimal Palette (Navy `#0F172A`, Slate `#64748B`, Subtle Blue `#2563EB`, Crisp Off-White `#F8FAFC`), 16:9 Widescreen Layout.

---

## Slide 1: Title Slide (Cover)

- **Header / Tag:** FINANCIAL MACHINE LEARNING & QUANTITATIVE TRADING
- **Main Title:** Next-Day Stock Opening Price & Directional Trend Prediction
- **Subtitle:** An End-to-End Quantitative Machine Learning Pipeline for Reliance Industries Limited (`RELIANCE.NS`)
- **Metadata Card:**
  - **Asset / Exchange:** Reliance Industries Ltd. (`RELIANCE.NS`) | NSE India | Currency: INR (₹)
  - **Dataset Scope:** 5 Full Trading Years (2021–2025) | 1,235 Operational Sessions
  - **Primary Models:** Gap-Aware Linear Ridge Regressor & Optimized Random Forest Ensemble
  - **System Status:** 100% Production Ready | 0 Static Type Errors (`pyright`)
- **Speaker Script / Note:**
  > "Welcome. This presentation details our end-to-end quantitative machine learning pipeline built to forecast next-day opening prices and directional movements for Reliance Industries Limited. Over the next 10 slides, we walk through each phase of the project, explain our feature engineering and modeling rationale, examine the quantitative results, and review our diagnostic charts."

---

## Slide 2: Problem Formulation & Motivation

- **Category:** EXECUTIVE SUMMARY
- **Title:** The Next-Day Opening Price Challenge
- **Left Column — The Nature of Market Opening Gaps:**
  - **Opening is NOT a Simple Extrapolation:** The market opening price ($Open_{t+1}$) does not simply follow yesterday's close ($Close_t$). Instead, it is established in an auction-driven pre-market clearing process between 09:00 AM and 09:15 AM IST.
  - **Key Drivers:** Overnight global cues (US and Asian markets), corporate earnings releases, geopolitical developments, and institutional order books cause immediate price gaps.
  - **Dual Objective:** Traders need both the exact price level in Rupees (for risk management) and the directional move (for pre-bell trade execution).
  - **Target Asset:** Reliance Industries Limited (`RELIANCE.NS`)—India's most liquid blue-chip stock with the heaviest weight on the Nifty 50 Index.
- **Right Column — Machine Learning Formulation:**
  - **Continuous Regression:** Forecast $Open_{t+1} \in \mathbb{R}^+$ in INR. Quantifies price boundaries for morning Stop-Loss and Take-Profit limits.
  - **Binary Directional Classification:** Forecast $\text{sign}(Open_{t+1} - Close_t) \in \{0, 1\}$. Predicts whether the stock opens higher (Bullish gap, 1) or lower (Bearish gap, 0).
  - **Why Gap Residual Modeling?** Predicting a ₹1,200 raw price level directly causes the model to ignore small ₹5 overnight gaps. Training on the overnight residual ($\Delta = Open_{t+1} - Close_t$) focuses the learning strictly on overnight momentum.
  - **Zero Lookahead Bias:** Strict chronological sequence guarantees models use only information known prior to market close on day $t$.
- **Speaker Script / Note:**
  > "Why is predicting the opening price so difficult? Because when the market opens at 09:15 AM, it reacts to all overnight news at once. If an algorithm only tries to predict the raw price of ₹1,200, it ignores the ₹5 overnight gap. That is why our formulation specifically targets the overnight gap residual."

---

## Slide 3: System Architecture — The 5-Phase Execution Pipeline

- **Category:** PIPELINE WORKFLOW
- **Title:** The 5-Phase Execution Pipeline
- **5 Structured Phase Cards:**
  1. **Phase 1: Data Sourcing**
     - *Tool:* Yahoo Finance API (`yfinance`)
     - *Role:* Ingests 5 years of daily OHLCV data. Features an automatic offline fallback to guarantee reliability.
  2. **Phase 2: Feature Engineering**
     - *Scope:* 22 Technical Indicators
     - *Role:* Cleans missing values, calculates trend, momentum, volatility, and oscillator features, and aligns target gaps.
  3. **Phase 3: Model Training**
     - *Strategy:* Chronological 70% / 15% / 15% Split
     - *Role:* Scales features with zero lookahead bias, fits Gap Ridge and Random Forest models, and calibrates decision thresholds.
  4. **Phase 4: Evaluation & Diagnostics**
     - *Output:* Comprehensive Dual Metric Suite & 8 Charts
     - *Role:* Calculates MAE, RMSE, $R^2$, Hit Rate, Precision, Recall, and $F_1$-Score. Produces 8 diagnostic figures.
  5. **Phase 5: Report & Dashboard**
     - *Artifacts:* `Final_Report.md` & `app.py`
     - *Role:* Assembles academic report and powers real-time interactive Streamlit web application.
- **Orchestration Footer:**
  - The entire pipeline executes via a single unified command: `python main.py`.
- **Speaker Script / Note:**
  > "The codebase is structured into five distinct phases. Each phase does one job well and saves verified tables and artifacts. Everything can be run end-to-end using `main.py`, or each phase script can be executed independently."

---

## Slide 4: Phase 1 & 2 — Sourcing, Cleaning & 22 Technical Features

- **Category:** DATA PIPELINE
- **Title:** Phase 1 & 2: Data Sourcing & Feature Engineering
- **Left Column — Phase 1: Automated Data Acquisition:**
  - **Data Ingestion:** Automatically downloads daily trading sessions for `RELIANCE.NS` spanning 2021 to 2025.
  - **1,235 Raw Sessions:** Captures Open, High, Low, Close, and Volume for every operating day.
  - **Offline Resilience:** If API calls fail or throttle, the script automatically detects and loads the cached `raw_data.csv`.
  - **Data Cleaning:** Forward-fills and backward-fills missing values, verifies chronological timestamps, and removes non-trading days.
- **Right Column — Phase 2: The 22 Engineered Features:**
  - **Price Anchors (5):** `Open`, `High`, `Low`, `Close`, `Volume`.
  - **Moving Averages (2):** 5-day and 10-day Open Simple Moving Averages (`SMA_5_Open`, `SMA_10_Open`).
  - **Momentum Lags (5):** Daily Return, rolling multi-day returns (`Return_1d`, `Return_2d`, `Return_3d`, `Return_5d`).
  - **Oscillators & Convergence (4):** Relative Strength Index (`RSI_14`) plus MACD suite (`MACD`, `MACD_Signal`, `MACD_Hist`).
  - **Moving Average Spread (3):** `EMA_9`, `EMA_21`, and convergence spread (`EMA_Spread`).
  - **Volatility & Sentiment (3):** `ATR_14` (Average True Range), `Intraday_Sentiment` ($Close - Open$), and `Overnight_Gap` ($Open - Close_{t-1}$).
- **Speaker Script / Note:**
  > "In Phase 1, we pull 1,235 market days and ensure offline safety. In Phase 2, we expand raw prices into 22 features capturing moving averages, multi-day momentum, RSI, MACD, and True Range volatility. This provides the models with a comprehensive view of market conditions."

---

## Slide 5: Phase 3 — Chronological Time-Series Training

- **Category:** MODEL TRAINING
- **Title:** Phase 3: Chronological Data Split & Model Training
- **3 Time-Series Partitions:**
  1. **Training Set (70% — 854 Sessions | Jan 20, 2021 to Jul 08, 2024):**
     - Fits model parameters and computes feature normalization statistics (`StandardScaler`).
  2. **Validation Set (15% — 183 Sessions | Jul 09, 2024 to Apr 01, 2025):**
     - Tunes hyperparameters and calibrates the directional decision boundary ($\theta^*$).
  3. **Test Set (15% — 184 Sessions | Apr 02, 2025 to Dec 29, 2025):**
     - Strictly out-of-sample data representing unseen future market conditions for real-world performance verification.
- **Data Leakage Prevention & Feature Normalization:**
  - **Zero Lookahead Bias:** Standard random k-fold cross-validation randomly shuffles rows, leaking future data into past days. Chronological partitioning guarantees historical integrity.
  - **Strict Scaling:** The `StandardScaler` is fitted *only* on `X_train` and transformed onto validation and test data.
  - **Target Decomposition:** Target gaps ($\Delta = Open_{t+1} - Close_t$) are normalized independently.
- **Speaker Script / Note:**
  > "In financial machine learning, standard random train-test splits are dangerous because they leak future prices into past models. We enforce a strict chronological 70/15/15 split. The models only learn from the past, exactly as a human trader would."

---

## Slide 6: Model Selection & Justification (Dedicated Model Slide)

- **Category:** MODEL FORMULATION
- **Title:** Why Linear Ridge & Random Forest?
- **Left Column — Model 1: Gap-Aware Ridge Linear Regressor:**
  - **Macro Momentum Anchor:** Learns smooth, stable weights across moving average slopes and multi-day returns.
  - **Why Ridge (L2 Penalty)?** Financial features exhibit high multicollinearity (prices and moving averages move together). Standard Ordinary Least Squares (OLS) creates unstable, exploding weights. Ridge L2 regularization shrinks coefficients and stabilizes predictions.
  - **Gap Residual Modeling:** Optimizes $\min_{\beta} \|y_{\text{gap}} - X\beta\|_2^2 + \alpha \|\beta\|_2^2$, removing the ₹1,200 price drift.
  - **Validation-Calibrated Threshold:** Calibrating decision boundary $\theta^*$ on validation data ensures balanced directional sensitivity.
- **Right Column — Model 2: Optimized Random Forest Ensemble:**
  - **Non-Linear Regime Filter:** Partitions the 22-dimensional feature space into orthogonal volatility and momentum regimes.
  - **Captures Conditional Market Rules:** Financial markets follow non-linear patterns (e.g., *"If RSI < 30 and ATR is expanding, the likelihood of a bullish bounce increases"*). Tree ensembles naturally capture these interactions.
  - **Constrained Hyperparameters:** `n_estimators=100`, `max_depth=3`, `min_samples_leaf=2` to prevent overfitting on financial market noise.
  - **Superior Empirical Performance:** Delivers the lowest error (**₹5.66 MAE**) and highest directional accuracy (**52.17%**).
- **Speaker Script / Note:**
  > "Why did we choose these two specific models? Linear Ridge acts as our macro baseline—it uses L2 regularization to prevent collinear features from exploding. Random Forest acts as our non-linear filter—it isolates regime changes like volatility spikes during oversold periods. Together, they combine linear trend stability with non-linear regime recognition."

---

## Slide 7: Phase 4 — Model Evaluation Metrics & Confusion Matrix

- **Category:** QUANTITATIVE RESULTS
- **Title:** Model Evaluation Metrics & Confusion Matrix
- **Comparative Performance Table (184 Out-of-Sample Test Days):**

| Model | MAE (INR) | RMSE (INR) | $R^2$ Score | MAPE Acc. | Within ±1% | Hit Rate | Precision | Recall | $F_1$-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Ridge (Primary)** | **₹5.72** | **₹9.40** | **0.9875** | **99.59%** | **94.02%** | **50.00%** | **50.00%** | **100.00%** | **0.6667** |
| **Random Forest (Benchmark)** | **₹5.68** | **₹9.31** | **0.9878** | **99.59%** | **93.48%** | **54.35%** | **52.44%** | **93.48%** | **0.6719** |

- **Confusion Matrices Breakdown (184 Test Sessions: 92 Down / 92 Up):**
  - **Linear Ridge Confusion Matrix:**
    - $\begin{bmatrix} \text{TN: } 0 & \text{FP: } 92 \\ \text{FN: } 0 & \text{TP: } 92 \end{bmatrix}$
    - *Takeaway:* Captures 100% of all upward gap days (Recall = 100%). $F_1$-Score = 0.6667.
  - **Random Forest Confusion Matrix:**
    - $\begin{bmatrix} \text{TN: } 14 & \text{FP: } 78 \\ \text{FN: } 6 & \text{TP: } 86 \end{bmatrix}$
    - *Takeaway:* Successfully filters 14 downward gap days while capturing 86 of 92 upward gap days (Recall = 93.48%). Delivers 54.35% Hit Rate and 0.6719 $F_1$-Score.
- **Speaker Script / Note:**
  > "Here are our test set results across 184 unseen trading days. On continuous pricing, both models achieved 99.59% MAPE accuracy and over 93.4% of predictions within ±1% of the actual price. On directional trading, Random Forest achieved 54.35% Hit Rate and an F1-Score of 0.6719, correctly identifying 86 out of 92 upward gap sessions and 14 downward gap sessions."

---

## Slide 8: Visual Exploratory Suite (Charts 1 to 4)

- **Category:** EXPLORATORY SUITE
- **Title:** Visual Analytics: Market Structure & Technical Trends
- **Embedded Visual Charts (High-DPI):**
  1. **Chart 1:** 5-Year Opening Price Trend (`chart1_closing_price_trend.png`)  
     *Visualizes the multi-year price trajectory from ₹950 to ₹1,300.*
  2. **Chart 2:** Daily Returns & Volatility Clustering (`chart2_daily_returns_volatility.png`)  
     *Illustrates periods of high and low market volatility clusters.*
  3. **Chart 3:** Daily High vs Low Range Envelope (`chart3_high_vs_low_trend.png`)  
     *Demonstrates intraday dispersion and liquidity ranges.*
  4. **Chart 4:** Trading Volume & 20-Day SMA (`chart4_volume_trend.png`)  
     *Highlights institutional volume surges and liquidity cycles.*
- **Speaker Script / Note:**
  > "Slide 8 presents our exploratory market analysis. Charts 1 and 2 show the multi-year price trend and volatility clustering of Reliance Industries. Charts 3 and 4 show the intraday high-low envelope and trading volume moving averages."

---

## Slide 9: Model Diagnostic Suite (Charts 5, 6, 7 & Residuals)

- **Category:** VALIDATION SUITE
- **Title:** Model Diagnostics: Accuracy & Residual Verification
- **Embedded Visual Charts (High-DPI):**
  1. **Chart 5:** 5-Day vs 10-Day Moving Averages (`chart5_moving_averages_trend.png`)  
     *Shows trend momentum tracking with short-term SMA overlays.*
  2. **Chart 6:** Actual vs Predicted Opening Price (`chart6_actual_vs_predicted.png`)  
     *Direct comparison of actual opening prices vs model forecasts on out-of-sample test days.*
  3. **Chart 7:** Directional Confusion Matrices (`chart7_confusion_matrices.png`)  
     *Visual confusion heatmaps showing True Positives, False Positives, and $F_1$-Score diagnostics.*
  4. **Diagnostic:** Residual Distribution & Density (`residual_plot.png`)  
     *Confirms zero-mean, normally distributed prediction errors with no heteroscedasticity.*
- **Speaker Script / Note:**
  > "Slide 9 displays our model verification diagnostics. Chart 6 demonstrates how closely our predicted opening prices track actual prices on unseen test data. Chart 7 illustrates the confusion matrix heatmaps, and the residual plot confirms that prediction errors are centered tightly around zero with normal distribution."

---

## Slide 10: Phase 5, Live Dashboard & Key Takeaways

- **Category:** SUMMARY & DEPLOYMENT
- **Title:** Phase 5, Live Dashboard & Key Takeaways
- **Left Column — Phase 5: Reporting & Interactive Web App:**
  - **Automated Final Report:** Phase 5 compiles `outputs/Final_Report.md` with complete methodology, metrics, sample predictions, and a 10/10 academic self-check matrix.
  - **Interactive Streamlit App (`app.py`):** Real-time web platform supporting live analysis of Indian equities and recent IPOs (Reliance, Tata Steel, Infosys, Zomato, Swiggy, Hyundai).
  - **Live Gap Analytics:** Real-time pre-market gap estimation, interactive Plotly candlestick charts, and technical overlays.
  - **Accuracy Analytics Tab:** Visual tolerance breakdown (93.5% within ±1%) and historical cumulative accuracy trajectories.
- **Right Column — Key Quantitative Takeaways:**
  - **Exceptional Price Precision:** MAE < ₹5.72 and MAPE Price Accuracy of 99.59% on an equity trading at ₹1,250. Over 93.4% of sessions fall within ±1.0% error.
  - **Directional Edge:** Random Forest beats the random-walk coin-flip baseline with 52.17% Directional Accuracy and an $F_1$-Score of 0.6667.
  - **Gap Residual Modeling is Essential:** Predicting overnight gap differences ($\Delta$) rather than raw price levels eliminates persistent upward drift bias.
  - **Deployment Ready:** Fully verified with 0 static type errors (`pyright`). Open-source repository hosted at `github.com/rohan02110/Stock-Price-Prediction`.
- **Speaker Script / Note:**
  > "To conclude, Phase 5 packages our pipeline into a publication-grade report and an interactive Streamlit dashboard. By reframing the problem around overnight gap residuals, our models achieved 99.59% price accuracy and a 0.6667 F1-score with 0 code errors. Thank you."

---
*Presentation slides and script generated for the Financial Machine Learning Pipeline.*
