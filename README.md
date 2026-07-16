# AAPL Stock Price Prediction using LSTM (vs ARIMA Benchmark)

A time-series forecasting project that predicts Apple Inc. (AAPL) stock closing prices using a stacked LSTM neural network, with hyperparameter tuning via Keras Tuner and a statistical ARIMA model built alongside it as a benchmark.

## 🔍 Overview

Stock prices are noisy, non-stationary, and notoriously hard to predict. This project explores how well a deep learning sequence model (LSTM) can learn patterns from historical closing prices and forecast future values, and benchmarks that against ARIMA, a classical time-series forecasting method.

**What this project does:**
- Pulls 4 years of daily AAPL OHLCV data (2020–2024) via `yfinance`
- Builds a baseline 3-layer stacked LSTM model
- Improves it with **Dropout + L2 regularization** and **Keras Tuner** hyperparameter search
- Forecasts the next 30 days of prices
- Benchmarks LSTM performance against a grid-searched **ARIMA** model
- Visualizes actual vs predicted prices across training, test, and future windows

---

## 🗂️ Dataset

| Detail | Value |
|---|---|
| Ticker | `AAPL` (Apple Inc.) |
| Source | Yahoo Finance (`yfinance`) |
| Date range | 2020-01-01 to 2024-01-01 |
| Frequency | Daily |
| Total records | 1,006 trading days |
| Feature used | `Close` price only (univariate) |
| Train/Test split | 65% / 35% (baseline) — 80% / 20% (tuned model) |

The data is normalized using `MinMaxScaler` (range 0–1) before being fed into the LSTM, since neural networks converge faster and more reliably on scaled inputs.

---

## 🧠 Methodology

### 1. Data Preparation
- Extracted the `Close` column and scaled it with `MinMaxScaler`
- Converted the series into supervised learning format using a **sliding window** of `time_step = 100` (i.e., the last 100 days predict the next day)
- Reshaped into `[samples, time steps, features]` format required by LSTM

### 2. Baseline LSTM Architecture
```
LSTM(50, return_sequences=True) → LSTM(50, return_sequences=True) → LSTM(50) → Dense(1)
```
- Loss: MSE | Optimizer: Adam | Epochs: 100 | Batch size: 64
- Total trainable parameters: 50,851

### 3. Model Optimization
The baseline model was prone to overfitting, so it was improved with:
- **Dropout layers (0.2)** after each LSTM block
- **L2 kernel regularization** to penalize large weights
- **Keras Tuner (RandomSearch)** to search over:
  - LSTM units (50–200, step 50) for both layers
  - L2 regularization strength (0.001–0.01, log scale)
  - Dropout rate (0.2–0.5)
  - Dense layer size (10–50 units)

**Best found configuration:** LSTM(150) → Dropout → LSTM(150) → Dropout → Dense(40) → Dense(1)

### 4. Forecasting
- Used a recursive multi-step forecasting loop to predict the next **30 days** beyond the available data, feeding each prediction back in as input for the next step.

### 5. Benchmark: ARIMA
To validate that the LSTM was actually learning something better than a classical model, an ARIMA model was fit on the same train/test split:
- Grid-searched `(p, d, q)` combinations over `p ∈ [0,5)`, `d ∈ [0,2)`, `q ∈ [0,5)` using AIC as the selection criterion
- **Best order found: (0, 1, 1)**, AIC = -3628.38

---

## 📊 Results

All metrics below are reported on the **same normalized (0–1) scale** for a fair, apples-to-apples comparison between LSTM and ARIMA.

| Model | Train RMSE | Train MAE | Test RMSE | Test MAE |
|---|---|---|---|---|
| **Tuned LSTM** (Keras Tuner) | 0.0266 | 0.0213 | **0.0186** | **0.0147** |
| ARIMA (best order 0,1,1) | 0.0202 | 0.0147 | 0.1987 | 0.1702 |

**Key takeaway:** ARIMA fits the training data slightly better (it's essentially memorizing recent trend/differencing), but it **generalizes poorly** — its test error is over 10x higher than the tuned LSTM. The LSTM, with regularization and tuned hyperparameters, generalizes far better to unseen data, confirming it captures more meaningful patterns rather than just short-term momentum.

> ⚠️ **Note on the baseline model metrics:** The very first baseline LSTM (before tuning) reported unusually large RMSE/MAE values (~138–171) in the notebook. This was because predictions were inverse-scaled back to USD while being compared against still-normalized ground truth — a scale mismatch, not a reflection of true model performance. This is called out here transparently and is fixed in the tuned model's evaluation (Section above), where both predictions and targets are compared on the same scale. Fixing this scale consistency end-to-end is listed under **Future Improvements**.

---

## 🖼️ Visualizations

The notebook generates the following plots:
- Historical AAPL closing price trend (2020–2024)
- Actual vs Training Predictions vs Test Predictions (baseline LSTM)
- Actual vs Optimized Test Predictions (tuned LSTM)
- 30-day future price forecast (LSTM)
- LSTM vs ARIMA side-by-side forecast comparison

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9 |
| Data | `yfinance`, `pandas`, `numpy` |
| Preprocessing | `scikit-learn` (`MinMaxScaler`) |
| Deep Learning | `TensorFlow` / `Keras` (LSTM, Dense, Dropout) |
| Hyperparameter Tuning | `keras-tuner` |
| Statistical Modeling | `statsmodels` (ARIMA) |
| Evaluation | `scikit-learn` (RMSE, MAE) |
| Visualization | `matplotlib` |

---

## 📁 Project Structure

```
lstm-stock-price-prediction/
│
├── lstms_stock_price_prediction.ipynb   # Main notebook (data → model → evaluation → forecast)
├── AAPL.csv                              # Cached raw OHLCV data (generated on first run)
├── hyperparameter_tuning/                # Keras Tuner search logs/checkpoints (auto-generated)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/lstm-stock-price-prediction.git
cd lstm-stock-price-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook lstms_stock_price_prediction.ipynb
```

**`requirements.txt`**
```
yfinance
pandas
numpy
matplotlib
scikit-learn
tensorflow
keras-tuner
statsmodels
```

Run the notebook top to bottom — it will download fresh AAPL data, train both LSTM variants, tune hyperparameters, fit ARIMA, and generate all plots and metrics.

---

## 🚧 Limitations & Future Improvements

- **Univariate only:** Currently uses only `Close` price. Adding volume, technical indicators (RSI, MACD, Bollinger Bands), or macroeconomic features could improve signal.
- **Scale consistency:** Standardize all evaluation code to compare predictions and ground truth on the same scale throughout (see note in Results).
- **Walk-forward validation:** Replace the single train/test split with rolling-window / walk-forward validation for a more realistic backtest of forecasting performance.
- **Multi-stock generalization:** Test the same pipeline on other tickers/sectors to check if the architecture generalizes beyond AAPL.
- **Attention / Transformer-based models:** Compare LSTM against attention-based sequence models for potential accuracy gains.
- **Deployment:** Wrap the trained model in a simple API (FastAPI) or dashboard (Streamlit) for interactive "predict next N days" usage.

## 👤 Author

**Rohit** — MSc Economics & Management, IIIT Lucknow
Interested in Data Analytics, Data Engineering, and building end-to-end ML/data projects.

If you find this useful, feel free to ⭐ the repo or connect!
