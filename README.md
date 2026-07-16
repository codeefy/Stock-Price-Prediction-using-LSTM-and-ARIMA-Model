# Stock Price Prediction using LSTM & ARIMA

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-D00000?logo=keras)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-DataAnalysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-ScientificComputing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Statsmodels](https://img.shields.io/badge/ARIMA-Statsmodels-2C5AA0)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 🚀 Project Overview

This project predicts **Apple Inc. (AAPL)** stock closing prices using a **Stacked Long Short-Term Memory (LSTM)** network optimized with **Keras Tuner**, and compares its forecasting performance against a classical **ARIMA** model.

The objective is to evaluate whether deep learning can better capture nonlinear temporal dependencies in financial time-series data compared to traditional statistical forecasting techniques.

> **Result:** The tuned LSTM achieved approximately **90% lower test RMSE** than the optimized ARIMA benchmark.

---

## ✨ Key Highlights

- 📊 Forecasts AAPL closing prices using Deep Learning
- 🧠 Three-layer Stacked LSTM architecture
- ⚙️ Hyperparameter Optimization using Keras Tuner
- 📉 ARIMA Benchmark for statistical comparison
- 📈 Recursive 30-day future forecasting
- 🔍 Comprehensive evaluation using RMSE & MAE
- 📊 Multiple visualizations of predictions and forecasts

---

# 🛠 Tech Stack

| Category | Technologies |
|------------|----------------|
| Programming | Python |
| Data Collection | yfinance |
| Data Processing | Pandas, NumPy |
| Machine Learning | TensorFlow, Keras |
| Hyperparameter Tuning | Keras Tuner |
| Statistical Modeling | Statsmodels (ARIMA) |
| Evaluation | Scikit-Learn |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

# 📚 Table of Contents

- Overview
- Project Workflow
- Dataset
- Methodology
- Model Architecture
- Hyperparameter Optimization
- Results
- Forecasting
- Visualizations
- Installation
- Future Improvements
- Project Structure
- Author

---

# 🔄 Project Workflow

```
Yahoo Finance
      │
      ▼
Data Collection
      │
      ▼
Preprocessing
(MinMax Scaling)
      │
      ▼
Sliding Window
(Time Step = 100)
      │
      ▼
Stacked LSTM
      │
      ▼
Hyperparameter Tuning
(Keras Tuner)
      │
      ▼
Prediction
      │
      ├────────► Test Prediction
      │
      └────────► 30-Day Forecast
      │
      ▼
Evaluation
(RMSE • MAE)

ARIMA Model
      │
      ▼
Benchmark Comparison
```

# 🧠 LSTM Architecture

```
Input (100 Days)

        │

LSTM (150 Units)

        │

Dropout (0.2)

        │

LSTM (150 Units)

        │

Dropout (0.2)

        │

Dense (40)

        │

Dense (1)

        │

Predicted Closing Price
```

# 📊 Performance

| Model | Test RMSE | Test MAE |
|---------|-----------|-----------|
| ⭐ Tuned LSTM | **0.0186** | **0.0147** |
| ARIMA | 0.1987 | 0.1702 |

## 🏆 Performance Summary

✅ 90% lower RMSE than ARIMA

✅ Better generalization

✅ Reduced overfitting using Dropout + L2 Regularization

✅ Successfully predicts future stock movement trends

# 📁 Repository Structure

```
📦 lstm-stock-price-prediction
│
├── 📓 lstms_stock_price_prediction.ipynb
├── 📄 README.md
├── 📄 requirements.txt
├── 📊 AAPL.csv
├── 📂 hyperparameter_tuning/
└── 📂 figures/
```

# 📷 Visualizations

The notebook produces the following visualizations:

- 📈 Historical Stock Price Trend
- 📉 Training vs Testing Prediction
- 📊 Optimized LSTM Prediction
- 🔮 30-Day Future Forecast
- ⚖️ LSTM vs ARIMA Comparison

> Replace the placeholders below with screenshots after running the notebook.

```
images/

├── stock_price.png
├── lstm_prediction.png
├── future_forecast.png
└── comparison.png
```

# ⚙️ Installation

```bash
git clone https://github.com/username/lstm-stock-price-prediction.git

cd lstm-stock-price-prediction

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

jupyter notebook
```

# 🚀 Future Work

- Incorporate Technical Indicators (RSI, MACD, EMA)
- Add Trading Volume
- Include Macroeconomic Variables
- Walk-Forward Cross Validation
- Compare with GRU and Transformer Models
- Deploy with FastAPI
- Streamlit Dashboard
- Dockerize the Application
- CI/CD using GitHub Actions
