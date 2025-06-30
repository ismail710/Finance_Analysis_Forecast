# Financial Analysis & Forecasting Dashboard

![Demo](demo.gif)

This Streamlit dashboard allows you to analyze and forecast stock data using interactive charts and machine learning models. It provides financial statistics, candlestick charts, support level visualization, and advanced forecasting (ARIMA, XGBoost, LSTM).

## Features
- **Stock Ticker & Date Selection:** Choose any stock and date range for analysis and forecasting.
- **Candlestick Chart:** Visualize price action with support levels.
- **Statistical Analysis:** Daily returns, normality tests, QQ-plots, and moving averages.
- **Stock Comparison:** Compare multiple stocks on price, volatility, beta, Sharpe ratio, and key financial ratios.
- **Forecasting:** Predict future prices using ARIMA, XGBoost, and LSTM models.
- **Download Data:** Export historical data as CSV.

## Requirements
- Python 3.8+
- streamlit
- yfinance
- pandas
- numpy
- plotly
- scipy
- statsmodels
- scikit-learn
- xgboost (optional, for XGBoost forecasts)
- keras, tensorflow (optional, for LSTM forecasts)

Install requirements with:
```
pip install -r requirements.txt
```

## How to Run
1. Place your files in a folder (e.g., `Finance`).
2. Open a terminal in that folder.
3. Run:
```
streamlit run Home.py
```
4. Use the sidebar and navigation buttons to explore analysis and forecasting features.

## Usage
- Enter a stock ticker (e.g., `AAPL`, `TSLA`) and select a date range.
- Navigate between **Analysis** and **Forecast** pages using the buttons.
- Use the comparison section to analyze multiple stocks.
- Download data as needed.

---

**Note:**
- Some features (like XGBoost and LSTM) require additional libraries. Install them if you want to use those models.
