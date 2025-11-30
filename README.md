<h2>**📈 Asset Price Prediction Using LSTM**</h2>

A Deep Learning System for Multi-Interval Financial Market Forecasting

This project is a full-stack machine learning application that predicts future prices of financial assets such as stocks, cryptocurrencies, forex pairs, and commodities using LSTM (Long Short-Term Memory) neural networks.
The system integrates real-time data ingestion, technical analysis, risk analytics, and interactive visualization into a unified Streamlit dashboard.

🎯 Project Objectives

✔ Develop a robust LSTM model for multi-step financial time-series forecasting
✔ Support multiple assets and intervals (1h, 4h, 1d)
✔ Incorporate technical indicators for more stable predictions
✔ Provide real-time market risk analysis (Volatility, VaR, Sharpe Ratio, Drawdown)
✔ Build a user-friendly dashboard for traders, analysts, and researchers
✔ Maintain a modular, scalable codebase suitable for deployment and extension
✔ Cache data efficiently using SQLite to avoid redundant API calls
✔ Enable live visualization of past data, predictions, and confidence intervals

📌 Key Features
🔄 Dynamic Data Fetching

Pulls the latest historical OHLCV data using yfinance

Automatically updates based on the chosen asset and interval

Uses SQLite caching to reduce network calls

🧠 LSTM-Based Price Prediction

Trains a dedicated LSTM model per asset + time interval

Predicts the next price and provides confidence bounds

Computes evaluation metrics:

R² Score

MAE

RMSE

📊 Technical Indicators (ta / TA-Lib)

Integrated indicators include:

SMA / EMA

RSI

MACD

Bollinger Bands

Volatility

Daily Returns

These features improve the model’s predictive capability beyond raw prices.

📉 Risk Metrics Dashboard

Real-time risk evaluation:

Volatility (%)

Sharpe Ratio

Maximum Drawdown

Value at Risk (VaR 95%)

📈 Interactive Visualization

Built using Plotly + Streamlit, including:

Historical price chart

Future price predictions

Confidence bands

Indicator overlays

Real-time metric cards
