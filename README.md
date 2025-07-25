**📈 Asset Price Prediction Using LSTM**

A deep learning-powered web application that predicts future prices of financial assets such as stocks, crypto, forex, and commodities using LSTM (Long Short-Term Memory) neural networks and technical analysis indicators. This project combines time-series forecasting, financial data engineering, and interactive visualization into a single real-time Streamlit dashboard.

**📌 Project Overview**
This system allows users to:
Select any asset (e.g., BTC-USD, AAPL, EUR-USD, GOLD)
Choose time intervals (1h, 4h, or 1 day)
View predictions with confidence intervals
Analyze risk using real-time metrics (Volatility, Sharpe Ratio, Drawdown, VaR)

**✅ Built using:**
Python
TensorFlow/Keras (LSTM model)
yfinance (data)
TA-Lib / ta library (technical indicators)
SQLite (caching historical data)
Streamlit + Plotly (visualization)

**⚙️ Features**
🔄 Dynamic Data Fetching: Fetches live historical data from Yahoo Finance using yfinance
🧠 LSTM-Based Price Prediction: Trains a fresh LSTM model per asset and time frame
📊 Technical Indicators: Uses features like RSI, MACD, SMA, Volatility, and Returns
📉 Risk Metrics Display: Shows Sharpe Ratio, Value at Risk (VaR), Volatility, and Max Drawdown
📈 Interactive Visualization: Real-time charts with prediction lines and confidence intervals
🧪 Model Evaluation: R² Score, MAE, and RMSE are computed for each asset
🧩 Modular Codebase: Clean separation of data, modeling, visualization, and app logic

**🧠 Model Architecture**
LSTM layer (128 units)
LSTM layer (64 units)
Dropout (30%)
Dense layer (32)
Output layer (1 neuron for price)

Optimizer: Adam | Loss: MSE

Evaluation: R² Score, MAE, RMSE

The model learns from the past N timesteps of price and indicators to predict the next price point.
