import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
import ta  # technical analysis library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def add_ta_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(10).std()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()

    # Optional TA indicators from ta library
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    
    df = df.dropna()
    return df

def prepare_lstm_data(df, lookback=60):
    df = add_ta_features(df)  # Add technical indicators

    feature_cols = ['close', 'return', 'volatility', 'sma_10', 'sma_20', 'rsi', 'macd']
    df = df[feature_cols].dropna()

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create X and y
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 0])  # Predicting 'close' price

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def build_lstm_model(lookback, n_features):
    """
    Build LSTM model
    
    Args:
        lookback (int): Number of time steps to look back
        
    Returns:
        tensorflow.keras.models.Sequential: LSTM model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(historical_data, prediction_days=7, interval="1day"):
    # Determine lookback
    if interval == "1hour":
        lookback = 24 * 7
    elif interval == "4hour":
        lookback = 6 * 7
    else:
        lookback = 60

    if len(historical_data) <= lookback:
        lookback = max(5, len(historical_data) // 5)

    # Prepare data
    X, y, scaler = prepare_lstm_data(historical_data, lookback)
    n_features = X.shape[2]
    
    # Split into training and validation sets
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Build model
    model = build_lstm_model(lookback, n_features)

   # Epochs
    if len(X) > 1000:
        epochs = 10
    elif len(X) > 500:
        epochs = 20
    else:
        epochs = 50

    # Train
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)

    # Evaluate
    y_pred = model.predict(X_val, verbose=0)
    # Create dummy arrays to match scaler's expected shape
    y_val_extended = np.zeros((len(y_val), 7))  # 7 = number of features
    y_val_extended[:, 0] = y_val  # 'close' was the first feature
    
    y_pred_extended = np.zeros((len(y_pred), 7))
    y_pred_extended[:, 0] = y_pred.flatten()
    
    # Only extract inverse of the 'close' column
    y_val_inv = scaler.inverse_transform(y_val_extended)[:, 0]
    y_pred_inv = scaler.inverse_transform(y_pred_extended)[:, 0]


    mae = mean_absolute_error(y_val_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))
    r2 = r2_score(y_val_inv, y_pred_inv)

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")

    return model, scaler, r2



'''def train_lstm_model(historical_data, prediction_days=7, interval="1day"):
    """
    Train an LSTM model for price prediction
    
    Args:
        historical_data (pandas.DataFrame): DataFrame with historical price data
        prediction_days (int): Number of days to predict
        interval (str): Time interval - "1hour", "4hour", or "1day"
        
    Returns:
        tuple: (model, scaler)
    """
    # Determine lookback period based on interval
    if interval == "1hour":
        lookback = 24 * 7  # 1 week of hourly data
    elif interval == "4hour":
        lookback = 6 * 7  # 1 week of 4-hour data
    else:  # 1day
        lookback = 60  # 60 days of daily data
    
    # Make sure we have enough data
    if len(historical_data) <= lookback:
        # If not enough data, use smaller lookback
        lookback = max(5, len(historical_data) // 5)
    
    # Prepare data
    X, y, scaler = prepare_lstm_data(historical_data, lookback)
    
    # Build model
    model = build_lstm_model(lookback)
    
    # Determine training epochs based on data size
    if len(X) > 1000:
        epochs = 10
    elif len(X) > 500:
        epochs = 20
    else:
        epochs = 50
    
    # Train model
    model.fit(
        X,
        y,
        batch_size=32,
        epochs=epochs,
        verbose=0
    )
    
    return model, scaler'''

def make_predictions(model, scaler, historical_data, prediction_days=7, interval="1day"):
    from ml_service import add_ta_features  # reuse feature function

    # Recreate technical indicators
    df = add_ta_features(historical_data)

    # Ensure feature order matches training
    feature_cols = ['close', 'return', 'volatility', 'sma_10', 'sma_20', 'rsi', 'macd']
    df = df[feature_cols].dropna()

    # Scale all features
    scaled_data = scaler.transform(df)

    # Determine lookback window
    if interval == "1hour":
        lookback = 24 * 7
    elif interval == "4hour":
        lookback = 6 * 7
    else:
        lookback = 60

    if len(scaled_data) <= lookback:
        lookback = max(5, len(scaled_data) // 5)

    # Get the last lookback window
    x_input = scaled_data[-lookback:]
    x_input = x_input.reshape(1, lookback, len(feature_cols))  # shape: (1, lookback, n_features)

    predictions = []

    for _ in range(prediction_days):
        pred_scaled = model.predict(x_input, verbose=0)

        # Build dummy array for inverse transform
        dummy_input = np.zeros((1, len(feature_cols)))
        dummy_input[0, 0] = pred_scaled[0, 0]  # only 'close'

        pred_actual = scaler.inverse_transform(dummy_input)[0, 0]
        predictions.append(pred_actual)

        # Update x_input for the next prediction
        next_input = np.append(x_input[:, 1:, :], dummy_input.reshape(1, 1, -1), axis=1)
        x_input = next_input

    return np.array(predictions)


def calculate_confidence_interval(model, scaler, historical_data, predictions, prediction_days, confidence=0.95, interval="1day"):
    """
    Calculate confidence intervals for LSTM predictions
    
    Args:
        model: Trained LSTM model
        scaler: Fitted scaler
        historical_data (pandas.DataFrame): DataFrame with historical price data
        predictions (numpy.ndarray): Array of predicted prices
        prediction_days (int): Number of days predicted
        confidence (float): Confidence level (0.0-1.0)
        interval (str): Time interval - "1hour", "4hour", or "1day"
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    # Determine uncertainty factors based on time interval
    if interval == "1hour":
        # Hourly predictions have faster-growing uncertainty but smaller magnitude
        uncertainty_factor = 0.01
    elif interval == "4hour":
        # 4-hour predictions have medium uncertainty
        uncertainty_factor = 0.015
    else:  # 1day
        # Daily predictions have standard uncertainty
        uncertainty_factor = 0.02
    
    # Calculate historical volatility
    returns = historical_data['close'].pct_change().dropna()
    historical_volatility = returns.std()
    
    # Calculate z-score for the given confidence level
    if confidence == 0.95:
        z_score = 1.96
    elif confidence == 0.99:
        z_score = 2.576
    else:
        z_score = 1.645  # 90% confidence
    
    # Increase uncertainty for predictions further into the future
    uncertainty_multiplier = np.array([1 + (uncertainty_factor * i) for i in range(prediction_days)])
    
    # Calculate the margin of error
    margin_of_error = historical_volatility * z_score * predictions * uncertainty_multiplier
    
    # Calculate lower and upper bounds
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    
    return lower_bound, upper_bound