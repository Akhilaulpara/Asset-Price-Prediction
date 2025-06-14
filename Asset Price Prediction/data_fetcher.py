import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_real_time_price(ticker, asset_type):
    """
    Fetch real-time price for a given ticker
    
    Args:
        ticker (str): Ticker symbol
        asset_type (str): Type of asset (Cryptocurrencies, Stocks, Forex, Commodities)
        
    Returns:
        tuple: (current_price, timestamp)
    """
    try:
        # Get data from yfinance
        ticker_data = yf.Ticker(ticker)
        
        # Get latest price info
        latest_data = ticker_data.history(period="1d")
        
        if latest_data.empty:
            return 0.0, datetime.now()
        
        # Get the latest close price
        current_price = latest_data['Close'].iloc[-1]
        
        # Get the timestamp
        timestamp = latest_data.index[-1]
        
        return current_price, timestamp
    
    except Exception as e:
        # Log the error
        print(f"Error fetching real-time price for {ticker}: {str(e)}")
        
        # Return a placeholder value
        return 0.0, datetime.now()

def fetch_historical_data(ticker, asset_type, days=30, interval="1day"):
    """
    Fetch historical price data for a given ticker
    
    Args:
        ticker (str): Ticker symbol
        asset_type (str): Type of asset (Cryptocurrencies, Stocks, Forex, Commodities)
        days (int): Number of days of history to fetch
        interval (str): Time interval - "1hour", "4hour", or "1day"
        
    Returns:
        pandas.DataFrame: DataFrame with historical price data
    """
    try:
        # Map our internal interval names to yfinance interval format
        yf_interval = {
            "1hour": "1h",
            "4hour": "4h",
            "1day": "1d"
        }.get(interval, "1d")
        
        # Calculate start date based on days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Calculate period to fetch
        # For hourly data, yfinance requires period="xd" format
        if interval == "1hour" or interval == "4hour":
            period = f"{days}d"
        else:
            period = f"{days}d"
            
        # Fetch data using yfinance
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=yf_interval
        )
        
        # If we got data, clean and return it
        if not data.empty:
            # Rename columns to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            return data
        else:
            print(f"No historical data found for {ticker} with interval {interval}")
            return pd.DataFrame()
    
    except Exception as e:
        # Log the error
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        
        # Return an empty DataFrame
        return pd.DataFrame()
