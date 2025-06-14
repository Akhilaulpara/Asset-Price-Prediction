import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database connection URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///local_data.db")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Asset(Base):
    """Table for storing information about financial assets being tracked"""
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True)
    name = Column(String(100))
    asset_type = Column(String(20), nullable=False)  # Cryptocurrencies, Stocks, Forex, Commodities
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    prices = relationship("PriceData", back_populates="asset", cascade="all, delete-orphan")
    watchlists = relationship("WatchlistItem", back_populates="asset", cascade="all, delete-orphan")

class PriceData(Base):
    """Table for storing historical price data for assets"""
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    interval = Column(String(10), default="1day")  # "1hour", "4hour", "1day"
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    
    # Relationships
    asset = relationship("Asset", back_populates="prices")
    
    # Composite unique constraint
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )

class Watchlist(Base):
    """Table for storing user-created watchlists"""
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    items = relationship("WatchlistItem", back_populates="watchlist", cascade="all, delete-orphan")

class WatchlistItem(Base):
    """Table for storing assets in watchlists"""
    __tablename__ = "watchlist_items"
    
    id = Column(Integer, primary_key=True)
    watchlist_id = Column(Integer, ForeignKey("watchlists.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    added_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    watchlist = relationship("Watchlist", back_populates="items")
    asset = relationship("Asset", back_populates="watchlists")
    
    # Composite unique constraint to prevent duplicates in a watchlist
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )

class UserSettings(Base):
    """Table for storing user settings"""
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True)
    setting_key = Column(String(50), nullable=False, unique=True)
    setting_value = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

def init_db():
    """Initialize the database by creating tables and setting up initial data"""
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create a session
    session = Session()
    
    try:
        # Check if we already have assets, if not add default assets
        if session.query(Asset).count() == 0:
            # Add default cryptocurrencies
            cryptos = [
                {"symbol": "BTC-USD", "name": "Bitcoin", "asset_type": "Cryptocurrencies"},
                {"symbol": "ETH-USD", "name": "Ethereum", "asset_type": "Cryptocurrencies"},
                {"symbol": "XRP-USD", "name": "Ripple", "asset_type": "Cryptocurrencies"},
                {"symbol": "LTC-USD", "name": "Litecoin", "asset_type": "Cryptocurrencies"},
                {"symbol": "ADA-USD", "name": "Cardano", "asset_type": "Cryptocurrencies"},
            ]
            
            # Add default stocks
            stocks = [
                {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "Stocks"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "asset_type": "Stocks"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "asset_type": "Stocks"},
                {"symbol": "AMZN", "name": "Amazon.com, Inc.", "asset_type": "Stocks"},
                {"symbol": "TSLA", "name": "Tesla, Inc.", "asset_type": "Stocks"},
            ]
            
            # Add default forex pairs
            forex = [
                {"symbol": "EURUSD=X", "name": "Euro / US Dollar", "asset_type": "Forex"},
                {"symbol": "GBPUSD=X", "name": "British Pound / US Dollar", "asset_type": "Forex"},
                {"symbol": "USDJPY=X", "name": "US Dollar / Japanese Yen", "asset_type": "Forex"},
                {"symbol": "AUDUSD=X", "name": "Australian Dollar / US Dollar", "asset_type": "Forex"},
                {"symbol": "USDCAD=X", "name": "US Dollar / Canadian Dollar", "asset_type": "Forex"},
            ]
            
            # Add default commodities
            commodities = [
                {"symbol": "GC=F", "name": "Gold", "asset_type": "Commodities"},
                {"symbol": "SI=F", "name": "Silver", "asset_type": "Commodities"},
                {"symbol": "CL=F", "name": "Crude Oil", "asset_type": "Commodities"},
                {"symbol": "HG=F", "name": "Copper", "asset_type": "Commodities"},
                {"symbol": "NG=F", "name": "Natural Gas", "asset_type": "Commodities"},
            ]
            
            # Add all assets to the database
            for asset_data in cryptos + stocks + forex + commodities:
                asset = Asset(**asset_data)
                session.add(asset)
            
            # Create a default watchlist
            default_watchlist = Watchlist(name="Default Watchlist", description="Default watchlist with popular assets", is_default=True)
            session.add(default_watchlist)
            
            # Commit to save assets before adding watchlist items
            session.commit()
            
            # Get the default watchlist
            default_watchlist = session.query(Watchlist).filter_by(is_default=True).first()
            
            # Add some assets to the default watchlist
            default_assets = ["BTC", "ETH", "AAPL", "MSFT", "GC"]
            for symbol in default_assets:
                asset = session.query(Asset).filter_by(symbol=symbol).first()
                if asset:
                    watchlist_item = WatchlistItem(watchlist_id=default_watchlist.id, asset_id=asset.id)
                    session.add(watchlist_item)
            
            # Commit all changes
            session.commit()
            
            print("Database initialized with default data.")
        else:
            print("Database already contains data, skipping initialization.")
    
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        session.rollback()
        raise e
    finally:
        session.close()

def get_asset_by_symbol(symbol):
    """Get an asset by its symbol"""
    session = Session()
    try:
        return session.query(Asset).filter_by(symbol=symbol).first()
    finally:
        session.close()

def get_assets_by_type(asset_type):
    """Get all assets of a specific type"""
    session = Session()
    try:
        return session.query(Asset).filter_by(asset_type=asset_type).all()
    finally:
        session.close()

def get_all_watchlists():
    """Get all watchlists"""
    session = Session()
    try:
        return session.query(Watchlist).all()
    finally:
        session.close()

def get_default_watchlist():
    """Get the default watchlist"""
    session = Session()
    try:
        default_watchlist = session.query(Watchlist).filter_by(is_default=True).first()
        if default_watchlist is None:
            # Create a default watchlist if none exists
            default_watchlist = Watchlist(name="Default Watchlist", description="Default watchlist", is_default=True)
            session.add(default_watchlist)
            session.commit()
        return default_watchlist
    finally:
        session.close()

def get_watchlist_assets(watchlist_id):
    """Get all assets in a watchlist"""
    session = Session()
    try:
        if watchlist_id is None:
            # If no watchlist_id is provided, return an empty list
            return []
            
        items = session.query(WatchlistItem).filter_by(watchlist_id=watchlist_id).all()
        asset_ids = [item.asset_id for item in items]
        return session.query(Asset).filter(Asset.id.in_(asset_ids)).all() if asset_ids else []
    finally:
        session.close()

def add_price_data(asset_id, df, interval="1day"):
    """
    Add price data from a DataFrame to the database
    
    Args:
        asset_id (int): The ID of the asset
        df (pandas.DataFrame): DataFrame with price data (columns: open, high, low, close, volume)
        interval (str): Time interval - "1hour", "4hour", or "1day"
    """
    session = Session()
    try:
        # Check if the asset exists
        asset = session.query(Asset).filter_by(id=asset_id).first()
        if not asset:
            raise ValueError(f"Asset with ID {asset_id} not found")
        
        # Add each row as a PriceData entry
        for date, row in df.iterrows():
            # Check if this data point already exists
            date_val = date if isinstance(date, datetime) else pd.to_datetime(date)
            existing = session.query(PriceData).filter_by(
                asset_id=asset_id,
                date=date_val,
                interval=interval
            ).first()
            
            if existing:
                # Update existing entry
                existing.open_price = row.get('Open', None)
                existing.high_price = row.get('High', None)
                existing.low_price = row.get('Low', None)
                existing.close_price = row.get('Close', None)
                existing.volume = row.get('Volume', 0)
            else:
                # Create new entry
                price_data = PriceData(
                    asset_id=asset_id,
                    date=date_val,
                    interval=interval,
                    open_price=row.get('Open', None),
                    high_price=row.get('High', None),
                    low_price=row.get('Low', None),
                    close_price=row.get('Close', None),
                    volume=row.get('Volume', 0)
                )
                session.add(price_data)
        
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_price_data(asset_id, start_date=None, end_date=None, interval="1day"):
    """
    Get price data for an asset from the database
    
    Args:
        asset_id (int): The ID of the asset
        start_date (datetime, optional): Start date for price data
        end_date (datetime, optional): End date for price data
        interval (str, optional): Time interval - "1hour", "4hour", or "1day"
        
    Returns:
        pandas.DataFrame: DataFrame with price data
    """
    session = Session()
    try:
        query = session.query(PriceData).filter_by(
            asset_id=asset_id,
            interval=interval
        )
        
        if start_date:
            query = query.filter(PriceData.date >= start_date)
        
        if end_date:
            query = query.filter(PriceData.date <= end_date)
        
        # Order by date
        query = query.order_by(PriceData.date)
        
        # Get all price data
        price_data = query.all()
        
        # Convert to DataFrame
        data = {
            'open': [p.open_price for p in price_data],
            'high': [p.high_price for p in price_data],
            'low': [p.low_price for p in price_data],
            'close': [p.close_price for p in price_data],
            'volume': [p.volume for p in price_data]
        }
        
        # Create DataFrame with date as index
        df = pd.DataFrame(data, index=[p.date for p in price_data])
        
        return df
    except Exception as e:
        print(f"Error getting price data: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def fetch_latest_data(asset_id, start_date, end_date, interval="1day"):
    """
    Fetch the latest data for an asset, using database data if available
    and fetching latest data from yfinance when needed.
    
    Args:
        asset_id (int): The ID of the asset
        start_date (datetime): Start date for price data
        end_date (datetime): End date for price data
        interval (str): Time interval - "1hour", "4hour", or "1day"
        
    Returns:
        pandas.DataFrame: DataFrame with price data
    """
    # Get asset information
    session = Session()
    try:
        asset = session.query(Asset).filter_by(id=asset_id).first()
        if not asset:
            return None
        
        # First try to get data from the database
        db_data = get_price_data(asset_id, start_date, end_date, interval)
        
        # Map interval to yfinance format
        yf_interval = {
            "1hour": "1h",
            "4hour": "4h",
            "1day": "1d"
        }.get(interval, "1d")
        
        # If we have no data or data is outdated, fetch from yfinance
        if db_data.empty or db_data.index[-1].date() < datetime.now().date():

            # Fetch from yfinance
            symbol = asset.symbol
            
            # Calculate the period to fetch based on the interval
            if interval == "1hour":
                period = "60d"
            elif interval == "4hour":
                period = "60d"
            else:  # 1day
                period = "1y"
            
            # Fetch data from yfinance
            yf_data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval
            )
            
            # Check if we got data
            if not yf_data.empty:
                # Rename columns to match our database structure
                yf_data = yf_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Store this data in the database
                add_price_data(asset_id, yf_data, interval)
                
                # Convert to lowercase column names for consistency
                yf_data.columns = ['open', 'high', 'low', 'close', 'volume']
                
                return yf_data
            
            # If no yfinance data, return what we have in the database
            return db_data
        
        # If database data is up to date, return it
        return db_data
    
    except Exception as e:
        print(f"Error fetching data for {asset.symbol if asset else asset_id}: {str(e)}")
        return None
    finally:
        session.close()
