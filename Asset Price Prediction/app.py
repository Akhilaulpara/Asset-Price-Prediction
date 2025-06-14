import streamlit as st
import pandas as pd
import numpy as np
import pytz
import os
import time
from datetime import datetime, timedelta
import streamlit.components.v1 as components  # Required for embedding HTML (TradingView)

from data_service import (
    init_db, 
    get_asset_by_symbol, 
    get_assets_by_type, 
    get_all_watchlists, 
    get_default_watchlist, 
    get_watchlist_assets, 
    get_price_data, 
    fetch_latest_data
)

from ml_service import (
    train_lstm_model,
    make_predictions,
    calculate_confidence_interval
)

from visualization import (
    plot_historical_data,
    plot_predictions_with_confidence,
    plot_risk_metrics_gauge
)


st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_database():
    try:
        init_db()
        return True
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return False


def map_to_tradingview_symbol(asset_type, symbol):
    if asset_type == "Cryptocurrencies":
        return f"BINANCE:{symbol.replace('-', '')}"
    elif asset_type == "Stocks":
        return f"NASDAQ:{symbol}"
    elif asset_type == "Forex":
        tv_forex_map = {
            "EURUSD=X": "FX_IDC:EURUSD",
            "GBPUSD=X": "FX_IDC:GBPUSD",
            "USDJPY=X": "FX_IDC:USDJPY",
            "AUDUSD=X": "FX_IDC:AUDUSD",
            "USDCAD=X": "FX_IDC:USDCAD",
        }
        return tv_forex_map.get(symbol, "FX_IDC:EURUSD")
    elif asset_type == "Commodities":
        tv_commodities_map = {
            "GC=F": "TVC:GOLD",
            "SI=F": "TVC:SILVER",
            "CL=F": "TVC:USOIL",
            "NG=F": "TVC:NATGAS",
        }
        return tv_commodities_map.get(symbol, "TVC:GOLD")
    return symbol




st.title("Financial Dashboard")
st.markdown("Real-time asset prices and ML-based price predictions")


db_initialized = initialize_database()

if not db_initialized:
    st.error("Database initialization failed. Please check your DATABASE_URL environment variable.")
    st.stop()


with st.sidebar:
    st.header("Settings")
    
    asset_type = st.selectbox(
        "Asset Type",
        ["Cryptocurrencies", "Stocks", "Forex", "Commodities"]
    )
    
    assets = get_assets_by_type(asset_type)
    if not assets:
        st.warning(f"No {asset_type} found in the database.")
        st.stop()
    
    asset_symbols = [asset.symbol for asset in assets]
    selected_symbol = st.selectbox("Select Asset", asset_symbols)
    selected_asset = get_asset_by_symbol(selected_symbol)
    
    interval = st.selectbox(
        "Time Interval",
        ["1hour", "4hour", "1day"],
        format_func=lambda x: {"1hour": "1 Hour", "4hour": "4 Hours", "1day": "1 Day"}[x]
    )
    
    prediction_days = st.slider("Prediction Days", 1, 30, 7)
    
    st.header("Watchlists")
    watchlists = get_all_watchlists()
    watchlist_names = [w.name for w in watchlists]
    selected_watchlist = st.selectbox("Select Watchlist", watchlist_names)
    selected_watchlist_obj = next((w for w in watchlists if w.name == selected_watchlist), None)
    
    if selected_watchlist_obj:
        watchlist_assets = get_watchlist_assets(selected_watchlist_obj.id)
        watchlist_symbols = [asset.symbol for asset in watchlist_assets]
        
        st.subheader("Watchlist Assets")
        for symbol in watchlist_symbols:
            if st.button(symbol, key=f"watchlist_{symbol}"):
                st.session_state.selected_symbol = symbol
                st.rerun()


if selected_asset:
    end_date = datetime.now()
    
    if interval == "1hour":
        start_date = end_date - timedelta(days=60 + prediction_days)
    elif interval == "4hour":
        start_date = end_date - timedelta(days=120 + prediction_days)
    else:
        start_date = end_date - timedelta(days=365 + prediction_days)
    
    with st.spinner("Fetching latest data..."):
        historical_data = fetch_latest_data(selected_asset.id, start_date, end_date, interval)
    
    if historical_data is None or historical_data.empty:
        st.error(f"No historical data available for {selected_asset.symbol}. Please select another asset.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            current_price = historical_data['close'].iloc[-1]
            previous_price = historical_data['close'].iloc[-2]
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            st.subheader("Current Price")
            price_color = "green" if price_change >= 0 else "red"
            st.markdown(f"<h2 style='color: {price_color};'>${current_price:.2f}</h2>", unsafe_allow_html=True)
            
            change_icon = "⬆️" if price_change >= 0 else "⬇️"
            st.markdown(f"{change_icon} ${abs(price_change):.2f} ({price_change_pct:.2f}%)")
            #st.text(f"Last updated: {historical_data.index[-1]}")

            ist = pytz.timezone("Asia/Kolkata")
            last_updated = pd.to_datetime(historical_data.index[-1])
            
            # Ensure UTC
            if last_updated.tzinfo is None:
                last_updated = last_updated.tz_localize("UTC")
            else:
                last_updated = last_updated.tz_convert("UTC")
            
            # Convert to IST
            last_updated_ist = last_updated.astimezone(ist)

            # Display the timestamp
            now_ist = datetime.now(ist)
            st.text(f"Your Local Time (IST): {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
                        
        
        with col2:
            with st.spinner("Training model and making predictions..."):
                model, scaler, r2 = train_lstm_model(historical_data, prediction_days, interval)
                predictions = make_predictions(model, scaler, historical_data, prediction_days, interval)
                lower_bound, upper_bound = calculate_confidence_interval(model, scaler, historical_data, predictions, prediction_days, interval=interval)
            
            st.subheader("Predicted Price (Next Period)")
            next_prediction = predictions[0]
            pred_change = next_prediction - current_price
            pred_change_pct = (pred_change / current_price) * 100
            pred_color = "green" if pred_change >= 0 else "red"
            pred_icon = "⬆️" if pred_change >= 0 else "⬇️"
            
            st.markdown(f"<h2 style='color: {pred_color};'>${next_prediction:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"{pred_icon} ${abs(pred_change):.2f} ({pred_change_pct:.2f}%)")
            st.text(f"Confidence Interval: ${lower_bound[0]:.2f} - ${upper_bound[0]:.2f}")
            st.subheader("Model Accuracy")
            accuracy_pct = r2 * 100
            st.markdown(f"<h4 style='color: teal;'>R² Accuracy: {accuracy_pct:.2f}%</h4>", unsafe_allow_html=True)

        
        st.subheader("Historical Price Data with Predictions")
        last_date = historical_data.index[-1]
        if interval == "1hour":
            future_dates = [last_date + timedelta(hours=i+1) for i in range(prediction_days)]
        elif interval == "4hour":
            future_dates = [last_date + timedelta(hours=(i+1)*4) for i in range(prediction_days)]
        else:
            future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        fig = plot_historical_data(historical_data, future_dates, predictions, lower_bound, upper_bound)
        st.plotly_chart(fig, use_container_width=True)
        
       
        st.subheader("TradingView Chart")
        tv_symbol = map_to_tradingview_symbol(asset_type, selected_asset.symbol)
        tradingview_iframe = f"""
        <iframe src="https://www.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=60&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Asia/Kolkata"
                width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no">
        </iframe>
        """
        components.html(tradingview_iframe, height=520)

        

        
        
        st.subheader("Risk Metrics")
        returns = historical_data['close'].pct_change().dropna()
        if interval == "1hour":
            annualization_factor = np.sqrt(252 * 24)
        elif interval == "4hour":
            annualization_factor = np.sqrt(252 * 6)
        else:
            annualization_factor = np.sqrt(252)
        
        volatility = returns.std() * annualization_factor
        var_95 = np.percentile(returns, 5)
        cumulative_returns = (1 + returns).cumprod()
        max_return = cumulative_returns.cummax()
        drawdown = (cumulative_returns / max_return) - 1
        max_drawdown = drawdown.min()
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
        
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        with risk_col1:
            fig_vol = plot_risk_metrics_gauge(value=volatility * 100, title="Annualized Volatility", max_value=100, suffix="%")
            st.plotly_chart(fig_vol, use_container_width=True)
        with risk_col2:
            fig_var = plot_risk_metrics_gauge(value=abs(var_95) * 100, title="Value at Risk (95%)", max_value=50, suffix="%")
            st.plotly_chart(fig_var, use_container_width=True)
        with risk_col3:
            fig_mdd = plot_risk_metrics_gauge(value=abs(max_drawdown) * 100, title="Maximum Drawdown", max_value=100, suffix="%")
            st.plotly_chart(fig_mdd, use_container_width=True)
        with risk_col4:
            fig_sharpe = plot_risk_metrics_gauge(value=sharpe_ratio, title="Sharpe Ratio", max_value=5, suffix="")
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        with st.expander("Asset Information"):
            st.write(f"**Name:** {selected_asset.name}")
            st.write(f"**Symbol:** {selected_asset.symbol}")
            st.write(f"**Type:** {selected_asset.asset_type}")
            if hasattr(selected_asset, 'description') and selected_asset.description:
                st.write(f"**Description:** {selected_asset.description}")
            
            now = datetime.now()
            weekday = now.weekday()
            market_status = "Unknown"
            if weekday >= 5:
                market_status = "Market Closed (Weekend)"
            else:
                if asset_type == "Stocks":
                    et_hour = (now.hour - 5) % 24
                    if (9 <= et_hour < 16) or (et_hour == 16 and now.minute == 0):
                        market_status = "Market Open"
                    else:
                        market_status = "Market Closed"
                elif asset_type == "Cryptocurrencies":
                    market_status = "Market Open (24/7)"
            st.write(f"**Market Status:** {market_status}")

if st.button("Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Financial Dashboard**")
