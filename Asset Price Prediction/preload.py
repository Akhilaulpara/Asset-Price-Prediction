from data_service import get_assets_by_type, add_price_data
from data_fetcher import fetch_historical_data

# Load assets of all types
asset_types = ["Cryptocurrencies", "Stocks", "Forex", "Commodities"]

for asset_type in asset_types:
    assets = get_assets_by_type(asset_type)
    for asset in assets:
        for interval in ["4hour", "1day"]:  # Add more if needed
            print(f"Fetching {interval} data for {asset.symbol}")
            df = fetch_historical_data(asset.symbol, asset.asset_type, days=120, interval=interval)
            if not df.empty:
                add_price_data(asset.id, df, interval)
                print(f"Saved {interval} data for {asset.symbol}")
            else:
                print(f"No {interval} data found for {asset.symbol}")
