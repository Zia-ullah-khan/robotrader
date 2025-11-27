from datetime import datetime, timedelta
import yfinance as yf
import os
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

api = REST(API_KEY, API_SECRET, BASE_URL) # type: ignore

def get_all_us_tickers():
    """Fetch all tradable US equity tickers from Alpaca."""
    assets = api.list_assets(status='active', asset_class='us_equity')
    return [asset.symbol for asset in assets if asset.tradable]

def get_latest_indicators(symbol):
    """Fetch the latest stock data and calculate technical indicators."""
    import pandas_ta as ta
    data = yf.download(symbol, period="1y", interval="1d")
    if data is None or data.empty:
        return None
    
    data.ta.rsi(append=True)
    data.ta.macd(append=True)
    data.ta.bbands(append=True)
    data.ta.atr(append=True)
    data.ta.stoch(append=True)
    data.ta.log_return(append=True)
    data.ta.realized_volatility(append=True)
    data.ta.parkinson(append=True)
    data.ta.custom_volatility(append=True)
    
    latest_data = data.iloc[-1].to_dict()
    return latest_data

def get_top_performing_stocks(num_stocks=10, interval_minutes=10, universe=None):
    """
    Fetch the top performing stocks in the last `interval_minutes` from the given universe.
    If universe is None, it fetches all tradable US tickers.
    """
    import pandas as pd
    import pytz
    from datetime import time as dt_time
    if universe is None:
        print("Fetching all tradable US tickers...")
        universe = get_all_us_tickers()
        if not universe:
            print("Could not fetch any tickers.")
            return []
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    is_weekday = now_eastern.weekday() < 5
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    if not (is_weekday and market_open <= now_eastern.time() <= market_close):
        print("[INFO] US stock market is currently closed. Top performing stocks can only be fetched during market hours (9:30am-4:00pm US/Eastern, Mon-Fri).")
        return []
    utc = pytz.UTC
    end = datetime.now(utc)
    start = end - timedelta(minutes=interval_minutes)
    data = yf.download(
        universe,
        start=start,
        end=end,
        interval="1m",
        progress=False,
        group_by='ticker',
        threads=True
    )
    perf = {}
    if data is not None and hasattr(data, 'columns') and hasattr(data.columns, 'levels'):
        for symbol in universe:
            try:
                if (symbol, 'Close') in data.columns:
                    closes = data[(symbol, 'Close')].dropna()
                    if len(closes) >= 2:
                        pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
                        perf[symbol] = pct_change
            except Exception:
                continue
    elif data is not None and hasattr(data, 'columns') and 'Close' in data.columns:
        try:
            closes = data['Close'].dropna()
            if len(closes) >= 2:
                pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
                perf[universe[0]] = pct_change
        except Exception:
            pass
    top = sorted(perf.items(), key=lambda x: x[1], reverse=True)[:num_stocks]
    return [s[0] for s in top]
