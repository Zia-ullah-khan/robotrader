from datetime import datetime, timedelta
import yfinance as yf

def get_top_performing_stocks(num_stocks=10, interval_minutes=10, universe=None):
    """
    Fetch the top performing stocks in the last `interval_minutes` from the given universe.
    If universe is None, use a default list of popular tickers (S&P 500 subset for demo).
    """
    import pandas as pd
    import pytz
    from datetime import time as dt_time
    if universe is None:
        universe = [
            "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "ADBE",
            "ORCL", "CRM", "V", "JNJ", "JPM", "KO", "PG", "PYPL", "SMCI", "AEHR", "RMTI", "BLUE"
        ]
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
