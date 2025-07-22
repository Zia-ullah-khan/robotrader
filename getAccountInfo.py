import os
from alpaca_trade_api import REST
from dotenv import load_dotenv
load_dotenv('.env')

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

api = REST(API_KEY, API_SECRET, BASE_URL) # type: ignore
def get_account_info():
    """Fetch and print account information."""
    account = api.get_account()
    available_balance = account.cash
    buying_power = account.buying_power

    positions = api.list_positions()
    stocks = [(position.symbol, position.qty, position.market_value) for position in positions]

    is_up = float(account.equity) > float(account.last_equity)

    print(f"Available balance: ${available_balance}")
    print("Current stocks:")
    for symbol, qty, market_value in stocks:
        print(f"  {symbol}: {qty} shares, Market Value: ${market_value}")
    print(f"Account is {'up' if is_up else 'down'} compared to last equity.")
    print(f"Account status: {account.status}")
    return {
        "available_balance": available_balance,
        "buying_power": buying_power,
        "stocks": stocks,
        "is_up": is_up,
        "status": account.status
    }
def get_portfolio_history(timeframe='1D', period='1M'):
    try:
        history = api.get_portfolio_history(timeframe=timeframe, period=period)
        return {
            'timestamp': history.timestamp,
            'equity': history.equity,
            'profit_loss': history.profit_loss,
            'base_value': history.base_value,
            'timeframe': history.timeframe
        }
    except Exception as e:
        print(f"Error fetching portfolio history: {e}")
        return None