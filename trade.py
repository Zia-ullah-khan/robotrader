from alpaca_trade_api import REST
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

api = REST(API_KEY, API_SECRET, BASE_URL) # type: ignore

def place_order(symbol, qty, side, order_type, time_in_force):
    """Place an order with Alpaca."""
    try:
        side = side.lower()
        if side == "hold":
            print(f"Action is 'hold' - no order will be placed for {symbol}")
            return None
            
        if side not in ['buy', 'sell']:
            print(f"Invalid side: {side}. Must be 'buy' or 'sell'")
            return None
        if qty <= 0:
            print(f"Skipping order: Invalid quantity {qty}")
            return None
            
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        print(f"Order placed: {order.id}") # type: ignore
        return order.id # type: ignore
    except Exception as e:
        print(f"Error placing order: {e}")
        return None
