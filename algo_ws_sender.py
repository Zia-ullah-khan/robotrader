import socketio
import time
import uuid
import json
from main import process_stock
from getAccountInfo import get_account_info
from utils import get_top_performing_stocks

SOCKETIO_SERVER_URL = 'http://localhost:5000'

sio = socketio.Client()

SERVER_ID = str(uuid.uuid4())

@sio.event
def connect():
    print('Connected to WebSocket server')

@sio.event
def disconnect():
    print('Disconnected from WebSocket server')

def send_trade_decisions(results):
    payload = {
        'server_id': SERVER_ID,
        'decisions': {}
    }
    for result in results:
        if result['success']:
            payload['decisions'][result['symbol']] = {
                'ticker': result['symbol'],
                'quantity': result['trade_data'].get('qty', 0),
                'reason': result['decision'].get('reason', '')
            }
    sio.emit('algo_trade_decision', payload)
    print('Sent trade decisions:', json.dumps(payload, indent=2))

def main_loop():
    start_date = "2020-01-01"
    from datetime import datetime
    while True:
        print("Fetching top 10 performing stocks in the last 10 minutes...")
        STOCKS = get_top_performing_stocks(num_stocks=10, interval_minutes=10)
        if not STOCKS:
            print("No stocks found. Waiting for next interval.")
            time.sleep(600)
            continue

        end_date = datetime.now().strftime("%Y-%m-%d")
        account_info = get_account_info()
        results = []
        print(f"Processing {len(STOCKS)} stocks: {', '.join(STOCKS)}")
        for stock in STOCKS:
            result = process_stock(stock, account_info, start_date, end_date)
            results.append(result)
        send_trade_decisions(results)
        print("Sleeping for 10 minutes...")
        time.sleep(600)

if __name__ == "__main__":
    sio.connect(SOCKETIO_SERVER_URL)
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Exiting...")
    sio.disconnect()
