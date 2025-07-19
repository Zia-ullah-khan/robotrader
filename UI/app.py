from flask import Flask, render_template, request, jsonify, session
import os
import sys
import json
from datetime import datetime
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from getAccountInfo import get_account_info
from LLM import llm
from SMVF.dataset import generate_dataset
from SMVF.testCnnLstmAttn import predict_next_hour_volatility
from trade import place_order
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'hehehe'

processing_status = {
    'is_running': False,
    'current_stock': '',
    'progress': 0,
    'results': [],
    'total_stocks': 0,
    'errors': []
}

def process_stock(symbol, account_info, start_date, end_date, demo_mode=False):
    """Process a single stock through the complete pipeline."""
    try:
        print(f"\n=== Processing {symbol} ===")
        processing_status['current_stock'] = symbol
        data = generate_dataset([symbol], start_date, end_date)
        volatility = predict_next_hour_volatility(symbol, f'datasets/{symbol}_dataset.csv')
        volatility = float(volatility) if volatility is not None else None
        prompt = f"Based on the current market conditions and the predicted volatility, what would be a good trading strategy for {symbol}? return a JSON object with 'action' (buy/sell/hold), 'reason', 'amount' (number of shares, not dollar amount), 'notion', 'type' (market, limit, stop, stop_limit, trailing_stop), 'time_in_force' (day, gtc, opg, cls, ioc, fok). Take into account the current account status and available balance. Make sure the number of shares * current stock price doesn't exceed the available balance. Consider the stock's typical price range when suggesting share amounts."
        stock_data = {symbol: volatility, "data": data}
        response = llm(account_info, stock_data, prompt, volatility)
        try:
            if isinstance(response, str):
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                    response_dict = json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON object found", response, 0)
            else:
                response_dict = response
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response for {symbol}: {e}")
            response_dict = {
                "action": "hold",
                "amount": 0,
                "type": "market",
                "time_in_force": "day",
                "reason": "Failed to parse LLM response"
            }
        trade_data = {
            "symbol": symbol,
            "qty": response_dict.get("amount", 0), # type: ignore
            "side": response_dict.get("action", "hold"), # type: ignore
            "order_type": response_dict.get("type", "market"), # type: ignore
            "time_in_force": response_dict.get("time_in_force", "day") # type: ignore
        }
        order_id = None
        if not demo_mode:
            try:
                #order_id = place_order(**trade_data)
                print("place order")
            except Exception as e:
                print(f"Error placing order for {symbol}: {e}")
        
        return {
            "symbol": symbol,
            "volatility": volatility,
            "decision": response_dict,
            "trade_data": trade_data,
            "order_id": order_id,
            "success": True
        }
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        processing_status['errors'].append(f"Error processing {symbol}: {str(e)}")
        return {
            "symbol": symbol,
            "volatility": None,
            "decision": None,
            "trade_data": None,
            "order_id": None,
            "success": False,
            "error": str(e)
        }

def run_trading_pipeline(stocks, start_date, end_date, demo_mode=False):
    """Run the trading pipeline for multiple stocks."""
    global processing_status
    
    processing_status['is_running'] = True
    processing_status['current_stock'] = ''
    processing_status['progress'] = 0
    processing_status['results'] = []
    processing_status['total_stocks'] = len(stocks)
    processing_status['errors'] = []
    
    try:
        account_info = get_account_info()
        results = []
        for i, stock in enumerate(stocks):
            if not processing_status['is_running']:
                break
            result = process_stock(stock, account_info, start_date, end_date, demo_mode)
            results.append(result)
            if isinstance(result.get('volatility'), (bytes,)):
                result['volatility'] = None
            processing_status['results'].append(result)
            processing_status['progress'] = int(((i + 1) / len(stocks)) * 100)
            time.sleep(1)
        
        processing_status['is_running'] = False
        processing_status['current_stock'] = 'Complete'
        
    except Exception as e:
        processing_status['is_running'] = False
        processing_status['errors'].append(f"Pipeline error: {str(e)}")
        processing_status['current_stock'] = 'Error'

@app.route('/')
def index():
    """Main page with trading interface."""
    api_key = os.getenv('ALPACA_API_KEY', '')
    api_secret = os.getenv('ALPACA_API_SECRET', '')
    base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    return render_template('index.html', 
                         api_key_masked='*' * (len(api_key) - 4) + api_key[-4:] if api_key else '',
                         base_url=base_url)

@app.route('/start_trading', methods=['POST'])
def start_trading():
    """Start the trading pipeline."""
    if processing_status['is_running']:
        return jsonify({'error': 'Trading pipeline is already running'}), 400
    stocks_input = request.form.get('stocks', '').strip()
    start_date = request.form.get('start_date', '2020-01-01')
    end_date = request.form.get('end_date', datetime.now().strftime("%Y-%m-%d"))
    demo_mode = request.form.get('demo_mode') == 'on'
    if not stocks_input:
        return jsonify({'error': 'Please enter at least one stock symbol'}), 400
    
    stocks = [stock.strip().upper() for stock in stocks_input.split(',') if stock.strip()]
    
    if not stocks:
        return jsonify({'error': 'Please enter valid stock symbols'}), 400
    thread = threading.Thread(target=run_trading_pipeline, args=(stocks, start_date, end_date, demo_mode))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Trading pipeline started', 'stocks': stocks})

@app.route('/stop_trading', methods=['POST'])
def stop_trading():
    """Stop the trading pipeline."""
    processing_status['is_running'] = False
    return jsonify({'message': 'Trading pipeline stopped'})

@app.route('/status')
def get_status():
    """Get current processing status."""
    return jsonify(processing_status)

@app.route('/results')
def get_results():
    """Get detailed results."""
    return jsonify({
        'results': processing_status['results'],
        'errors': processing_status['errors'],
        'is_complete': not processing_status['is_running'] and processing_status['current_stock'] in ['Complete', 'Error']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
