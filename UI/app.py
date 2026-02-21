from flask import Flask, render_template, jsonify
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from getAccountInfo import get_account_info, get_portfolio_history as get_alpaca_history
from trade import get_previous_transactions
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'hehehe'

# Path to the trade history JSON file
TRADE_HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trade_history.json')

def load_trade_history():
    """Load trade history from JSON file."""
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def get_portfolio_history():
    """Get portfolio value history from Alpaca API."""
    try:
        # Fetch 1 month of daily history
        history = get_alpaca_history(timeframe='1D', period='1M')
        if not history:
            return []
            
        portfolio_data = []
        timestamps = history['timestamp']
        equities = history['equity']
        
        # Combine timestamps and equity values
        for ts, eq in zip(timestamps, equities):
            if eq is not None:  # Filter out None values
                portfolio_data.append({
                    'timestamp': ts,  # Alpaca returns unix timestamp
                    'equity': float(eq),
                    'cash': 0  # Alpaca history doesn't give historical cash easily, so we'll omit or fetch differently if needed
                })
        
        return portfolio_data
    except Exception as e:
        print(f"Error getting portfolio history: {e}")
        return []

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/account')
def get_account():
    """Get current account information."""
    try:
        account_info = get_account_info()
        return jsonify({
            'success': True,
            'data': account_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/trades')
def get_trades():
    """Get trade history."""
    try:
        history = load_trade_history()
        # Return most recent first
        return jsonify({
            'success': True,
            'data': list(reversed(history[-100:]))  # Last 100 trades
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/portfolio-history')
def get_portfolio_history_api():
    """Get portfolio value history for charting."""
    try:
        data = get_portfolio_history()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/recent-orders')
def get_recent_orders():
    """Get recent orders from Alpaca."""
    try:
        orders = get_previous_transactions(limit=20)
        return jsonify({
            'success': True,
            'data': orders
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stats')
def get_stats():
    """Get trading statistics."""
    try:
        history = load_trade_history()
        
        total_trades = len(history)
        successful_trades = sum(1 for t in history if t.get('success', False))
        
        buy_count = sum(1 for t in history if t.get('decision', {}).get('action') == 'buy')
        sell_count = sum(1 for t in history if t.get('decision', {}).get('action') == 'sell')
        hold_count = sum(1 for t in history if t.get('decision', {}).get('action') == 'hold')
        
        # Get unique symbols traded
        symbols = set(t.get('symbol') for t in history if t.get('symbol'))
        
        return jsonify({
            'success': True,
            'data': {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'failed_trades': total_trades - successful_trades,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'unique_symbols': len(symbols),
                'symbols_list': list(symbols)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Path to pipeline status file
PIPELINE_STATUS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipeline_status.json')

@app.route('/api/pipeline-status')
def get_pipeline_status():
    """Get current pipeline processing status."""
    try:
        if os.path.exists(PIPELINE_STATUS_FILE):
            with open(PIPELINE_STATUS_FILE, 'r') as f:
                status = json.load(f)
            return jsonify({
                'success': True,
                'data': status
            })
        else:
            return jsonify({
                'success': True,
                'data': {
                    'status': 'idle',
                    'message': 'Pipeline not running',
                    'current_stock': None,
                    'progress': 0,
                    'total_stocks': 0,
                    'completed_stocks': []
                }
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/export-training-data')
def export_training_data():
    """Export trade history in a format suitable for ML training."""
    try:
        history = load_trade_history()
        return jsonify({
            'success': True,
            'count': len(history),
            'data': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Path to scheduler log file
SCHEDULER_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scheduler_log.json')

@app.route('/api/scheduler-log')
def get_scheduler_log():
    """Get scheduler run history."""
    try:
        if os.path.exists(SCHEDULER_LOG_FILE):
            with open(SCHEDULER_LOG_FILE, 'r') as f:
                log = json.load(f)
            return jsonify({
                'success': True,
                'data': list(reversed(log[-20:]))  # Last 20 runs
            })
        else:
            return jsonify({
                'success': True,
                'data': []
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
