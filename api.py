
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from getAccountInfo import get_account_info
from main import process_stock
from getAccountInfo import get_portfolio_history
try:
    from trade import get_previous_transactions
except ImportError:
    get_previous_transactions = None

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
@app.route('/api/portfolio_history', methods=['GET'])
def api_portfolio_history():
    timeframe = request.args.get('timeframe', '1D')
    period = request.args.get('period', '1M')
    history = get_portfolio_history(timeframe, period)
    if history is None:
        return jsonify({'error': 'Could not fetch portfolio history'}), 500
    return jsonify(history)

@app.route('/api/account', methods=['GET'])
def api_account_info():
    info = get_account_info()
    return jsonify(info)


@app.route('/api/status', methods=['GET'])
def api_status():
    info = get_account_info()
    status = {
        'available_balance': info.get('available_balance', 0),
        'buying_power': info.get('buying_power', 0),
        'stocks': info.get('stocks', []),
        'is_up': info.get('is_up', False),
        'status': info.get('status', 'UNKNOWN')
    }
    return jsonify(status)


@app.route('/api/transactions', methods=['GET'])
def api_transactions():
    if get_previous_transactions is None:
        return jsonify({'error': 'Transaction history not available'}), 501
    transactions = get_previous_transactions()
    return jsonify(transactions)


@app.route('/api/trade_decision', methods=['POST'])
def api_trade_decision():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Missing or invalid JSON'}), 400
    symbol = data.get('symbol')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    if not symbol or not start_date or not end_date:
        return jsonify({'error': 'Missing required fields: symbol, start_date, end_date'}), 400
    account_info = get_account_info()
    result = process_stock(symbol, account_info, start_date, end_date)
    return jsonify(result)

@socketio.on('trade_decision')
def handle_trade_decision(data):
    symbol = data.get('symbol')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    if not symbol or not start_date or not end_date:
        emit('trade_decision_response', {'error': 'Missing required fields: symbol, start_date, end_date'})
        return
    account_info = get_account_info()
    result = process_stock(symbol, account_info, start_date, end_date)
    emit('trade_decision_response', result)

@socketio.on('algo_trade_decision')
def handle_algo_trade_decision(data):
    socketio.emit('algo_trade_decision_broadcast', data)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
