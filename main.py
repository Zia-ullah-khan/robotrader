from getAccountInfo import get_account_info
from LLM import llm
from SMVF.dataset import generate_dataset
from datetime import datetime
from SMVF.predict import predict_next_hour_volatility
from trade import place_order
import json
import os
from utils import get_top_performing_stocks, get_latest_indicators

# Path to trade history file for training data
TRADE_HISTORY_FILE = 'trade_history.json'
# Path to pipeline status file for UI monitoring
PIPELINE_STATUS_FILE = 'pipeline_status.json'

def load_trade_history():
    """Load existing trade history from JSON file."""
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_trade_history(history):
    """Save trade history to JSON file."""
    with open(TRADE_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)

def update_pipeline_status(status_data):
    """Update the pipeline status file for UI monitoring."""
    try:
        with open(PIPELINE_STATUS_FILE, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
    except Exception as e:
        print(f"[WARNING] Could not update pipeline status: {e}")

def log_trade_result(result, account_info, latest_indicators):
    """Log a trade result to the history file for training."""
    history = load_trade_history()
    
    # Create a comprehensive log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": result.get("symbol"),
        "volatility": result.get("volatility"),
        "decision": result.get("decision"),
        "trade_data": result.get("trade_data"),
        "success": result.get("success"),
        "error": result.get("error"),
        "latest_indicators": latest_indicators,
        "account_snapshot": {
            "available_balance": account_info.get("available_balance"),
            "buying_power": account_info.get("buying_power"),
            "equity": account_info.get("buying_power"),  # Using buying_power as equity proxy
            "cash": account_info.get("available_balance"),
            "status": account_info.get("status"),
            "is_up": account_info.get("is_up")
        }
    }
    
    history.append(log_entry)
    save_trade_history(history)
    print(f"[LOG] Trade result saved to {TRADE_HISTORY_FILE}")

def process_stock(symbol, account_info, start_date, end_date):
    """Process a single stock through the complete pipeline."""
    try:
        print(f"\n=== Processing {symbol} ===")
        
        latest_indicators = get_latest_indicators(symbol)
        
        try:
            data = generate_dataset([symbol], start_date, end_date)
            print(f"[DEBUG] Dataset generated for {symbol}: {type(data)}")
        except Exception as e:
            print(f"[ERROR] generate_dataset failed for {symbol}: {e}")
            data = None
        try:
            volatility = predict_next_hour_volatility(symbol, f'datasets/{symbol}_dataset.csv')
            print(f"[DEBUG] Volatility for {symbol}: {volatility}")
        except Exception as e:
            print(f"[ERROR] predict_next_hour_volatility failed for {symbol}: {e}")
            volatility = None
        prompt = f"Based on the current market conditions and the predicted volatility, what would be a good trading strategy for {symbol}? return a JSON object with 'action' (buy/sell/hold), 'reason', 'amount' (number of shares, not dollar amount), 'notion', 'type' (market, limit, stop, stop_limit, trailing_stop), 'time_in_force' (day, gtc, opg, cls, ioc, fok). If the order type is 'limit', you must provide a 'limit_price'. If the order type is 'stop' or 'stop_limit', you must provide a 'stop_price'. Take into account the current account status and available balance. Make sure the number of shares * current stock price doesn't exceed the available balance. Consider the stock's typical price range when suggesting share amounts. Take into account the current portfolio of the user and make decisions based on that."
        stock_data = {
            "symbol": symbol,
            "predicted_volatility": volatility,
            "latest_indicators": latest_indicators
        }
        try:
            response = llm(account_info, stock_data, prompt, volatility)
            print(f"[DEBUG] LLM response for {symbol}: {response}")
        except Exception as e:
            print(f"[ERROR] LLM call failed for {symbol}: {e}")
            response = None
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
            print(f"Raw response: {response}")
            response_dict = {
                "action": "hold",
                "amount": 0,
                "type": "market",
                "time_in_force": "day",
                "reason": "Failed to parse LLM response"
            }
        trade_data = {
            "symbol": symbol,
            "qty": response_dict.get("amount", 0),
            "side": response_dict.get("action", "hold"),
            "order_type": response_dict.get("type", "market"),
            "time_in_force": response_dict.get("time_in_force", "day")
        }

        # Add limit_price or stop_price if applicable
        order_type = trade_data["order_type"]
        if order_type == 'limit' and 'limit_price' in response_dict:
            trade_data['limit_price'] = response_dict['limit_price']
        elif order_type == 'limit':
            print("[WARNING] LLM suggested a limit order without a limit_price. Defaulting to market order.")
            trade_data['order_type'] = 'market'
        
        if (order_type == 'stop' or order_type == 'stop_limit') and 'stop_price' in response_dict:
            trade_data['stop_price'] = response_dict['stop_price']
        elif order_type == 'stop' or order_type == 'stop_limit':
            print(f"[WARNING] LLM suggested a {order_type} order without a stop_price. Defaulting to market order.")
            trade_data['order_type'] = 'market'
    
        print(f"Trade decision for {symbol}: {response_dict.get('action', 'hold')} {response_dict.get('amount', 0)} shares")
        print(f"Reason: {response_dict.get('reason', 'No reason provided')}")
        place_order(**trade_data)
        
        result = {
            "symbol": symbol,
            "volatility": volatility,
            "decision": response_dict,
            "trade_data": trade_data,
            "success": True
        }
        
        # Log the trade result for training data
        log_trade_result(result, account_info, latest_indicators)
        
        return result
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        result = {
            "symbol": symbol,
            "volatility": None,
            "decision": None,
            "trade_data": None,
            "success": False,
            "error": str(e)
        }
        
        # Log even failed trades for analysis
        log_trade_result(result, account_info, {})
        
        return result

if __name__ == "__main__":
    print("Fetching top performing stocks...")
    
    # Update status: starting
    update_pipeline_status({
        "status": "starting",
        "message": "Fetching top performing stocks...",
        "current_stock": None,
        "progress": 0,
        "total_stocks": 0,
        "completed_stocks": [],
        "last_update": datetime.now().isoformat()
    })
    
    STOCKS = get_top_performing_stocks(num_stocks=50, interval_minutes=10)
    if not STOCKS:
        print("No stocks to process. Market might be closed or no performing stocks found.")
        update_pipeline_status({
            "status": "idle",
            "message": "No stocks to process. Market might be closed.",
            "current_stock": None,
            "progress": 100,
            "total_stocks": 0,
            "completed_stocks": [],
            "last_update": datetime.now().isoformat()
        })
        exit()

    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    account_info = get_account_info()
    print(f"Processing {len(STOCKS)} stocks: {', '.join(STOCKS)}")
    
    # Update status: running
    update_pipeline_status({
        "status": "running",
        "message": f"Processing {len(STOCKS)} stocks",
        "current_stock": None,
        "progress": 0,
        "total_stocks": len(STOCKS),
        "completed_stocks": [],
        "stocks_to_process": STOCKS,
        "last_update": datetime.now().isoformat()
    })
    
    results = []
    completed_stocks = []
    for i, stock in enumerate(STOCKS):
        # Update status: processing stock
        update_pipeline_status({
            "status": "running",
            "message": f"Processing {stock} ({i+1}/{len(STOCKS)})",
            "current_stock": stock,
            "progress": int((i / len(STOCKS)) * 100),
            "total_stocks": len(STOCKS),
            "completed_stocks": completed_stocks.copy(),
            "stocks_to_process": STOCKS,
            "last_update": datetime.now().isoformat()
        })
        
        result = process_stock(stock, account_info, start_date, end_date)
        results.append(result)
        completed_stocks.append({
            "symbol": stock,
            "action": result.get("decision", {}).get("action") if result.get("decision") else None,
            "success": result.get("success")
        })
        print(f"Completed processing {stock}")
        
    # Update status: complete
    update_pipeline_status({
        "status": "complete",
        "message": f"Completed processing {len(STOCKS)} stocks",
        "current_stock": None,
        "progress": 100,
        "total_stocks": len(STOCKS),
        "completed_stocks": completed_stocks,
        "stocks_to_process": STOCKS,
        "last_update": datetime.now().isoformat()
    })
    
    print("\n=== TRADING SUMMARY ===")
    successful_trades = 0
    failed_trades = 0
    for result in results:
        if result["success"]:
            successful_trades += 1
            action = result["decision"].get("action", "hold") if result["decision"] else "hold"
            amount = result["decision"].get("amount", 0) if result["decision"] else 0
            if result['volatility'] is not None:
                print(f"{result['symbol']}: {action.upper()} {amount} shares (Volatility: {result['volatility']:.4f})")
            else:
                print(f"{result['symbol']}: {action.upper()} {amount} shares (Volatility: N/A)")
        else:
            failed_trades += 1
            print(f"{result['symbol']}: FAILED - {result['error']}")
    print(f"\nTotal: {successful_trades} successful, {failed_trades} failed")