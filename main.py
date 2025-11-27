from getAccountInfo import get_account_info
from LLM import llm
from SMVF.dataset import generate_dataset
from datetime import datetime
from SMVF.predict import predict_next_hour_volatility
from trade import place_order
import json
from utils import get_top_performing_stocks, get_latest_indicators

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
        
        return {
            "symbol": symbol,
            "volatility": volatility,
            "decision": response_dict,
            "trade_data": trade_data,
            "success": True
        }
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return {
            "symbol": symbol,
            "volatility": None,
            "decision": None,
            "trade_data": None,
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    print("Fetching top performing stocks...")
    STOCKS = get_top_performing_stocks(num_stocks=50, interval_minutes=10)
    if not STOCKS:
        print("No stocks to process. Market might be closed or no performing stocks found.")
        exit()

    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    account_info = get_account_info()
    print(f"Processing {len(STOCKS)} stocks: {', '.join(STOCKS)}")
    results = []
    for stock in STOCKS:
        result = process_stock(stock, account_info, start_date, end_date)
        results.append(result)
        print(f"Completed processing {stock}")
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