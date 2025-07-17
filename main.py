from getAccountInfo import get_account_info
from LLM import llm
from SMVF.dataset import generate_dataset
from datetime import datetime
from SMVF.testCnnLstmAttn import predict_next_hour_volatility
from trade import place_order
import json

def process_stock(symbol, account_info, start_date, end_date):
    """Process a single stock through the complete pipeline."""
    try:
        print(f"\n=== Processing {symbol} ===")
        data = generate_dataset([symbol], start_date, end_date)
        volatility = predict_next_hour_volatility(symbol, f'datasets/{symbol}_dataset.csv')
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
            "qty": response_dict.get("amount", 0), # type: ignore
            "side": response_dict.get("action", "hold"), # type: ignore
            "order_type": response_dict.get("type", "market"), # type: ignore
            "time_in_force": response_dict.get("time_in_force", "day") # type: ignore
        }
    
        print(f"Trade decision for {symbol}: {response_dict.get('action', 'hold')} {response_dict.get('amount', 0)} shares") # type: ignore
        print(f"Reason: {response_dict.get('reason', 'No reason provided')}") # type: ignore
        #place_order(**trade_data)
        
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
    STOCKS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
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
            action = result["decision"].get("action", "hold")
            amount = result["decision"].get("amount", 0)
            print(f"{result['symbol']}: {action.upper()} {amount} shares (Volatility: {result['volatility']:.4f})")
        else:
            failed_trades += 1
            print(f"{result['symbol']}: FAILED - {result['error']}")
    print(f"\nTotal: {successful_trades} successful, {failed_trades} failed")