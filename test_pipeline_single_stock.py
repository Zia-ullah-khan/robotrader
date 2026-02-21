import sys
import os
from datetime import datetime
import json
from getAccountInfo import get_account_info
from LLM import llm
from main import process_stock

def test_single_stock_pipeline(symbol='AAPL'):
    print(f"Testing pipeline for single stock: {symbol}")
    
    # 1. Get Account Info
    try:
        account_info = get_account_info()
        print("Account info retrieved.")
    except Exception as e:
        print(f"Error getting account info: {e}")
        return

    # 2. Define Date Range
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # 3. Run Pipeline for the Stock
    print(f"Running process_stock for {symbol}...")
    result = process_stock(symbol, account_info, start_date, end_date)
    
    print("\nPipeline execution completed.")
    print("Result:", json.dumps(result, indent=2, default=str))

    # 4. Send Result to LLM for Analysis
    print("\nSending result to LLM for analysis...")
    
    # Prepare data for LLM
    stock_data = {
        "symbol": symbol,
        "predicted_volatility": result.get("volatility"),
        # We don't have latest_indicators in the result dict from process_stock, 
        # but we can pass what we have.
        "decision": result.get("decision"),
        "trade_data": result.get("trade_data"),
        "success": result.get("success"),
        "error": result.get("error")
    }
    
    volatility = result.get("volatility")
    if volatility is None:
        volatility = 0.0

    prompt = (
        f"I have just run the trading pipeline for {symbol}. "
        f"The system calculated a volatility of {volatility} and made the following decision: "
        f"{json.dumps(result.get('decision'), default=str)}. "
        f"The trade execution data was: {json.dumps(result.get('trade_data'), default=str)}. "
        f"Was this a successful run? Please analyze the decision."
    )

    try:
        llm_response = llm(account_info, stock_data, prompt, volatility)
        print("\nLLM Analysis Response:")
        print(llm_response)
    except Exception as e:
        print(f"Error calling LLM: {e}")

if __name__ == "__main__":
    # Allow symbol to be passed as argument
    symbol_to_test = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    test_single_stock_pipeline(symbol_to_test)
