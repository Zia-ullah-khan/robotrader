from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv('.env')

API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def llm(accountInfo, stockData, prompt, volatility):
    """Call the OpenAI API with the provided account info, stock data, and prompt."""
    
    stock_data_content = ""
    if "latest_indicators" in stockData and stockData["latest_indicators"]:
        stock_data_content += "Latest Indicators:\n"
        for key, value in stockData["latest_indicators"].items():
            stock_data_content += f"- {key}: {value}\n"
    
    if "predicted_volatility" in stockData and stockData["predicted_volatility"]:
        stock_data_content += f"Predicted Volatility: {stockData['predicted_volatility']}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial advisor."},
            {"role": "user", "content": f"Account Info: {accountInfo}\nStock Data for {stockData['symbol']}:\n{stock_data_content}\nPrompt: {prompt}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content