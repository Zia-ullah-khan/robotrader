from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv('.env')

API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def llm(accountInfo, stockData, prompt, volatility):
    """Call the OpenAI API with the provided account info, stock data, and prompt."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial advisor."},
            {"role": "user", "content": f"Account Info: {accountInfo}\nStock Data: {stockData}\nPrompt: {prompt}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content