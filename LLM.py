import re
import json
import os
from dotenv import load_dotenv
import ollama
load_dotenv('.env')

def llm(accountInfo, stockData, prompt, volatility):
    """Call the OpenAI API with the provided account info, stock data, and prompt."""
    
    stock_data_content = ""
    if "latest_indicators" in stockData and stockData["latest_indicators"]:
        stock_data_content += "Latest Indicators:\n"
        for key, value in stockData["latest_indicators"].items():
            stock_data_content += f"- {key}: {value}\n"
    
    if "predicted_volatility" in stockData and stockData["predicted_volatility"]:
        stock_data_content += f"Predicted Volatility: {stockData['predicted_volatility']}\n"

    # Use ollama chat API
    response = ollama.chat(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a financial advisor. Always respond with plain ASCII characters only - no special Unicode symbols."},
            {"role": "user", "content": f"Account Info: {accountInfo}\nStock Data for {stockData['symbol']}:\n{stock_data_content}\nPrompt: {prompt}"}
        ],
        options={"temperature": 0.7, "max_tokens": 150}
    )
    raw_response = response['message']['content']
    
    # Normalize Unicode characters to ASCII equivalents
    unicode_replacements = {
        '\u2248': '~',      # ≈ approximately
        '\u2011': '-',      # non-breaking hyphen
        '\u2212': '-',      # minus sign
        '\u2013': '-',      # en dash
        '\u2014': '-',      # em dash
        '\u2018': "'",      # left single quote
        '\u2019': "'",      # right single quote
        '\u201c': '"',      # left double quote
        '\u201d': '"',      # right double quote
        '\u2026': '...',    # ellipsis
        '\u00b7': '*',      # middle dot
    }
    for unicode_char, ascii_char in unicode_replacements.items():
        raw_response = raw_response.replace(unicode_char, ascii_char)
    
    # Extract JSON from the response
    json_match = re.search(r'```json\n({.*?})\n```|({.*})', raw_response, re.S)
    if json_match:
        json_str = json_match.group(1) or json_match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return raw_response  # Return raw on failure
    return raw_response