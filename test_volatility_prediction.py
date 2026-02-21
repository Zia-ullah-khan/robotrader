import unittest
import os
import pandas as pd
from SMVF.predict import predict_next_hour_volatility
from SMVF.dataset import generate_dataset
from datetime import datetime
import numpy as np
from LLM import llm

class TestVolatilityPrediction(unittest.TestCase):
    def test_predict_next_hour_volatility(self):
        # Generate a sample dataset
        symbol = 'AAPL'
        start_date = "2020-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        dataset = generate_dataset([symbol], start_date, end_date)
        dataset_path = f'datasets/{symbol}_dataset.csv'

        # Test the prediction function
        volatility = predict_next_hour_volatility(symbol, dataset_path)
        self.assertIsInstance(volatility, (float, np.float32))
        self.assertGreater(volatility, 0.0)

if __name__ == '__main__':
    # Run tests programmatically so that we can continue execution afterwards
    test_loader = unittest.defaultTestLoader
    test_suite = test_loader.loadTestsFromTestCase(TestVolatilityPrediction)
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)
    # After running tests, send the file content to the LLM for analysis
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            file_content = f.read()
        # Prepare dummy context for LLM
        account_info = None
        volatility = 0.0
        stock_data = {
            "symbol": "AAPL",
            "latest_indicators": {},
            "predicted_volatility": volatility
        }
        prompt = "Analyze the following test file and provide insights."
        response = llm(account_info, stock_data, prompt, volatility)
        print("\nLLM Response:\n", response)
    except Exception as e:
        print("Error calling LLM:", e)
