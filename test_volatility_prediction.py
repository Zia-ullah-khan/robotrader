import unittest
import os
import pandas as pd
from SMVF.predict import predict_next_hour_volatility
from SMVF.dataset import generate_dataset
from datetime import datetime

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
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0.0)

if __name__ == '__main__':
    unittest.main()
