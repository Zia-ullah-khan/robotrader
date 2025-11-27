import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import os

WINDOW_SIZES = [20, 40, 60]

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical financial data from Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        
    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data is None or data.empty:
        raise ValueError(f"No data fetched for ticker {ticker} between {start_date} and {end_date}.")
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] != '' else col[1] for col in data.columns]
    print("Column names after flattening:", data.columns)
    data.rename(columns={
        "Adj Close": "Close",
        "High": "High",
        "Low": "Low",
        "Open": "Open",
        "Volume": "Volume"
    }, inplace=True)
    return data


def calculate_technical_indicators(data):
    """
    Calculates technical indicators as described in the paper.
    
    Args:
        data (pd.DataFrame): The input financial data.
        
    Returns:
        pd.DataFrame: The data with added technical indicators.
    """
    data.ta.rsi(append=True)
    data.ta.macd(append=True)
    data.ta.atr(append=True)
    data.ta.bbands(append=True)
    data.ta.stoch(append=True)
    data.ta.adx(append=True)
    data.ta.obv(append=True)
    data.ta.sma(length=20, append=True)
    data.ta.sma(length=50, append=True)
    data.ta.ema(length=20, append=True)
    data.ta.ema(length=50, append=True)
    data.dropna(inplace=True)
    return data

def calculate_volatility(data):
    """
    Calculates different volatility measures as described in the paper.
    
    Args:
        data (pd.DataFrame): The input financial data.
        
    Returns:
        pd.DataFrame: The data with added volatility measures.
    """
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['realized_volatility'] = data['log_returns'].rolling(window=20).std() * np.sqrt(252)

    data['parkinson_volatility'] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(data['High'] / data['Low']))**2).rolling(window=20).sum()) * np.sqrt(252)

    rolling_std = data['log_returns'].rolling(window=20).std()
    atr = data['ATRr_14']
    normalized_returns = data['log_returns'] / rolling_std
    data['custom_volatility'] = (rolling_std + atr + normalized_returns.abs()) / 3
    
    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    """
    Normalizes the data and creates rolling windows.
    
    Args:
        data (pd.DataFrame): The input data with features and targets.
        
    Returns:
        pd.DataFrame: The preprocessed data.
    """
    if 'Date' in data.columns:
        date_col = data['Date']
        data_numeric = data.drop(columns=['Date'])
    else:
        date_col = None
        data_numeric = data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_numeric)
    scaled_df = pd.DataFrame(scaled_data, columns=data_numeric.columns, index=data_numeric.index)
    if date_col is not None:
        scaled_df.insert(0, 'Date', date_col)
    return scaled_df

def generate_dataset(TICKERS, START_DATE, END_DATE):
    """
    Main function to generate the dataset for all tickers. (This function should be able to receive 1 or more tickers)
    Saves the datasets to the 'dataset' folder.
    """
    os.makedirs("datasets", exist_ok=True)
    for ticker in TICKERS:
        try:
            raw_data = fetch_data(ticker, START_DATE, END_DATE)
            data_with_indicators = calculate_technical_indicators(raw_data)
            data_with_volatility = calculate_volatility(data_with_indicators)
            processed_data = preprocess_data(data_with_volatility)
            file_name = os.path.join("datasets", f"{ticker.replace('^','')}_dataset.csv")
            processed_data.to_csv(file_name)
            print(f"Dataset for {ticker} saved to {file_name}")
            return processed_data
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
            continue
