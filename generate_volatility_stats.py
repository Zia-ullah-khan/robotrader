
import yfinance as yf
import pandas as pd
import numpy as np


def get_data_and_stats():
    # Define tickers match: S&P 500, NASDAQ, NIFTY 50, Jakarta Composite
    tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "NIFTY 50": "^NSEI",
        "Jakarta Composite": "^JKSE"  
    }

    start_date = "2018-01-01"
    end_date = "2024-12-31"

    stats_list = []

    print(f"Fetching data from {start_date} to {end_date}...")

    for name, ticker in tickers.items():
        print(f"Processing {name} ({ticker})...")
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(data) == 0:
                print(f"Warning: No data found for {ticker}")
                continue
                
            # Handle Flattening of MultiIndex columns if necessary (yfinance update)
            if isinstance(data.columns, pd.MultiIndex):
                # Keep only 'Close' and flatten
                data = data['Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                 # If just a series or other format
                pass
            
            # Ensure we have a Series for Close prices
            if isinstance(data, pd.DataFrame):
                # If it's still a dataframe, try to get the ticker column or just the first column
                # With yfinance 0.2+, downloading single ticker might return DataFrame with 'Close'
                if ticker in data.columns:
                    close = data[ticker]
                else:
                    close = data.iloc[:, 0]
            else:
                close = data

            # Calculate Log Returns
            log_returns = np.log(close / close.shift(1))

            # Calculate Realized Volatility (Annualized, 20-day rolling window)
            # Matching the logic in dataset.py: rolling(window=20).std() * np.sqrt(252)
            realized_volatility = log_returns.rolling(window=20).std() * np.sqrt(252)

            # Drop NaNs created by rolling window
            realized_volatility = realized_volatility.dropna()

            # Calculate Statistics
            n_obs = len(realized_volatility)
            mean_val = realized_volatility.mean()
            std_dev = realized_volatility.std()
            min_val = realized_volatility.min()
            max_val = realized_volatility.max()
            skew_val = realized_volatility.skew()
            kurt_val = realized_volatility.kurtosis() # Pandas kurtosis is Fisher (normal=0), but commonly 'excess kurtosis'
            # The prompt asks if Kurtosis > 3 (Leptokurtic). 
            # Standard Pearson kurtosis of normal is 3. Excess is 0.
            # Pandas .kurtosis() returns Excess Kurtosis (Fisher). 
            # So if Pandas Kurtosis > 0, it is Leptokurtic.
            # However, prompt says "Leptokurtic (Kurtosis > 3)". This implies they might mean raw Pearson kurtosis.
            # I will calculate Raw Kurtosis to be safe and match the condition "Kurtosis > 3".
            # Raw Kurtosis = Excess Kurtosis + 3.
            
            raw_kurtosis = kurt_val + 3

            stats_list.append({
                "Index": name,
                "Count": n_obs,
                "Mean": mean_val,
                "Standard Deviation": std_dev,
                "Min": min_val,
                "Max": max_val,
                "Skewness": skew_val,
                "Kurtosis": raw_kurtosis 
            })

        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Create DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # Reorder columns
    cols = ["Index", "Count", "Mean", "Standard Deviation", "Min", "Max", "Skewness", "Kurtosis"]
    stats_df = stats_df[cols]

    output_lines = []
    output_lines.append("Descriptive Statistics Table for Daily Realized Volatility:")
    output_lines.append(stats_df.to_string(index=False, float_format="%.4f"))
    output_lines.append("\nKurtosis Analysis:")
    
    # Summary Analysis for Kurtosis
    for i, row in stats_df.iterrows():
        k_val = row['Kurtosis']
        index_name = row['Index']
        if k_val > 3:
            output_lines.append(f"- {index_name}: Kurtosis = {k_val:.4f} (> 3). Distribution is Leptokurtic. High likelihood of extreme volatility events ('fat tails'). Justifies Attention Mechanism.")
        else:
            output_lines.append(f"- {index_name}: Kurtosis = {k_val:.4f} (<= 3). Distribution is not Leptokurtic.")

    with open("volatility_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        
    print("Analysis saved to volatility_analysis.txt")

if __name__ == "__main__":
    get_data_and_stats()
