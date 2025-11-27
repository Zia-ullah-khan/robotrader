from yahoo_fin.stock_info import tickers_sp500

sp500_list = tickers_sp500()

# The list of tickers is now in the `sp500_list` variable.
print(sp500_list[:10]) # Print the first 10 tickers
