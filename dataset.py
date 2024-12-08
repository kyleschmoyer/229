from twelvedata import TDClient
import pandas as pd
import time

# Initialize client - apikey parameter is requiered
td = TDClient(apikey="ad0b9d7d58e54f3b9cbc7260ac54ada7")

df = pd.read_csv('/Users/kyleschmoyer/Downloads/cs229project/congress-trading-all.csv')
stocks = df['Ticker'].unique().tolist()
# Get the top 200 most common stock tickers
top_200_stocks = df['Ticker'].value_counts().nlargest(300).index.tolist()
stocks = top_200_stocks


final_df = pd.DataFrame()

batch_size = 7
i = 0

# Split the stocks list into groups of 99
for stock in stocks:
    time.sleep(10)
    i += 1
    
    # Construct the necessary time series for the stock
    try:
        ts = td.time_series(
            symbol=stock,
            interval="1day",
            outputsize=365*12,
            timezone="America/New_York",
        )
        stock_df = ts.as_pandas()
        stock_df['Ticker'] = stock
        final_df = pd.concat([final_df, stock_df])
        final_df.to_csv('/Users/kyleschmoyer/Downloads/cs229project/all_data_final.csv', index=True)
        print(f"Processed stock: {stock}, number {i}")
    except Exception as e:
        print(f"Error processing stock {stock}: {e}")
        continue
    
    


# Save the final dataframe to a CSV file
final_df.to_csv('/Users/kyleschmoyer/Downloads/cs229project/all_data_final.csv', index=True)
