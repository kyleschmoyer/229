import pandas as pd
import numpy as np

# Function to calculate the average trade size from a range string
def parse_trade_size(trade_size):
    try:
        trade_size = trade_size.replace(',', '').replace('$', '').replace(" ", "").split('-')
        return (float(trade_size[0]) + float(trade_size[1])) / 2
    except Exception as e:
        print(f"Error parsing trade size '{trade_size}': {e}")
        return 0

# Read CSV files
stock_path = '/Users/kyleschmoyer/Downloads/cs229project/all_data_final.csv'
congress_path = '/Users/kyleschmoyer/Downloads/cs229project/congress-trading-all.csv'

stock = pd.read_csv(stock_path)
congress = pd.read_csv(congress_path)

# Filter congress data to only include matching tickers
matching_tickers = stock['Ticker'].unique()
congress_filtered = congress[congress['Ticker'].isin(matching_tickers)].copy()

# Apply the parse_trade_size function
congress_filtered['Trade_Size_USD'] = congress_filtered['Trade_Size_USD'].apply(parse_trade_size)

# Convert date columns to datetime
congress_filtered['Filed'] = pd.to_datetime(congress_filtered['Filed'], errors='coerce')
congress_filtered['Traded'] = pd.to_datetime(congress_filtered['Traded'], errors='coerce')
stock['datetime'] = pd.to_datetime(stock['datetime'], errors='coerce')

# Calculate the time difference in days
congress_filtered['time_diff'] = (congress_filtered['Filed'] - congress_filtered['Traded']).dt.days

# Count the number of transactions for each type and calculate additional metrics
transaction_types = {
    'Purchase': 'purchase_count',
    'Sale': 'sale_count',
    'Sale (Full)': 'salefull_count',
    'Sale (Partial)': 'salepartial_count'
}

all_counts = pd.DataFrame()

for transaction, col_name in transaction_types.items():
    counts = (
        congress_filtered[congress_filtered['Transaction'] == transaction]
        .groupby(['Ticker', 'Filed']).size()
        .reset_index(name=col_name)
    )
    if all_counts.empty:
        all_counts = counts
    else:
        all_counts = all_counts.merge(counts, how='left', on=['Ticker', 'Filed'])

# Add average trade size and time difference
average_metrics = congress_filtered.groupby(['Ticker', 'Filed']).agg(
    average_trade_size_per_day=('Trade_Size_USD', 'mean'),
    average_time_diff=('time_diff', 'mean')
).reset_index()

# Merge all the metrics
all_counts = all_counts.merge(average_metrics, how='left', on=['Ticker', 'Filed'])

# Merge the combined data back into the stock dataframe
stock = stock.merge(all_counts, how='left', left_on=['Ticker', 'datetime'], right_on=['Ticker', 'Filed']).fillna(0)

# Calculate additional metrics for stock data
stock['price_range'] = stock['high'] - stock['low']
stock['volatility'] = stock['close'].rolling(window=10).std()
stock['volatility_10day_ma'] = stock['volatility'].rolling(window=10).mean()

# Fill NaN values in the stock DataFrame with zero
stock.fillna(0, inplace=True)
# Save the updated DataFrame to a CSV file
output_path = '/Users/kyleschmoyer/Downloads/cs229project/withcounts.csv'
stock.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
