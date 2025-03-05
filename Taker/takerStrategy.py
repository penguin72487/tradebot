import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
file_path = 'C:\\gitproject\\tradebot\\Taker\\BITSTAMP_BTCUSD, 240Taker.csv'
data = pd.read_csv(file_path)

# Ensure proper column formatting (adjust column names if needed)
data.columns = [col.strip() for col in data.columns]

# Example column names adjustment (if required)
data.rename(columns={"timestamp": "time", "close": "close_price"}, inplace=True)

# Parse timestamps (if the time column is in string format)
data['time'] = pd.to_datetime(data['time'])

# Define the strategy function
def apply_strategy(data, leverage, alpha_percent):
    # Strategy Variables
    mhull = data['MHULL']  # Placeholder for MHULL calculation
    shull = data['SHULL']  # Placeholder for SHULL calculation

    sm_diffpercent = (shull - mhull) / shull * 100.0 - 100.0  # Calculate the difference percentage
    bull_m = sm_diffpercent * alpha_percent  # Calculate the Bull Market Signal

    # Initialize columns
    data['Equity'] = 1000  # Placeholder for Equity
    data['position_size'] = 0  # Placeholder for Position Size

    for i in range(1, len(data)):
        delta_price = data.loc[i, 'close_price'] - data.loc[i - 1, 'close_price']
        data.loc[i, 'Equity'] = data.loc[i - 1, 'position_size'] * delta_price + data.loc[i - 1, 'Equity']
        data.loc[i, 'position_size'] = (
            data.loc[i, 'Equity'] * bull_m[i] * leverage / data.loc[i, 'close_price']
        )

    return data

# Function to split data
def split_data(data, split_ratio):
    split_index = int(len(data) * split_ratio)
    return data.iloc[:split_index]

# Apply the strategy to the first 10% of the data
subset_data = split_data(data, 0.01)
strategy_result = apply_strategy(
    data=subset_data,
    leverage=1.0,
    alpha_percent=1.0
)

# Visualize the key strategy outputs
plt.figure(figsize=(12, 8))
plt.plot(strategy_result['time'], strategy_result['position_size'], label="Position Size")
plt.plot(strategy_result['time'], strategy_result['Equity'], label="Equity")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.title("Strategy Backtest Results")
plt.grid()
plt.show()

# Save the results to a new CSV
output_file_path = 'C:\\gitproject\\tradebot\\Taker\\results.csv'
strategy_result.to_csv(output_file_path, index=False)

output_file_path
