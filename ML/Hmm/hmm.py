import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def monte_carlo_simulation(file_path, num_simulations=1000, num_days=10000):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Calculate daily returns based on 'close' prices
    df['return'] = df['close'].pct_change()

    # Calculate mean and standard deviation of returns
    mean_return = df['return'].mean()
    std_return = df['return'].std()

    # Start from the latest close price
    start_price = df['close'].iloc[-1]

    # Step 3: Run Monte Carlo simulation for the given number of days
    simulations = np.zeros((num_days, num_simulations))

    for i in range(num_simulations):
        price_series = [start_price]
        for _ in range(num_days):
            random_return = np.random.normal(mean_return, std_return)
            next_price = price_series[-1] * (1 + random_return)
            price_series.append(next_price)
        simulations[:, i] = price_series[1:]

    # Step 4: Plot the Monte Carlo simulated price paths
    plt.figure(figsize=(12, 6))
    plt.plot(simulations, alpha=0.05, color='blue')
    plt.title('Monte Carlo Simulation of Future BTC/USD Prices ({} Days, {} Simulations)'.format(num_days, num_simulations))
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # File path to your local CSV file
    file_path = "BITSTAMP_BTCUSD,240more.csv"
    
    # Run the Monte Carlo simulation
    monte_carlo_simulation(file_path, num_simulations=1000, num_days=10000)
