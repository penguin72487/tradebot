from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(args):
    start_price, state_means, state_stds, hidden_states, model, num_days = args
    price_series = [start_price]
    current_state = hidden_states[-1]  # Start with the latest detected market state

    for _ in range(num_days):
        # Randomly generate returns based on current state mean and std deviation
        random_return = np.random.normal(state_means[current_state], state_stds[current_state])
        next_price = price_series[-1] * (1 + random_return)
        price_series.append(next_price)

        # Predict the next hidden state based on the current return
        next_hidden_state = model.predict(np.array([[random_return]]))
        current_state = next_hidden_state[0]

    return price_series[1:]

# Main entry point
if __name__ == "__main__":
    # Load the data
    file_path = 'C:\\gitproject\\tradebot\\ML\\btcLSTMmore\\BITSTAMP_BTCUSD,240more.csv'
    df = pd.read_csv(file_path)

    # Step 1: Prepare data for Hidden Markov Model (HMM)
    # Calculate daily returns based on 'close' prices
    df['return'] = df['close'] / df['close'].shift(1)
    print(df.isnull().sum())
    df['return'] = df['return'].fillna(0.0)  # Fill NaN values with 0.0
    returns = df['return'].values.reshape(-1, 1)

    # Step 2: Fit an HMM to model different market states (e.g., Bullish, Bearish ,sideways)
    model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=100)
    model.fit(returns)

    # Predict the hidden states for each time step
    hidden_states = model.predict(returns)

    # Step 3: Calculate mean and standard deviation for each hidden state
    state_means = []
    state_stds = []

    for state in range(model.n_components):
        state_returns = returns[hidden_states == state]
        state_means.append(np.mean(state_returns))
        state_stds.append(np.std(state_returns))

    # Step 4: Monte Carlo simulation using multiprocessing
    num_simulations = 100
    num_days = 1000
    start_price = df['close'].iloc[-1]

    # Prepare arguments for each simulation
    args = [(start_price, state_means, state_stds, hidden_states, model, num_days) for _ in range(num_simulations)]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(monte_carlo_simulation, args))

    # Converting results to a numpy array for plotting
    simulations = np.array(results).T

    # Step 5: Plot the Monte Carlo simulated price paths
    plt.figure(figsize=(12, 6))
    plt.plot(simulations, alpha=0.05, color='blue')
    plt.title('HMM-based Monte Carlo Simulation with 5 States of Future BTC/USD Prices ({} Days, {} Simulations)'.format(num_days, num_simulations))
    plt.show()
