import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('date', inplace=True)
    return df[['close']]

# Create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Main function to run analysis
def run_analysis(input_csv, output_txt):
    btc_close = load_and_preprocess_data(input_csv)
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_close_scaled = scaler.fit_transform(btc_close)
    
    time_step = 100
    X, y = create_dataset(btc_close_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(100, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit model (assuming 80% data for training and 20% for testing)
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Predict future prices
    last_100_days = btc_close_scaled[-100:]
    future_prices = []
    for _ in range(365 * 4):  # Predict next 4 years
        x_input = last_100_days[-100:].reshape(1, -1)
        x_input = x_input.reshape((1, 100, 1))
        price = model.predict(x_input, verbose=0)
        future_prices.append(price[0][0])
        last_100_days = np.append(last_100_days, price)
    
    # Scale back the predicted prices
    future_prices_scaled = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    
    # Write predictions to output txt file
    start_date = btc_close.index[-1] + timedelta(days=1)
    with open(output_txt, 'w') as f:
        for i, price in enumerate(future_prices_scaled):
            date = start_date + timedelta(days=i)
            f.write(f"{date.strftime('%Y-%m-%d')}, {price[0]}\n")

# Example usage
run_analysis('C:\\gitproject\\tradebot\\anolize\\BITFINEX_BTCUSD.csv', 'future_prices.txt')
