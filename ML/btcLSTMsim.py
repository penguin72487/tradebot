import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import gc
# from btcLSTMsim import LSTMModel

# Load configuration from config.json
with open('C:\\gitproject\\tradebot\\ML\\configlite.json', 'r') as f:
    config = json.load(f)


sequence_length = config['seq_len']
batch_size = config['batch_size']
hidden_size = config['hidden_dim']
num_layers = config['num_layers']
output_size = config['output_dim']
epochs = config['epochs']
learning_rate = config['learning_rate']

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)

        out, (hn, _) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

# Load the dataset
file_path = config['file_path']
df = pd.read_csv(file_path)

# Use only the 'close' price column
price_data = df['close'].values

# Hyperparameters
sequence_length = config['seq_len']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = LSTMModel(input_size=1, hidden_size=config['hidden_dim'], num_layers=config['num_layers'], output_size=config['output_dim']).to(device)
model.load_state_dict(torch.load(config['model_save_path']))
model.eval()

# Function to predict optimal position size
def predict_optimal_position(data, model, sequence_length, device):
    input_sequence = torch.FloatTensor(data[-sequence_length:].reshape(-1, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_value = model(input_sequence).cpu().item()
    return predicted_value

# Backtesting parameters
initial_balance = 10000  # Initial cash balance in USD
position_size = 0  # BTC holdings
balance = initial_balance

# Backtest logic
for i in range(sequence_length, len(price_data)):
    latest_data = price_data[i-sequence_length:i]
    predicted_position = predict_optimal_position(latest_data, model, sequence_length, device)
    current_price = price_data[i]
    position_size = predicted_position* balance / price_data[i-1]  # Calculate position size based on predicted position and previous price
    deltaPrice = current_price - price_data[i-1]
    # If model suggests to buy (positive position size) and we have cash
    balance += position_size * deltaPrice
    hold_Strategy = current_price/price_data[0]
    print(f'Predicted Position: {predicted_position:.2f}, Current Price: {current_price:.2f}, Position Size: {position_size:.2f}, Balance: {balance:.2f}, Hold Strategy: {hold_Strategy:.2f}')

# Final Balance

print(f'Final Balance: ${balance:.2f}')
