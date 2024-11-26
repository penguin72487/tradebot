import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import gc
import os

torch.cuda.empty_cache()
gc.collect()

# Load configuration from config.json
with open('C:\\gitproject\\tradebot\\ML\\configlite.json', 'r') as f:
    config = json.load(f)

# Load the dataset
file_path = config['file_path']
df = pd.read_csv(file_path)

# Keep only the 'close' price column (use real prices)
price_data = df['close'].values

# Hyperparameters
sequence_length = config['seq_len']
batch_size = config['batch_size']
hidden_size = config['hidden_dim']
num_layers = config['num_layers']
output_size = config['output_dim']
epochs = config['epochs']
learning_rate = config['learning_rate']

model_save_path = config['model_save_path']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the dataset
class PriceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])

dataset = PriceDataset(price_data, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Define the LSTM model
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

# Instantiate the model, loss function, and optimizer
model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
criterion = nn.L1Loss()  # Use MAE for more stable training with real prices
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to predict future prices for a given number of steps
def predict_future_prices(data, model, sequence_length, num_predictions, device):
    model.eval()
    predictions = []

    # Use the most recent data to make predictions iteratively
    input_sequence = torch.FloatTensor(data[-sequence_length:].reshape(-1, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(num_predictions):
            predicted_value = model(input_sequence).cpu().item()
            predictions.append(predicted_value)
            print(predicted_value)
            # Update the input sequence by appending the predicted value and removing the oldest value
            input_sequence = torch.cat((input_sequence[:, 1:, :], torch.FloatTensor([[predicted_value]]).unsqueeze(0).to(device)), dim=1)


    return predictions

# Number of steps to predict into the future
num_predictions = 1000

# Generate predictions
latest_data = price_data[-sequence_length:]
predicted_prices = predict_future_prices(latest_data, model, sequence_length, num_predictions, device)

# Plot the actual and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(range(len(price_data)), price_data, label='Actual Price', color='blue')
plt.plot(range(len(price_data), len(price_data) + num_predictions), predicted_prices, label='Predicted Price', color='red')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.title('Actual Price vs Predicted Price')
plt.legend()
plt.grid(True)
plt.show()
