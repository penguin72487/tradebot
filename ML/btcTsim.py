import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load model and scaler
config_path = "C:\\gitproject\\tradebot\\ML\\config.json"
model_save_path = "./transformer_kline_model.pth"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Load model
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, seq_length, num_heads, num_layers, hidden_dim):
        super(TransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerPredictor(1, config["seq_len"], config["model_params"]["nhead"], config["model_params"]["num_layers"], config["model_params"]["d_model"]).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()

# Load data
scaler = StandardScaler()
data_path = config["file_path"]
df = pd.read_csv(data_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
data = scaler.fit_transform(df[['close']].values)

# Prepare to store actual and predicted values
actual_prices = []
predicted_prices = []

# Sliding window to simulate the prediction process
seq_length = config["seq_len"]
with torch.no_grad():
    for i in range(len(data) - seq_length - 1):
        # Prepare input sequence for the model
        input_seq = data[i:i + seq_length]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Model prediction
        predicted = model(input_tensor).item()
        predicted_prices.append(predicted)
        
        # Get the actual next value
        actual_value = data[i + seq_length]
        actual_prices.append(actual_value[0])

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices', linestyle='--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price (Standardized)')
plt.legend()
plt.show()
