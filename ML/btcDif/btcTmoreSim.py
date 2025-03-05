import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Load model and scaler
config_path = "C:\\gitproject\\tradebot\\ML\\btcDif\\configBTCmore.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

model_save_path = config["save_path"] + config["model_name"] + ".pth"
best_model_save_path = model_save_path.replace('.pth', '_best.pth')

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

# Load data from the CSV file
data_path = config["file_path"]
df = pd.read_csv(data_path)

# Count null values in each column
# print(df.isnull().sum())

# Select features and apply Min-Max Scaling
features = config["features"].replace("'", "").replace(", ", ",").split(",")
data = df[features].values
min_value = np.min(data)
max_value = np.max(data)
data = (data - min_value) / (max_value - min_value)

model = TransformerPredictor(input_dim=len(features), 
                             seq_length=config["seq_len"], 
                             num_heads=config["nhead"], 
                             num_layers=config["num_layers"], 
                             hidden_dim=config["hidden_dim"]).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resuming training from epoch {start_epoch}')
model.eval()

# Prepare to store actual and predicted values
actual_prices = []
predicted_delta = []
predicted_prices = []

balance_history = []
balance = 1000

# Sliding window to simulate the prediction process
seq_length = config["seq_len"]
with torch.no_grad():
    # Initialize sliding window with the first sequence of data
    sliding_window = data[:seq_length]

    for i in range(len(data) - seq_length - 1):
        # Prepare input sequence for the model
        input_tensor = torch.tensor(sliding_window, dtype=torch.float32).unsqueeze(0).to(device)

        # Model prediction
        predicted = model(input_tensor).item()
        predicted_delta.append(predicted)
        predicted = predicted + 1

        predicted_prices.append(input_tensor[0, -1, 0].item() * predicted)

        # Get the actual next value
        actual_value = data[i + seq_length][0]  # 取出下一個時間步的實際 'close' 價格
        actual_prices.append(actual_value)

        # update balance
        prev_price = data[i + seq_length - 1][0]* (max_value - min_value) + min_value
        current_price = data[i + seq_length][0]* (max_value - min_value) + min_value
        delta_price = current_price - prev_price
        kelly_fraction = 1/ (predicted-1)
        balance += balance/prev_price * kelly_fraction * delta_price
        balance_history.append(balance)

        if balance < 0:
            balance = 1000

        # 更新滑動窗口：用實際的特徵進行更新，將舊的數據丟棄
        new_entry = data[i + seq_length]  # 獲取新的時間步的所有特徵
        sliding_window = np.vstack((sliding_window[1:], new_entry))  # 保持窗口大小不變

# Convert actual_prices and predicted_prices to numpy arrays for scaling
actual_prices = np.array(actual_prices)
predicted_delta = np.array(predicted_delta)
predicted_prices = np.array(predicted_prices)
balance_history = np.array(balance_history)

# 反標準化處理
actual_prices_unscaled = actual_prices * (max_value - min_value) + min_value
predicted_prices_unscaled = predicted_prices * (max_value - min_value) + min_value

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices_unscaled, label='Actual Prices', color='blue')
plt.plot(predicted_prices_unscaled, label='Predicted Prices', linestyle='dotted', color='orange')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()


# Plot the balance history
plt.figure(figsize=(10, 6))
plt.plot(balance_history, label='Balance', color='green')
plt.title('Balance History')
plt.xlabel('Time Steps')
plt.ylabel('Balance')
plt.legend()
plt.show()


# Output to csv
df_result = pd.DataFrame(data={"Actual": actual_prices_unscaled, "Predicted": predicted_prices_unscaled, "predicted_delta": predicted_delta})
df_result.to_csv(config["save_path"] + config["model_name"] + "_result.csv", index=False)
