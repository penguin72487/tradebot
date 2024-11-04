import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
# Load model and scaler
config_path = "C:\\gitproject\\tradebot\\ML\\btcT\\configBTCmore.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

model_save_path = config["save_path"]+ config["model_name"]+".pth"
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

# Load data
scaler_standard = StandardScaler()
features = df.columns
#幫我數每個column有null的數量
print(df.isnull().sum())
# df = df.dropna()
features = config["features"].replace("'", "").replace(", ", ",").split(",")
data = scaler_standard.fit_transform(df[features].values)
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
    # best_loss = checkpoint.get('best_loss', best_loss)
    print(f'Resuming training from epoch {start_epoch}')
model.eval()

# Prepare to store actual and predicted values
actual_prices = []
predicted_prices = []

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
        predicted_prices.append(predicted)

        # Get the actual next value
        actual_value = data[i + seq_length][0]  # 取出下一個時間步的實際 'close' 價格
        actual_prices.append(actual_value)

        # 更新滑動窗口：用實際的特徵進行更新，將舊的數據丟棄
        new_entry = data[i + seq_length]  # 獲取新的時間步的所有特徵
        sliding_window = np.vstack((sliding_window[1:], new_entry))  # 保持窗口大小不變

# 反標準化處理
actual_prices_unscaled = scaler_standard.inverse_transform(
    np.column_stack([actual_prices] + [np.zeros(len(actual_prices)) for _ in range(len(features) - 1)]))[:, 0]

predicted_prices_unscaled = scaler_standard.inverse_transform(
    np.column_stack([predicted_prices] + [np.zeros(len(predicted_prices)) for _ in range(len(features) - 1)]))[:, 0]

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices_unscaled, label='Actual Prices', color='blue')
plt.plot(predicted_prices_unscaled, label='Predicted Prices', linestyle= 'dotted', color='orange')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# Output to csv
df_result = pd.DataFrame(data={"Actual": actual_prices_unscaled, "Predicted": predicted_prices_unscaled})
df_result.to_csv("C:\\gitproject\\tradebot\\ML\\btcT\\TAIEXTmoreSim{}.csv".format(start_epoch), sep=',', index=False)
