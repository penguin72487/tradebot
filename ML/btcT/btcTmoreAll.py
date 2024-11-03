import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

# 定義初始化權重的函數
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.TransformerEncoderLayer):
        nn.init.xavier_uniform_(m.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(m.linear1.weight)
        nn.init.xavier_uniform_(m.linear2.weight)

# Load configuration from the JSON file
config_path = "C:\\gitproject\\tradebot\\ML\\btcT\\configmore.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Load data from the CSV file
data_path = config["file_path"]
df = pd.read_csv(data_path)

# Standardize the data
scaler_standard = StandardScaler()
features = ['close', 'PMA12', 'PMA144', 'PMA169', 'PMA576', 'PMA676', 'MHULL', 'SHULL', 'KD', 'J', 'RSI', 'MACD', 'Signal Line', 'Histogram', 'QQE Line', 'Histo2', 'volume', 'Bullish Volume Trend', 'Bearish Volume Trend']
data = scaler_standard.fit_transform(df[features].values)

# Dataset class to handle the input sequence data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index + self.seq_length, 0]  # Predicting next 'close' price
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Transformer model definition
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
        # Create embeddings with positional encodings
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)  # No need to permute if batch_first=True
        x = x[:, -1, :]  # Get the last time step
        out = self.fc(x)
        return out

# Parameters from config file
seq_length = config["seq_len"]
input_dim = len(features)
hidden_dim = config["hidden_dim"]
num_heads = config["nhead"]
num_layers = config["num_layers"]
batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]

# Prepare dataset and dataloader
dataset = TimeSeriesDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerPredictor(input_dim, seq_length, num_heads, num_layers, hidden_dim).to(device)

# Initialize weights
model.apply(init_weights)

criterion = nn.SmoothL1Loss()  # Changed to Smooth L1 Loss
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Mixed precision scaler
scaler = torch.amp.GradScaler()

# Training and backtesting loop
for epoch in range(epochs):
    # Training
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    scheduler.step(avg_loss)

    # Backtesting after each epoch
    model.eval()
    actual_prices = []
    predicted_prices = []

    with torch.no_grad():
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
    actual_prices_unscaled = scaler_standard.inverse_transform(np.array(actual_prices).reshape(-1, 1))[:, 0]
    predicted_prices_unscaled = scaler_standard.inverse_transform(np.array(predicted_prices).reshape(-1, 1))[:, 0]

    # Save backtesting results to CSV
    df_result = pd.DataFrame(data={"Actual": actual_prices_unscaled, "Predicted": predicted_prices_unscaled})
    backtest_save_path = f"C:\\gitproject\\tradebot\\ML\\btcT\\btcTmoreSim_epoch_{epoch+1}.csv"
    df_result.to_csv(backtest_save_path, sep=',', index=False)

    print(f"Backtesting results saved for epoch {epoch+1}.")

# Save the trained model
model_save_path = config["model_save_path"]
torch.save(model.state_dict(), model_save_path)

print("Training complete.")
