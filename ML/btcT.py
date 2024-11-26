import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.preprocessing import StandardScaler
import torch
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
config_path = "C:\\gitproject\\tradebot\\ML\\config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Load data from the CSV file
data_path = config["file_path"]
df = pd.read_csv(data_path)

# Assume we have columns: ['timestamp', 'open', 'high', 'low', 'close', 'Volume']
# We will use the 'close' as our input feature
if df.isnull().values.any():
    df = df.dropna()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(df[['close']].values)

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
input_dim = 1  # Changed to 1 since we only have 'close' as input feature
hidden_dim = config["model_params"]["d_model"]
num_heads = config["model_params"]["nhead"]
num_layers = config["model_params"]["num_layers"]
batch_size = 512  # Decreased batch size
epochs = config["epochs"]
learning_rate = 1e-5  # Lower learning rate

# Prepare dataset and dataloader
dataset = TimeSeriesDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerPredictor(input_dim, seq_length, num_heads, num_layers, hidden_dim).to(device)

# Initialize weights
model.apply(init_weights)

criterion = nn.SmoothL1Loss()  # Changed to Smooth L1 Loss
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
scaler = torch.amp.GradScaler('cuda')
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
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

# Save the trained model
model_save_path = config["model_save_path"]
torch.save(model.state_dict(), model_save_path)

print("Training complete.")
