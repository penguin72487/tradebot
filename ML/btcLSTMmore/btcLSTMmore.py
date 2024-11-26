import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import gc
import os
import torch_optimizer as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.empty_cache()
gc.collect()

# Load configuration from config.json
with open('C:\\gitproject\\tradebot\\ML\\btcLSTMmore\\configlite.json', 'r') as f:
    config = json.load(f)

# Load the dataset
file_path = config['file_path']
df = pd.read_csv(file_path)

# Keep only the relevant columns as features (including 'close' for prediction)
#time,open,high,low,close,PMA12,PMA144,PMA169,PMA576,PMA676,MHULL,SHULL,KD,J,RSI,MACD,Signal Line,Histogram,QQE Line,Histo2,volume,Bullish Volume Trend,Bearish Volume Trend
features = ['close', 'PMA12', 'PMA144', 'PMA169', 'PMA576', 'PMA676', 'MHULL', 'SHULL', 'KD', 'J', 'RSI', 'MACD', 'Signal Line', 'Histogram', 'QQE Line', 'Histo2', 'volume', 'Bullish Volume Trend', 'Bearish Volume Trend']
price_data = df[features].values

# Split the dataset
train_size =  config['train_Percent']
train_data = price_data[:int(len(price_data) * train_size)]

# Hyperparameters
sequence_length = config['seq_len']
batch_size = config['batch_size']
hidden_size = config['hidden_dim']
num_layers = config['num_layers']
output_size = config['output_dim']
epochs = config['epochs']
learning_rate = config['learning_rate']

model_save_path = config['save_path']+config['model_name']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the dataset
class PriceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length,:]  # Input features excluding 'close'
        y = self.data[index + self.seq_length, 0]  # Predicting 'close' price
        return torch.FloatTensor(x), torch.FloatTensor([y])

train_dataset = PriceDataset(train_data, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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
input_size = len(features)  # Number of features excluding 'close'
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# Replace Huber Loss with Mean Absolute Percentage Error (MAPE) Loss
def mape_loss(pred, target):
    epsilon = 1e-8  # 防止除零
    pred = torch.clamp(pred, min=1e-3)  # 將預測值限制在最小正值
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100

# 定義 Log-Cosh Loss
def log_cosh_loss(pred, target):
    loss = torch.log(torch.cosh(pred - target))
    return torch.mean(loss)


# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.HuberLoss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.NLLLoss()
# criterion = nn.KLDivLoss()
# criterion = nn.PoissonNLLLoss()
# criterion = nn.KLDivLoss()
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MarginRankingLoss()
# criterion = nn.HingeEmbeddingLoss()
# criterion = nn.MultiLabelMarginLoss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.MultiLabelSoftMarginLoss()
# criterion = nn.CosineEmbeddingLoss()
# criterion = nn.MultiMarginLoss()
# criterion = nn.TripletMarginLoss()
# criterion = nn.TripletMarginWithDistanceLoss()
# criterion = nn.MarginRankingLoss()
# criterion = nn.SoftMarginLoss()
criterion = mape_loss
# criterion = log_cosh_loss

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Load the model if a checkpoint exists
start_epoch = 0
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resuming training from epoch {start_epoch}')

# Record predictions during training
record = []

# Train the model
model.train()
for epoch in range(start_epoch, epochs):
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        predicted_price = model(x_batch)
        loss = criterion(predicted_price, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        torch.cuda.empty_cache()
        record.append([predicted_price[0].item(), y_batch[0].item(), loss.item()])
        # if (i) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Predicted Price: {predicted_price[0].item()}, Actual Price: {y_batch[0].item()}')

    # Save the model checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    output = pd.DataFrame(record, columns=['Predicted', 'Actual', 'Loss'])
    output.to_csv('{model_save_path}{epoch}.csv', index=False)
    record = []

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print('Training Completed.')
