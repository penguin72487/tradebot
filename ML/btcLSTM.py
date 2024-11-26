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
with open('C:\\gitproject\\tradebot\\ML\\configlite.json', 'r') as f:
    config = json.load(f)

# Load the dataset
file_path = config['file_path']
df = pd.read_csv(file_path)

# Keep only the 'close' price column (use real prices)
price_data = df['close'].values

# Split the dataset
train_size =  config['train_Percent']
validation_size = config['validation_Percent']
test_size = config['test_Percent']

train_data = price_data[:int(len(price_data) * train_size)]
# validation_data = price_data[int(len(price_data) * train_size):int(len(price_data) * (train_size + validation_size))]
# test_data = price_data[int(len(price_data) * (train_size + validation_size))]

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

train_dataset = PriceDataset(train_data, sequence_length)
# validation_dataset = PriceDataset(validation_data, sequence_length)
# test_dataset = PriceDataset(test_data, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
# criterion = nn.SmoothL1Loss()
 

# Replace Huber Loss with Mean Absolute Percentage Error (MAPE) Loss
def mape_loss(pred, target):
    epsilon = 1e-8  # To avoid division by zero
    return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100

criterion = mape_loss




optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
# optimizer = optim.Lamb(model.parameters(), lr=learning_rate)
# optimizer = optim.Ranger(model.parameters(), lr=learning_rate)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=2, patience=5, verbose=True)



# Load the model if a checkpoint exists
start_epoch = 0
if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resuming training from epoch {start_epoch}')


# 紀錄訓練過程的prediction price

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
        loss = torch.mean(torch.abs((predicted_price - y_batch) / y_batch))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        torch.cuda.empty_cache()
        record.append([predicted_price[0].item(), y_batch[0].item()])
        if (i) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f} ,Predicted Price: {predicted_price[0].item()}, Actual Price: {y_batch[0].item()}')
    # scheduler.step(loss)
    # Save the model checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    output = pd.DataFrame(record, columns=['Predicted', 'Actual'])
    output.to_csv('C:\\gitproject\\tradebot\\ML\\record_epoch_{}.csv'.format(epoch))

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
