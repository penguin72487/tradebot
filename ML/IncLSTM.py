import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'ML\\IncLSTM.py'
data = pd.read_csv(file_path)

# Extract only the 'close' column for the LSTM model
close_prices = data['close']

# Normalize the close prices to make them suitable for LSTM training
scaler = MinMaxScaler(feature_range=(-1, 1))
close_prices_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# Prepare the data for LSTM model
def prepare_data(sequence, seq_length):
    data = []
    for i in range(len(sequence) - seq_length):
        data.append((sequence[i:i + seq_length], sequence[i + seq_length]))
    return data

SEQ_LENGTH = 10
prepared_data = prepare_data(close_prices_scaled, SEQ_LENGTH)

# Hyperparameters
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 50

# Convert prepared data to DataLoader for batch processing
train_data = torch.utils.data.DataLoader(prepared_data, batch_size=16, shuffle=True)

# Define the IncrementalLSTM model with dynamic output scaling
class IncrementalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(IncrementalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_activation = nn.Tanh()  # Add tanh activation to ensure output is in the desired range
    
    def forward(self, x):
        h_0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).float().to(x.device)
        c_0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).float().to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Use the last time step's output
        out = self.fc(out)
        out = self.output_activation(out) * 100  # Scale output to range [-100, 100]
        return out

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, criterion, and optimizer
model = IncrementalLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Custom loss function based on equity change (simplified for demonstration purposes)
def custom_loss(preds, targets, prices):
    # Calculate the difference in equity based on the predictions and actual price movement
    # Assume the model's output is the position (between -100 and 100)
    price_change = prices[:, -1] - prices[:, -2]  # Simple price difference between last two time steps
    equity_change = preds.squeeze() * price_change  # Equity change based on predicted position and price change
    return -torch.mean(equity_change)  # We want to maximize equity, so minimize the negative equity

# Training loop
loss_history = []
for epoch in range(EPOCHS):
    for sequences, targets in train_data:
        sequences = torch.Tensor(sequences).view(-1, SEQ_LENGTH, INPUT_SIZE).float().to(device)  # Reshape to (batch, seq_length, input_size) and convert to float32
        targets = torch.Tensor(targets).float().to(device)  # Shape (batch, 1) and convert to float32
        
        # Forward pass
        outputs = model(sequences)
        loss = custom_loss(outputs, targets, sequences)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_history.append(loss.item())
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}')

# Plot the training loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()
