import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

# Load config
with open("C:\\gitproject\\tradebot\\ML\\config.json", 'r') as f:
    config = json.load(f)

file_path = config["file_path"]
model_save_path = config["model_save_path"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]
batch_size = config["batch_size"]
seq_len = config["seq_len"]
model_params = config["model_params"]

# Hyperparameters
d_model = model_params["d_model"]
nhead = model_params["nhead"]
dim_feedforward = model_params["dim_feedforward"]
num_layers = model_params["num_layers"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load price data from CSV
import pandas as pd
data = pd.read_csv(file_path)['close'].values  # Assuming 'close' column contains the price data

# Transformer model definition
class TransformerTradingModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, num_layers):
        super(TransformerTradingModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        self.linear_in = nn.Linear(input_dim, d_model)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.linear_in(src).permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        transformer_output = self.transformer(src, src)
        output = self.linear_out(transformer_output).permute(1, 0, 2)  # (seq_len, batch_size, d_model) -> (batch_size, seq_len, 1)
        return output[:, -1, 0]  # Return only the last time step prediction

# Training setup
model = TransformerTradingModel(model_params["input_dim"], d_model, nhead, dim_feedforward, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
cash = 10000.0  # Initial cash

for epoch in range(epochs):
    cash = 10000.0  # Reset cash for each epoch
    all_position = 0.0  # Initial all-in position
    total_Loss = 0.0  # Initial total loss
    for i in range(0, len(data) - seq_len, batch_size):
        batch_indices = range(i, min(i + batch_size, len(data) - seq_len))
        past_prices = torch.stack([torch.tensor(data[j:j + seq_len], dtype=torch.float32) for j in batch_indices]).view(len(batch_indices), seq_len, 1).to(device)
        current_prices = torch.tensor([data[j + seq_len] for j in batch_indices], dtype=torch.float32).to(device)
        previous_prices = torch.tensor([data[j + seq_len - 1] for j in batch_indices], dtype=torch.float32).to(device)

        # Forward pass
        model.train()
        predicted_position_percent = model(past_prices)

        # Clamp predicted_position_percent to be between -1 and 1
        predicted_position_percent = torch.clamp(predicted_position_percent, -1.0, 1.0)

        # Calculate delta percentage
        delta_percentage = (current_prices - previous_prices) / previous_prices - 1

        # Create target position percent based on delta price
        target_position_percent = torch.where(delta_percentage > 1, torch.tensor(1.0).to(device), torch.tensor(-1.0).to(device))

        # Compute loss as the product of delta percentage and target position percent
        loss = torch.mean((target_position_percent - predicted_position_percent) * delta_percentage)
        print(f"Loss: {loss}, Predicted Position: {predicted_position_percent}, delta_percentage: {delta_percentage}")
        if torch.isnan(predicted_position_percent).any():
            print("Predicted Position has NaN values.")
            break

        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate total loss for reporting
        total_Loss += loss.item()

    # Print epoch info
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_Loss:.4f}, Cash: {cash:.2f}")

# Save model
torch.save(model.state_dict(), model_save_path)

print("Training complete.")