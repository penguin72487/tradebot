import torch
import torch.nn as nn
import pandas as pd
import json

# Load config
with open("C:\\gitproject\\tradebot\\ML\\config.json", 'r') as f:
    config = json.load(f)

# Load model parameters
file_path = config["file_path"]
model_save_path = config["model_save_path"]
seq_len = config["seq_len"]
model_params = config["model_params"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load price data from CSV
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
            batch_first=True  # Set batch_first to True for better performance
        )
        self.linear_in = nn.Linear(input_dim, d_model)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.linear_in(src)  # (batch_size, seq_len, d_model)
        transformer_output = self.transformer(src, src)
        output = self.linear_out(transformer_output)  # (batch_size, seq_len, 1)
        return output[:, -1, 0]  # Return only the last time step prediction

# Load the trained model
model = TransformerTradingModel(model_params["input_dim"], 
                                model_params["d_model"], 
                                model_params["nhead"], 
                                model_params["dim_feedforward"], 
                                model_params["num_layers"]).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=True))  # Set weights_only=True for safer loading
model.eval()

# Backtesting setup
initial_cash = 10000.0
cash = initial_cash
position = 0.0  # Start without any position

# Iterate through the data for backtesting, processing one price point at a time
for i in range(seq_len, len(data) - 1):
    # Prepare the past price data as input
    past_prices = torch.tensor(data[i-seq_len:i], dtype=torch.float32).view(1, seq_len, 1).to(device)

    # Get the predicted position percentage from the model
    with torch.no_grad():
        predicted_position_percent = model(past_prices)

    # Clamp predicted_position_percent to be between -1 and 1
    predicted_position_percent = torch.clamp(predicted_position_percent, -1.0, 1.0).item()

    # Calculate the change in price
    current_price = data[i]
    next_price = data[i + 1]
    price_change = (next_price - current_price) / current_price

    # Update position based on predicted position percentage
    position = cash * predicted_position_percent / current_price

    # Update cash based on position and price change
    cash += position * price_change

    # Print backtesting information for each step
    print(f"Step [{i}/{len(data)}], Cash: {cash:.2f}", position, price_change)

# Print final backtesting results
print(f"Initial Cash: {initial_cash:.2f}, Final Cash: {cash:.2f}, Profit: {(cash - initial_cash):.2f}")
