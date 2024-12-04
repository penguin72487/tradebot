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
print(df.isnull().sum())

# Select features and apply Min-Max Scaling
features = config["features"].replace("'", "").replace(", ", ",").split(",")
data = df[features].values
min_value = np.min(data)
max_value = np.max(data)
data = (data - min_value) / (max_value - min_value)

train_percent = config["train_Percent"]
test_percent = 1 - train_percent
train_size = int(len(data) * train_percent)
train_data = data[:train_size]
data = data[train_size:]


model = TransformerPredictor(input_dim=len(features), 
                             seq_length=config["seq_len"], 
                             num_heads=config["nhead"], 
                             num_layers=config["num_layers"], 
                             hidden_dim=config["hidden_dim"]).to(device)

if os.path.exists(best_model_save_path):
    checkpoint = torch.load(best_model_save_path, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f'Model loaded from {best_model_save_path}')
model.eval()

# Prepare to store actual and predicted values
actual_prices = []
predicted_prices = []
initial_balance = 1000.0  # Initial capital in USD
balance = initial_balance
position = 0  # Current position (in units of the asset)
balance_history = []  # Store balance over time

# Generalized Kelly Criterion function
def generalized_kelly(predicted, prob_dist, leverage=10000):
    """
    Calculate the optimal fraction to invest based on expected returns and probability distribution
    using the generalized Kelly criterion.
    """
    # Calculate expected return
    expected_return = torch.sum(predicted * prob_dist)
    # Calculate variance
    variance = torch.sum(prob_dist * (predicted - expected_return) ** 2)
    # Kelly formula (f = (mean / variance))
    kelly_fraction = expected_return / variance
    return kelly_fraction * leverage  # 返回每個可能性對應的分配比例，並乘上一個常數以避免過度投資

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
        predicted_prices.append(input_tensor[0, -1, 0].item() * predicted)

        # Get the actual next value
        actual_value = data[i + seq_length][0]  # 取出下一個時間步的實際 'close' 價格
        actual_prices.append(actual_value)

        # Calculate the return (change rate) and decide position size using Kelly Criterion
        outputs = model(input_tensor).squeeze()
        for i in range(outputs):
            predicted_value = outputs[i].item()  # 預測的數值

            # 將預測值轉換為機率
            prob_dist = torch.softmax(outputs, dim=-1)  # 這會產生每個樣本的機率分佈
            probability = prob_dist[i].item()  # 這是模型的機率值（可能需要根據需求調整）

            # 打印出預測和機率
            print(f"Prediction: {predicted_value:.4f}, Probability: {probability:.4f}")

        kelly_fraction = generalized_kelly(outputs, prob_dist)

        # Update position and balance
        prev_price = data[i + seq_length - 1][0]  # Previous price
        current_price = data[i + seq_length][0]
        delta_price = current_price - prev_price  # Price change

        balance += balance/prev_price*kelly_fraction*delta_price

        if(balance < 0):
            print(f"balance < 0, balance={balance}, prev_price={prev_price}, current_price={current_price}, kelly_fraction={kelly_fraction}, delta_price={delta_price}")
            break

        # Store balance history
        balance_history.append(balance)  # Balance plus current position value
        print(f"balance={balance}, prev_price={prev_price}, current_price={current_price}, kelly_fraction={kelly_fraction}, delta_price={delta_price}")

        # 更新滑動窗口：用實際的特徵進行更新，將舊的數據丟棄
        new_entry = data[i + seq_length]  # 獲取新的時間步的所有特徵
        sliding_window = np.vstack((sliding_window[1:], new_entry))  # 保持窗口大小不變

# Convert actual_prices and predicted_prices to numpy arrays for scaling
actual_prices = np.array(actual_prices)
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

# Plot the balance over time
plt.figure(figsize=(10, 6))
plt.plot(balance_history, label='Portfolio Balance', color='green')
plt.title('Portfolio Balance Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Balance (USD)')
plt.legend()
plt.show()

# Output to csv
df_result = pd.DataFrame(data={"Actual": actual_prices_unscaled, "Predicted": predicted_prices_unscaled, "Balance": balance_history})
df_result.to_csv(config["save_path"] + config["model_name"] + "_result.csv", index=False)

# Print final balance
print(f"Initial Balance:  ${initial_balance:.2f}")
print(f"Final Balance:    ${balance:.2f}")
print(f"holding strategy: ${actual_prices_unscaled[-1] / actual_prices_unscaled[0]*1000:.2f}")
print("Backtesting complete.")
