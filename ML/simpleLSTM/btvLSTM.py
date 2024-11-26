import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load config
with open("C:\\gitproject\\tradebot\\ML\\simpleLSTM\\config.json", 'r') as f:
    config = json.load(f)

# 確認 CUDA 是否可用
file_path = config["file_path"]
model_save_path = config["model_save_path"]
seq_len = config["seq_len"]
hidden_layer_size = config["hidden_layer"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load price data from CSV
data = pd.read_csv(file_path)['close'].values  # Assuming 'close' column contains the price data

# 定義 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=hidden_layer_size, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size, device=device),
                            torch.zeros(1, 1, self.hidden_layer_size, device=device))
    
    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(1, input_seq.size(1), self.hidden_layer_size, device=device),
                            torch.zeros(1, input_seq.size(1), self.hidden_layer_size, device=device))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1].view(-1)  # Ensure the output size matches the target size

# 參數設置
learning_rate = config["learning_rate"]
epochs = config["epochs"]
train_window = config["seq_len"]

# 函數: 創建序列資料
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw].reshape(-1, 1)  # 改成 (seq_len, input_size)
        train_label = input_data[i+tw]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# 數據轉換為 PyTorch tensor
data_normalized = (data - np.mean(data)) / np.std(data)
data_normalized = torch.FloatTensor(data_normalized).to(device)
train_inout_seq = create_inout_sequences(data_normalized, train_window)

# 初始化模型和優化器
model = LSTMModel().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        # 初始化 hidden_cell 的形狀為 (num_layers, batch_size, hidden_size)
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                             torch.zeros(1, 1, model.hidden_layer_size, device=device))
        
        y_pred = model(seq)
        labels = labels.view(-1)  # Ensure labels have the same size as y_pred
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f'Epoch {i} loss: {single_loss.item()}, y_pred: {y_pred.item()}, labels: {labels.item()}')

# 測試模型 (用訓練資料進行預測)
model.eval()
predictions = []
for i in range(len(data_normalized) - train_window):
    seq = data_normalized[i:i + train_window]
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                             torch.zeros(1, 1, model.hidden_layer_size, device=device))
        predictions.append(model(seq).item())

# 顯示最終的 loss
print(f'Final loss: {single_loss.item()}')

# 繪製預測結果
plt.plot(data[train_window:], label="True Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title("Price Prediction using LSTM")
plt.show()