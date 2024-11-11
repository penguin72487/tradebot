import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Avoiding OpenMP runtime error by setting environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Parameters to adjust easily
PARAMS = {
    'hidden_size': 50,
    'num_layers': 4,
    'learning_rate': 0.001,
    'epochs': 100,
    'sequence_length': 100,
    'simulation_days': 100
}

# 1. 定義下載市場數據的函數
def get_market_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    # 選擇需要的欄位，如收盤價和交易量
    data = data[['Close', 'Volume']]
    return data

# 2. 標記市場情境
def label_market_conditions(data):
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['Condition'] = 'Neutral'

    # 設定牛市與熊市的條件
    data.loc[data['Close'] > data['SMA_30'] * 1.02, 'Condition'] = 'Bull'  # 當價格高於30日均線2%為牛市
    data.loc[data['Close'] < data['SMA_30'] * 0.98, 'Condition'] = 'Bear'  # 低於30日均線2%為熊市

    return data[['Close', 'Volume', 'Condition']]

# 3. 準備數據，用於LSTM模型訓練
def prepare_data(data):
    X, y = [], []
    sequence_length = PARAMS['sequence_length']
    for i in range(sequence_length, len(data)):  # 以 sequence_length 天為一組數據
        X.append(data[i-sequence_length:i, 0])  # Close價作為輸入
        y.append(data[i, 0])  # 預測的目標
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

# 4. 建立LSTM模型 (PyTorch)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(PARAMS['num_layers'], x.size(0), PARAMS['hidden_size']).to(x.device)  # 初始 hidden state
        c_0 = torch.zeros(PARAMS['num_layers'], x.size(0), PARAMS['hidden_size']).to(x.device)  # 初始 cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# 5. 模擬不同市場情境
def simulate_market_scenario(model, data, scenario='Bull', device='cuda'):
    if scenario == 'Bull':
        initial_conditions = data[data['Condition'] == 'Bull'].iloc[-PARAMS['sequence_length']:]['Close'].values
    elif scenario == 'Bear':
        initial_conditions = data[data['Condition'] == 'Bear'].iloc[-PARAMS['sequence_length']:]['Close'].values
    else:
        initial_conditions = data[data['Condition'] == 'Neutral'].iloc[-PARAMS['sequence_length']:]['Close'].values

    initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32).to(device).view(1, -1, 1)
    simulated_data = list(initial_conditions.cpu().numpy().flatten())

    model.eval()
    with torch.no_grad():
        for _ in range(PARAMS['simulation_days']):  # 模擬 simulation_days 天
            pred = model(initial_conditions)
            pred_value = pred.item()  # 獲取標量值
            simulated_data.append(pred_value)
            initial_conditions = torch.cat((initial_conditions[:, 1:, :], torch.tensor(pred_value, dtype=torch.float32).to(device).view(1, 1, 1)), dim=1)

    return np.array(simulated_data)

# 6. 執行流程：下載數據、標記情境、訓練模型並模擬市場情境
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 下載市場數據並標記情境
    market_data = get_market_data('^GSPC', '2000-01-01', '2024-11-01')
    labeled_data = label_market_conditions(market_data)
    print(labeled_data['Condition'].value_counts())

    # 準備數據並訓練LSTM模型
    close_prices = labeled_data[['Close']].values
    X_train, y_train = prepare_data(close_prices)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)

    model = LSTMModel(input_size=1, hidden_size=PARAMS['hidden_size'], output_size=1, num_layers=PARAMS['num_layers']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'])

    # 開始訓練
    model.train()
    for epoch in range(PARAMS['epochs']):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{PARAMS['epochs']}], Loss: {loss.item():.4f}')

    # 模擬不同市場情境
    bull_market = simulate_market_scenario(model, labeled_data, scenario='Bull', device=device)
    bear_market = simulate_market_scenario(model, labeled_data, scenario='Bear', device=device)
    neutral_market = simulate_market_scenario(model, labeled_data, scenario='Neutral', device=device)

    # 繪製結果
    plt.figure(figsize=(12, 8))
    plt.plot(bull_market, label='Bull Market Simulation', color='green')
    plt.plot(bear_market, label='Bear Market Simulation', color='red')
    plt.plot(neutral_market, label='Neutral Market Simulation', color='blue')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Simulated Close Price')
    plt.title('Market Scenario Simulation with LSTM (PyTorch)')
    plt.show()

# 7. 執行主函數
if __name__ == "__main__":
    main()