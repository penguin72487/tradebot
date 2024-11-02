import numpy as np
import pandas as pd
import torch

# 使用 GPU 加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載你的加密貨幣價格資料，請替換成實際的文件路徑
file_path = 'C:\\gitproject\\tradebot\\ML\\BINANCE_BTCUSDT, 15.csv'
df = pd.read_csv(file_path)

# 檢查欄位名稱
print(df.columns)

# 確保 time 欄位是 UNIX 時間戳並轉換為日期格式
df['Date'] = pd.to_datetime(df['time'], unit='s')

# 計算 'close' 價格的變化率
df['Price_Change'] = df['close'].diff().fillna(0)

# 檢查 Price_Change 列的內容
print("Price_Change head:")
print(df['Price_Change'].head())

# 將價格變化率進行離散化，分成若干區間（這裡我們分成 3 個區間，可以根據需要調整）
n_observations = 3
try:
    df['Discrete_Price_Change'] = pd.qcut(df['Price_Change'], n_observations, labels=False)
except ValueError as e:
    print(f"Error in qcut: {e}")
    df['Discrete_Price_Change'] = pd.cut(df['Price_Change'], bins=n_observations, labels=False)

# 檢查離散化後的結果
print("Discrete_Price_Change head:")
print(df['Discrete_Price_Change'].head())

# 使用離散後的價格變化值作為觀測序列
observations = torch.tensor(df['Discrete_Price_Change'].dropna().values, dtype=torch.long).to(device)

# 假設我們有 8 個隱藏狀態（可以根據需求調整）
n_hidden_states = 8

# 隨機生成狀態轉移矩陣 (Transition Matrix) 和發射矩陣 (Emission Matrix)
transition_matrix = torch.randn(n_hidden_states, n_hidden_states, device=device).softmax(dim=-1)
emission_matrix = torch.randn(n_hidden_states, n_observations, device=device).softmax(dim=-1)

# 初始狀態分佈 (Initial state distribution)
initial_state_prob = torch.randn(n_hidden_states, device=device).softmax(dim=-1)

# 進行前向算法計算 (Forward algorithm)
def forward_algorithm(transition_matrix, emission_matrix, initial_state_prob, observations):
    n_steps = len(observations)
    n_states = transition_matrix.shape[0]

    # 初始化 alpha，表示 t=0 時各個狀態的前向概率
    alpha = initial_state_prob * emission_matrix[:, observations[0]]

    # 遞迴更新 alpha
    for t in range(1, n_steps):
        alpha = torch.matmul(alpha, transition_matrix) * emission_matrix[:, observations[t]]

    return torch.sum(alpha)

# 執行前向算法
probability = forward_algorithm(transition_matrix, emission_matrix, initial_state_prob, observations)
print(f"Total probability of the observation sequence: {probability.item()}")

# 將結果寫入到檔案中
output_file = 'output.txt'
with open(output_file, 'w') as file:
    file.write(f"Total probability of the observation sequence: {probability.item()}\n")

print(f"Results saved to {output_file}")
