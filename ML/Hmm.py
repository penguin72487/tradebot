import numpy as np
import pandas as pd
from hmmlearn import hmm
import torch

# 加載你的加密貨幣價格資料，請替換成實際的文件路徑
file_path = 'C:\\gitproject\\tradebot\\ML\\BINANCE_BTCUSDT, 15.csv'
df = pd.read_csv(file_path)

# 檢查欄位名稱
print(df.columns)

# 確保 time 欄位是 UNIX 時間戳並轉換為日期格式
df['Date'] = pd.to_datetime(df['time'], unit='s')

# 確保 'close' 欄位存在，並計算價格變化率
df['Price_Change'] = df['close'].diff().fillna(0)

# 只使用 Price_Change 作為模型的輸入數據
X = df['Price_Change'].values.reshape(-1, 1)

# 使用 PyTorch 將數據移動到 GPU 進行加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_torch = torch.tensor(X, dtype=torch.float32).to(device)

# 將數據返回到 CPU 用於 hmmlearn 訓練
X_cpu = X_torch.cpu().numpy()

# 構建隱馬可夫模型
model = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=1000)

# 訓練模型
model.fit(X_cpu)

# 使用模型來預測隱藏狀態
hidden_states = model.predict(X_cpu)

# 將隱藏狀態添加到 DataFrame
df['Hidden_State'] = hidden_states

# 輸出隱藏狀態與價格變化的對應表
print(df[['Date', 'close', 'Price_Change', 'Hidden_State']])

# 保存結果為 CSV 檔案
output_file_path = 'C:\\gitproject\\tradebot\\ML\\hmm_results.csv'
df.to_csv(output_file_path, index=False)

# 查看每個隱藏狀態的均值與方差
for i in range(model.n_components):
    print(f"State {i}: Mean = {model.means_[i][0]}, Variance = {np.diag(model.covars_[i])[0]}")
