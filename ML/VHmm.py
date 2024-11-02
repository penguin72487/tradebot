import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取之前保存的結果 CSV 檔案
input_file_path = 'C:\\gitproject\\tradebot\\ML\\hmm_results.csv'
df = pd.read_csv(input_file_path)

# 檢查是否成功讀取
print(df.head())

# 定義一個顏色列表，用於不同隱藏狀態
colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'
]

# 創建圖形
plt.figure(figsize=(15, 8))

# 為每個隱藏狀態繪製散點圖
for state in range(df['Hidden_State'].nunique()):
    state_data = df[df['Hidden_State'] == state]
    plt.scatter(state_data['Date'], state_data['close'], color=colors[state], label=f'State {state}', alpha=0.6, s=10)

# 添加標題和軸標籤
plt.title('Hidden States vs Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)

# 顯示圖例
plt.legend()

# 顯示圖形
plt.tight_layout()
plt.show()
