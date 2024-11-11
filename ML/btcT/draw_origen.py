import pandas as pd
import matplotlib.pyplot as plt
import os

# 獲取當前檔案的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
data_Set = ['spxTmore_result.csv', 'btcTmore_result.csv', 'taiexTmore_result.csv']
# 讀取 CSV 檔案

for data in data_Set:
    csv_file_path = os.path.join(current_dir, data)
    df = pd.read_csv(csv_file_path)

    # 繪製圖表
    plt.figure(figsize=(10, 6))
    plt.plot(df['Actual'], label='Actual Price')

    if data == 'btcTmore_result.csv':
        plt.xlabel('Time (4 hours)')
    else:
        plt.xlabel('Time (Days)')
    plt.ylabel('Price')
    plt.title(f'Price ({data.replace("Tmore_result.csv", "").upper()})')
    plt.legend()
    plt.grid(True)
    plt.show()

