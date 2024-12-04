import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 獲取當前檔案的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

data_Set = ['spxTmore_result.csv', 'btcTmore_result.csv', 'taiexTmore_result.csv']
for data in data_Set:
    # 讀取預測結果
    csv_file_path = os.path.join(current_dir, data)
    df = pd.read_csv(csv_file_path)

    # 分割資料集為訓練集和測試集
    split_index = int(len(df) * 0.8)
    train_df = df[:split_index]
    test_df = df[split_index:]

    # 計算訓練集指標
    train_mae = mean_absolute_error(train_df['Actual'], train_df['Predicted'])
    train_rmse = np.sqrt(mean_squared_error(train_df['Actual'], train_df['Predicted']))
    train_r2 = r2_score(train_df['Actual'], train_df['Predicted'])

    # 計算測試集指標
    test_mae = mean_absolute_error(test_df['Actual'], test_df['Predicted'])
    test_rmse = np.sqrt(mean_squared_error(test_df['Actual'], test_df['Predicted']))
    test_r2 = r2_score(test_df['Actual'], test_df['Predicted'])

    # 計算測試集的 Predicted / Actual 比例
    test_df.loc[:, 'Predicted/Actual'] = (test_df['Predicted'] / test_df['Actual'] - 1.0) * 100

    # 繪製 Actual vs Predicted 圖表
    plt.figure(figsize=(10, 6))
    plt.plot(df['Actual'], label='Actual Price', color='blue')
    plt.plot(df['Predicted'], label='Predicted Price', color='orange')
    plt.axvline(x=split_index, color='blue', linestyle='--', label='80% Point')

    # 顯示訓練集和測試集指標
    plt.text(0.05, 0.95, f'Train MAE: {train_mae:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.90, f'Train RMSE: {train_rmse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.85, f'Train R²: {train_r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.80, f'Test MAE: {test_mae:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.75, f'Test RMSE: {test_rmse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.70, f'Test R²: {test_r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    if data == 'btcTmore_result.csv':
        plt.xlabel('Time (4 hours)')
    else:
        plt.xlabel('Time (Days)')
    plt.ylabel('Price')
    plt.title(f'Actual vs Predicted Price ({data})')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # 獨立繪製 Predicted / Actual 比例圖
    plt.figure(figsize=(10, 6))
    plt.plot(test_df['Predicted/Actual'], label='Predicted / Actual Ratio', color='green', linestyle='--')
    plt.axhline(y=0, color='red', linestyle='-', label='Ratio = 0')  # 顯示基準線
    plt.xlabel('Time')
    plt.ylabel('Predicted / Actual Ratio')
    plt.title(f'Predicted / Actual Ratio ({data})')
    plt.legend()
    plt.grid(True)
    plt.show()
