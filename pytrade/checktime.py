import pandas as pd
import os

# 打印當前工作目錄
print("當前工作目錄:", os.getcwd())

# 嘗試加載CSV文件
try:
    df = pd.read_csv('BINANCE_BTCUSDT, 1.csv')
    print(df.head())
    # 假設你已經知道日期列的名稱
    # print(df['日期列名稱'].min(), df['日期列名稱'].max())
except FileNotFoundError:
    print("無法找到文件。請檢查文件路徑是否正確。")
