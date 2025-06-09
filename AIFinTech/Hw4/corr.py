import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# 設定資料夾路徑
folder_path = "FinTechML/Hw4/correlationDataset"
file_names = [
    "correlationCoefficientExample1.xlsx",
    "correlationCoefficientExample2.xlsx",
    "correlationCoefficientExample3.xlsx",
    "correlationCoefficientExample4.xlsx",
    "correlationCoefficientExample5.xlsx"
]

# 確保輸出資料夾存在
output_folder = os.path.join(folder_path, "output")
os.makedirs(output_folder, exist_ok=True)

# 開始處理每個檔案
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_excel(file_path)

    # 轉存為CSV
    csv_name = file_name.replace(".xlsx", ".csv")
    csv_path = os.path.join(output_folder, csv_name)
    df.to_csv(csv_path, index=False)

    # 畫圖
    x = df.columns[0]
    y = df.columns[1]

    # 計算相關係數
    corr = df[x].corr(df[y])

    # 線性回歸
    X = df[[x]]
    Y = df[y]
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, data=df)
    plt.plot(df[x], Y_pred, color='red', linewidth=2, label=f'Linear Fit')
    plt.title(f'{file_name} | Correlation: {corr:.2f}')
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)

    plot_name = file_name.replace(".xlsx", "_plot.png")
    plot_path = os.path.join(output_folder, plot_name)
    plt.savefig(plot_path)
    plt.close()

    print(f"{file_name} 轉檔與畫圖完成，相關係數為 {corr:.2f} 喵♡")

print("全部檔案處理完成囉，記得去output資料夾看圖圖喔～喵～")
