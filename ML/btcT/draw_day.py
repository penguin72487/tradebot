import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 獲取當前檔案的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 讀取 CSV 檔案
csv_file_path = os.path.join(current_dir, 'spxTmore_result.csv')
# csv_file_path = os.path.join(current_dir, 'taiexTmore_result.csv')
df = pd.read_csv(csv_file_path)

# 計算評估指標
mae = mean_absolute_error(df['Actual'], df['Predicted'])
rmse = np.sqrt(mean_squared_error(df['Actual'], df['Predicted']))
r2 = r2_score(df['Actual'], df['Predicted'])

# 打印評估指標
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Square Error (RMSE): {rmse}')
print(f'R-squared: {r2}')

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(df['Actual'], label='Actual Price')
plt.plot(df['Predicted'], label='Predicted Price')
plt.xlabel('Time (Days)')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()