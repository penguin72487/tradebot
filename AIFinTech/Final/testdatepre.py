import pandas as pd
import numpy as np
import os
# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_test_cleaned_utf8.csv')
base_dir = os.path.dirname(file_path)

df = pd.read_csv(file_path)
# df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()

df = df.replace([np.inf, -np.inf], pd.NA).dropna()
# 儲存
output_path = os.path.join(base_dir, 'top200_test_cleaned_utf8_final.csv')
print(f"✅ 清理後的資料儲存到：{output_path} 喵♡")
df.to_csv(output_path, index=False, encoding='utf-8-sig')