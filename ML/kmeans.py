import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# 讀取 CSV 檔案
file_path = 'C:\\gitproject\\tradebot\\ML\\BINANCE_BTCUSDT, 15.csv'
df = pd.read_csv(file_path)

# 選取所需欄位：time, open, high, low, close
features = df[['time', 'open', 'high', 'low', 'close']]

# 用平均值填補缺失值
features_filled = features.fillna(features.mean())

# 初始化 KMeans 分群，設定 16 個類別
kmeans = KMeans(n_clusters=16, random_state=42)

# 執行 KMeans 分群
kmeans.fit(features_filled)

# 將分群結果加回 DataFrame
df['cluster'] = kmeans.labels_

# 儲存結果到新的 CSV 檔案
output_file = 'kmeans_clustering_results.csv'
df.to_csv(output_file, index=False)

# 取得分群中心點
centers = kmeans.cluster_centers_

# 輸出中心點
print("16個分群的中心點:")
for i, center in enumerate(centers):
    print(f"Cluster {i}: {center}")

# 若你想將中心點保存到 CSV 檔案
centers_df = pd.DataFrame(centers, columns=['time', 'open', 'high', 'low', 'close'])
centers_output_file = 'kmeans_centers.csv'
centers_df.to_csv(centers_output_file, index=False)

print(f"分群中心點已儲存到 {centers_output_file}")
