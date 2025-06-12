import pandas as pd
import os
import re
from glob import glob

# 路徑設置
base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, 'gooddata')
result_dir = os.path.join(base_dir, 'gooddata')
os.makedirs(result_dir, exist_ok=True)

# 搜尋所有 StockList 開頭的 CSV 檔
file_paths = sorted(glob(os.path.join(file_dir, 'StockList*.csv')))

# 儲存所有轉換後的資料（格式為：代號, 名稱, 年, 指標, 值）
long_data = []

for file_path in file_paths:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 嘗試去掉「排名」欄位
    if '排名' in df.columns:
        df = df.drop(columns=['排名'])

    id_name_cols = ['代號', '名稱']
    data_cols = [col for col in df.columns if col not in id_name_cols]
    
    # 根據欄位名稱抽出年與指標
    for col in data_cols:
        m = re.match(r"(\d{4})(?:年度)?(.+)", col)
        if m:
            year = int(m.group(1))
            metric = m.group(2).strip()
            for _, row in df.iterrows():
                stock_id = str(row['代號'])[2:-1].zfill(4) if isinstance(row['代號'], str) and row['代號'].startswith('="') else str(row['代號']).zfill(4)
                long_data.append({
                    '代號': stock_id,
                    '名稱': row['名稱'],
                    '年': year,
                    '指標': metric,
                    '值': row[col]
                })

# 長格式資料轉為寬格式（pivot）
long_df = pd.DataFrame(long_data)
long_df['值'] = (
    long_df['值']
    .astype(str)  # 確保是字串
    .str.replace(',', '', regex=False)  # 去除逗號
    .replace('', pd.NA)  # 空字串轉為 NA
    .astype(float)  # 轉 float
)

pivot_df = long_df.pivot_table(
    index=['代號', '名稱', '年'],
    columns='指標',
    values='值',
    aggfunc='first'  # 改成不平均，只取第一個值
).reset_index()

# 移除含有空值的列
pivot_df = pivot_df.dropna(how='any')

# 儲存為最終整理後的 CSV
output_path = os.path.join(result_dir, 'merged_all_metrics.csv')
pivot_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f'💖 完美合併完成～結果儲存在：{output_path} 喵♡')
