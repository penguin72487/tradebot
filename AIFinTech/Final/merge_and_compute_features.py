import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 路徑設定 ===
base_dir = 'AIFinTech/Final/gooddata'
csv_path = os.path.join(base_dir, 'merged_all_metrics.csv')
price_dir = os.path.join(base_dir, 'price_csv')
price_output_path = os.path.join(base_dir, 'closing_price_map.csv')
merged_path = os.path.join(base_dir, 'merged_all_metrics_with_price.csv')
final_output_path = os.path.join(base_dir, 'financial_features_all.csv')

# === 定義處理每個股票 CSV 的函數 ===
def process_price_csv(stock_file):
    if not stock_file.startswith("Price_") or not stock_file.endswith(".csv"):
        return []

    sid = stock_file[6:10]
    price_path = os.path.join(price_dir, stock_file)
    try:
        df_price = pd.read_csv(price_path, parse_dates=['Date'])
        df_price['year'] = df_price['Date'].dt.year
        last_prices = df_price.sort_values('Date').groupby('year').last()

        stock_price_data = []
        for year, row in last_prices.iterrows():
            stock_price_data.append({
                '代號': sid,
                'year': year,
                'closing_price_year': row['Close'],
                'adj_closing_price_year': row['Adj Close']
            })
            # if year == 2025:
            #     print(f"⚠️ {sid} 2025 年的收盤{row['Close']}，可能是 YTD 資料")

        return stock_price_data
    except Exception as e:
        print(f"⚠️ 無法處理 {stock_file}：{e}")
        return []

# === 多執行緒處理所有檔案 ===
print("📦 開始讀取 price_csv 中的年底收盤價與調整收盤價（多執行緒）...")
all_files = sorted(os.listdir(price_dir))
price_data = []

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(process_price_csv, f): f for f in all_files}
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        price_data.extend(result)

# === 匯出資料（選擇性）===
df_price_map = sorted(price_data, key=lambda x: (x['代號'], x['year']))
df_price_map = pd.DataFrame(price_data)
df_price_map.to_csv(price_output_path, index=False)
print("✅ 處理完成並存入：", price_output_path)

# # 儲存收盤價對照表
# df_price_map = pd.DataFrame(price_data)
# df_price_map.to_csv(price_output_path, index=False, encoding='utf-8-sig')
# print(f"✅ 收盤價對照表儲存到：{price_output_path} 喵♡")

# === 合併財報資料與收盤價 ===
df = pd.read_csv(csv_path)
df_price_map = pd.read_csv(price_output_path)
# Step 1：讀入原始 df
df = pd.read_csv(csv_path)
df['year'] = df['年'].astype(int)
df['代號'] = df['代號'].astype(str).str.zfill(4)

# Step 2：找出所有代號，建立補充的 2025 資料
stock_ids = df['代號'].unique()
columns_to_fill = df.columns.drop(['代號', '年', 'year'])  # 除了這些其他都補 0

# 建立補資料的 DataFrame
df_2025 = pd.DataFrame({
    '代號': np.repeat(stock_ids, 1),
    '年': 2025,
    'year': 2025
})

for col in columns_to_fill:
    df_2025[col] = 0  # 補值填 0

# Step 3：合併進原本的 df
df = pd.concat([df, df_2025], ignore_index=True)
df = df.sort_values(by=['代號', 'year']).reset_index(drop=True)

print(f"✅ 每個股票都補上了 2025 年空白財報資料，共新增 {len(df_2025)} 筆喵♡")


df_price_map['代號'] = df_price_map['代號'].astype(str).str.zfill(4)
df_merged = pd.merge(df, df_price_map, how='left', on=['代號', 'year'])


# 移除沒有收盤價的資料
df_merged = df_merged.dropna(subset=['closing_price_year'])



# 儲存中間結果
df_merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
print(f"✅ 合併後的資料儲存到：{merged_path} 喵♡")

# === 計算財務特徵 ===
df = df_merged.copy()
df['market_cap_mil'] = df['closing_price_year'] * df['發行量(萬張)'] * 10000000 / 1000000
df['pb_ratio'] = df['closing_price_year'] / (df['股東權益總額(億)'] * 100 / (df['發行量(萬張)']*10000000))
df['ps_ratio'] = df['market_cap_mil'] / (df['營收(億)'] * 100) 
df['roe_m'] = df['淨利率(%)'] / 100 * df['營收(億)'] / df['股東權益總額(億)']
df['opm'] = df['營益(億)'] / df['營收(億)']
df['npm'] = df['淨利率(%)'] / 100
df['roa'] = df['ROA(%)'] / 100
df['debt_to_equity'] = df['負債總額(億)'] / df['股東權益總額(億)']
df['current_ratio_m'] = df['流動資產對流動負債(%)'] / 100
df['quick_ratio_m'] = df['速動資產對流動負債(%)'] / 100
df['inventory_turnover_m'] = df['存貨週轉率']
df['ar_turnover_m'] = df['應收帳款週轉率']
df['year'] = df['year'].astype(int)  # 保證是數字年份
df = df.sort_values(by=['代號', 'year'])  # 排序保證正確順序

df['op_growth_m'] = df.groupby('代號')['營益(億)'].pct_change()
df['net_income_growth_m'] = df.groupby('代號')['淨利率(%)'].pct_change()
df['future_adj_price'] = df.groupby('代號')['adj_closing_price_year'].shift(-1)
df['return'] = (df['future_adj_price'] / df['adj_closing_price_year']) - 1
df['return'] *= 100  # 百分比
df['return_label'] = (df['return'] > 0).astype(int) * 2 - 1  # 轉換為標籤 
# 1 表示上漲，-1 表示下跌或持平
# df.drop(columns=['future_adj_price'], inplace=True)


# === 最後只保留指定英文欄位並輸出 ===
df['stock_id'] = df['代號']
df['year_month'] = df['year'].astype(str) + '12'  # 預設為每年年底

final_cols = [
    'stock_id', 'year_month', 'market_cap_mil', 'closing_price_year',
    'pb_ratio', 'ps_ratio', 'roe_m', 'roa', 'opm', 'npm',
    'debt_to_equity', 'current_ratio_m', 'quick_ratio_m',
    'inventory_turnover_m', 'ar_turnover_m',
    'op_growth_m', 'net_income_growth_m', 'return', 'return_label'
]

df_final = df[final_cols].copy()

# 💥 移除 NaN, inf, -inf 的 row
df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()


# 儲存到 base_dir 的上一層
parent_dir = os.path.dirname(base_dir)
df_final.to_csv(os.path.join(parent_dir, 'final_features.csv'), index=False, encoding='utf-8-sig')
print("🌟 已成功儲存精簡版英文特徵表：final_features.csv 喵♡")

# ## 輸出統計結果
# print("\n📊 特徵統計資訊：")
# print(df_final.describe().T)
# 每年的股票數量
yearly_counts = df_final.groupby('year_month')['stock_id'].nunique()
print("\n📅 每年股票數量統計：")
print(yearly_counts
            .to_frame(name='stock_count'
                      ).reset_index()
                        .sort_values(by='year_month'))