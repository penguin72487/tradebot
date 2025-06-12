import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# === 路徑設定 ===
base_dir = 'AIFinTech/Final/gooddata'
csv_path = os.path.join(base_dir, 'merged_all_metrics.csv')
price_dir = os.path.join(base_dir, 'price_csv')
price_output_path = os.path.join(base_dir, 'closing_price_map.csv')
merged_path = os.path.join(base_dir, 'merged_all_metrics_with_price.csv')
final_output_path = os.path.join(base_dir, 'financial_features_all.csv')

# === 建立 closing_price_year 對照表 ===
# === 建立 closing_price_year 對照表 ===
price_data = []
print("📦 開始讀取 price_csv 中的年底收盤價與調整收盤價...")
for stock_file in tqdm(sorted(os.listdir(price_dir))):
    if not stock_file.startswith("Price_") or not stock_file.endswith(".csv"):
        continue
    sid = stock_file[6:10]
    price_path = os.path.join(price_dir, stock_file)
    try:
        df_price = pd.read_csv(price_path, parse_dates=['Date'])
        df_price['year'] = df_price['Date'].dt.year
        last_prices = df_price.sort_values('Date').groupby('year').last()

        for year, row in last_prices.iterrows():
            price_data.append({
                '代號': sid,
                'year': year,
                'closing_price_year': row['Close'],          # 原始價格
                'adj_closing_price_year': row['Adj Close']  # 調整後價格（拿來算報酬）
            })
    except Exception as e:
        print(f"⚠️ 無法處理 {stock_file}：{e}")


# 儲存收盤價對照表
df_price_map = pd.DataFrame(price_data)
df_price_map.to_csv(price_output_path, index=False, encoding='utf-8-sig')
print(f"✅ 收盤價對照表儲存到：{price_output_path} 喵♡")

# === 合併財報資料與收盤價 ===
df = pd.read_csv(csv_path)
df_price_map = pd.read_csv(price_output_path)

df['year'] = df['年'].astype(int)
df['代號'] = df['代號'].astype(str).str.zfill(4)
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
df = df.sort_values(by=['代號', 'year'])
df['op_growth_m'] = df.groupby('代號')['營益(億)'].pct_change()
df['net_income_growth_m'] = df.groupby('代號')['淨利率(%)'].pct_change()
df['future_adj_price'] = df.groupby('代號')['adj_closing_price_year'].shift(-1)
df['return'] = (df['future_adj_price'] / df['adj_closing_price_year']) - 1
df['return'] *= 100  # 百分比
df['return_label'] = (df['return'] > 0).astype(int)
df.drop(columns=['future_adj_price'], inplace=True)


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


df_final.to_csv(os.path.join(base_dir, 'final_features.csv'), index=False, encoding='utf-8-sig')
print("🌟 已成功儲存精簡版英文特徵表：final_features.csv 喵♡")