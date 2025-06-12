import pandas as pd
import os
from tqdm import tqdm

# === 路徑設定 ===
base_dir = 'AIFinTech/Final/gooddata'
csv_path = os.path.join(base_dir, 'merged_all_metrics.csv')
price_dir = os.path.join(base_dir, 'price_csv')
output_path = os.path.join(base_dir, 'financial_features_all.csv')

# === 讀入財報資料 ===
df = pd.read_csv(csv_path)
df['year'] = df['年'].astype(int)
df['代號'] = df['代號'].astype(str).str.zfill(4)

# === 建立 closing_price_year ===
price_map = {}

print("📦 開始讀取 price_csv 中的年底收盤價...")
for stock_id in tqdm(df['代號'].unique()):
    price_path = os.path.join(price_dir, f"Price_{stock_id}.csv")
    if not os.path.exists(price_path):
        continue
    try:
        price_df = pd.read_csv(price_path, parse_dates=['Date'])
        price_df['year'] = price_df['Date'].dt.year
        year_groups = price_df.groupby('year')
        for y, g in year_groups:
            if not g.empty:
                last_close = g.sort_values(by='Date').iloc[-1]['Close']
                price_map[f"{stock_id}_{y}"] = last_close
    except Exception as e:
        print(f"⚠️ 無法處理 {stock_id}：{e}")

# === 加入收盤價欄位 ===
df['closing_price_year'] = df.apply(
    lambda x: price_map.get(f"{x['代號']}_{x['year']}"), axis=1
)

# 💡 移除沒有收盤價的行
# df = df[df['closing_price_year'].notna()]

# === 計算特徵 ===
df['market_cap_mil'] = df['closing_price_year'] * df['發行量(萬張)'] * 1000
df['pb_ratio'] = df['closing_price_year'] / (df['股東權益總額(億)'] * 100 / df['發行量(萬張)'])
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
df['future_price'] = df.groupby('代號')['closing_price_year'].shift(-1)
df['return'] = (df['future_price'] / df['closing_price_year']) - 1
df['return_label'] = (df['return'] > 0).astype(int)
df.drop(columns=['future_price'], inplace=True)

# === 輸出結果 ===
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n🎉 收盤價與財務特徵整合完成：{output_path}，可以開始做模型了喵♡")
