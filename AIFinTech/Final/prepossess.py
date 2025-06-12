import pandas as pd
import numpy as np
import yfinance as yf
import os
import re

# 路徑設置
base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, 'gooddata')
result_dir = os.path.join(base_dir, 'gooddata')
os.makedirs(result_dir, exist_ok=True)

# 讀資料
raw = pd.read_csv(os.path.join(file_dir, 'StockList.csv'))

# 只保留會用到的欄位（根據你貼圖）
base_cols = ['代號', '名稱']
data_cols = [col for col in raw.columns if col not in base_cols]

# 正則抓欄位：「2024年度ROE(%)」→ 年度, 項目
# 或是 "2024發行量(萬張)


long_data = []
for col in data_cols:
    m = re.match(r"(\d{4})(?:年度|年)?(.+)", col)  # 同時支援「年度」「年」或沒有的情況
    if m:
        year = int(m.group(1))
        for idx, val in enumerate(raw[col]):
            row = {
                '股票代號': str(raw.loc[idx, '代號']).zfill(4),
                '公司名稱': raw.loc[idx, '名稱'],
                '年度': year,
            }
            long_data.append(row)

df = pd.DataFrame(long_data)
df = df.pivot_table(index=['股票代號', '公司名稱', '年度'], 
                    aggfunc='first').reset_index()

# 欄位轉換（億 → 百萬、萬張 → 股本）
def to_float(x):
    try:
        return float(str(x).replace(',', ''))
    except:
        return np.nan



df['營收(百萬)'] = df['營收(億)'].apply(to_float) * 100
df['營業利益(百萬)'] = df['營益(億)'].apply(to_float) * 100
df['股東權益(百萬)'] = df['股東權益總額(億)'].apply(to_float) * 100
df['負債總額(百萬)'] = df['負債總額(億)'].apply(to_float) * 100
df['流動資產(百萬)'] = df['流動資產對流動負債(%)'].apply(to_float) / 100  # 假設你有流動負債
df['速動比率'] = df['速動資產對流動負債(%)'].apply(to_float) / 100
df['應收帳款週轉率'] = df['應收帳款週轉率'].apply(to_float)
df['存貨週轉率'] = df['存貨週轉率'].apply(to_float)
df['稅後淨利率'] = df['淨利率(%)'].apply(to_float) / 100
df['稅後淨利(百萬)'] = df['營收(百萬)'] * df['稅後淨利率']
df['OPM(%)'] = df['營業利益(百萬)'] / df['營收(百萬)']
df['營業利益(百萬)'] = df['營業利益(百萬)'].fillna(df['營收(百萬)'] * df['OPM(%)'].apply(to_float) / 100)

# 發行股數（萬張 → 百萬）×10000
df['股本(百萬)'] = df['發行量(萬張)']  * 10
df['發行股數'] = df['股本(百萬)'] * 100_000

df.to_csv(os.path.join(result_dir, 'cleaned_data.csv'), index=False)
# 將清理後的資料讀入

# Yahoo Finance 抓年度收盤價
def fetch_close(stock_id, year):
    try:
        ticker = yf.Ticker(f"{stock_id}.TW")
        hist = ticker.history(start=f"{year}-01-01", end=f"{year+1}-01-10")
        return hist['Close'].resample("YE").last().values[0]
    except Exception as e:
        print(f"❌ {stock_id}.TW 抓不到收盤價：{e}")
        return np.nan

# 移除股票代號和年度的第一行

df['收盤價'] = df.apply(lambda row: fetch_close(row['股票代號'], row['年度']), axis=1)
df = df.dropna(subset=['收盤價'])

# 計算指標
df['EPS'] = df['稅後淨利(百萬)'] * 1e6 / df['發行股數']
df['PER'] = df['收盤價'] / df['EPS']
df['BVPS'] = df['股東權益(百萬)'] * 1e6 / df['發行股數']
df['PBR'] = df['收盤價'] / df['BVPS']
df['PS'] = df['收盤價'] / (df['營收(百萬)'] * 1e6 / df['發行股數'])
df['市值(百萬元)'] = df['收盤價'] * df['發行股數'] / 1e6
df['ROE'] = df['稅後淨利(百萬)'] / df['股東權益(百萬)']
df['ROA'] = df['稅後淨利(百萬)'] / (df['股東權益(百萬)'] + df['負債總額(百萬)'])
df['OPM'] = df['營業利益(百萬)'] / df['營收(百萬)']
df['NPM'] = df['稅後淨利(百萬)'] / df['營收(百萬)']
df['Debt_to_Equity'] = df['負債總額(百萬)'] / df['股東權益(百萬)']
df['Current_Ratio'] = df['流動資產(百萬)']  # 流動負債不明，暫略
df['Quick_Ratio'] = df['速動比率']
df['Inventory_Turnover'] = df['存貨週轉率']
df['AR_Turnover'] = df['應收帳款週轉率']

# 成長率（稅後淨利、營業利益）
df = df.sort_values(['股票代號', '年度'])
df['營業利益成長率'] = df.groupby('股票代號')['營業利益(百萬)'].pct_change()
df['稅後淨利成長率'] = df.groupby('股票代號')['稅後淨利(百萬)'].pct_change()

# 輸出
df.to_csv(os.path.join(result_dir, '完整指標計算.csv'), index=False)
print("✅ 已輸出『完整指標計算.csv』，每年每股一行，所有指標都算好了喵♡")
df = df.dropna()  # 最後一層清潔，所有欄位都不能有 NaN
