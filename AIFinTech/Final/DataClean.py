import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, 'Each_CSV')
result_dir = os.path.join(base_dir, 'cleaned_data')
os.makedirs(result_dir, exist_ok=True)

# 匯入你的 CSV 財報（header=None 因為你資料沒標題）
df = pd.read_csv(os.path.join(file_dir, "financial_1101.csv"), header=None)

# 年度欄位對應（注意原始欄位是 int，不是 str）
year_map = {
    0: 'Name',
    1: 2017, 2: 2018, 3: 2019,
    4: 2014, 5: 2015, 6: 2016,
    7: 2011, 8: 2012, 9: 2013,
}
df.columns = [year_map.get(i, i) for i in df.columns]

# 將指標名稱當作 index，並轉置（行列對調）
df = df.set_index('Name').T
df.index = df.index.astype(int)
df = df.sort_index()

# 把數值字串轉成浮點數
df = df.replace("-", None).replace(",", "", regex=True).astype(float)

# --- 加這段處理稅後淨利 ---
if '稅後淨利' not in df.columns:
    if '稅前淨利（淨損）' in df.columns and '所得稅費用（利益）' in df.columns:
        df['稅後淨利'] = df['稅前淨利（淨損）'] - df['所得稅費用（利益）']
        print("✅ 已成功建立『稅後淨利』欄位喵～")
    else:
        print("⚠️ 找不到計算稅後淨利所需的欄位喵！")

# 抓 TSMC 每年最後一個交易日收盤價
ticker = yf.Ticker("2330.TW")
time.sleep(5)  # 小小休息一下，讓Yahoo冷靜點喵～
price_df = ticker.history(start="2011-01-01", end="2023-01-01")
yearly_close = price_df.groupby(price_df.index.year)['Close'].last()
df['Closing_Price'] = df.index.map(yearly_close)

# 計算市值（假設股本單位是億）
try:
    capital = df['股本']  # 若單位是億元，則 *100
    df['Market_Cap_Mil'] = df['Closing_Price'] * capital * 1000 / 1e6
except:
    df['Market_Cap_Mil'] = None


# 欄位名稱對應（根據實際資料調整）
# df['



# 財務比率計算（部分欄位請依實際名稱檢查）
df['ROE'] = df['稅後淨利'] / df['股東權益合計']
df['ROA'] = df['稅後淨利'] / df['資產總計']
df['OPM'] = df['營業利益'] / df['營業收入']
df['NPM'] = df['稅後淨利'] / df['營業收入']
df['Debt_to_Equity'] = df['負債總計'] / df['股東權益合計']
df['Current_Ratio'] = df['流動資產'] / df['流動負債']
df['Quick_Ratio'] = (df['流動資產'] - df['存貨']) / df['流動負債']
df['Inventory_Turnover'] = df['營業成本'] / df['存貨']
df['AR_Turnover'] = df['營業收入'] / df['應收帳款']
df['PB_Ratio'] = df['Closing_Price'] / (df['股東權益合計'] / capital / 1000)
df['PS_Ratio'] = df['Closing_Price'] / (df['營業收入'] / capital / 1000)

# 成長率
df['OPM_Growth'] = df['營業利益'].pct_change()
df['NetIncome_Growth'] = df['稅後淨利'].pct_change()

# Unknown 欄位補留空
df['Unknown_Param'] = None

# 重新整理欄位順序與英文命名
output = df[[
    'Market_Cap_Mil', 'Closing_Price', 'Unknown_Param', 'PB_Ratio', 'PS_Ratio',
    'ROE', 'ROA', 'OPM', 'NPM', 'Debt_to_Equity',
    'Current_Ratio', 'Quick_Ratio', 'Inventory_Turnover', 'AR_Turnover',
    'OPM_Growth', 'NetIncome_Growth'
]]

# 輸出結果
print(output)
output.to_csv(os.path.join(result_dir, "financial_indicators_cleaned.csv"))
