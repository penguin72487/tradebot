import pandas as pd
import yfinance as yf
import os
import time
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(base_dir, 'Each_CSV')
result_dir = os.path.join(base_dir, 'cleaned_data')
os.makedirs(result_dir, exist_ok=True)


def safe_divide(numerator, denominator):
    return numerator.divide(denominator).replace([float('inf'), -float('inf')], None)





all_results = []

for filename in os.listdir(file_dir):
    if not filename.endswith('.csv'):
        continue

    stock_id = filename.replace("financial_", "").replace(".csv", "")
    print(f"🐱 處理中：{stock_id}...")

    try:
        df = pd.read_csv(os.path.join(file_dir, filename), header=None)

        # 對應年度欄位
        year_map = {
            0: 'Name',
            1: 2017, 2: 2018, 3: 2019,
            4: 2014, 5: 2015, 6: 2016,
            7: 2011, 8: 2012, 9: 2013,
        }
        df.columns = [year_map.get(i, i) for i in df.columns]
        df = df.set_index('Name').T
        df.index = df.index.astype(int)
        df = df.sort_index()
        df = df.replace("-", None).replace(",", "", regex=True).astype(float)

        # 補「稅後淨利」
        if '稅後淨利' not in df.columns:
            if '稅前淨利（淨損）' in df.columns and '所得稅費用（利益）' in df.columns:
                df['稅後淨利'] = df['稅前淨利（淨損）'] - df['所得稅費用（利益）']

        # 加上每年收盤價（抓不到就填 NaN）
        ticker = yf.Ticker(f"{stock_id}.TW")
        time.sleep(1.5)  # Yahoo 很膽小喵～慢慢來
        try:
            price_df = ticker.history(start="2011-01-01", end="2023-01-01")
            yearly_close = price_df.groupby(price_df.index.year)['Close'].last()
            df['Closing_Price'] = df.index.map(yearly_close)
        except:
            df['Closing_Price'] = None

        # 市值
        try:
            capital = df['股本']
            df['Market_Cap_Mil'] = df['Closing_Price'] * capital * 1000 / 1e6
        except:
            df['Market_Cap_Mil'] = None

        # 指標們
        capital = df.get('股本')
        df['ROE'] = safe_divide(df.get('稅後淨利'), df.get('股東權益合計'))
        df['ROA'] = safe_divide(df.get('稅後淨利'), df.get('資產總計'))
        df['OPM'] = safe_divide(df.get('營業利益'), df.get('營業收入'))
        df['NPM'] = safe_divide(df.get('稅後淨利'), df.get('營業收入'))
        df['Debt_to_Equity'] = safe_divide(df.get('負債總計'), df.get('股東權益合計'))
        df['Current_Ratio'] = safe_divide(df.get('流動資產'), df.get('流動負債'))
        df['Quick_Ratio'] = safe_divide(df.get('流動資產') - df.get('存貨'), df.get('流動負債'))
        df['Inventory_Turnover'] = safe_divide(df.get('營業成本'), df.get('存貨'))
        df['AR_Turnover'] = safe_divide(df.get('營業收入'), df.get('應收帳款'))
        df['PB_Ratio'] = safe_divide(df.get('Closing_Price'), df.get('股東權益合計') / capital / 1000)
        df['PS_Ratio'] = safe_divide(df.get('Closing_Price'), df.get('營業收入') / capital / 1000)
        df['OPM_Growth'] = df.get('營業利益').pct_change()
        df['NetIncome_Growth'] = df.get('稅後淨利').pct_change()
        df['Unknown_Param'] = None

        df['stock_id'] = stock_id
        df['year'] = df.index

        # 擷取指定欄位
        output = df[[
            'stock_id', 'year',
            'Market_Cap_Mil', 'Closing_Price', 'Unknown_Param', 'PB_Ratio', 'PS_Ratio',
            'ROE', 'ROA', 'OPM', 'NPM', 'Debt_to_Equity',
            'Current_Ratio', 'Quick_Ratio', 'Inventory_Turnover', 'AR_Turnover',
            'OPM_Growth', 'NetIncome_Growth'
        ]]

        all_results.append(output)

    except Exception as e:
        print(f"❌ 失敗：{stock_id} - {e}")

# 合併所有結果
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(os.path.join(result_dir, "AllStocks_FinancialIndicators.csv"), index=False)

print("✅ 所有股票處理完成，檔案已輸出到 cleaned_data 資料夾喵♡")
