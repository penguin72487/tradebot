import pandas as pd
import os

def main():
    # 讀取CSV檔案路徑
    data_dir = os.path.join("FinTechML", "Dataset")
    ko_df = pd.read_csv(os.path.join(data_dir, "KO.csv"))
    pep_df = pd.read_csv(os.path.join(data_dir, "PEP.csv"))

    # 日期轉換
    ko_df['Date'] = pd.to_datetime(ko_df['Date'])
    pep_df['Date'] = pd.to_datetime(pep_df['Date'])

    # 篩選資料區間
    start_date = '2008-01-01'
    end_date = '2024-12-31'
    ko_df = ko_df[(ko_df['Date'] >= start_date) & (ko_df['Date'] <= end_date)]
    pep_df = pep_df[(pep_df['Date'] >= start_date) & (pep_df['Date'] <= end_date)]

    # 選取調整收盤價
    ko_df = ko_df[['Date', 'Adj Close']].rename(columns={'Adj Close': 'KO'})
    pep_df = pep_df[['Date', 'Adj Close']].rename(columns={'Adj Close': 'PEP'})

    # 合併資料集
    merged_df = pd.merge(ko_df, pep_df, on='Date', how='inner').sort_values('Date')

    # 計算 KO 與 PEP 的 Z-score（256日）
    ko_mean = merged_df['KO'].rolling(window=256, min_periods=1).mean()
    ko_std = merged_df['KO'].rolling(window=256, min_periods=1).std()
    ko_z = (merged_df['KO'] - ko_mean) / ko_std

    pep_mean = merged_df['PEP'].rolling(window=256, min_periods=1).mean()
    pep_std = merged_df['PEP'].rolling(window=256, min_periods=1).std()
    pep_z = (merged_df['PEP'] - pep_mean) / pep_std

    # 計算 Z-score 差
    z_diff = ko_z - pep_z

    # 組裝結果
    result_df = pd.DataFrame({
        'Date': merged_df['Date'],
        'KO_price': merged_df['KO'],
        'KO_zscore': ko_z,
        'PEP_price': merged_df['PEP'],
        'PEP_zscore': pep_z,
        'Zscore_diff': z_diff
    })

    # 輸出資料夾與檔案
    output_dir = os.path.join("FinTechML", "mid", "KOPEPEresult")
    os.makedirs(output_dir, exist_ok=True)  # 自動建立資料夾
    output_path = os.path.join(output_dir, "zscore_result.csv")
    result_df.to_csv(output_path, index=False)

    print(f"Z-score 計算結果已儲存到：{output_path}")

if __name__ == "__main__":
    main()
