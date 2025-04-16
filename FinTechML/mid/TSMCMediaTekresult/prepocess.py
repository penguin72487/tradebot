import pandas as pd
import os

def main():
    # 讀取CSV檔案路徑
    data_dir = os.path.join("FinTechML", "Dataset")
    tsmc_df = pd.read_csv(os.path.join(data_dir, "TSMC.csv"))
    mediatek_df = pd.read_csv(os.path.join(data_dir, "MediaTek.csv"))

    # 日期轉換
    tsmc_df['Date'] = pd.to_datetime(tsmc_df['Date'])
    mediatek_df['Date'] = pd.to_datetime(mediatek_df['Date'])

    # 轉換 Adj Close 成 float（去掉逗號）
    tsmc_df['Adj Close'] = tsmc_df['Adj Close'].replace(',', '', regex=True).astype(float)
    mediatek_df['Adj Close'] = mediatek_df['Adj Close'].replace(',', '', regex=True).astype(float)


    # 篩選資料區間
    start_date = '2008-01-01'
    end_date = '2024-12-31'
    tsmc_df = tsmc_df[(tsmc_df['Date'] >= start_date) & (tsmc_df['Date'] <= end_date)]
    mediatek_df = mediatek_df[(mediatek_df['Date'] >= start_date) & (mediatek_df['Date'] <= end_date)]

    # 選取調整收盤價
    tsmc_df = tsmc_df[['Date', 'Adj Close']].rename(columns={'Adj Close': 'TSMC'})
    mediatek_df = mediatek_df[['Date', 'Adj Close']].rename(columns={'Adj Close': 'MediaTek'})

    # 合併資料集
    merged_df = pd.merge(tsmc_df, mediatek_df, on='Date', how='inner').sort_values('Date')

    # 計算 TSMC 與 MediaTek 的 Z-score（256日）
    tsmc_mean = merged_df['TSMC'].rolling(window=256, min_periods=1).mean()
    tsmc_std = merged_df['TSMC'].rolling(window=256, min_periods=1).std()
    tsmc_z = (merged_df['TSMC'] - tsmc_mean) / tsmc_std

    mediatek_mean = merged_df['MediaTek'].rolling(window=256, min_periods=1).mean()
    mediatek_std = merged_df['MediaTek'].rolling(window=256, min_periods=1).std()
    mediatek_z = (merged_df['MediaTek'] - mediatek_mean) / mediatek_std

    # 計算 Z-score 差
    z_diff = tsmc_z - mediatek_z

    # 組裝結果
    result_df = pd.DataFrame({
        'Date': merged_df['Date'],
        'TSMC_price': merged_df['TSMC'],
        'TSMC_zscore': tsmc_z,
        'MediaTek_price': merged_df['MediaTek'],
        'MediaTek_zscore': mediatek_z,
        'Zscore_diff': z_diff
    })

    # 輸出資料夾與檔案
    output_dir = os.path.join("FinTechML", "mid", "TSMCMediaTekresult")
    os.makedirs(output_dir, exist_ok=True)  # 自動建立資料夾
    output_path = os.path.join(output_dir, "zscore_result.csv")
    result_df.to_csv(output_path, index=False)

    print(f"Z-score 計算結果已儲存到：{output_path}")

if __name__ == "__main__":
    main()
