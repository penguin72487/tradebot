import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # 設定檔案位置
    input_path = os.path.join("FinTechML", "mid", "KOPEPEresult", "zscore_result.csv")

    # 讀取資料
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # 建立圖表資料夾
    output_dir = os.path.join("FinTechML", "mid", "KOPEPEresult")
    os.makedirs(output_dir, exist_ok=True)

    # 第0張圖：累積報酬率圖 + SMA256
    df['KO_return'] = df['KO_price'].pct_change().fillna(0)
    df['PEP_return'] = df['PEP_price'].pct_change().fillna(0)

    df['KO_cum_return'] = (1 + df['KO_return']).cumprod()
    df['PEP_cum_return'] = (1 + df['PEP_return']).cumprod()

    # 累積報酬率的 256 日移動平均
    df['KO_cum_return_sma256'] = df['KO_cum_return'].rolling(window=256, min_periods=1).mean()
    df['PEP_cum_return_sma256'] = df['PEP_cum_return'].rolling(window=256, min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['KO_cum_return'], label='KO Cumulative Return', color='blue', alpha=0.5)
    plt.plot(df['Date'], df['KO_cum_return_sma256'], label='KO SMA256', color='blue', linestyle='--')

    plt.plot(df['Date'], df['PEP_cum_return'], label='PEP Cumulative Return', color='green', alpha=0.5)
    plt.plot(df['Date'], df['PEP_cum_return_sma256'], label='PEP SMA256', color='green', linestyle='--')

    plt.title('Cumulative Return of KO and PEP (with 256-day SMA)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_return = os.path.join(output_dir, "cumulative_return_sma256_plot.png")
    plt.savefig(output_return)
    plt.show()



    # 第一張圖：各自的 Z-score
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['KO_zscore'], label='KO Z-score', color='blue')
    plt.plot(df['Date'], df['PEP_zscore'], label='PEP Z-score', color='green')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Z-score (256-day) of KO and PEP')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_each = os.path.join(output_dir, "zscore_each_plot.png")
    plt.savefig(output_each)
    plt.show()

    # 第二張圖：Z-score 差
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Zscore_diff'], label='KO Z - PEP Z', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(1, color='green', linestyle='--', label='Z Diff = 1')
    plt.axhline(-1, color='red', linestyle='--', label='Z Diff = -1')
    plt.title('Z-score Difference (KO - PEP)')
    plt.xlabel('Date')
    plt.ylabel('Z-score Diff')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_diff = os.path.join(output_dir, "zscore_diff_plot.png")
    plt.savefig(output_diff)
    plt.show()

    print("✅ 所有圖表已成功儲存：")
    print(f"1️⃣ 累積報酬率圖：{output_return}")
    print(f"2️⃣ Z-score 圖：{output_each}")
    print(f"3️⃣ Z-score 差異圖：{output_diff}")

if __name__ == "__main__":
    main()
