import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_annual_return_summary():
    # 讀取 annual return 資料
    summary_path = os.path.join("FinTechML", "mid", "KOPEPEresult", "annual_return_summary_with_hold.csv")
    df = pd.read_csv(summary_path)

    # 計算 Test 年數
    df['Test_Years_Length'] = df['Test_Years'].apply(lambda x: int(x.split('-')[1]) - int(x.split('-')[0]) + 1)

    # 計算年化報酬率
    df['Strategy_Annual_Return'] = (df['Strategy_Final_Value'] / 10000) ** (1 / df['Test_Years_Length']) - 1
    df['Hold_Annual_Return'] = (df['Hold_Final_Value'] / 10000) ** (1 / df['Test_Years_Length']) - 1
    df['Test_Annual_Return'] = (df['Test_Final_Value'] / 10000) ** (1 / df['Test_Years_Length']) - 1

    # 畫圖
    plt.figure(figsize=(14, 6))
    plt.plot(df['Train_Years'], df['Strategy_Annual_Return'], label='Strategy Annual Return', marker='o')
    plt.plot(df['Train_Years'], df['Hold_Annual_Return'], label='Hold Annual Return', marker='o')
    plt.plot(df['Train_Years'], df['Test_Annual_Return'], label='Test Annual Return', marker='o')

    plt.title('Annualized Return Comparison')
    plt.xlabel('Train Years')
    plt.ylabel('Annual Return')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_annual = os.path.join("FinTechML", "mid", "KOPEPEresult", "annual_return_plot.png")
    plt.savefig(output_annual)
    plt.show()

    print(f"📈 年化報酬圖已儲存：{output_annual} 喵")

def plot_combined_cumulative_return():
    # 檔案位置
    base_dir = os.path.join("FinTechML", "mid", "KOPEPEresult")
    strat_hold_path = os.path.join(base_dir, "strategy_vs_hold_cumulative_return.csv")
    test_path = os.path.join(base_dir, "time_value.csv")

    # 讀資料
    df_strat_hold = pd.read_csv(strat_hold_path)
    df_test = pd.read_csv(test_path)

    # 轉換時間格式
    df_strat_hold['time'] = pd.to_datetime(df_strat_hold['time'])
    df_test['time'] = pd.to_datetime(df_test['time'])

    # 合併資料（以 strategy 為主，右邊用前向填充）
    df = pd.merge_asof(df_strat_hold.sort_values('time'),
                       df_test.sort_values('time'),
                       on='time',
                       direction='backward')  # 若對應不到則用前一筆 test_value

    # 畫圖
    plt.figure(figsize=(14, 6))
    plt.plot(df['time'], df['strategy_value'], label='Strategy', color='blue')
    plt.plot(df['time'], df['hold_value'], label='Hold', color='green')
    plt.plot(df['time'], df['test_value'], label='Test', color='orange')

    plt.title('Cumulative Return Comparison (Strategy vs Hold vs Test)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_combined = os.path.join(base_dir, "cumulative_return_combined_plot.png")
    plt.savefig(output_combined)
    plt.show()

    print(f"🎉 策略與現金對照圖儲存完成：{output_combined} 喵")


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

    # 第三張圖：年化報酬率圖
    plot_annual_return_summary()

    # 第四張圖：年化報酬率圖
    plot_combined_cumulative_return()


if __name__ == "__main__":
    main()
