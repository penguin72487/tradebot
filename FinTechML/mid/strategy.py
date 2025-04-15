import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def simulate(df):
    capital = 10_000
    total_value = capital
    last_ko_price = None
    last_pep_price = None
    ko_position = 0.0
    pep_position = 0.0

    history = []

    for _, row in df.iterrows():
        x = row['Zscore_diff']
        if pd.isna(x):
            x = 0  # 預設為 0（中性狀態）
        ko_price = row['KO_price']
        pep_price = row['PEP_price']

        ko_ratio = 0.5 - 0.25 * x
        pep_ratio = 0.5 + 0.25 * x

        if last_ko_price is not None and last_pep_price is not None:
            ko_pnl = ko_position * (ko_price - last_ko_price)
            pep_pnl = pep_position * (pep_price - last_pep_price)
            total_value += ko_pnl + pep_pnl

        ko_position = (total_value * ko_ratio) / ko_price
        pep_position = (total_value * pep_ratio) / pep_price

        history.append({
            'Date': row['Date'],
            'TotalValue': total_value,
            'KORatio': ko_ratio,
            'PEPRatio': pep_ratio,
            'KOPosition': ko_position,
            'PEPPosition': pep_position
        })

        last_ko_price = ko_price
        last_pep_price = pep_price

    hist_df = pd.DataFrame(history)
    return capital, total_value, hist_df




def simulate50(df):
    capital = 10_000
    total_value = capital
    last_ko_price = None
    last_pep_price = None
    ko_position = 0.0
    pep_position = 0.0

    history = []

    for _, row in df.iterrows():
        ko_price = row['KO_price']
        pep_price = row['PEP_price']

        if last_ko_price is not None and last_pep_price is not None:
            ko_pnl = ko_position * (ko_price - last_ko_price)
            pep_pnl = pep_position * (pep_price - last_pep_price)
            total_value += ko_pnl + pep_pnl

        ko_position = (total_value * 0.5) / ko_price
        pep_position = (total_value * 0.5) / pep_price

        history.append({
            'Date': row['Date'],
            'TotalValue': total_value,
            'KORatio': 0.5,
            'PEPRatio': 0.5,
            'KOPosition': ko_position,
            'PEPPosition': pep_position
        })

        last_ko_price = ko_price
        last_pep_price = pep_price

    hist_df = pd.DataFrame(history)
    return capital, total_value, hist_df



def main():
    input_path = os.path.join("FinTechML", "mid", "KOPEPEresult", "zscore_result.csv")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    output_dir = os.path.join("FinTechML", "mid", "KOPEPEresult")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for split_year in range(2009, 2025):
        train_df = df[df['Year'] <= split_year - 1].copy()
        test_df = df[df['Year'] >= split_year].copy()

        if len(train_df) < 100 or len(test_df) < 100:
            continue

        # train_initial, train_value, train_history = simulate(train_df)
        test_initial, test_value, test_history = simulate(test_df)
        hold_initial, hold_value, hold_history = simulate50(test_df)



        train_years = f"{train_df['Year'].min()}-{train_df['Year'].max()}"
        test_years = f"{test_df['Year'].min()}-{test_df['Year'].max()}"

        # # 📊 畫策略 vs 靜態持有比較圖（每組測試集）
        # plt.figure(figsize=(14, 6))
        # plt.plot(test_history['Date'], test_history['TotalValue'], label='Strategy', color='black', linewidth=2)
        # plt.plot(hold_history['Date'], hold_history['TotalValue'], label='Hold 50% KO + 50% PEP', color='gray', linestyle='--')
        # plt.title(f"Test Comparison: Strategy vs Hold ({train_years} vs {test_years})")
        # plt.xlabel('Date')
        # plt.ylabel('Portfolio Value')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # compare_filename = f"compare_test_{train_years.replace('-', '_')}_vs_{test_years.replace('-', '_')}_strategy_vs_hold.png"
        # plt.savefig(os.path.join(output_dir, compare_filename))
        # plt.close()


        # 儲存當次結果
        all_results.append({
            'Train_Years': train_years,
            'Test_Years': test_years,
            'Test_Initial': 10_000,
            'Hold_Initial': 10_000,
            'Test_Final_Value': round(test_value, 2),
            'Hold_Final_Value': round(hold_value, 2),
        })



    result_df = pd.DataFrame(all_results)
    result_csv = os.path.join(output_dir, "strategy_result.csv")
    result_df.to_csv(result_csv, index=False)

    print(f"✅ 策略回測結果已儲存到：{result_csv}")

        # 年化報酬率圖
    
    test_annual_returns = []
    hold_annual_returns = []
    print("\n📊 年化報酬率列表：")
    print(f"{'Index':<5} {'Train Years':<20} {'Test Years':<20} {'Train Return (%)':>18} {'Test Return (%)':>18}")
    print("-" * 85)

    for i, row in result_df.iterrows():
        train_range = row['Train_Years'].split('-')
        test_range = row['Test_Years'].split('-')
        n_train_years = int(train_range[1]) - int(train_range[0]) + 1
        n_test_years = int(test_range[1]) - int(test_range[0]) + 1

        # train_return = (row['Train_Final_Value'] / row['Train_Initial']) ** (1 / n_train_years) - 1
        test_return = (row['Test_Final_Value'] / row['Test_Initial']) ** (1 / n_test_years) - 1
        hold_return = (row['Hold_Final_Value'] / row['Hold_Initial']) ** (1 / n_train_years) - 1

        print(f"{i+1:<5} {row['Train_Years']:<20} {row['Test_Years']:<20} {hold_return*100:>17.2f}% {test_return*100:>17.2f}%")

    for i, row in result_df.iterrows():
        train_range = row['Train_Years'].split('-')
        test_range = row['Test_Years'].split('-')
        n_train_years = int(train_range[1]) - int(train_range[0]) + 1
        n_test_years = int(test_range[1]) - int(test_range[0]) + 1

        hold_return = (row['Hold_Final_Value'] / 10_000) ** (1 / n_train_years) - 1
        test_return = (row['Test_Final_Value'] / 10_000) ** (1 / n_test_years) - 1

        
        test_annual_returns.append(test_return * 100)
        hold_annual_returns.append(hold_return * 100)

    x = list(range(1, len(hold_annual_returns) + 1))  # 測試集編號 1 ~ 16

    plt.figure(figsize=(12, 6))
    plt.plot(x, test_annual_returns, marker='s', label='Test Annual Return (%)')
    plt.plot(x, hold_annual_returns, marker='o', label='Train Annual Return (%)')
    plt.xticks(x)
    plt.title('Annualized Return per Test Set (Train vs Test)')
    plt.xlabel('Test Set Index')
    plt.ylabel('Annualized Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_return_plot = os.path.join(output_dir, "annualized_return_comparison_simple.png")
    plt.savefig(output_return_plot)
    plt.show()

    print(f"📊 年化報酬率圖已儲存：{output_return_plot}")
    # 💾 輸出年化報酬率資料（含持有）

    hold_final_value = round(df_sorted['Hold_Value'].iloc[-1], 2)

    annual_return_output = pd.DataFrame({
        'Train_Years': result_df['Train_Years'],
        'Test_Years': result_df['Test_Years'],
        'Test_Final_Value': result_df['Test_Final_Value'],
        'hold_Final_Value': [hold_final_value] * len(result_df)
    })

    annual_return_csv = os.path.join(output_dir, "annual_return_summary_with_hold.csv")
    annual_return_output.to_csv(annual_return_csv, index=False)
    print(f"📁 年化報酬率 + 持有結果 已輸出成檔案：{annual_return_csv} 喵～")



    # 🔁 不分割數據：整體策略回測（累積報酬率）圖
    all_initial, all_value, all_history = simulate(df)

    # 計算策略累積報酬率
    all_history['Strategy_CumReturn'] = all_history['TotalValue'] / all_initial - 1

    # 計算 KO 和 PEP 的累積報酬率（用最早價格當起點）
        # 計算 KO 和 PEP 的累積報酬率（用最早價格當起點）
    df_sorted = df.sort_values('Date')
    ko_base_price = df_sorted['KO_price'].iloc[0]
    pep_base_price = df_sorted['PEP_price'].iloc[0]

    df_sorted['KO_CumReturn'] = df_sorted['KO_price'] / ko_base_price - 1
    df_sorted['PEP_CumReturn'] = df_sorted['PEP_price'] / pep_base_price - 1

    # ➕ 加入靜態持有 50% KO + 50% PEP 組合
    df_sorted['Hold_CumReturn'] = (df_sorted['KO_CumReturn'] + df_sorted['PEP_CumReturn']) / 2
    df_sorted['Hold_Value'] = 10_000 * (1 + df_sorted['Hold_CumReturn'])

    # 合併資料（對齊日期）
    merged_df = pd.merge(
        all_history[['Date', 'TotalValue']],
        df_sorted[['Date', 'Hold_Value']],
        on='Date', how='inner'
    )
    merged_df['time'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    merged_df = merged_df[['time', 'TotalValue', 'Hold_Value']]
    merged_df.columns = ['time', 'strategy_value', 'hold_value']

       # 📈 畫累積報酬比較圖
    plt.figure(figsize=(14, 6))
    plt.plot(merged_df['time'], merged_df['strategy_value'], label='Strategy', linewidth=2)
    plt.plot(merged_df['time'], merged_df['hold_value'], label='Hold 50% KO + 50% PEP', linestyle='--')
    plt.title("Cumulative Return Comparison (Strategy vs Hold)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    cum_plot_path = os.path.join(output_dir, "cumulative_return_strategy_vs_hold.png")
    plt.savefig(cum_plot_path)
    plt.show()

    print(f"📊 策略 vs 靜態持有 比較圖已儲存：{cum_plot_path} 喵～")

        # 💾 輸出累積報酬 CSV（策略 + 靜態持有）
    strategy_vs_hold_csv = os.path.join(output_dir, "strategy_vs_hold_cumulative_return.csv")
    merged_df.to_csv(strategy_vs_hold_csv, index=False)
    print(f"📁 策略與持有報酬已輸出成檔案：{strategy_vs_hold_csv} 喵～")


        # 📊 持倉比例變化圖
    plt.figure(figsize=(14, 6))
    plt.plot(all_history['Date'], all_history['KORatio'], label='KO Ratio', color='red')
    plt.plot(all_history['Date'], all_history['PEPRatio'], label='PEP Ratio', color='blue')

    plt.title("Daily Portfolio Allocation (KO vs PEP)")
    plt.xlabel("Date")
    plt.ylabel("Position Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ratio_plot_path = os.path.join(output_dir, "daily_position_ratio.png")
    plt.savefig(ratio_plot_path)
    plt.show()

    print(f"📊 每日持倉比例圖已儲存：{ratio_plot_path} 喵～")






if __name__ == "__main__":
    main()
