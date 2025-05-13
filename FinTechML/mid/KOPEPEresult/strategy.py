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
        x = row['Zscore_diff'] if not pd.isna(row['Zscore_diff']) else 0
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
        })

        last_ko_price = ko_price
        last_pep_price = pep_price

    return capital, total_value, pd.DataFrame(history)

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
        })

        last_ko_price = ko_price
        last_pep_price = pep_price

    return capital, total_value, pd.DataFrame(history)

def main():
    input_path = os.path.join("FinTechML", "mid", "KOPEPEresult", "zscore_result.csv")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    output_dir = os.path.join("FinTechML", "mid", "KOPEPEresult")
    os.makedirs(output_dir, exist_ok=True)

    # 全體模擬
    _, _, strategy_hist = simulate(df)
    _, _, hold_hist = simulate50(df)
    
    strategy_hist['time'] = strategy_hist['Date'].dt.strftime('%Y-%m-%d')
    hold_hist['time'] = hold_hist['Date'].dt.strftime('%Y-%m-%d')
    merged_df = pd.merge(
        strategy_hist[['time', 'TotalValue']],
        hold_hist[['time', 'TotalValue']],
        on='time', suffixes=('_strategy', '_hold')
    )
    merged_df.columns = ['time', 'strategy_value', 'hold_value']

    merged_df.to_csv(os.path.join(output_dir, "strategy_vs_hold_cumulative_return.csv"), index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(merged_df['time'], merged_df['strategy_value'], label='Strategy', linewidth=2)
    plt.plot(merged_df['time'], merged_df['hold_value'], label='Hold 50% KO + 50% PEP', linestyle='--')
    plt.title("Cumulative Return Comparison (Strategy vs Hold)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_return_strategy_vs_hold.png"))
    plt.show()

    # 比例圖
    plt.figure(figsize=(14, 6))
    plt.plot(strategy_hist['Date'], strategy_hist['KORatio'], label='KO Ratio', color='red')
    plt.plot(strategy_hist['Date'], strategy_hist['PEPRatio'], label='PEP Ratio', color='blue')
    plt.title("Daily Portfolio Allocation (KO vs PEP)")
    plt.xlabel("Date")
    plt.ylabel("Position Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "daily_position_ratio.png"))
    plt.show()

    # 年化報酬率比較圖（每年切一段）
    df['Year'] = df['Date'].dt.year
    all_results = []
    for split_year in range(2009, 2025):
        train_df = df[df['Year'] <= split_year - 1].copy()
        test_df = df[df['Year'] >= split_year].copy()
        if len(train_df) < 100 or len(test_df) < 100:
            continue
        _, test_value, _ = simulate(test_df)
        _, hold_value, _ = simulate50(test_df)
        all_results.append({
            'Train_Years': f"{train_df['Year'].min()}-{train_df['Year'].max()}",
            'Test_Years': f"{test_df['Year'].min()}-{test_df['Year'].max()}",
            'Strategy_Final_Value': round(test_value, 2),
            'Hold_Final_Value': round(hold_value, 2)
        })

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(os.path.join(output_dir, "annual_return_summary_with_hold.csv"), index=False)

    test_returns = []
    hold_returns = []
    for _, row in result_df.iterrows():
        test_range = list(map(int, row['Test_Years'].split('-')))
        train_range = list(map(int, row['Train_Years'].split('-')))
        test_years = test_range[1] - test_range[0] + 1
        # train_years = train_range[1] - train_range[0] + 1
        test_return = (row['Strategy_Final_Value'] / 10_000) ** (1 / test_years) - 1
        hold_return = (row['Hold_Final_Value'] / 10_000) ** (1 / test_years) - 1
        test_returns.append(test_return * 100)
        hold_returns.append(hold_return * 100)

    plt.figure(figsize=(12, 6))
    x = list(range(1, len(test_returns) + 1))
    plt.plot(x, test_returns, marker='s', label='Strategy Annual Return (%)')
    plt.plot(x, hold_returns, marker='o', label='Hold Annual Return (%)')
    plt.title('Annualized Return per Test Set')
    plt.xlabel('Test Set Index')
    plt.ylabel('Annualized Return (%)')
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annualized_return_comparison_simple.png"))
    plt.show()

if __name__ == "__main__":
    main()