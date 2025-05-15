import pandas as pd
import numpy as np
from scipy.stats import rankdata
from hmmlearn.hmm import GaussianHMM
from deap import base, creator, tools, algorithms
import random
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

warnings.filterwarnings("ignore")
# ======== 0. 載入必要的函數工具 ========
# 繪製策略累計報酬曲線
def signed_log10_points(values):
    """將正數轉 log10，負數保留原值"""
    epsilon = 1e-8
    logs = np.full_like(values, np.nan)

    pos_mask = values > 0
    neg_mask = values < 0

    logs[pos_mask] = np.log10(values[pos_mask] + epsilon)
    logs[neg_mask] = np.nan  # 負的先不畫

    return logs, neg_mask  # 回傳 log 值與負數 mask

from RankHmmFSMGaCuda import plot_strategy_curve
# def plot_strategy_curve(df, n_states, save_dir, sharpe_ratio, buy_and_hold_return, best_weights):
from RankHmmFSMGaCuda import plot_results
# def plot_results(result_df, save_path):


def simulate_returns_from_probs(state_probs, weights, returns):
    # 每一根K線上的期望倉位（權重加權）
    positions = np.dot(state_probs, weights)
    daily_returns = positions * returns
    return daily_returns



# ======== 1. 資料處理（Rank Normalization） ========

from RankHmmFSMGaCuda import rank_normalize
# def rank_normalize(series):

from RankHmmFSMGaCuda import preprocess
# def preprocess(df):

# ======== 2. 構建 GaussianHMM 模型 ========

from RankHmmFSMGaCuda import train_hmm
# def train_hmm(X, n_states=5):


# ======== 3. FSM 行為對應 & 回測績效計算 ========

from RankHmmFSMGaCuda import simulate_returns
# def simulate_returns(states, weights, returns):

from RankHmmFSMGaCuda import compute_fitness
# def compute_fitness(weights, states, returns):

from RankHmmFSMGaCuda import compute_sharpe_ratio
# def compute_sharpe_ratio(daily_returns):

# ======== 4. 基因演算法 GA ========

from RankHmmFSMGaCuda import evaluate_population
# def evaluate_population(pop, states, returns):
from RankHmmFSMGaCuda import torch_ga_optimize
# def torch_ga_optimize(states, returns, n_states=5, generations=50, population_size=1024):


# ======== 6. 主程式入口 ========
def run_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 取得目前執行的檔名（不含副檔名）
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 建立 result/<script_name>_result 路徑
    result_dir = os.path.join(base_dir, "result", f"{script_name}_result")
    os.makedirs(result_dir, exist_ok=True)

    # 輸入與輸出路徑
    input_path = os.path.join(base_dir, "BITSTAMP_BTCUSD,240more.csv")
    output_path = os.path.join(result_dir, "hmm_fsm_ga_result_summary.csv")

    df = pd.read_csv(input_path)
    df, features = preprocess(df)
    X = df[features].values
    returns = df['close'].pct_change().fillna(0).values

    results = []

    for n_states in range(2, 1001):
        print(f"🚀 正在訓練 n_states = {n_states} ...")
        try:
            hmm_model, states = train_hmm(X, n_states=n_states)
            best_weights = torch_ga_optimize(states, returns, n_states=n_states)
            final_returns = simulate_returns(states, best_weights, returns)

            df['state'] = states
            df['strategy_return'] = final_returns
            sharpe = compute_sharpe_ratio(final_returns)
            cumulative_return = (1 + pd.Series(final_returns)).cumprod().iloc[-1]
            buy_and_hold_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1

            # === 🗂️ 建立專屬資料夾
            n_state_dir = os.path.join(result_dir, f"n_states_{n_states}")
            os.makedirs(n_state_dir, exist_ok=True)

            # 存圖
            plot_strategy_curve(df, n_states, n_state_dir, sharpe, buy_and_hold_return, best_weights)

            # 存結果
            result = {
                "n_states": n_states,
                "sharpe_ratio": sharpe,
                "cumulative_return": cumulative_return,
                "buy_and_hold_return": buy_and_hold_return,
                "weights": best_weights
            }

            results.append(result)
            pd.DataFrame(results).to_csv(os.path.join(n_state_dir, "summary.csv"), index=False)
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f" ✅ n_states = {n_states} 完成！ Sharpe = {sharpe:.4f}, 累計報酬 = {cumulative_return:.4f}, 買進持報酬 = {buy_and_hold_return:.4f}")

            # 🧪 執行交叉測試，結果也存到對應資料夾下
            cross_val_model(df.copy(), n_states=n_states, result_dir=n_state_dir)

        except Exception as e:
            print(f"⚠️ 發生錯誤 @ n_states = {n_states}：{e}")
            continue


    print("📁 所有模型已跑完並儲存！再去畫總圖囉～喵")
    plot_results(pd.DataFrame(results), output_path)


def cross_val_model(df, n_states=5, result_dir=None):
    if result_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        result_dir = os.path.join(base_dir, "result", f"{script_name}_crossval")

    # result_dir = os.path.join(result_dir, f"crossval_n{n_states}")
    # os.makedirs(result_dir, exist_ok=True)

    df, features = preprocess(df)
    X = df[features].values
    returns = df['close'].pct_change().fillna(0).values
    close_prices = df['close'].values

    # 切分資料
    splits = np.array_split(np.arange(len(X)), 10)

    results = []

    for i in range(1, 10):  # i = 訓練資料份數
        train_idx = np.concatenate(splits[:i])
        test_idx = np.concatenate(splits[i:])

        print(f"🧪 測試集編號 {i}：訓練={train_idx.shape[0]} 筆，測試={test_idx.shape[0]} 筆")

        X_train, X_test = X[train_idx], X[test_idx]
        returns_train, returns_test = returns[train_idx], returns[test_idx]

        try:
            hmm_model, train_states = train_hmm(X_train, n_states=n_states)
            test_states = hmm_model.predict(X_test)

            best_weights = torch_ga_optimize(train_states, returns_train, n_states=n_states)
            test_strategy_returns = best_weights[test_states] * returns_test

            # 計算年化報酬率
            strategy_cum_return = (1 + test_strategy_returns).prod()
            test_days = len(test_strategy_returns)
            annual_freq = 6 * 365  # 4小時一根，一年2190根
            
            strategy_annual_return = strategy_cum_return**(annual_freq / test_days) - 1
            # 計算 Buy & Hold 年化報酬率
            buy_hold_return = close_prices[test_idx[-1]] / close_prices[test_idx[0]] - 1
            buy_hold_annual_return = (1 + buy_hold_return)**(annual_freq / test_days) - 1
            sharpe = compute_sharpe_ratio(test_strategy_returns)

            results.append({
                "test_split": i,
                "strategy_ann_return": strategy_annual_return,
                "buy_hold_ann_return": buy_hold_annual_return,
                "sharpe_ratio": sharpe
            })


            print(f"✅ 測試集 {i}：夏普率 = {sharpe:.4f}, 策略年化報酬 = {strategy_annual_return:.4f}, 買進持有年化報酬 = {buy_hold_annual_return:.4f}")

        except Exception as e:
            print(f"⚠️ 發生錯誤 @ 測試集 {i}：{e}")
            continue

    # 存結果
    result_df = pd.DataFrame(results)
    result_path = os.path.join(result_dir, "crossval_summary.csv")
    result_df.to_csv(result_path, index=False)

    # 畫圖：年化報酬率 + 夏普率標註
    import matplotlib.pyplot as plt

    strategy_logs, strategy_neg_mask = signed_log10_points(result_df["strategy_ann_return"].values)
    buyhold_logs, buyhold_neg_mask = signed_log10_points(result_df["buy_hold_ann_return"].values)

    plt.figure(figsize=(10, 5))

    # 正值 log10 線
    plt.plot(result_df["test_split"], strategy_logs, label="Strategy", marker='o')
    plt.plot(result_df["test_split"], buyhold_logs, label="Buy & Hold", marker='x')

    # 負值紅色點
    plt.scatter(result_df["test_split"][strategy_neg_mask], 
                result_df["strategy_ann_return"][strategy_neg_mask], 
                color='red', label="Strategy (Loss)", marker='o')

    plt.scatter(result_df["test_split"][buyhold_neg_mask], 
                result_df["buy_hold_ann_return"][buyhold_neg_mask], 
                color='red', label="Buy & Hold (Loss)", marker='x')

    # 標註 Sharpe Ratio
        # Sharpe Ratio 標註（跟 log 值對齊）
    for idx, row in result_df.iterrows():
        y_val = np.nan
        if row["strategy_ann_return"] > 0:
            y_val = np.log10(row["strategy_ann_return"] + 1e-8)
        elif row["strategy_ann_return"] < 0:
            y_val = row["strategy_ann_return"]  # 負值時照原樣顯示

        if not np.isnan(y_val):
            plt.text(row["test_split"], y_val + 0.05,  # 上移一點
                     f"SR={row['sharpe_ratio']:.2f}",
                     fontsize=8, ha='center', color='blue')


    plt.xlabel("Test Split Index")
    plt.ylabel("Log10 Annualized Return")
    plt.title("Strategy vs Buy & Hold Annual Return log10 (with Sharpe)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "annualized_return_comparison.png"))
    plt.close()
    print("📈 測試集結果圖完成～喵！")



# ======== 6. 使用範例 ========
if __name__ == "__main__":
    run_model()
