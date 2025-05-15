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

            # 存圖也放到對應 result 資料夾
            plot_strategy_curve(df, n_states, result_dir, sharpe, buy_and_hold_return, best_weights)

            result = {
                "n_states": n_states,
                "sharpe_ratio": sharpe,
                "cumulative_return": cumulative_return,
                "buy_and_hold_return": buy_and_hold_return,
                "weights": best_weights
            }

            results.append(result)
            pd.DataFrame(results).to_csv(output_path, index=False)

            print(f" ✅ n_states = {n_states} 完成！ Sharpe = {sharpe:.4f}, 累計報酬 = {cumulative_return:.4f}, 買進持報酬 = {buy_and_hold_return:.4f}")
        except Exception as e:
            print(f"⚠️ 發生錯誤 @ n_states = {n_states}：{e}")
            continue

    print("📁 所有模型已跑完並儲存！再去畫總圖囉～喵")
    plot_results(pd.DataFrame(results), output_path)


# ======== 6. 使用範例 ========
if __name__ == "__main__":
    run_model()
