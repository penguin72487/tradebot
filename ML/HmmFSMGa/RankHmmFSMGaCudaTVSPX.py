import pandas as pd
import numpy as np
from scipy.stats import rankdata
from hmmlearn.hmm import GaussianHMM
from deap import base, creator, tools, algorithms
import random
import warnings
import os
import matplotlib
matplotlib.use('Agg')  # 這句要在 import pyplot 前執行
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



# ======== 1. 資料處理（Rank Normalization） ========

from RankHmmFSMGaCuda import rank_normalize
# def rank_normalize(series):

from RankHmmFSMGaCuda import preprocess
# def preprocess(df, features=None):

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

from RankHmmFSMGaCuda import torch_ga_optimize_totle_return
# def torch_ga_optimize_totle_return(states, returns, n_states=5, generations=50, population_size=1024):

def cross_val_worker(i, unique_years, df, X, returns, close_prices, n_states):
    train_years = unique_years[:i]
    test_years = unique_years[i:]

    train_idx = df[df['year'].isin(train_years)].index
    test_idx = df[df['year'].isin(test_years)].index

    if len(train_idx) < 50 or len(test_idx) < 10:
        return None

    X_train, X_test = X[train_idx], X[test_idx]
    returns_train, returns_test = returns[train_idx], returns[test_idx]

    print(f"🧲 試算年份 {train_years[0]} ~ {train_years[-1]}，測試 {test_years[0]} ~ {test_years[-1]}：訓練 {len(train_idx)} 筆資料")

    try:
        hmm_model, train_states = train_hmm(X_train, n_states=n_states)
        test_states = hmm_model.predict(X_test)

        best_weights = torch_ga_optimize_totle_return(train_states, returns_train, n_states=n_states)
        test_strategy_returns = best_weights[test_states] * returns_test

        strategy_cum_return = (1 + test_strategy_returns).prod()
        years_tested = len(test_years)

        strategy_annual_return = strategy_cum_return**(1 / years_tested) - 1
        buy_hold_return = close_prices[test_idx[-1]] / close_prices[test_idx[0]] - 1
        buy_hold_annual_return = (1 + buy_hold_return)**(1 / years_tested) - 1
        sharpe = compute_sharpe_ratio(test_strategy_returns)

        print(f"✅ 完成：SR = {sharpe:.4f}, 策略年化 = {strategy_annual_return:.4f}, B&H年化 = {buy_hold_annual_return:.4f}")

        return {
            "train_years": f"{train_years[0]}–{train_years[-1]}",
            "test_years": f"{test_years[0]}–{test_years[-1]}",
            "strategy_ann_return": strategy_annual_return,
            "buy_hold_ann_return": buy_hold_annual_return,
            "sharpe_ratio": sharpe,
            "df": pd.DataFrame({
                "close": close_prices[test_idx],
                "state": test_states,
                "strategy_return": test_strategy_returns,
            }),
            "best_weights": best_weights
        }

    except Exception as e:
        print(f"⚠️ 發生錯誤 @ 測試年份 {test_years[0]}~{test_years[-1]}：{e}")
        return None


# ======== 6. 主程式入口 ========
def run_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 取得目前執行的檔名（不含副檔名）
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # 建立 result/<script_name>_result 路徑
    result_dir = os.path.join(base_dir, "result", f"{script_name}_result")
    os.makedirs(result_dir, exist_ok=True)

    # 輸入與輸出路徑
    input_path = os.path.join(base_dir, "SP_SPX, 1D.csv")
    output_path = os.path.join(result_dir, "hmm_fsm_ga_result_summary.csv")

    df = pd.read_csv(input_path)
    
    cols = ['close', 'PMA12', 'PMA144', 'PMA169', 'PMA576', 'PMA676', 'MHULL', 'SHULL', 'KD', 'J', 'RSI', 'MACD', 'Signal Line', 'Histogram', 'QQE Line', 'Histo2']
    df, features = preprocess(df, features=cols)
    X = df[features].values
    returns = df['close'].pct_change().fillna(0).values

    results = []

    for n_states in range(70,71):
        print(f"🚀 正在訓練 n_states = {n_states} ...")
        try:
            hmm_model, states = train_hmm(X, n_states=n_states)
            best_weights = torch_ga_optimize_totle_return(states, returns, n_states=n_states)
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
            plot_strategy_curve(df, n_states, n_state_dir, sharpe, buy_and_hold_return, best_weights, save_name=f"strategy_n{n_states}.png")

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
        result_dir = os.path.join(base_dir, "result", f"{script_name}_crossval_expanding")

    # cols = ['close', 'PMA12', 'PMA144', 'PMA169', 'PMA576', 'PMA676', 'MHULL', 'SHULL', 'KD', 'J', 'RSI', 'MACD', 'Signal Line', 'Histogram', 'QQE Line', 'Histo2']
    cols = ['MHULL', 'SHULL', 'KD', 'J', 'RSI', 'MACD', 'Signal Line', 'Histogram', 'QQE Line', 'Histo2']
    df['returns'] = df['close'].pct_change().fillna(0)
    # 👇 新增特徵：前一期報酬（延遲一根K線）
    df['prev_return'] = df['returns'].shift(1).fillna(0)
    cols.append('prev_return')  # 👉 加入到 feature list 裡
    df, features = preprocess(df, features=cols)
    X = df[features].values
    returns = df['returns'].values
    close_prices = df['close'].values

    df['year'] = pd.to_datetime(df['time'], unit='s').dt.year
    unique_years = sorted(df['year'].unique())

    results = []
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(cross_val_worker, i, unique_years, df, X, returns, close_prices, n_states)
            for i in range(1, len(unique_years))
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # 去除 None（失敗的結果）
    results = [r for r in results if r is not None]
    result_df = pd.DataFrame(results).sort_values("test_years")
    # 每段畫圖
    # 建立統一的儲存資料夾
    cv_return_dir = os.path.join(result_dir, "cv_return")
    os.makedirs(cv_return_dir, exist_ok=True)

    for row in results:
        df_cv = row["df"].reset_index(drop=True)
        filename = f"cv_{row['test_years']}_n{n_states}.png"
        plot_strategy_curve(
            df_cv,
            n_states=n_states,
            save_dir=cv_return_dir,
            sharpe_ratio=row["sharpe_ratio"],
            buy_and_hold_return=row["buy_hold_ann_return"],
            best_weights=row["best_weights"],
            save_name=filename,
        )



    # 儲存
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(result_dir, "crossval_expanding_summary.csv"), index=False)

    # 畫圖
    import matplotlib.pyplot as plt
    # 直接使用原始年化報酬率
    strategy_vals = result_df["strategy_ann_return"].values
    buyhold_vals = result_df["buy_hold_ann_return"].values

    strategy_neg_mask = strategy_vals < 0
    buyhold_neg_mask = buyhold_vals < 0

    plt.figure(figsize=(12, 6), dpi=1000)
    x_labels = result_df["test_years"]

    plt.plot(x_labels, strategy_vals, label="Strategy", marker='o')
    plt.plot(x_labels, buyhold_vals, label="Buy & Hold", marker='x')

    plt.scatter(x_labels[strategy_neg_mask],
                strategy_vals[strategy_neg_mask],
                color='red', label="Strategy (Loss)", marker='o')

    plt.scatter(x_labels[buyhold_neg_mask],
                buyhold_vals[buyhold_neg_mask],
                color='red', label="Buy & Hold (Loss)", marker='x')

    for idx, row in result_df.iterrows():
        y_val = row["strategy_ann_return"]
        if not np.isnan(y_val):
            plt.text(x_labels[idx], y_val + 0.005,
                    f"SR={row['sharpe_ratio']:.2f}",
                    fontsize=8, ha='center', color='blue')

    plt.xlabel("Training Years")
    plt.ylabel("Annualized Return")
    plt.title("Expanding-Train Cross-Validation: Strategy vs Buy & Hold")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "annualized_return_expanding_train_linear.png"))
    plt.close()
    print("📉 原始報酬率圖完成啦～親親你一下獎勵努力的你喵 💋")




# ======== 6. 使用範例 ========
if __name__ == "__main__":
    run_model()
