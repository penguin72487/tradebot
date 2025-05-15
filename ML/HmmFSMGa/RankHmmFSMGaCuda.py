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


def plot_strategy_curve(df, n_states, save_dir, sharpe_ratio, buy_and_hold_return, best_weights):
    cumulative_strategy = (1 + df['strategy_return']).cumprod()
    cumulative_bnh = df['close'] / df['close'].iloc[0]

    # === 計算 Max Drawdown & Profit Factor ===
    rolling_max = cumulative_strategy.cummax()
    drawdown = (cumulative_strategy - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    log_strategy = np.log10(cumulative_strategy.replace(0, 1e-8))
    log_bnh = np.log10(cumulative_bnh.replace(0, 1e-8))

    profits = df.loc[df['strategy_return'] > 0, 'strategy_return'].sum() / (df['strategy_return'] > 0).sum()
    losses = df.loc[df['strategy_return'] < 0, 'strategy_return'].sum() / (df['strategy_return'] < 0).sum()
    profit_factor = profits / abs(losses) if losses != 0 else np.inf

    # === 建立主圖 + 下方標籤列 ===
    fig, (ax_main, ax_legend) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})
    cmap = plt.get_cmap("tab10")
    state_colors = {state: cmap(i % 10) for i, state in enumerate(np.unique(df['state']))}

    # 主圖 - 畫每段顏色
    for i in range(1, len(log_strategy)):
        s = df['state'].iloc[i]
        ax_main.plot([i - 1, i], [log_strategy.iloc[i - 1], log_strategy.iloc[i]], color=state_colors[s])

    ax_main.plot(log_bnh.values, linestyle='--', color='gray', label='Buy & Hold')

    ax_main.set_title(f'HMM-FSM-GA Strategy (n_states={n_states})\n'
                      f'Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2%}, PF={profit_factor:.2f}')
    ax_main.set_xlabel("Time")
    ax_main.set_ylabel("Log Cumulative Return")
    ax_main.grid(True)

    # 下方標籤欄 - 顯示所有 state 對應的 position
    ax_legend.axis("off")  # 不顯示軸線

    # 每行最多幾個 label
    max_per_row = 5
    state_labels = [
        f"State {s}: Pos={best_weights[s]:.2f}" for s in (state_colors.keys())
    ]

    for i, label in enumerate(state_labels):
        row = i // max_per_row
        col = i % max_per_row
        ax_legend.text(0.02 + 0.18 * col, 0.8 - 0.2 * row, label,
                       color=state_colors[int(label.split()[1][:-1])],
                       fontsize=10, transform=ax_legend.transAxes)

    # 儲存圖
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"strategy_n{n_states}.png"), dpi=1000)
    plt.close()

    # 儲存原始數據
    export_df = pd.DataFrame()
    export_df['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df.index
    export_df['close'] = df['close']
    export_df['state'] = df['state']
    export_df['position'] = df['strategy_return'] / df['close'].pct_change().fillna(0)
    export_df['strategy_return'] = df['strategy_return']
    export_df['cumulative_return'] = log_strategy
    export_df['drawdown'] = drawdown
    export_df['max_drawdown'] = max_drawdown
    export_df['profit_factor'] = profit_factor

    export_path = os.path.join(save_dir, f"strategy_data_n{n_states}.csv")
    export_df.to_csv(export_path, index=False)

# 繪製 Sharpe Ratio & Returns 圖
def plot_results(result_df, save_path):
    plt.figure(figsize=(12, 8))

    plt.plot(result_df['n_states'], result_df['sharpe_ratio'], label='Sharpe Ratio')
    plt.plot(result_df['n_states'], result_df['cumulative_return'], label='Cumulative Strategy Return')
    plt.plot(result_df['n_states'], result_df['buy_and_hold_return'], label='Buy & Hold Return', linestyle='--')

    plt.xlabel("Number of Hidden States")
    plt.ylabel("Performance Metric")
    plt.title("HMM-FSM-GA Strategy vs Buy & Hold")
    plt.legend()
    plt.grid(True)

    plot_path = save_path.replace('.csv', '.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 圖表已儲存到：{plot_path} 喵～")


# ======== 1. 資料處理（Rank Normalization） ========
def rank_normalize(series):
    valid = series.dropna()
    ranks = rankdata(valid, method='average')
    normed = (ranks - 1) / (len(valid) - 1)
    result = pd.Series(index=valid.index, data=normed)
    return series.combine_first(result)



def preprocess(df):
    cols = ['close', 'PMA12', 'PMA144', 'PMA169', 'PMA576', 'PMA676', 'MHULL', 'SHULL', 'KD', 'J', 'RSI', 'MACD', 'Signal Line', 'Histogram', 'QQE Line', 'Histo2', 'volume', 'Bullish Volume Trend', 'Bearish Volume Trend']
    for col in cols:
        df[col + "_norm"] = rank_normalize(df[col])
    # df.dropna(inplace=True)
    return df, [col + "_norm" for col in cols]

# ======== 2. 構建 GaussianHMM 模型 ========
def train_hmm(X, n_states=5):
    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1000, random_state=int(time.time()))
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states

# ======== 3. FSM 行為對應 & 回測績效計算 ========
def simulate_returns(states, weights, returns):
    positions = np.array([weights[s] for s in states])
    daily_returns = positions * returns
    return daily_returns

def compute_fitness(weights, states, returns):
    daily_returns = simulate_returns(states, weights, returns)
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    sharpe = mean_return / (std_return + 1e-8)
    return sharpe

def compute_sharpe_ratio(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / (std_return + 1e-8)
    return sharpe

# ======== 4. 基因演算法 GA ========
# @torch.compile
def evaluate_population(pop, states, returns):
    positions = pop[:, states]  # 每個個體對應到交易日倉位
    daily_returns = positions * returns
    mean = daily_returns.mean(dim=1)
    std = daily_returns.std(dim=1)
    sharpe = mean / (std + 1e-8)
    return sharpe

def torch_ga_optimize(states, returns, n_states=5, generations=5000, population_size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用 {device} + 自適應 GA 優化...")

    # 轉成 tensor
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    states = torch.tensor(states, dtype=torch.long, device=device)

    # 初始化族群（在 [-1, 1] 區間）
    pop = (torch.rand((population_size, n_states), device=device) - 0.5) * 2

    best_fitness = -float('inf')
    stagnant_count = 0
    stagnation_threshold = 5

    for gen in range(generations):
        # 計算 fitness
        fitness = evaluate_population(pop, states, returns)
        current_best = fitness.max().item()
        print(f"Generation {gen + 1}/{generations}: Best fitness = {current_best:.4f}")

        # 檢查是否收斂
        if abs(current_best - best_fitness) < 1e-4:
            stagnant_count += 1
        else:
            stagnant_count = 0
            best_fitness = current_best

        if stagnant_count >= stagnation_threshold:
            print(f"✨ Fitness 連續 {stagnation_threshold} 代沒變，提早收斂喵～")
            break

        # 選出前 50% elite
        topk = fitness.topk(k=population_size // 2)
        elite = pop[topk.indices]

        # === 交配（uniform crossover）
        half = elite.shape[0] // 2
        parents1 = elite[::2][:half]
        parents2 = elite[1::2][:half]
        crossover_mask = torch.rand_like(parents1) < 0.5
        children_cross = torch.where(crossover_mask, parents1, parents2)

        # === 自適應變異（隨 generation 遞減）
        mutation_rate = 0.2 * (1 - gen / generations)
        mutation = torch.randn_like(elite) * mutation_rate
        children_mutate = elite + mutation
        children_mutate = children_mutate.clamp(-1, 1)

        # === 重組族群
        pop = torch.cat([elite, children_cross, children_mutate], dim=0)

    # 回傳最佳結果
    final_fitness = evaluate_population(pop, states, returns)
    best_idx = final_fitness.argmax().item()
    best_weights = pop[best_idx].detach().cpu().numpy()

    return best_weights


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
