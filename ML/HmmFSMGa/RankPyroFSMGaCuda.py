import pandas as pd
import numpy as np
from scipy.stats import rankdata
from hmmlearn.hmm import GaussianHMM
from deap import base, creator, tools, algorithms
import random
import warnings
import os
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


warnings.filterwarnings("ignore")
# ======== 0. 載入必要的函數工具 ========
# 繪製策略累計報酬曲線
def plot_strategy_curve(df, n_states, save_dir, sharpe_ratio, buy_and_hold_return):
    cumulative_strategy = (1 + df['strategy_return']).cumprod()
    cumulative_bnh = df['close'] / df['close'].iloc[0]

    # === Log for visual ===
    log_strategy = np.log(cumulative_strategy.replace(0, 1e-8))
    log_bnh = np.log(cumulative_bnh.replace(0, 1e-8))

    # === 計算 Max Drawdown ===
    rolling_max = cumulative_strategy.cummax()
    drawdown = (cumulative_strategy - rolling_max) / rolling_max
    max_drawdown = drawdown.min()  # 是負的值

    # === 計算 Profit Factor ===
    profits = df.loc[df['strategy_return'] > 0, 'strategy_return'].sum()
    losses = df.loc[df['strategy_return'] < 0, 'strategy_return'].sum()
    profit_factor = profits / abs(losses) if losses != 0 else np.inf

    # === 繪圖 ===
    plt.figure(figsize=(10, 6))
    plt.plot(log_strategy, label=f'Strategy (Sharpe={sharpe_ratio:.2f})')
    plt.plot(log_bnh, linestyle='--', label=f'Buy & Hold (Return={buy_and_hold_return:.2f})')

    plt.title(f'HMM-FSM-GA Log Return Curve (n_states={n_states})\n'
              f'MaxDD={max_drawdown:.2%}, PF={profit_factor:.2f}')
    plt.xlabel("Time")
    plt.ylabel("Log Cumulative Return")
    plt.legend()
    plt.grid(True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, f"strategy_n{n_states}.png"))
    plt.close()

    # === 儲存原始數據 CSV（補 maxDD、PF） ===
    export_df = pd.DataFrame()
    export_df['timestamp'] = df['timestamp'] if 'timestamp' in df.columns else df.index
    export_df['close'] = df['close']
    export_df['state'] = df['state']
    export_df['position'] = df['strategy_return'] / df['close'].pct_change().fillna(0)
    export_df['strategy_return'] = df['strategy_return']
    export_df['cumulative_return'] = cumulative_strategy
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
def pyro_hmm(X, n_states=5, device="cuda"):
    """
    模擬一個簡單的 HMM 用 Pyro 表達（GPU 上運行）
    回傳最可能的 state 序列（Viterbi 路徑）
    """

    T, D = X.shape
    X = torch.tensor(X, dtype=torch.float32, device=device)

    # 初始狀態機率
    probs_init = torch.ones(n_states, device=device) / n_states

    # 狀態轉移機率矩陣
    trans_probs = torch.rand(n_states, n_states, device=device)
    trans_probs = trans_probs / trans_probs.sum(dim=1, keepdim=True)

    # 每個 state 對應到一個高斯分布（均值+共變異數）
    emission_means = torch.randn(n_states, D, device=device)
    emission_stds = torch.abs(torch.randn(n_states, D, device=device)) + 1e-2

    # ===== 計算 log likelihood (手動 Viterbi for inference) =====
    log_delta = torch.zeros((T, n_states), device=device)
    psi = torch.zeros((T, n_states), dtype=torch.long, device=device)

    # 初始觀測
    for s in range(n_states):
        log_prob = dist.Normal(emission_means[s], emission_stds[s]).log_prob(X[0]).sum()
        log_delta[0, s] = torch.log(probs_init[s]) + log_prob

    # 遞迴
    for t in range(1, T):
        for s in range(n_states):
            emission_logp = dist.Normal(emission_means[s], emission_stds[s]).log_prob(X[t]).sum()
            trans_logp = log_delta[t - 1] + torch.log(trans_probs[:, s])
            psi[t, s] = torch.argmax(trans_logp)
            log_delta[t, s] = torch.max(trans_logp) + emission_logp

    # 回溯最可能的隱狀態序列
    states = torch.zeros(T, dtype=torch.long, device=device)
    states[-1] = torch.argmax(log_delta[-1])
    for t in reversed(range(1, T)):
        states[t - 1] = psi[t, states[t]]

    return states.detach().cpu().numpy()  # 讓後續 numpy 相容

def train_hmm(X, n_states=5):
    return None, pyro_hmm(X, n_states=n_states, device="cuda" if torch.cuda.is_available() else "cpu")


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
    return sharpe,

# ======== 4. 基因演算法 GA ========
def optimize_ga(states, returns, n_states=5, ngen=50, pop_size=30):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: random.uniform(-1, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_states)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: compute_fitness(ind, states, returns))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, verbose=False)
    best_ind = tools.selBest(pop, k=1)[0]
    return best_ind

# @torch.compile
def evaluate_population(pop, states, returns):
    positions = pop[:, states]  # 每個個體對應到交易日倉位
    daily_returns = positions * returns
    mean = daily_returns.mean(dim=1)
    std = daily_returns.std(dim=1)
    sharpe = mean / (std + 1e-8)
    return sharpe

def torch_ga_optimize(states, returns, n_states=5, generations=50, population_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用 {device} + torch.compile 進行 GA 優化...")

    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    states = torch.tensor(states, dtype=torch.long, device=device)

    # 初始化族群
    pop = (torch.rand((population_size, n_states), device=device) - 0.5) * 2  # [-1, 1]

    for gen in range(generations):
        sharpe = evaluate_population(pop, states, returns)

        # 選 elite
        topk = sharpe.topk(k=population_size // 2)
        elite = pop[topk.indices]

        # 向量化 mutation（高效）
        mutation = torch.randn_like(elite) * 0.1
        children = elite + mutation
        children = children.clamp(-1, 1)

        # 新族群
        pop = torch.cat([elite, children], dim=0)

    # 最佳結果
    final_sharpe = evaluate_population(pop, states, returns)
    best_idx = final_sharpe.argmax().item()
    return pop[best_idx].detach().cpu().numpy()


# ======== 6. 主程式入口 ========
def run_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "BITSTAMP_BTCUSD,240more.csv")
    output_path = os.path.join(base_dir, "hmm_fsm_ga_result_summary.csv")

    df = pd.read_csv(input_path)
    df, features = preprocess(df)
    X = df[features].values
    returns = df['close'].pct_change().fillna(0).values

    results = []

    for n_states in range(2, 1001):  # 2 到 1000
        print(f"🚀 正在訓練 n_states = {n_states} ...")
        try:
            # 1. 模型訓練 & 策略產出
            hmm_model, states = train_hmm(X, n_states=n_states)
            best_weights = torch_ga_optimize(states, returns, n_states=n_states)
            final_returns = simulate_returns(states, best_weights, returns)

            # 2. 計算績效
            df['state'] = states
            df['strategy_return'] = final_returns
            sharpe = compute_fitness(best_weights, states, returns)[0]
            cumulative_return = (1 + pd.Series(final_returns)).cumprod().iloc[-1]
            buy_and_hold_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1

            # 3. 畫圖（加入 sharpe & bnh 回報率）
            plot_strategy_curve(df, n_states, os.path.join(base_dir, "result"), sharpe, buy_and_hold_return)

            # 4. 紀錄 summary
            results.append({
                "n_states": n_states,
                "sharpe_ratio": sharpe,
                "cumulative_return": cumulative_return,
                "buy_and_hold_return": buy_and_hold_return,
                "weights": best_weights
            })

            print(f" ✅ n_states = {n_states} 訓練完成！  Results: Sharpe Ratio = {sharpe:.4f}, Cumulative Return = {cumulative_return:.4f}, Buy & Hold Return = {buy_and_hold_return:.4f}")
        except Exception as e:
            print(f"⚠️ 發生錯誤 @ n_states = {n_states}：{e}")
            continue

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print("✅ 全部儲存完畢囉～等你來分析喵💕")
    plot_results(result_df, output_path)


# ======== 6. 使用範例 ========
if __name__ == "__main__":
    run_model()
