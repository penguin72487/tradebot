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
import time


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
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import copy

def eval_hmm_instance(X, returns, n_states, df_raw, base_dir, seed=None):
    try:
        # 確保每個子任務隨機不同
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        hmm_model, states = train_hmm(X, n_states=n_states)
        weights = torch_ga_optimize(states, returns, n_states=n_states)
        final_returns = simulate_returns(states, weights, returns)

        df = copy.deepcopy(df_raw)
        df['state'] = states
        df['strategy_return'] = final_returns
        sharpe = compute_fitness(weights, states, returns)[0]
        cumulative_return = (1 + pd.Series(final_returns)).cumprod().iloc[-1]
        buy_and_hold_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1

        return {
            'sharpe_ratio': sharpe,
            'cumulative_return': cumulative_return,
            'buy_and_hold_return': buy_and_hold_return,
            'states': states,
            'weights': weights,
            'df': df
        }
    except Exception as e:
        return {'sharpe_ratio': -999, 'error': str(e)}

def run_parallel_hmm_for_n_states(n_states=30, n_jobs=16):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "BITSTAMP_BTCUSD,240more.csv")
    df_raw = pd.read_csv(input_path)
    df, features = preprocess(df_raw)
    X = df[features].values
    returns = df['close'].pct_change().fillna(0).values

    results = []
    print(f"🚀 執行 n_states = {n_states}，平行跑 {n_jobs} 組 HMM ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用 {device} + torch.compile 進行 GA 優化...")
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(eval_hmm_instance, X, returns, n_states, df_raw, base_dir, seed=i) for i in range(n_jobs)]
        for future in futures:
            res = future.result()
            if 'error' in res:
                print(f"⚠️ 發生錯誤：{res['error']}")
            else:
                # print(f"✅ 完成一組 HMM，Sharpe = {res['sharpe_ratio']:.4f}")
                results.append(res)

    # 選出最佳 HMM 組合

    best = max(results, key=lambda x: x['sharpe_ratio'])
    print(f"🎯 最佳 HMM 組合 Sharpe = {best['sharpe_ratio']:.4f}, Cumulative Return = {best['cumulative_return']:.4f}, Buy & Hold Return = {best['buy_and_hold_return']:.4f}")

    # 畫圖與儲存
    plot_strategy_curve(best['df'], n_states, os.path.join(base_dir, "result_best_hmm"), best['sharpe_ratio'], best['buy_and_hold_return'])

    return best



# ======== 6. 使用範例 ========
if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    n_states = 1000  # 可以調整
    n_jobs = 15  # 可以調整
    for i in range(2, n_states):
        print(f"🚀 正在訓練 n_states = {i} ...")
        run_parallel_hmm_for_n_states(n_states=i, n_jobs=n_jobs)
