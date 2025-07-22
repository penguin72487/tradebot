import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import random
import os
from joblib import Parallel, delayed


# 設定隨機種子（讓結果穩定）
np.random.seed(42)
random.seed(42)

file_path = os.path.join(os.path.dirname(__file__), 'BITSTAMP_BTCUSD,240more.csv')
base_dir = os.path.dirname(file_path)
result_dir = os.path.join(base_dir, 'results')
os.makedirs(result_dir, exist_ok=True)  # 確保資料夾存在

# 讀取價格資料，CSV 檔案需包含欄位 'time', 'close'
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values("time").reset_index(drop=True)
price = df['close'].values
import matplotlib.pyplot as plt

# 額外：畫出累積報酬圖
def plot_equity_curve(price, best_params):
    trigger_pct, take_profit_pct, max_adds, leverage, scale_factor = best_params
    usdt = 100.0
    balance = usdt
    init_pos = usdt * 0.1
    pos = 0.0
    avg_price = 0.0
    adds = 0
    equity_curve = []

    for i in range(1, len(price)):
        current = price[i]

        if pos == 0:
            pos = init_pos * leverage
            avg_price = current

        elif (current < avg_price * (1 - trigger_pct)) and adds < max_adds:
            adds += 1
            add_pos = init_pos * leverage
            pos += add_pos
            avg_price = (avg_price * (pos - add_pos) + current * add_pos) / pos

        elif current >= avg_price * (1 + take_profit_pct):
            balance += pos * (current / avg_price - 1)
            pos = 0.0
            avg_price = 0.0
            adds = 0

        total = balance
        if pos > 0:
            total += pos * (current / avg_price - 1)
        equity_curve.append(total)

    equity_curve = np.array(equity_curve)

    # Buy & Hold 策略
    bh = price[1:] / price[1] * usdt  # 用第二筆價格為基準

    # 畫圖
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'][1:], equity_curve, label="💼 Martingale Strategy")
    plt.plot(df['time'][1:], bh, label="📈 Buy & Hold", linestyle="--")
    plt.title("累積資產報酬曲線（vs Buy & Hold）", fontsize=16)
    plt.xlabel("時間")
    plt.ylabel("資產（USDT）")
    plt.yscale('log')  # 使用對數尺度顯示資產變化
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "equity_curve.png"))
    # plt.show()


# 馬丁格策略回測函數
def martingale_backtest(price, trigger_pct, take_profit_pct, max_adds, leverage, scale_factor=1.0):
    usdt = 10000.0

    max_uni = ((scale_factor ** max_adds) - 1) / (scale_factor - 1)

    init_pos = usdt / max_uni
    pos = 0.0
    avg_price = 0.0
    adds = 0
    equity_curve = []
    entry_price = []
    balance = usdt
    ep = 0.0

    for i in range(1, len(price)):
        current = price[i]

        # 初次建倉
        if pos == 0:
            pos = init_pos * leverage
            avg_price = current
            adds = 0
            entry_price = []
            ep = current
            for _ in range(max_adds):
                ep = ep * (1 - trigger_pct)
                entry_price.append(ep)

        # 檢查是否要觸發加倉（用 entry_price 判斷是否穿越）
        while adds < max_adds and current <= entry_price[adds]:
            add_pos = init_pos * (scale_factor ** adds) * leverage
            new_total_pos = add_pos
            avg_price = (avg_price * pos + current * add_pos) / new_total_pos
            pos = new_total_pos
            adds += 1

        # 達止盈條件平倉
        if pos > 0 and current >= avg_price * (1 + take_profit_pct):
            balance += pos * (current / avg_price - 1)
            pos = 0.0
            avg_price = 0.0
            adds = 0
            entry_price = []

        # 記錄資產變化
        total = balance
        if pos > 0:
            total += pos * (current / avg_price - 1)
        equity_curve.append(total)

    return np.array(equity_curve) if equity_curve else np.array([usdt])



# 計算適應度（越高越好）
def evaluate_individual(params):
    trigger_pct, take_profit_pct, max_adds, leverage, scale_factor = params
    equity = martingale_backtest(
        price,
        trigger_pct=trigger_pct,
        take_profit_pct=take_profit_pct,
        max_adds=int(max_adds),
        leverage=leverage,
        scale_factor=scale_factor
    )

    returns = np.diff(equity) / equity[:-1]
    if len(returns) == 0:
        return -np.inf  # 無交易，給很低評分

    # downside_returns = returns[returns < 0]
    downside_returns = returns
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
    sortino_ratio = np.mean(returns) / downside_std
    return sortino_ratio


# 初始化族群
def initialize_population(n):
    return [
        [
            random.uniform(1e-4, 0.5),   # trigger_pct
            random.uniform(1e-4, 1.0),   # take_profit_pct
            random.randint(1, 100),     # max_adds
            random.uniform(1e-4, 100),   # leverage
            random.uniform(0.5, 4.0),    # scale_factor ✅
        ]
        for _ in range(n)
    ]


# 遺傳演算法主流程
def run_ga(generations=1000, pop_size=1024):
    
    threshold = 300
    delta = 0.001
    init_mutation_rate = 0.2

    checkpoint_path = os.path.join(result_dir, 'best_params.npy')
    if os.path.exists(checkpoint_path):
        best_params = np.load(checkpoint_path, allow_pickle=True)
        print(f"已載入先前最佳參數：{best_params}")
        return best_params
    else:
        print("未找到先前最佳參數，開始新的遺傳演算法搜尋")
        best_params = None
        best_score = -np.inf
        no_improvement = 0



    population = initialize_population(pop_size)
    for gen in range(generations):


        if no_improvement >= threshold:
            print(f"已達到 {threshold} 代無改善，提前終止")
            break

        fitness = np.array(Parallel(n_jobs=-1)(
            delayed(evaluate_individual)(ind) for ind in population
        ))

        if np.max(fitness) > best_score + delta:
            best_score = np.max(fitness)
            best_params = population[np.argmax(fitness)]
            no_improvement = 0
            print(f"🚀第 {gen+1} 表現{best_score}代找到新最佳參數：下跌{best_params[0]*100:.2f}%，止盈{best_params[1]*100:.2f}%，最大加倉次數{int(best_params[2])}，槓桿{best_params[3]:.2f}x")
            np.save(checkpoint_path, best_params)
            print(f"已儲存最佳參數至 {checkpoint_path}")
            plot_equity_curve(price, best_params)
            print("📈 馬丁格策略回測完成，已儲存結果圖表！")
        else:
            no_improvement += 1
            print(f"第 {gen+1} 代，已經 {no_improvement} 代無改善，最佳資產仍為 ${best_score:.2f}")

        top_indices = np.argsort(fitness)[-pop_size//2:]
        parents = [population[i] for i in top_indices]

        # 交配+突變產生新個體

        # 突變
        mutation_rate = init_mutation_rate + (1 - init_mutation_rate) * (gen / generations)

        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = [
                np.mean([p1[0], p2[0]]),  # 跌幅觸發加倉%
                np.mean([p1[1], p2[1]]),  # 止盈%
                int(np.mean([p1[2], p2[2]])),  # 最大加倉次數
                np.mean([p1[3], p2[3]]),  # 槓桿
                np.mean([p1[4], p2[4]]),  # scale_factor
            ]
            # 突變
            if random.random() < mutation_rate:
                child[random.randint(0, 4)] *= random.uniform(0.8, 1.2)
            children.append(child)

        population = children
        best_score = max(fitness)
        print(f"第 {gen+1} 代：最佳資產 = ${best_score:.2f} 🪙")

    # 找出最強個體
    best_index = np.argmax([evaluate_individual(ind) for ind in population])
    return population[best_index]

# 主程式
# 主程式最後呼叫
if __name__ == "__main__":
    best = run_ga()
    print("\\n🌟 找到最棒的馬丁格參數了喵：")
    print(f"➤ 跌幅觸發加倉: {best[0]*100:.2f}%")
    print(f"➤ 止盈百分比: {best[1]*100:.2f}%")
    print(f"➤ 最大加倉次數: {int(best[2])}")
    print(f"➤ 槓桿倍數: {best[3]:.2f}x")
    print(f"➤ 加倉倍率: {best[4]:.2f}x")


    # 畫圖
    plot_equity_curve(price, best)
    print("📈 馬丁格策略回測完成，已儲存結果圖表！")
