import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_cleaned_noname.csv')
base_dir = os.path.dirname(file_path)
df = pd.read_csv(file_path)

# 預處理
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100

# 特徵欄位
all_features = df.drop(columns=['stock_id', 'year_month', 'year', 'return', 'return_label']) \
                 .select_dtypes(include=[np.number]).columns.tolist()

# 篩選完整年份
years = sorted(df['year'].unique())[:-1]
df = df[df['year'].isin(years)]

# 回測策略：回傳不同 TopN 組合的策略報酬序列
def backtest_strategy(df, selected_features):
    strategy_returns = {n: {'long': [], 'short': [], 'long_short': []} for n in [1, 10, 20, 30]}
    for i in range(len(years) - 1):
        train_years = years[:i + 1]
        test_year = years[i + 1]

        train_df = df[df['year'].isin(train_years)]
        test_df = df[df['year'] == test_year]

        X_train = train_df[selected_features]
        y_train = train_df['return']
        X_test = test_df[selected_features]
        y_test = test_df['return']

        model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=4, eta=0.1, n_estimators=300)
        model.fit(X_train, y_train)

        test_df = test_df.copy()
        test_df['predicted_return'] = model.predict(X_test)
        test_df['true_return'] = y_test

        for n in [1, 10, 20, 30]:
            top_n = test_df.nlargest(n, 'predicted_return')
            bottom_n = test_df.nsmallest(n, 'predicted_return')

            long_return = top_n['true_return'].mean()
            short_return = -bottom_n['true_return'].mean()
            long_short = (long_return + short_return) / 2

            strategy_returns[n]['long'].append(long_return)
            strategy_returns[n]['short'].append(short_return)
            strategy_returns[n]['long_short'].append(long_short)

    return strategy_returns

# --- GA 設定 ---
from joblib import Parallel, delayed
from tqdm import tqdm

population_size = 64
num_generations = 50
mutation_rate = 0.2
num_features = len(all_features)

population = np.random.randint(0, 2, size=(population_size, num_features))

# 平行化個體評估
def evaluate_individual(individual):
    selected = [f for i, f in enumerate(all_features) if individual[i] == 1]
    if not selected:
        return 0, None
    try:
        result = backtest_strategy(df, selected)
        best_cumret = max(
            (1 + pd.Series(result[n][k])).cumprod().iloc[-1]
            for n in [1, 10, 20, 30]
            for k in ['long', 'short', 'long_short']
        )
        return best_cumret, result
    except Exception as e:
        print(f"❌ Error with features {selected}: {str(e)}")
        return 0, None

# 用多核心平行跑一整群個體
def evaluate_population(population):
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_individual)(ind) for ind in tqdm(population, desc="Evaluating")
    )
    fitness_scores, strategy_history = zip(*results)
    return np.array(fitness_scores), list(strategy_history)

# GA 主流程
best_score = -np.inf
best_features = None
best_strategies = None

no_improvement_count = 0
threshold = 5  # 停止條件：連續5代沒有改進
delta = 0.001  # 改進幅度太小也算沒改進

for gen in range(num_generations):
    fitness, all_strategies = evaluate_population(population)

    if np.max(fitness) - best_score <= delta:
        no_improvement_count += 1
        if no_improvement_count >= threshold:
            print(f"Stopping early at generation {gen+1} due to no improvement.")
            break
    else:
        no_improvement_count = 0

    best_idx = np.argmax(fitness)
    if fitness[best_idx] > best_score:
        best_score = fitness[best_idx]
        best_features = [f for i, f in enumerate(all_features) if population[best_idx][i] == 1]
        best_strategies = all_strategies[best_idx]

    # 選擇（Roulette wheel）
    prob = fitness / fitness.sum() if fitness.sum() > 0 else np.ones(population_size) / population_size
    selected = population[np.random.choice(population_size, size=population_size, p=prob)]

    # 交配（single-point crossover）
    next_gen = []
    for i in range(0, population_size, 2):
        p1 = selected[i]
        p2 = selected[(i + 1) % population_size]
        cp = np.random.randint(1, num_features)
        c1 = np.concatenate([p1[:cp], p2[cp:]])
        c2 = np.concatenate([p2[:cp], p1[cp:]])
        next_gen.extend([c1, c2])

    # 突變（bit flip）
    next_gen = np.array(next_gen)
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            mp = np.random.randint(num_features)
            next_gen[i][mp] = 1 - next_gen[i][mp]

    population = next_gen
    print(f"Generation {gen+1}: Best cumulative return = {best_score:.4f}")

# 繪圖：所有策略的累積報酬圖
def plot_strategies(strategies):
    plt.figure(figsize=(14, 8))
    best_label = ""
    best_cumret = -np.inf

    markers = ['o', 's', 'D']
    for i, n in enumerate([1, 10, 20, 30]):
        for kind in ['long', 'short', 'long_short']:
            returns = pd.Series(strategies[n][kind])
            cumret = (1 + returns).cumprod()
            label = f'{kind.capitalize()} Top {n}'
            plt.plot(years[1:], cumret, label=label, marker=markers[i % len(markers)])
            if cumret.iloc[-1] > best_cumret:
                best_cumret = cumret.iloc[-1]
                best_label = label


    plt.title(f'GA Feature Selection - Best Strategy: {best_label} ({best_cumret:.2f})')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.yscale('log')  # 使用對數刻度更好地顯示累積報酬
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'best_strategy_cumulative_returns.png'))
    plt.show()

# 輸出結果
print("\n✅ 最佳累積報酬率：", round(best_score, 4))
print("✅ 最佳特徵組合：", best_features)
plot_strategies(best_strategies)
