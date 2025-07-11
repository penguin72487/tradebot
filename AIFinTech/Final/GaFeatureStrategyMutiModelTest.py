import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone  # 加在最上面
from sklearn.preprocessing import StandardScaler


# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_test_cleaned_utf8_final.csv')
base_dir = os.path.dirname(file_path)
result_dir = os.path.join(base_dir, 'results_Test')
os.makedirs(result_dir, exist_ok=True)  # 確保資料夾存在

df = pd.read_csv(file_path)

# 預處理
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100

df['current_return'] = df['return'].shift(1)  # 當前報酬率
df['current_return_label'] = (df['current_return'] > 0).astype(int)  # 當前報酬率標籤

# 特徵欄位
all_features = df.drop(columns=['stock_id', 'year_month', 'year','return','return_label']) \
                 .select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
df[all_features] = scaler.fit_transform(df[all_features])
# scaler = StandardScaler()
# df[all_features] = scaler.fit_transform(df[all_features])

# 篩選完整年份
years = sorted(df['year'].unique())[1:-1]
df = df[df['year'].isin(years)]

# 模型集合
models = {
    'Ridge': Ridge(
        alpha=10.0,
        fit_intercept=True,
        solver='auto'
    ),
    'SVR': SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.01
    ),
    'KNN': KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        algorithm='auto',
        leaf_size=20,
        p=2,
        metric='minkowski'
    ),
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True
    ),
    'BayesianRidge': BayesianRidge(
        max_iter=300,
        tol=1e-4
    ),
    'Linear': LinearRegression(
        fit_intercept=True,
        copy_X=True
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True
    ),

    'XGBoost': xgb.XGBRegressor(
        objective='reg:squarederror', 
        max_depth=4, 
        eta=0.1, 
        n_estimators=300
    ),
    'CatBoost': cb.CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        verbose=0
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15
    )
}


# 回測策略：回傳不同 TopN 組合的策略報酬序列
def backtest_strategy(df, selected_features,model):

    strategy_returns = {n: {'long': [], 'short': [], 'long_short': []} for n in [1, 10, 20, 30, 200]}
    for i in range(len(years) - 1):
        train_years = years[:i + 1]
        test_year = years[i + 1]

        train_df = df[df['year'].isin(train_years)]
        test_df = df[df['year'] == test_year]

        X_train = train_df[selected_features]
        y_train = train_df['return']
        X_test = test_df[selected_features]
        y_test = test_df['return']

        model_clone = clone(model)  # 每次都要 clone，避免被 overwrite
        model_clone.fit(X_train, y_train)

        test_df = test_df.copy()
        test_df['predicted_return'] = model_clone.predict(X_test)
        test_df['true_return'] = y_test

        for n in [1, 10, 20, 30, 200]:
            top_n = test_df.nlargest(n, 'predicted_return')
            bottom_n = test_df.nsmallest(n, 'predicted_return')

            long_return = top_n['true_return'].mean()
            short_return = -bottom_n['true_return'].mean()
            long_short = (long_return + short_return) / 2

            strategy_returns[n]['long'].append(long_return)
            strategy_returns[n]['short'].append(short_return)
            strategy_returns[n]['long_short'].append(long_short)

    return strategy_returns

def plot_strategies(strategies, best_features, model_name='XGBoost'):
    plt.figure(figsize=(14, 10))
    best_label = ""
    best_cumret = -np.inf

    markers = {"long": 'o', "short": 'v', "long_short": 's'}
    line_styles = {"long": '-', "short": '--', "long_short": ':'}
    color_map = {'1': 'blue', '10': 'orange', '20': 'green', '30': 'red', '200': 'purple'}
    results_df = pd.DataFrame({'Year': years[1:]})  

    for n in [1, 10, 20, 30, 200]:
        for kind in ['long', 'short', 'long_short']:
            returns = pd.Series(strategies[n][kind])
            cumret = (1 + returns).cumprod()
            label = f'{kind.capitalize()} Top {n}'

            # 畫圖
            plt.plot(years[1:], cumret, label=label, marker=markers[kind], linestyle=line_styles[kind], color=color_map[str(n)])

            # 儲存最好的策略名稱
            if cumret.iloc[-1] > best_cumret:
                best_cumret = cumret.iloc[-1]
                best_label = label

            # 加入結果到 DataFrame
            results_df[label] = cumret.values

    # 主標題與標籤
    plt.title(f'{model_name} GA Feature Selection - Best Strategy: {best_label} ({best_cumret:.2f})')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    feature_text = 'Best Features:\n' + ', '.join(best_features)
    plt.gcf().text(0.01, 0.01, feature_text, fontsize=10, va='bottom', ha='left', wrap=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 儲存圖片
    plt.savefig(os.path.join(result_dir, f'{model_name}_best_strategy_cumulative_returns.png'))
    plt.close()  # Close the figure to free memory

    # 儲存 CSV
    csv_path = os.path.join(result_dir, f'{model_name}_cumulative_returns.csv')
    results_df.to_csv(csv_path, index=False)


def backtest_cross_validation(df, selected_features, model):
    results = []

    for i in range(len(years) - 1):
        train_years = years[:i + 1]
        test_years = years[i + 1:]

        train_df = df[df['year'].isin(train_years)]
        strategy_returns = {n: {'long': [], 'short': [], 'long_short': []} for n in [1, 10, 20, 30, 200]}

        for test_year in test_years:
            test_df = df[df['year'] == test_year]
            if test_df.empty:
                continue

            X_train = train_df[selected_features]
            y_train = train_df['return']
            X_test = test_df[selected_features]
            y_test = test_df['return']

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)

            test_df = test_df.copy()
            test_df['predicted_return'] = model_clone.predict(X_test)
            test_df['true_return'] = y_test

            for n in [1, 10, 20, 30, 200]:
                top_n = test_df.nlargest(n, 'predicted_return')
                bottom_n = test_df.nsmallest(n, 'predicted_return')

                long_return = top_n['true_return'].mean()
                short_return = -bottom_n['true_return'].mean()
                long_short = (long_return + short_return) / 2

                strategy_returns[n]['long'].append(long_return)
                strategy_returns[n]['short'].append(short_return)
                strategy_returns[n]['long_short'].append(long_short)

        # 結果記錄
        result_row = {
            'TrainYears': f"{train_years[0]}-{train_years[-1]}",
            'TestYears': f"{test_years[0]}-{test_years[-1]}"
        }

        if len(test_years) == 0:
            continue

        test_year_range = test_years[-1] - test_years[0] + 1

        for n in [1, 10, 20, 30, 200]:
            for strategy_name in ['long', 'short', 'long_short']:
                series = pd.Series(strategy_returns[n][strategy_name])
                if series.empty:
                    result_row[f'Top{n}_{strategy_name}_Cumulative'] = np.nan
                    result_row[f'Top{n}_{strategy_name}_Annual'] = np.nan
                else:
                    cum = (1 + series).prod() - 1
                    ann = float('-1')  # 預設年化報酬率為 NaN
                    if 1 + cum > 0:
                        ann = (1 + cum) ** (1 / test_year_range) - 1
                    else:
                        ann = float('-1')  # 或設為 0 或其他值

                    result_row[f'Top{n}_{strategy_name}_Cumulative'] = round(cum, 4)
                    result_row[f'Top{n}_{strategy_name}_Annual'] = round(ann, 4)

        results.append(result_row)

    return pd.DataFrame(results)





def plot_crossval_results(result_df, base_dir='.', model_name='Model'):

   
    # 儲存 CSV
    csv_path = os.path.join(base_dir, f"{model_name}_crossval_results.csv")
    result_df.to_csv(csv_path, index=False)

    # 畫年化報酬率圖
    markers = {"long": 'o', "short": 'v', "long_short": 's'}
    line_styles = {"long": '-', "short": '--', "long_short": ':'}
    color_map = {'1': 'blue', '10': 'orange', '20': 'green', '30': 'red', '200': 'purple'}
    plt.figure(figsize=(14, 10))
    for n in [1, 10, 20, 30, 200]:
        for strategy in ['long', 'short', 'long_short']:
            col_name = f'Top{n}_{strategy}_Annual'
            if col_name in result_df.columns:
                plt.plot(result_df['TestYears'], result_df[col_name], 
                         label=f'Top {n} {strategy.capitalize()}', 
                         marker=markers[strategy], linestyle=line_styles[strategy],
                         color=color_map[str(n)])

    plt.title(f'{model_name} Cross-Validation: Annualized Return')
    plt.xlabel('Test Period')
    plt.ylabel('Annualized Return')
    # plt.yscale('log')  # 使用對數刻度以便更好地顯示累積報酬率
    plt.yscale('linear')  # 使用線性刻度
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    png_path = os.path.join(base_dir, f"{model_name}_crossval_annual_return.png")
    plt.savefig(png_path)
    plt.close()  # Close the figure to free memory








# --- GA 設定 ---
from joblib import Parallel, delayed
from tqdm import tqdm

population_size = 64
num_generations = 100
mutation_rate = 0.2
num_features = len(all_features)

population = np.random.randint(0, 2, size=(population_size, num_features))

# 平行化個體評估
def evaluate_individual(individual, model):
    selected = [f for i, f in enumerate(all_features) if individual[i] == 1]
    if not selected:
        return 0, None
    try:
        result = backtest_strategy(df, selected, model)
        best_cumret = max(
            (1 + pd.Series(result[n][k])).cumprod().iloc[-1]
            for n in [10, 20, 30]
            for k in ['long', 'short', 'long_short']
        )
        return best_cumret, result
    except Exception as e:
        print(f"❌ Error with features {selected}: {str(e)}")
        return 0, None

# 用多核心平行跑一整群個體
def evaluate_population(population, model):
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_individual)(ind, model) for ind in tqdm(population, desc="Evaluating")
    )
    fitness_scores, strategy_history = zip(*results)
    return np.array(fitness_scores), list(strategy_history)






for model_name, model in models.items():
    print(f"Training model: {model_name}")
    results = []
    cv_result = pd.DataFrame()
    # GA 主流程
    best_score = -np.inf
    best_features = None
    best_strategies = None

    no_improvement_count = 0
    threshold = 10  # 停止條件：連續5代沒有改進
    delta = 0.001  # 改進幅度太小也算沒改進


    for gen in range(num_generations):
        fitness, all_strategies = evaluate_population(population, model)

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
        print(f"Best features: {best_features}")
        cv_result = backtest_cross_validation(df, best_features, model)




        
    # 輸出結果
    print(f"\n✅ {model_name} 最佳累積報酬率：", round(best_score, 4))
    print(f"✅ {model_name} 最佳特徵組合：", best_features)
    #不挑特徵畫圖
    plot_strategies(best_strategies, all_features, model_name+ ' (All Features)')
    plot_crossval_results(cv_result, result_dir, model_name + ' (All Features)')
    # 挑特徵畫圖
    plot_strategies(best_strategies, best_features, model_name)
    # 儲存交叉驗證結果
    plot_crossval_results(cv_result, result_dir, model_name)
    # 儲存最佳策略
 


