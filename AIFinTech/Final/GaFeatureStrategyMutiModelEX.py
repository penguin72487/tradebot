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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone  # 加在最上面

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'final_features.csv')
base_dir = os.path.dirname(file_path)
result_dir = os.path.join(base_dir, 'results_Ex')
os.makedirs(result_dir, exist_ok=True)  # 確保資料夾存在

df = pd.read_csv(file_path)

# 預處理
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100

df['current_return'] = df['return'].shift(1)  # 當前報酬率
df['current_return_label'] = (df['current_return'] > 0).astype(int)  # 當前報酬率標籤

# 特徵欄位

# 原始數值欄位（排除非數值或標籤）
base_features = df.drop(columns=['year_month', 'year', 'return', 'return_label']) \
                  .select_dtypes(include=[np.number]).columns.tolist()

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    Normalizer, PowerTransformer, QuantileTransformer
)
# 建立標準化版本
scalers = {
    'z': StandardScaler(),
    'minmax': MinMaxScaler(),
    'maxabs': MaxAbsScaler(),
    'robust': RobustScaler(),
    'l2norm': Normalizer(norm='l2'),
    'power': PowerTransformer(method='yeo-johnson'),
    'quantile': QuantileTransformer(output_distribution='normal', n_quantiles=100)
}

for name, scaler in scalers.items():
    try:
        scaled = scaler.fit_transform(df[base_features])
        scaled_df = pd.DataFrame(scaled, columns=[f"{col}_{name}" for col in base_features])
        df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print(f"⚠️ {name} scaling failed: {e}")

# # 加入變化率（差分比例變化）1~4階
# # 加入變化率（自訂前一筆為 0 的行為）
# for col in base_features:
#     for k in range(1, 5):
#         prev = df[col].shift(k)
#         curr = df[col]
#         # 如果 prev==0，讓它變成極小值以避免除以 0；或你可以自訂為 1.0
#         safe_prev = prev.replace(0, np.nan)
#         change = (curr - prev) / safe_prev
#         df[f"{col}_chg{k}"] = change.fillna(0)  # 也可以用 .fillna(1.0)


# 更新 all_features（選所有數值特徵，不含標籤）
exclude = ['year_month', 'year', 'return', 'return_label']
all_features = df.drop(columns=exclude).select_dtypes(include=[np.number]).columns.tolist()

# 篩選完整年份
years = sorted(df['year'].unique())[1:-1]
df = df[df['year'].isin(years)]
# 每年股票數量統計
yearly_stock_counts = df.groupby('year')['stock_id'].nunique()
print("每年股票數量：")
print(yearly_stock_counts)

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
    'HistGB' : HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        early_stopping=True
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
    )
}

param_spaces = {
    'Ridge': {
        'alpha': (1e-3, 1e8),  # 正則化參數
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
        'max_iter': (100, 10000),  # 最大迭代次數
    },
    'SVR': {
        'C': (0.1, 1000.0),              # 正則化參數
        'epsilon': (0.001, 1.0),        # 容許誤差
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto'],     # 核函數係數
        'shrinking': [True, False]
    },
    'KNN': {
        'n_neighbors': (1, 30),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': (10, 100),
        'p': (1, 2),  # 曼哈頓 or 歐式距離
        'metric': ['minkowski']
    },
    'ExtraTrees': {
        'n_estimators': (50, 500),
        'max_depth': (2, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'HistGB': {
        'max_iter': (100, 1000),
        'learning_rate': (0.01, 0.2),
        'max_depth': (2, 12),
        'l2_regularization': (0.0, 5.0),
        'max_bins': (128, 255),
        'early_stopping': [True, False]
    },
    'BayesianRidge': {
        'max_iter': (100, 500),
        'tol': (1e-6, 1e-2),
        'alpha_1': (1e-7, 1e-3),
        'alpha_2': (1e-7, 1e-3),
        'lambda_1': (1e-7, 1e-3),
        'lambda_2': (1e-7, 1e-3),
        'fit_intercept': [True, False],
        'compute_score': [True, False]
    },
    'Linear': {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False]
    },
    'RandomForest': {
        'n_estimators': (50, 500),
        'max_depth': (2, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'XGBoost': {
        'objective': ['reg:squarederror'],
        'n_estimators': (50, 500),
        'max_depth': (2, 12),
        'learning_rate': (0.01, 0.3),  # eta
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'gamma': (0, 5.0),
        'reg_alpha': (0.0, 5.0),
        'reg_lambda': (0.0, 5.0),
        'booster': ['gbtree', 'dart']
    },
    'CatBoost': {
        'iterations': (100, 1000),
        'learning_rate': (0.01, 0.3),
        'depth': (3, 10),
        'l2_leaf_reg': (1, 10),
        'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
        'bagging_temperature': (0.0, 1.0),
        'random_strength': (1e-9, 10.0),
        'loss_function': ['RMSE', 'MAE', 'Quantile', 'LogLinQuantile']
    }
}

def init_models_by_name (model_name, params):
    """
    根據模型名稱和參數初始化模型。
    """
    if model_name == 'XGBoost':
        model = xgb.XGBRegressor(**params)
    elif model_name == 'CatBoost':
        model = cb.CatBoostRegressor(verbose=0, **params)
    elif model_name == 'SVR':
        model = SVR(**params)
    elif model_name == 'Ridge':
        model = Ridge(**params)
    elif model_name == 'BayesianRidge':
        model = BayesianRidge(**params)
    elif model_name == 'Linear':
        model = LinearRegression(**params)
    elif model_name == 'KNN':
        model = KNeighborsRegressor(**params)
    elif model_name == 'ExtraTrees':
        model = ExtraTreesRegressor(**params)
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(**params)
    elif model_name == 'HistGB':
        model = HistGradientBoostingRegressor(**params)
    else:
        raise ValueError(f"未知模型：{model_name}")
    return model



# 回測策略：回傳不同 TopN 組合的策略報酬序列
def backtest_strategy(df, selected_features,model):

    strategy_returns = {n: {'long': [], 'short': [], 'long_short': []} for n in [10, 20, 30, 200]}
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

        for n in [10, 20, 30, 200]:
            top_n = test_df.nlargest(n, 'predicted_return')
            bottom_n = test_df.nsmallest(n, 'predicted_return')

            long_return = top_n['true_return'].mean()
            short_return = -bottom_n['true_return'].mean()
            long_short = (long_return + short_return) / 2

            strategy_returns[n]['long'].append(long_return)
            strategy_returns[n]['short'].append(short_return)
            strategy_returns[n]['long_short'].append(long_short)

    return strategy_returns

def plot_strategies(strategies, best_features, best_parameters, model_name='ridge'):
    plt.figure(figsize=(14, 11))
    best_label = ""
    best_cumret = -np.inf

    markers = {"long": 'o', "short": 'v', "long_short": 's'}
    line_styles = {"long": '-', "short": '--', "long_short": ':'}
    color_map = {'1': 'blue', '10': 'orange', '20': 'green', '30': 'red', '200': 'purple'}
    results_df = pd.DataFrame({'Year': years[1:]})  

    for n in [10, 20, 30, 200]:
        for kind in ['long', 'short', 'long_short']:
            returns = pd.Series(strategies[n][kind])
            cumret = (1 + returns).cumprod()
            label = f'{kind.capitalize()} Top {n}'

            plt.plot(years[1:], cumret, label=label,
                     marker=markers[kind],
                     linestyle=line_styles[kind],
                     color=color_map[str(n)])

            if cumret.iloc[-1] > best_cumret:
                best_cumret = cumret.iloc[-1]
                best_label = label

            results_df[label] = cumret.values

    # 主標題與圖例
    plt.title(f'{model_name} GA Optimization - Best Strategy: {best_label} ({best_cumret:.2f})')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # 加入特徵與參數說明
    feature_text = 'Best Features:\n' + ', '.join(best_features)
    param_text = 'Best Parameters:\n' + ', '.join([f'{k}={v}' for k, v in best_parameters.items()])

    plt.gcf().text(0.01, 0.01, feature_text, fontsize=10, va='bottom', ha='left', wrap=True)
    plt.gcf().text(0.01, 0.95, param_text, fontsize=10, va='top', ha='left', wrap=True)
    plt.tight_layout(rect=[0.01, 0.05, 1, 0.93])

    # 儲存圖與數據
    plt.savefig(os.path.join(result_dir, f'{model_name}_best_strategy_cumulative_returns.png'))
    plt.close()

    csv_path = os.path.join(result_dir, f'{model_name}_cumulative_returns.csv')
    results_df.to_csv(csv_path, index=False)



def backtest_cross_validation(df, selected_features, best_prameters, model_name='ridge'):
    results = []    

    model = init_models_by_name(model_name, best_prameters)

    for i in range(len(years) - 1):
        train_years = years[:i + 1]
        test_years = years[i + 1:]

        train_df = df[df['year'].isin(train_years)]
        strategy_returns = {n: {'long': [], 'short': [], 'long_short': []} for n in [10, 20, 30, 200]}

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

            for n in [10, 20, 30, 200]:
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

        for n in [10, 20, 30, 200]:
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
    for n in [10, 20, 30, 200]:
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

population_size = 1024
num_generations = 1000

num_features = len(all_features)

# population = np.random.randint(0, 2, size=(population_size, num_features))

# 平行化個體評估
def evaluate_individual(individual, model_name):
    num_params = len(param_spaces[model_name])
    selected = [f for i, f in enumerate(all_features) if individual[i] == 1]
    if not selected:
        return 0, None, None  # 沒選特徵直接淘汰

    try:
        # 解碼超參數（後半段）
        params = decode_params(param_spaces[model_name], individual[-num_params:])

        # 動態建模
        if model_name == 'XGBoost':
            model = xgb.XGBRegressor(**params)
        elif model_name == 'CatBoost':
            model = cb.CatBoostRegressor(verbose=0, **params)
        elif model_name == 'SVR':
            model = SVR(**params)
        elif model_name == 'Ridge':
            model = Ridge(**params)
        elif model_name == 'BayesianRidge':
            model = BayesianRidge(**params)
        elif model_name == 'Linear':
            model = LinearRegression(**params)
        elif model_name == 'KNN':
            model = KNeighborsRegressor(**params)
        elif model_name == 'ExtraTrees':
            model = ExtraTreesRegressor(**params)
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(**params)
        elif model_name == 'HistGB':
            model = HistGradientBoostingRegressor(**params)
        else:
            raise ValueError(f"未知模型：{model_name}")

        # 執行策略回測
        result = backtest_strategy(df, selected, model)
        best_cumret = max(
            (1 + pd.Series(result[n][k])).cumprod().iloc[-1]
            for n in [10, 20, 30]
            for k in ['long', 'short', 'long_short']
        )

        return best_cumret, result, selected, params

    except Exception as e:
        print(f"❌ Error in {model_name}: {str(e)}")
        return 0, None, None, None, None  # 評估失敗則返回 0 分數


# 用多核心平行跑一整群個體
def evaluate_population(population, model_name):
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_individual)(ind, model_name)
        for ind in tqdm(population, desc=f"Evaluating {model_name}")
    )
    fitness, strategy_history, selected_features_list, param_list = zip(*results)
    return np.array(fitness), list(strategy_history), list(selected_features_list), list(param_list)


def decode_params(param_space, gene_vector):
    param_vec = gene_vector[-len(param_space):]
    decoded = {}
    for i, (key, space) in enumerate(param_space.items()):
        val = param_vec[i]
        if isinstance(space, tuple):
            low, high = space
            decoded[key] = int(low + val * (high - low)) if isinstance(low, int) else low + val * (high - low)
        elif isinstance(space, list):
            idx = int(val * len(space)) % len(space)
            decoded[key] = space[idx]
    return decoded






for model_name, model in models.items():
    print(f"Training model: {model_name}")
    results = []
    cv_result = pd.DataFrame()
    # GA 主流程
    best_score = -np.inf
    best_features = None
    best_strategies = None

    no_improvement_count = 0
    threshold = 20  # 停止條件：連續5代沒有改進
    delta = 0.001  # 改進幅度太小也算沒改進
    init_mutation_rate = 0.2
    
    num_params = len(param_spaces[model_name])  # 超參數個數
    population = np.random.rand(population_size, num_features + num_params)
    # 前半段 0/1，後半段 0.0~1.0（需 decode 後套入超參數）
    population[:, :num_features] = (population[:, :num_features] > 0.5).astype(int)




    for gen in range(num_generations):
        fitness, all_strategies, selected_features_list, param_list = evaluate_population(population, model_name)


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
            best_features = selected_features_list[best_idx]
            best_params = param_list[best_idx]
            best_strategies = all_strategies[best_idx]

        # 選擇（Roulette wheel）
        # 確保 prob 是正確的機率分布
        prob = fitness / fitness.sum()  # 或其他 normalize 的方式
        population = np.array(population)  # 加這行，保證是 ndarray
        indices = np.random.choice(population_size, size=population_size, replace=True, p=prob)
        selected = population[indices]


        # 交配（single-point crossover）
        next_gen = []
        for i in range(0, population_size, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % population_size]
            cp = np.random.randint(1, num_features)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            next_gen.extend([c1, c2])

        # 動態突變率調整
        if no_improvement_count >= 5:
            mutation_rate = init_mutation_rate + ((1-init_mutation_rate) * (1 - no_improvement_count / threshold))
        else:
            mutation_rate = init_mutation_rate

        # # 突變（bit flip）
        # next_gen = np.array(next_gen)
        # for i in range(population_size):
        #     if np.random.rand() < mutation_rate:
        #         mp = np.random.randint(num_features)
        #         next_gen[i][mp] = 1 - next_gen[i][mp]

        # 多點突變（讓突變率平均分配到每個基因位點）
        for i in range(population_size):
            for j in range(num_features):
                if np.random.rand() < (mutation_rate / num_features):
                    next_gen[i][j] = 1 - next_gen[i][j]


        population = next_gen
        print(f"{model_name}Generation {gen+1}: Best cumulative return = {best_score:.4f}")
        print(f"Best features: {best_features}")
        print(f"Best parameters: {best_params}")
        cv_result = backtest_cross_validation(df, best_features, best_params, model_name)




        
    # 輸出結果
    print(f"\n✅ {model_name} 最佳累積報酬率：", round(best_score, 4))
    print(f"✅ {model_name} 最佳特徵組合：", best_features)
    #不挑特徵畫圖
    # plot_strategies(best_strategies, all_features, model_name+ ' (All Features)')
    # plot_crossval_results(cv_result, result_dir, model_name + ' (All Features)')
    # 挑特徵畫圖
    plot_strategies(best_strategies, best_features, best_params, model_name)
    # 儲存交叉驗證結果
    plot_crossval_results(cv_result, result_dir, model_name)
    # 儲存最佳策略
 


