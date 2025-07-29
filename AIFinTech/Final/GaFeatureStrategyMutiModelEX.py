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
df = pd.read_csv(file_path, dtype={'year_month': str})


df['year_month'] = df['year_month'].astype(str)

# 看看 year_month 欄位中不是 6 碼數字的有哪些
invalid = df[~df['year_month'].astype(str).str.match(r'^\d{6}$')]

if not invalid.empty:
    print("⚠️ 發現以下 year_month 欄位格式異常的資料：")
    print(invalid[['stock_id', 'year_month']].head(10))  # 先印前10筆
else:
    print("✅ 所有 year_month 格式正常喵～")
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100

df['current_return'] = df['return'].shift(1)  # 當前報酬率
df['current_return_label'] = (df['current_return'] > 0).astype(int)  # 當前報酬率標籤

# 特徵欄位

# 原始數值欄位（排除非數值或標籤）
base_features = df.drop(columns=['year_month', 'year', 'return', 'return_label']) \
                  .select_dtypes(include=[np.number]).columns.tolist()

# 加入變化率（差分比例變化）1~4階
# 加入變化率（自訂前一筆為 0 的行為）
chg_features = {}

for col in base_features:
    for k in range(1, 7):
        prev = df[col].shift(k)
        safe_prev = prev.replace(0, np.nan)
        change = (df[col] - prev) / safe_prev
        chg_features[f"{col}_chg{k}"] = change.fillna(0)

# 一次性加入所有新欄位，避免 fragmentation
chg_df = pd.DataFrame(chg_features)
df = pd.concat([df.reset_index(drop=True), chg_df.reset_index(drop=True)], axis=1)

# 如果你後面要繼續處理 df，也可以去除 fragmentation
df = df.copy()  # 去除 internal block fragmentation



changed_features = df.drop(columns=['year_month', 'year', 'return', 'return_label']) \
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
    # 'l2norm': Normalizer(norm='l2'),
    'power': PowerTransformer(method='yeo-johnson'),
    'quantile': QuantileTransformer(output_distribution='normal', n_quantiles=100)
}
df = df.sort_values(by='year').reset_index(drop=True)

# 只保留有 year 欄位的 row
years = sorted(df['year'].unique())

# 建立每一種 scaler 的滾動標準化結果
for name, scaler_obj in scalers.items():
    print(f"🔧 正在處理標準化方式：{name} ...")
    transformed_list = []
    
    for current_year in years:
        # 取得當前年以前的資料當作 fit 的 base
        fit_data = df[df['year'] <= current_year][changed_features]

        try:
            scaler = clone(scaler_obj)
            scaler.fit(fit_data)

            # transform 當年當下的資料
            current_data = df[df['year'] == current_year][changed_features]
            transformed = scaler.transform(current_data)

            transformed_df = pd.DataFrame(
                transformed,
                columns=[f"{col}_{name}" for col in changed_features],
                index=current_data.index
            )
            transformed_list.append(transformed_df)

        except Exception as e:
            print(f"⚠️ {name} 在 year={current_year} 標準化失敗：{e}")
            continue

    # 合併所有年度的標準化結果
    all_transformed = pd.concat(transformed_list).sort_index()
    df = pd.concat([df, all_transformed], axis=1)

print("✅ 滾動式標準化完成囉～避免未來資訊喵♡")

# 更新 all_features（選所有數值特徵，不含標籤）
exclude = ['year_month', 'year', 'return', 'return_label']
all_features = df.drop(columns=exclude).select_dtypes(include=[np.number]).columns.tolist()

# 篩選完整年份
df_2024 = df[df['year'] == 2024] 
years = sorted(df['year'].unique())[1:-1]
df = df[df['year'].isin(years)]
# 每年股票數量統計
yearly_stock_counts = df.groupby('year')['stock_id'].nunique()
print("每年股票數量：")
print(yearly_stock_counts)

# 模型集合
models = {
    # 'Ridge': Ridge(
    #     alpha=10.0,
    #     fit_intercept=True,
    #     solver='auto'
    # ),
    # 'KNN': KNeighborsRegressor(
    #     n_neighbors=7,
    #     weights='distance',
    #     algorithm='auto',
    #     leaf_size=20,
    #     p=2,
    #     metric='minkowski'
    # ),
    # 'ExtraTrees': ExtraTreesRegressor(
    #     n_estimators=300,
    #     max_depth=8,
    #     min_samples_split=5,
    #     min_samples_leaf=3,
    #     max_features='sqrt',
    #     bootstrap=True
    # ),
    # 'HistGB' : HistGradientBoostingRegressor(
    #     max_iter=300,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     l2_regularization=0.1,
    #     early_stopping=True
    # ),
    'BayesianRidge': BayesianRidge(
        max_iter=300,
        tol=1e-4
    ),
    # 'Linear': LinearRegression(
    #     fit_intercept=True,
    #     copy_X=True
    # ),
    # 'RandomForest': RandomForestRegressor(
    #     n_estimators=300,
    #     max_depth=8,
    #     min_samples_split=5,
    #     min_samples_leaf=3,
    #     max_features='sqrt',
    #     bootstrap=True
    # ),
    # 'SVR': SVR(
    #     kernel='rbf',
    #     C=1.0,
    #     epsilon=0.01
    # ),
    # 'XGBoost': xgb.XGBRegressor(
    #     objective='reg:squarederror',
    #     max_depth=4, 
    #     eta=0.1, 
    #     n_estimators=300
    # ),
    # 'CatBoost': cb.CatBoostRegressor(
    #     iterations=500,
    #     learning_rate=0.03,
    #     depth=6,
    #     l2_leaf_reg=3,
    #     loss_function='RMSE',
    #     verbose=0
    # )
}

param_spaces = {
    'Ridge': {
        'alpha': (1e-3, 1e8),  # 正則化參數
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr','sparse_cg', 'sag', 'saga'],
        'max_iter': (100, 1e8),  # 最大迭代次數
    },
    'SVR': {
        'C': (1e-3, 1e6),              # 正則化參數
        'epsilon': (0.001, 1.0),        # 容許誤差
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto'],     # 核函數係數
        'shrinking': [True, False]
    },
    'KNN': {
        'n_neighbors': (1, 200),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': (10, 1e8),
        'p': (1, 2),  # 曼哈頓 or 歐式距離
        'metric': ['minkowski']
    },
    'ExtraTrees': {
        'n_estimators': (50, 500),
        'max_depth': (2, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['sqrt', 'log2'],
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
        'max_iter': (100, 3000),
        'tol': (1e-6, 1),
        'alpha_1': (1e-7, 1e8),
        'alpha_2': (1e-7, 1e8),
        'lambda_1': (1e-7, 1e8),
        'lambda_2': (1e-7, 1e8),
        'fit_intercept': [True, False],
        'compute_score': [True, False],
        'copy_X': [True, False],
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
        'max_features': ['sqrt', 'log2'],
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

population_size =1024
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
        model = init_models_by_name(model_name, params)

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
def safe_evaluate(ind, model_name):
    try:
        return evaluate_individual(ind, model_name)
    except Exception as e:
        print(f"❌ 個體評估失敗：model={model_name}, ind={ind}, error={e}")
        # 回傳預設值：fitness -inf、空策略、空特徵、空參數（你也可以自訂）
        return -np.inf, None, None, None

def evaluate_population(population, model_name):
    results = Parallel(n_jobs=-1)(  # 建議先限制平行數量，例如4
        delayed(safe_evaluate)(ind, model_name)
        for ind in tqdm(population, desc=f"Evaluating {model_name}")
    )

    # 過濾掉錯誤的結果（你也可以選擇保留再後續處理）
    valid_results = [r for r in results if r[1] is not None]

    if not valid_results:
        raise RuntimeError("😭 所有個體評估都失敗喵... 沒有有效結果")

    fitness, strategy_history, selected_features_list, param_list = zip(*valid_results)
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


import pickle

def save_checkpoint(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

for model_name, model in models.items():
    print(f"Training model: {model_name}")
    checkpoint_path = os.path.join(result_dir, f"{model_name}_checkpoint.pkl")
    cv_result = pd.DataFrame()

    num_params = len(param_spaces[model_name])
    start_generation = 0
    init_density = 0.025

    if os.path.exists(checkpoint_path):
        
        checkpoint = load_checkpoint(checkpoint_path)
        population = checkpoint['population']
        best_score = checkpoint['best_score']
        best_features = checkpoint['best_features']
        best_params = checkpoint['best_params']
        best_strategies = checkpoint['best_strategies']
        no_improvement_count = checkpoint['no_improvement_count']
        start_generation = checkpoint['generation']
        print(f"🔄 從 checkpoint 恢復: {checkpoint_path} generation: {checkpoint['generation']}")
    else:
        print(f"🆕 開始新的訓練: {model_name}")
        population = np.random.rand(population_size, num_features + num_params)
        population[:, :num_features] = (population[:, :num_features] < init_density).astype(int)
        best_score = -np.inf
        best_features = None
        best_strategies = None
        best_params = None
        no_improvement_count = 0

    threshold = 300
    delta = 0.001
    init_mutation_rate = 0.2

    for gen in range(start_generation, num_generations):
        if no_improvement_count >= threshold:
            print(f"Stopping early at generation {gen+1} due to no improvement.")
            break
        fitness, all_strategies, selected_features_list, param_list = evaluate_population(population, model_name)

        if np.max(fitness) - best_score <= delta:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_score:
            old_best_features = best_features if best_features is not None else []
            new_best_features = selected_features_list[best_idx]

            # 計算特徵變化
            added_features = set(new_best_features) - set(old_best_features)
            removed_features = set(old_best_features) - set(new_best_features)
            
            # 打印特徵變化
            if added_features:
                print(f"+ Added {len(added_features)} features: {', '.join(sorted(added_features))}")
            if removed_features:
                print(f"- Removed {len(removed_features)} features: {', '.join(sorted(removed_features))}")
            best_score = fitness[best_idx]
            best_features = new_best_features
            best_params = param_list[best_idx]
            best_strategies = all_strategies[best_idx]

        # 選擇
        prob = fitness / fitness.sum()
        indices = np.random.choice(population_size, size=population_size, replace=True, p=prob)
        selected = population[indices]

        # 交配
        next_gen = []
        for i in range(0, population_size, 2):
            p1 = selected[i]
            p2 = selected[(i + 1) % population_size]
            cp = np.random.randint(1, num_features)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
            next_gen.extend([c1, c2])

        # 突變
        if no_improvement_count >= 100:
            mutation_rate = 1 + no_improvement_count
        elif no_improvement_count >=20:
            mutation_rate = 1 + no_improvement_count / 20
        elif no_improvement_count >= 5:
            mutation_rate = init_mutation_rate + ((0.8 - init_mutation_rate) * (1 - no_improvement_count / 20))
        else:
            mutation_rate = init_mutation_rate

        if mutation_rate/num_features >=1:
            break

        for i in range(population_size):
            for j in range(num_features):
                if np.random.rand() < (mutation_rate / num_features):
                    next_gen[i][j] = 1 - next_gen[i][j]

        population = np.array(next_gen)

        print(f"{model_name} Generation {gen+1}: Best cumulative return = {best_score:.4f}")
        print(f"number of All features: {num_features} no_improvement_count: {no_improvement_count} mutation_rate: {mutation_rate:.4f}")
        print(f"Best features: {best_features} with {len(best_features)} features")
        print(f"Best parameters: {best_params}")


        # 👉 每一代都存 checkpoint
        checkpoint_data = {
            'population': population,
            'best_score': best_score,
            'best_features': best_features,
            'best_params': best_params,
            'best_strategies': best_strategies,
            'no_improvement_count': no_improvement_count,
            'generation': gen + 1
        }
        save_checkpoint(checkpoint_path, checkpoint_data)

        # 可以選擇不要每一代都做交叉驗證（會很慢），必要時再開啟
        cv_result = backtest_cross_validation(df, best_features, best_params, model_name)
        plot_strategies(best_strategies, best_features, best_params, model_name)
        plot_crossval_results(cv_result, result_dir, model_name)
        #用模型跑2024的資料
        test_df = df_2024
        if not test_df.empty:
            X_test = test_df[best_features]
            y_test = test_df['return']
            model_clone = clone(model)
            model_clone.set_params(**best_params)
            model_clone.fit(df[df['year'] < 2024][best_features], df[df['year'] < 2024]['return'])
            test_df['predicted_return'] = model_clone.predict(X_test)
            test_df['true_return'] = y_test
            top_10 = test_df.nlargest(10, 'predicted_return')
            top_20 = test_df.nlargest(20, 'predicted_return')
            top_30 = test_df.nlargest(30, 'predicted_return')
            bottom_10 = test_df.nsmallest(10, 'predicted_return')
            bottom_20 = test_df.nsmallest(20, 'predicted_return')
            bottom_30 = test_df.nsmallest(30, 'predicted_return')

            print(f"2025年YTD Top 10 預測報酬率：{top_10['predicted_return'].mean():.4f}, Top 20：{top_20['predicted_return'].mean():.4f}, Top 30：{top_30['predicted_return'].mean():.4f}")

            print(f"2025年YTD Top 10 實際報酬率：{top_10['true_return'].mean():.4f}, Top 20：{top_20['true_return'].mean():.4f}, Top 30：{top_30['true_return'].mean():.4f}")

            col_names = ['stock_id', 'year_month', 'predicted_return', 'true_return']
            # 保留預測和實際報酬率的小數點後4位
            top_Actually = test_df.sort_values(by='true_return', ascending=False).copy()
            top_Actually = top_Actually[col_names].round(4)
            top_predicted = test_df.sort_values(by='predicted_return', ascending=False).copy()
            top_predicted = top_predicted[col_names].round(4)
            top_Actually.to_csv(os.path.join(result_dir, f"{model_name}_2024_top_actual.csv"), index=False)
            top_predicted.to_csv(os.path.join(result_dir, f"{model_name}_2024_top_predicted.csv"), index=False)
            print(f"✅ 2024年 {model_name} 模型預測結果已儲存")
        else:
            print(f"⚠️ 2024年沒有資料可供測試 {model_name} 模型")
                
                
               

    print(f"\n✅ {model_name} 最佳累積報酬率：", round(best_score, 4))
    print(f"✅ {model_name} 最佳特徵組合：", best_features)

    # 最終做一次完整交叉驗證與圖表儲存
    cv_result = backtest_cross_validation(df, best_features, best_params, model_name)
    plot_strategies(best_strategies, best_features, best_params, model_name)
    plot_crossval_results(cv_result, result_dir, model_name)



    # 刪除 checkpoint（或你也可以保留）
    # os.remove(checkpoint_path)
