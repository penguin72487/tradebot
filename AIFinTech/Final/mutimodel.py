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

# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_cleaned_noname.csv')
base_dir = os.path.dirname(file_path)
df = pd.read_csv(file_path)

# 預處理
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100

# 特徵欄位
# 在特徵工程那段重新開啟：


features = df.drop(columns=['stock_id', 'year_month', 'year', 'return', 'return_label']).select_dtypes(include=[np.number]).columns.tolist()
# Rank normalization Min-Max
# 對數值型欄位做 Min-Max 壓縮（除了 return 與 rank_score）
# norm_cols = [col for col in features if col in df.columns]
# df[norm_cols] = df[norm_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# 過濾資料，只使用完整年份
years = sorted(df['year'].unique())[:-1]
df = df[df['year'].isin(years)]

# 模型集合
models = {
    'XGBoost': xgb.XGBRegressor(
        objective='reg:squarederror', 
        max_depth=4, 
        eta=0.1, 
        n_estimators=300
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True
    ),
    'Ridge': Ridge(
        alpha=10.0,
        fit_intercept=True,
        solver='auto'
    ),
    'LightGBM': lgb.LGBMRegressor(
        max_depth=6,
        num_leaves=31,
        n_estimators=500,
        learning_rate=0.03,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    ),
    'CatBoost': cb.CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        verbose=0
    ),
    'Linear': LinearRegression(
        fit_intercept=True,
        copy_X=True
    ),
    'KNN': KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        algorithm='auto',
        leaf_size=20,
        p=2,
        metric='minkowski'
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



# 累積結果
all_results = []

# 每個模型做一次滾動訓練與預測
for model_name, model in models.items():
    results = []
    for i in range(len(years) - 1):
        train_years = years[:i+1]
        test_year = years[i+1]

        train_df = df[df['year'].isin(train_years)]
        test_df = df[df['year'] == test_year]

        X_train = train_df[features]
        y_train = train_df['return']
        X_test = test_df[features]
        y_test = test_df['return']

        model.fit(X_train, y_train)
        test_df = test_df.copy()
        test_df['predicted_return'] = model.predict(X_test)
        test_df['true_return'] = y_test

        for n in [1,10, 20, 30]:
            top_n = test_df.nlargest(n, 'predicted_return')
            bottom_n = test_df.nsmallest(n, 'predicted_return')

            long_return = top_n['true_return'].mean()
            short_return = -bottom_n['true_return'].mean()
            long_short = (long_return + short_return) / 2

            results.append({
                'model': model_name,
                'year': test_year,
                'n': n,
                'long_return': long_return,
                'short_return': short_return,
                'long_short_return': long_short
            })

    all_results.extend(results)

# 整理成 DataFrame
results_df = pd.DataFrame(all_results)

# 累積報酬函數
def cumulative(series):
    return (1 + series).cumprod()

# 畫圖
plt.figure(figsize=(16, 10))

markers = ['o', 's', 'D', '^', 'v', 'x', 'p', '*']
# 繪製每個模型的長期累積報酬
for i, (model_name, model) in enumerate(models.items()):
    for n in [10, 20, 30]:
        subset = results_df[(results_df['model'] == model_name) & (results_df['n'] == n)]
        if not subset.empty:
            years = subset['year'].tolist()
            cum_ret = cumulative(subset['long_return'])
            label = f'{model_name} Top{n} Long'
            plt.plot(years, cum_ret, label=label, marker=markers[i % len(markers)])

# for model_name in models.keys():
#     for n in [10, 20, 30]:
#         subset = results_df[(results_df['model'] == model_name) & (results_df['n'] == n)]
#         if not subset.empty:
#             years = subset['year'].tolist()
#             cum_ret = cumulative(subset['long_short_return'])
#             label = f'{model_name} Top{n} L+S'
#             plt.plot(years, cum_ret, label=label, marker='o')

plt.title('Cumulative Returns: Multiple Models (Long + Short Strategy)')
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
# plt.yscale('linear') # 使用線性刻度
plt.yscale('log')  # 使用對數刻度以便更好地顯示累積報酬率
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(base_dir + '/multi_model_strategy_comparison.png')
plt.show()

# 儲存結果
results_df.to_csv(base_dir + '/multi_model_strategy_results.csv', index=False)
