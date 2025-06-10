import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt

# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_cleaned_noname.csv')
base_dir = os.path.dirname(file_path)
df = pd.read_csv(file_path)



# 預處理
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100

# feature engineering
features = df.drop(columns=['stock_id', 'year_month', 'year']) \
                 .select_dtypes(include=[np.number]).columns.tolist()
# Rank normalization Min-Max

# 對數值型欄位做 Min-Max 壓縮（除了 return 與 rank_score）
# norm_cols = [col for col in features if col in df.columns]
# df[norm_cols] = df[norm_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))



# 過濾資料，只使用完整年份
years = sorted(df['year'].unique())
years = years[:-1]  # 去掉最後一年
df = df[df['year'].isin(years)]

results = []
# 使用 XGBoost 進行排名預測

for i in range(len(years) - 1):
    train_years = years[:i+1]
    test_year = years[i+1]

    train_df = df[df['year'].isin(train_years)]
    test_df = df[df['year'] == test_year]

    X_train = train_df[features]
    y_train = train_df['return']
    X_test = test_df[features]
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
        long_short = (long_return + short_return)/2

        results.append({
            'year': test_year,
            'n': n,
            'long_return': long_return,
            'short_return': short_return,
            'long_short_return': long_short
        })

# 整理成 DataFrame
results_df = pd.DataFrame(results)

# 畫圖用累積報酬函數
def cumulative(series):
    return (1 + series).cumprod()

# 畫圖
plt.figure(figsize=(14, 8))
for n in [10, 20, 30]:
    subset = results_df[results_df['n'] == n]
    years = subset['year'].tolist()
    plt.plot(years, cumulative(subset['long_return']), label=f'Long Top {n}', marker='o')
    plt.plot(years, cumulative(subset['short_return']), label=f'Short Bottom {n}', marker='x')
    plt.plot(years, cumulative(subset['long_short_return']), label=f'Long+Short {n}', marker='s')

plt.title('XGBoost Ranking Strategy Cumulative Returns')
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.yscale('log')
# plt.yscale('linear')  # 使用線性刻度
plt.legend()
plt.grid(True)
plt.tight_layout()

# 儲存圖片
plt.savefig(base_dir + '/xgboost_rank_strategy_cumulative_returns.png')
plt.show()