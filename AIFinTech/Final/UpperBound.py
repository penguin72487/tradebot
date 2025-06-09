import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_cleaned_noname.csv')
df = pd.read_csv(file_path)

# 確保 year_month 是字串格式，且提取年份
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100
years = df['year'].unique()
years.sort()
# 去掉最後一年
years = years[:-1]  # 去掉最後一年，因為沒有後續數據來計算報酬率

# 找到每年的前後1, 10,20,30名的平均報酬

def calculate_upper_bound(df, year, top_n):
    year_data = df[df['year'] == year]
    if len(year_data) < top_n:
        return None  # 如果該年數據不足，返回 None
    top_stocks = year_data.nlargest(top_n, 'return')
    return top_stocks['return'].mean()

def calculate_lower_bound(df, year, top_n):
    year_data = df[df['year'] == year]
    if len(year_data) < top_n:
        return None  # 如果該年數據不足，返回 None
    bottom_stocks = year_data.nsmallest(top_n, 'return')
    return bottom_stocks['return'].mean()

top_n_list = [1, 10, 20, 30]



# 整理數據 
years_top_returns = {n: [] for n in top_n_list}
years_bottom_returns = {n: [] for n in top_n_list}
years_top_bottom_returns = {n: [] for n in top_n_list}
for year in years:
    for n in top_n_list:
        upper_bound = calculate_upper_bound(df, year, n)
        lower_bound = calculate_lower_bound(df, year, n)
        upper_lower_bound = calculate_upper_bound(df, year, n) - calculate_lower_bound(df, year, n)
        years_top_returns[n].append(upper_bound)
        years_bottom_returns[n].append(lower_bound)
        years_top_bottom_returns[n].append(upper_lower_bound)

# 輸出
print("每年前後1, 10, 20, 30名的平均報酬率:")
for n in top_n_list:
    print(f"前 {n} 名的平均報酬率: {years_top_returns[n]}")
    print(f"後 {n} 名的平均報酬率: {years_bottom_returns[n]}")
    print(f"前-後 {n} 名的平均報酬率差: {years_top_bottom_returns[n]}")


years_top_cum_returns = {n: [] for n in top_n_list}
years_bottom_cum_returns = {n: [] for n in top_n_list}
years_top_bottom_cum_returns = {n: [] for n in top_n_list}
# 計算每年的累積報酬率
for n in top_n_list:
    cum_return = 1.0  # 初始累積報酬率
    # years_top_cum_returns[n].append(0)  # 初始年累積報酬率為0
    for returns in years_top_returns[n]:
        cum_return*= (1 + returns)
        years_top_cum_returns[n].append(cum_return - 1)  # 減去1得到累積報酬率

    cum_return = 1.0  # 初始累積報酬率
    # years_bottom_cum_returns[n].append(0)  # 初始年累積報酬率為0
    for returns in years_bottom_returns[n]:
        cum_return *= (1 + -returns)
        years_bottom_cum_returns[n].append(cum_return - 1)  # 減去1得到累積報酬率
    cum_return = 1.0  # 初始累積報酬率
    # years_top_bottom_cum_returns[n].append(0)  # 初始年累積報酬率為0
    for returns in years_top_bottom_returns[n]:
        cum_return *= (1 + returns/2)
        years_top_bottom_cum_returns[n].append(cum_return - 1)


print("\n每年前後1, 10, 20, 30名的累積報酬率:")
for n in top_n_list:
    print(f"前 {n} 名的累積報酬率: {years_top_cum_returns[n]}")
    print(f"後 {n} 名的累積報酬率: {years_bottom_cum_returns[n]}")
    print(f"前-後 {n} 名的累積報酬率差: {years_top_bottom_cum_returns[n]}")

# 繪製圖表
plt.figure(figsize=(14, 8))
for n in top_n_list:
    plt.plot(years, years_top_cum_returns[n], marker='o', label=f'top {n} cum returns')
    plt.plot(years, years_bottom_cum_returns[n], marker='x', linestyle='--', label=f'bottom {n} cum returns')
    plt.plot(years, years_top_bottom_cum_returns[n], marker='s', linestyle=':', label=f'top-bottom {n} cum returns')
plt.title('Cumulative Returns for Top & Bottom N Stocks')
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.yscale('log')  # 使用對數刻度以便更好地顯示累積報酬率
plt.xticks(years, rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(file_path.replace('.csv', '_cumulative_returns.png'))
plt.show()

        
