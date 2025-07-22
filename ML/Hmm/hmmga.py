from hmm import GaussianHMM
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    Normalizer, PowerTransformer, QuantileTransformer
)
# 檔案與資料夾設定
base_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
result_dir = os.path.join(base_dir, "result", f"{script_name}_result")
os.makedirs(result_dir, exist_ok=True)

input_path = os.path.join(base_dir, "BITSTAMP_BTCUSD,240more.csv")
output_path = os.path.join(result_dir, script_name + "_output.csv")

# 讀資料
df = pd.read_csv(input_path)
df['next_return'] = df['close'].pct_change().shift(-1).fillna(0)
# 🧠 選你想要的技術指標欄位（多一點沒關係）
df = df.dropna(axis=1)
features = df.drop(columns=['time','next_return']) \
                  .select_dtypes(include=[np.number]).columns.tolist()
# df = df.dropna(subset=features).reset_index(drop=True)
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
# 對每個特徵欄位進行標準化
print("🔍 正在標準化特徵欄位...")
for name, scaler in scalers.items():
    try:
        scaled = scaler.fit_transform(df[features])
        scaled_df = pd.DataFrame(scaled, columns=[f"{col}_{name}" for col in features])
        df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print(f"⚠️ {name} scaling failed: {e}")
# # 加入變化率（差分比例變化）1~4階
# # 加入變化率（自訂前一筆為 0 的行為）
# for col in features:
#     for k in range(1, 5):
#         prev = df[col].shift(k)
#         curr = df[col]
#         # 如果 prev==0，讓它變成極小值以避免除以 0；或你可以自訂為 1.0
#         safe_prev = prev.replace(0, np.nan)
#         change = (curr - prev) / safe_prev
#         df[f"{col}_chg{k}"] = change.fillna(0)  # 也可以用 .fillna(1.0)

select_features = [col for col in df.columns if col != 'time']
print(f"🐱‍👤 現在 df 中共有 {len(select_features)} 欄位：")
print(select_features)

print(f"📊 使用的特徵欄位：{select_features}")


# 🚀 初始化 GPU 上的 HMM 模型

# 回測參數
start_index = 100  # 至少要有 100 筆資料才能訓練第一輪
n_states = 5       # 固定隱藏狀態數量
positions_map = np.linspace(-1, 1, n_states)  # 狀態對應倉位（線性對應）

# 紀錄預測與報酬
pred_states = []
strategy_returns = []
buy_and_hold = []
# 在 for 迴圈外一次完成
full_scaled = scaler.fit_transform(df[features])

from tqdm import tqdm  # ✅ 記得先加這行引入 tqdm

for i in tqdm(range(start_index, len(df) - 1), desc="🐍 模型訓練進度"):
    # 訓練資料
    train_X = df.iloc[:i][features]
    train_scaled = train_scaled = full_scaled[:i]
    X_tensor = torch.tensor(train_scaled, dtype=torch.float32, device='cuda')

    # 訓練 HMM
    model = GaussianHMM(n_states=n_states, n_features=X_tensor.shape[1], device='cuda')
    states = model.viterbi(X_tensor).cpu().numpy()

    # 當前時間點的預測狀態
    current_X = df.iloc[i:i+1][features]
    current_scaled = full_scaled[i:i+1]
    current_tensor = torch.tensor(current_scaled, dtype=torch.float32, device='cuda')
    current_state = model.viterbi(current_tensor).item()
    pred_states.append(current_state)

    # 根據狀態對應倉位
    position = positions_map[current_state]
    ret = df.iloc[i]['next_return']
    strategy_returns.append(position * ret)

    # Buy & Hold（單純持有）
    buy_and_hold.append(ret)


# 計算累積報酬
strategy_cum_return = np.cumprod([1 + r for r in strategy_returns])
buy_cum_return = np.cumprod([1 + r for r in buy_and_hold])

# 畫圖
plt.figure(figsize=(12, 6))
plt.plot(strategy_cum_return, label='HMM 策略報酬', linewidth=2)
plt.plot(buy_cum_return, label='Buy & Hold', linestyle='--')
plt.title("📈 HMM 策略 vs Buy & Hold 累積報酬率")
plt.xlabel("時間")
plt.ylabel("累積報酬")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hmm_strategy_vs_hold.png", dpi=300)
plt.show()

print("✅ 策略執行完畢，圖片已儲存為 hmm_strategy_vs_hold.png 喵！")
