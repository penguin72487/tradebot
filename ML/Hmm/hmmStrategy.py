from hmm import GaussianHMM
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
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
# 加入變化率（差分比例變化）1~4階
# 加入變化率（自訂前一筆為 0 的行為）
for col in features:
    for k in range(1, 5):
        prev = df[col].shift(k)
        curr = df[col]
        # 如果 prev==0，讓它變成極小值以避免除以 0；或你可以自訂為 1.0
        safe_prev = prev.replace(0, np.nan)
        change = (curr - prev) / safe_prev
        df[f"{col}_chg{k}"] = change.fillna(0)  # 也可以用 .fillna(1.0)

select_features = [col for col in df.columns if col != 'time']
print(f"🐱‍👤 現在 df 中共有 {len(select_features)} 欄位：")
print(select_features)

print(f"📊 使用的特徵欄位：{select_features}")


# 🚀 初始化 GPU 上的 HMM 模型

for i in range(2, 31):
    # 🧪 標準化 & 放進 CUDA
    X_np = scaler.fit_transform(df[features].values)
    X = torch.tensor(X_np, dtype=torch.float32, device='cuda')

    model = GaussianHMM(n_states=i, n_features=X.shape[1], device='cuda')

    # 📈 跑 Viterbi 預測隱藏狀態
    states = model.viterbi(X)

    # 🐾 存回原始 DataFrame 並輸出
    df['hidden_state'] = states.cpu().numpy()
    df.to_csv(output_path, index=False)

    print(f"✅ 全部在 GPU 上完成！結果儲存到：{output_path}")
