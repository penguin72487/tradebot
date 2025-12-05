import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
    
INPUT_CSV = os.path.join(base_dir, "BYBIT_BTCUSDT15.csv")
OUTPUT_DIR = os.path.join(base_dir, "clean_csv_for_qlib")
SYMBOL = "BTCUSDT"   # 你可以改成任何 symbol

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 讀 CSV
df = pd.read_csv(INPUT_CSV)

# 2. time → date（Qlib 需要 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）
df["date"] = pd.to_datetime(df["time"], unit="s")
df = df.drop(columns=["time"])

# 3. 刪掉任何含 NaN 的整個欄位
df = df.dropna(axis=1, how="any")

# 4. 重新排序欄位：date 必須在最前面
cols = ["date"] + [c for c in df.columns if c != "date"]
df = df[cols]

# 5. 建議補上 factor 欄位（加密貨幣沒有除權息 → 全部填 1）
df["factor"] = 1.0

# 6. 儲存成 Qlib 單檔格式（每個 symbol 一檔）
output_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}.csv")
df.to_csv(output_path, index=False)

print(f"[OK] 清洗後檔案輸出到：{output_path}")
print(f"[INFO] 最終剩下 {len(df.columns)} 個 feature 欄位：")
print(df.columns.tolist())
