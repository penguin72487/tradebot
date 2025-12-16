import pandas as pd
import numpy as np
import os
from pathlib import Path
import qlib
from qlib.data.storage import FeatureStorage, CalendarStorage, InstrumentStorage

# ========= 使用者設定 =========

base_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(base_dir, "clean_csv_for_qlib", "BTCUSDT.csv")
OUTPUT_DIR = os.path.join(base_dir, "qlib_data")
SYMBOL = "BTCUSDT"
FREQ = "15min"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= Step 1: 讀 CSV + 時間欄位處理 =========
df = pd.read_csv(INPUT_CSV)

def _parse_time_column(df):
    if "time" in df.columns:
        series = df["time"]
        is_num = np.issubdtype(series.dtype, np.number)
        parsed = pd.to_datetime(series, unit="s" if is_num else None, errors="coerce")
        df = df.drop(columns=["time"])
        df["date"] = parsed
        return df
    if "timestamp" in df.columns:
        series = df["timestamp"]
        is_num = np.issubdtype(series.dtype, np.number)
        parsed = pd.to_datetime(series, unit="s" if is_num else None, errors="coerce")
        df = df.drop(columns=["timestamp"])
        df["date"] = parsed
        return df
    if "date" in df.columns:
        series = df["date"]
        is_num = np.issubdtype(series.dtype, np.number)
        parsed = pd.to_datetime(series, unit="s" if is_num else None, errors="coerce")
        df["date"] = parsed
        return df

    candidates = [c for c in df.columns if ("time" in c.lower() or "date" in c.lower())]
    if candidates:
        col = candidates[0]
        series = df[col]
        is_num = np.issubdtype(series.dtype, np.number)
        parsed = pd.to_datetime(series, unit="s" if is_num else None, errors="coerce")
        if col != "date":
            df = df.drop(columns=[col])
        df["date"] = parsed
        return df

    raise ValueError(f"No time/date column found. Available columns: {df.columns.tolist()}")

df = _parse_time_column(df)

if df["date"].isnull().any():
    raise ValueError("Some dates could not be parsed. Please check the CSV time/date column.")

# ========= 刪掉含 NaN 的整欄 feature =========
df = df.dropna(axis=1, how="any")

# ========= 加上 symbol / 排序欄位 / factor =========
df["symbol"] = SYMBOL
cols = ["symbol", "date"] + [c for c in df.columns if c not in ["symbol", "date"]]
df = df[cols]

# 加 factor（全部 1.0）
df["factor"] = 1.0

# ========= Step 2: 建 Calendar =========
calendar_dir = Path(OUTPUT_DIR) / "calendar" / FREQ
calendar_dir.mkdir(parents=True, exist_ok=True)
calendar = sorted(df["date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist())

use_calendar_store = True
try:
    # 很多版本的 CalendarStorage 是 CalendarStorage(root_dir)
    cal_store = CalendarStorage(calendar_dir)
except TypeError:
    # 有的版本是 CalendarStorage(freq, future, root_dir) or 其他順序
    try:
        cal_store = CalendarStorage(FREQ, False, calendar_dir)
    except Exception:
        cal_store = None
        use_calendar_store = False

if use_calendar_store and cal_store is not None:
    if hasattr(cal_store, "write"):
        cal_store.write(calendar)
    elif hasattr(cal_store, "save"):
        cal_store.save(calendar)
    elif hasattr(cal_store, "dump"):
        cal_store.dump(calendar)
    else:
        use_calendar_store = False

if not use_calendar_store:
    # fallback 純文字
    cal_path = calendar_dir / "calendar.txt"
    with open(cal_path, "w", encoding="utf-8") as fh:
        for line in calendar:
            fh.write(line + "\n")
    print(f"[Fallback] Wrote calendar file to: {cal_path}")

# ========= Step 3: 建 Instrument =========
inst_dir = Path(OUTPUT_DIR) / "instruments"
inst_dir.mkdir(parents=True, exist_ok=True)

use_inst_store = True
try:
    inst_store = InstrumentStorage(inst_dir)
except TypeError:
    try:
        inst_store = InstrumentStorage(inst_dir, FREQ)
    except Exception:
        inst_store = None
        use_inst_store = False

if use_inst_store and inst_store is not None:
    if hasattr(inst_store, "write"):
        inst_store.write([SYMBOL])
    elif hasattr(inst_store, "save"):
        inst_store.save([SYMBOL])
    else:
        use_inst_store = False

if not use_inst_store:
    inst_path = inst_dir / "instruments.txt"
    with open(inst_path, "w", encoding="utf-8") as fh:
        fh.write(SYMBOL + "\n")
    print(f"[Fallback] Wrote instrument file to: {inst_path}")

# ========= Step 4: 建 Feature Storage =========
features_root = Path(OUTPUT_DIR) / "features" / FREQ
features_root.mkdir(parents=True, exist_ok=True)

try:
    feature_store = FeatureStorage(root_dir=features_root)
    use_feature_store = True
except TypeError:
    try:
        feature_store = FeatureStorage(freq=FREQ, root_dir=features_root)
        use_feature_store = True
    except Exception:
        feature_store = None
        use_feature_store = False

for col in df.columns:
    if col in ["symbol", "date"]:
        continue
    series = pd.Series(df[col].values, index=df["date"])
    if use_feature_store and feature_store is not None:
        feature_store.write(f"{SYMBOL}/{col}", series)
    else:
        out_dir = features_root / SYMBOL
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{col}.pkl"
        series.to_pickle(out_path)

print("OK！Qlib 資料已產生：", OUTPUT_DIR)
print("有效 feature 欄位：", df.columns.tolist())
