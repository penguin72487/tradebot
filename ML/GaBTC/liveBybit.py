# -*- coding: utf-8 -*-
"""
liveBybit.py

Postgres(Price) -> 逐根輸入 -> 特徵工程(無展望偏誤; pandas_ta + TA-Lib)
-> 滾動標準化(用過去 STD_WIN) -> Parquet 快取(可續跑)
-> GA 訓練(Sharpe 或 幾何報酬) -> 儲存最佳模型(排除最近 unvalidated_gen)
-> 追上最新 -> 輪詢補K線並續訓
"""

import os, sys, time, json, math, warnings
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import pandas_ta as pta
import talib
from talib import abstract as tlab

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# ========== 參數 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result_live_bybit"); os.makedirs(RESULT_DIR, exist_ok=True)
FEATURE_PARQUET = os.path.join(RESULT_DIR, "features_240m.parquet")
CHECKPOINT_PATH = os.path.join(RESULT_DIR, "ga_checkpoint.npz")
ARCHIVE_PATH = os.path.join(RESULT_DIR, "models_archive.json")
LOG_DIR = os.path.join(RESULT_DIR, "logs"); os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "liveBybit.log")

# 日誌設定
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logging.info(f"Starting liveBybit at: {BASE_DIR}")

# ===== 執行模式（改這裡就好）=====
RUN_FOLLOW   = False   # True: 持續輪詢補K並續訓；False: 只跑一次
POLL_SECONDS = 30      # follow 輪詢秒數
GENS         = 100     # 單次/續訓要跑的 GA 代數；初次建議設較大（下方 N_GEN_BASE 也會用到）

# 特徵工程設定
STD_WIN = 2**10                 # 滾動標準化視窗 (1024)
change_rate_steps = 6           # 1~6次變化率   
MIN_HISTORY = 676               # 至少這麼多列才開始訓練

# GA 參數
POP_SIZE = 2**10            # 1024
N_GEN_BASE = 100            # 初次完整訓練代數（follow 有新K時用）
N_GEN_STEP = max(20, N_GEN_BASE // 5)  # follow 沒新K時短代續訓
ELITE_FRAC = 0.10
MUT_RATE = 0.12
MUT_SIGMA = 0.05
MAX_LEV = 5.0               # 槓桿上限
FEE = 0.000                 # 單邊費（含滑點）
SEED = 42
np.random.seed(SEED)

# ========= 交易品 =========
EXCHANGE = "bybit"
PRODUCT  = "perpetual"   # spot/future/perpetual
SYMBOL   = "BTCUSDT"
INTERVAL = 240           # 分鐘 (int)
ANN_FACTOR = math.sqrt(int(24*60/INTERVAL)*365)  # 240m: 一天約6根, 一年~2190根 → sqrt(年bar數)

# 模型選擇
fitness_metric = "sharpe"     # "sharpe" or "return"
unvalidated_gen = 7*6         # 避免過擬合：排除最近這些代的模型

# ========== DB 連線 ==========
load_dotenv()
DB_CONFIG = dict(
    dbname=os.getenv("DBNAME"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST", "127.0.0.1"),
    port=os.getenv("PORT", "5432"),
)

def db_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_fetch_data_from_db(EXCHANGE, PRODUCT, SYMBOL, INTERVAL):
    """
    讀取全量資料（方便確保滾動視窗正確）
    """
    df = fetch_new_data_from_db(EXCHANGE, PRODUCT, SYMBOL, INTERVAL, last_timestamp= datetime(1970,1,1,tzinfo=timezone.utc))
    return df

def fetch_new_data_from_db(EXCHANGE, PRODUCT, SYMBOL, INTERVAL, last_timestamp):
    """
    讀取比 last_timestamp 更新的資料
    """
    conn = db_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT open, high, low, close, volume, timestamp
        FROM price
        WHERE exchange = %s AND product = %s AND symbol = %s AND interval = %s
          AND timestamp > %s
        ORDER BY timestamp ASC
    """, (EXCHANGE, PRODUCT, SYMBOL, str(INTERVAL), last_timestamp))
    data = cur.fetchall()
    cur.close(); conn.close()
    logging.info(f"Fetched {len(data)} new rows after {last_timestamp}.")
    if not data:
        cols = ["open","high","low","close","volume","timestamp"]
        return pd.DataFrame(columns=cols).set_index("timestamp")

    df = pd.DataFrame(data, columns=["open","high","low","close","volume","timestamp"])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["current_returns"] = df["close"].pct_change().fillna(0.0)
    df["future_returns"]  = df["close"].pct_change().shift(-1).fillna(0.0)
    return df



df = init_fetch_data_from_db(EXCHANGE, PRODUCT, SYMBOL, INTERVAL)
# ========= 特徵工程（只用過去資料；窗長<=STD_WIN） =========
from collections import OrderedDict
import inspect

def _safe_name(prefix, name):
    return f"{prefix}_{name}".lower()

def _clip_len(val, max_len):
    try:
        v = int(val)
    except Exception:
        v = max_len
    return max(1, min(v, max_len))

def _align_df(out, prefix):
    if out is None:
        return None
    if isinstance(out, pd.Series):
        out = out.to_frame(_safe_name(prefix, out.name if out.name else "s"))
    elif isinstance(out, pd.DataFrame):
        # 防止沒有欄名
        cols = [c if isinstance(c, str) and len(str(c)) > 0 else f"c{i}" for i, c in enumerate(out.columns)]
        out.columns = [_safe_name(prefix, c) for c in cols]
    else:
        return None
    return out

def compute_Pandas_Ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    嘗試呼叫 pandas_ta 中所有「看起來需要 O/H/L/C/V」的函式。
    - 僅提供存在於 df 的對應欄位
    - 如函式包含 length/period 參數，會壓到 <= STD_WIN
    - 出錯跳過並記錄
    """
    logging.info("Computing pandas_ta features (best-effort, skip on error)...")
    feats = []
    tried, ok = 0, 0

    # 候選參數名（不同函式可能叫不一樣）
    price_arg_alias = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"]
    }

    # 先準備可用輸入
    inputs = {}
    for k, aliases in price_arg_alias.items():
        for a in aliases:
            if a in df.columns:
                inputs[k] = df[a]
                break
        if k not in inputs and k in df.columns:
            inputs[k] = df[k]

    # 動態找出 pandas_ta 裡可呼叫的函式
    for name in dir(pta):
        if name.startswith("_"): 
            continue
        func = getattr(pta, name)
        if not callable(func): 
            continue

        # 跳過類別與非技術類
        if inspect.isclass(func):
            continue

        # 只嘗試接受 O/H/L/C/V 任一參數的函式
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            continue

        params = sig.parameters
        has_price_input = any(p in params for p in ["open", "high", "low", "close", "volume", "o", "h", "l", "c", "v"])
        if not has_price_input:
            continue

        tried += 1
        kwargs = {}
        # 傳遞對應輸入（若該函式參數名是 o/h/l/c/v 也支援）
        for std_key, aliases in price_arg_alias.items():
            for a in aliases:
                if a in params and std_key in inputs:
                    kwargs[a] = inputs[std_key]

        # 常見窗長參數名稱（盡量不超過 STD_WIN）
        for wnd_name in ["length", "window", "timeperiod", "n", "fast", "slow", "long", "short", "lbp", "period"]:
            if wnd_name in params:
                kwargs[wnd_name] = _clip_len(params[wnd_name].default if params[wnd_name].default is not inspect._empty else 14, STD_WIN)

        # 有些函式有 append / mamode / talib 等參數；不主動開 talib 與 append
        if "append" in params:
            kwargs["append"] = False

        try:
            out = func(**kwargs)
            out = _align_df(out, f"pta_{name}")
            if out is not None:
                feats.append(out)
                ok += 1
        except Exception as e:
            logging.debug(f"[pandas_ta] skip {name}: {e}")

    if feats:
        feat_df = pd.concat(feats, axis=1)
        df = df.join(feat_df, how="left")
    logging.info(f"pandas_ta done. tried={tried}, ok={ok}, cols_now={df.shape[1]}")
    return df


def compute_TA_Lib_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    呼叫 TA-Lib 的所有函式（abstract）：
    - 輸入採用 'open','high','low','close','volume'
    - 將 timeperiod / fastperiod / slowperiod… 等窗長壓到 <= STD_WIN
    - 出錯即跳過並記錄
    """
    logging.info("Computing TA-Lib features (best-effort, skip on error)...")
    feats = []
    tried, ok = 0, 0

    inputs = {
        "open":   df["open"]   if "open"   in df.columns else None,
        "high":   df["high"]   if "high"   in df.columns else None,
        "low":    df["low"]    if "low"    in df.columns else None,
        "close":  df["close"]  if "close"  in df.columns else None,
        "volume": df["volume"] if "volume" in df.columns else None,
    }
    # 移除 None
    inputs = {k: v for k, v in inputs.items() if v is not None}

    # 沒有基本的 close 就不做
    if "close" not in inputs:
        logging.warning("TA-Lib skipped: 'close' not found.")
        return df

    for fname in talib.get_functions():
        tried += 1
        try:
            fn = tlab.Function(fname)
            # 調整參數內所有 period 值 <= STD_WIN
            params = fn.parameters
            adj_params = {}
            for k, v in params.items():
                if isinstance(v, (int, float)) and any(s in k for s in ["period", "window", "time", "fast", "slow", "short", "long"]):
                    adj_params[k] = _clip_len(v, STD_WIN)
                else:
                    adj_params[k] = v

            out = fn(inputs, **adj_params)
            out = _align_df(out, f"ta_{fname}")
            if out is not None:
                feats.append(out)
                ok += 1
        except Exception as e:
            logging.debug(f"[TA-Lib] skip {fname}: {e}")

    if feats:
        feat_df = pd.concat(feats, axis=1)
        df = df.join(feat_df, how="left")
    logging.info(f"TA-Lib done. tried={tried}, ok={ok}, cols_now={df.shape[1]}")
    return df


def compute_change_rate_features(df: pd.DataFrame, steps: int = 6) -> pd.DataFrame:
    """
    對所有數值欄做 1~steps 階變化率（百分比變化），僅用過去資料
    """
    logging.info(f"Adding change-rate features (1..{steps})...")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    chg_frames = []
    for k in range(1, steps + 1):
        # avoid FutureWarning by explicitly disabling fill_method padding
        chg = df[numeric_cols].pct_change(periods=k, fill_method=None)
        chg = chg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        chg.columns = [f"{c}_chg{k}" for c in numeric_cols]
        chg_frames.append(chg)
    if chg_frames:
        df = pd.concat([df] + chg_frames, axis=1)
    logging.info(f"Change-rate done. cols_now={df.shape[1]}")
    return df


# ===== 滾動標準化工具（全都只用現在以及過去） =====
def _roll_mean(s: pd.Series, win: int):
    return s.rolling(win, min_periods=max(2, win//8)).mean()

def _roll_std(s: pd.Series, win: int):
    return s.rolling(win, min_periods=max(2, win//8)).std(ddof=0)

def _roll_min(s: pd.Series, win: int):
    return s.rolling(win, min_periods=max(2, win//8)).min()

def _roll_max(s: pd.Series, win: int):
    return s.rolling(win, min_periods=max(2, win//8)).max()

def _roll_median(s: pd.Series, win: int):
    return s.rolling(win, min_periods=max(2, win//8)).median()

def _roll_q(s: pd.Series, win: int, q: float):
    return s.rolling(win, min_periods=max(2, win//8)).quantile(q)

def compute_Scaler_features(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """
    為所有數值欄計算多種「只用過去窗」的縮放版：
      - z：     (x - mean) / std
      - minmax: (x - min)  / (max - min)
      - maxabs:  x / max(|x|)
      - robust: (x - median) / IQR
      - tanh:   0.5 * (tanh(0.01 * z) + 1)
      - unit_vector: x / ||x|| 
    """
    logging.info(f"Adding rolling scalers (window={win})...")
    eps = 1e-12
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    z_frames, minmax_frames, maxabs_frames, robust_frames, tanh_frames, unit_frames = [], [], [], [], [], []

    for c in tqdm(numeric_cols, desc="rolling-scalers", leave=False):
        s = df[c].astype(float)

        mean = _roll_mean(s, win)
        std  = _roll_std(s, win)
        z    = (s - mean) / std.replace({0: np.nan})
        z    = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        z_frames.append(z.rename(f"{c}__z"))

        rmin = _roll_min(s, win); rmax = _roll_max(s, win)
        denom = (rmax - rmin).replace({0: np.nan})
        mm = ((s - rmin) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        minmax_frames.append(mm.rename(f"{c}__minmax"))

        mabs = s.abs().rolling(win, min_periods=max(2, win//8)).max().shift(1)
        ma = (s / mabs.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        maxabs_frames.append(ma.rename(f"{c}__maxabs"))

        med = _roll_median(s, win)
        q75 = _roll_q(s, win, 0.75)
        q25 = _roll_q(s, win, 0.25)
        iqr = (q75 - q25).replace({0: np.nan})
        rb = ((s - med) / iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        robust_frames.append(rb.rename(f"{c}__robust"))

        tanh = 0.5 * (np.tanh(0.01 * z) + 1.0)
        tanh_frames.append(pd.Series(tanh.values, index=z.index, name=f"{c}__tanh").astype(np.float32))

        # unit_vector: x / ||x|| where ||x|| is the L2 norm over the rolling window
        l2 = np.sqrt((s ** 2).rolling(win, min_periods=max(2, win//8)).sum())
        unit = (s / l2.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        unit_frames.append(unit.rename(f"{c}__unit_vector"))

    add_parts = []
    for part in (z_frames, minmax_frames, maxabs_frames, robust_frames, tanh_frames, unit_frames):
        if part:
            add_parts.append(pd.concat(part, axis=1))
    if add_parts:
        df = pd.concat([df] + add_parts, axis=1)

    logging.info(f"Scalers done. cols_now={df.shape[1]}")
    return df


# ====== 綜合計算：支援 parquet 恢復進度 ======
def compute_features_all(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    完整流程：
      1) 若 FEATURE_PARQUET 存在：讀舊特徵，找出新K線，串上尾段 STD_WIN 作為暖機重算
      2) 否則：由零開始計算
      3) 依序：pandas_ta → TA-Lib → 1~6階變化率 → 滾動縮放
      4) 存回 parquet
    """
    if os.path.exists(FEATURE_PARQUET):
        try:
            old = pd.read_parquet(FEATURE_PARQUET)
            old = old.sort_index()
            last_ts = old.index.max()
            logging.info(f"Found existing features parquet up to {last_ts}.")

            # 取得新資料（嚴格大於 last_ts）
            new_raw = df_raw[df_raw.index > last_ts]
            if new_raw.empty:
                logging.info("No new rows. Reusing existing features parquet.")
                return old

            # 暖機：舊資料尾端 + 新資料
            warm = df_raw[df_raw.index <= last_ts].tail(STD_WIN)
            chunk = pd.concat([warm, new_raw], axis=0)
            logging.info(f"Recompute window rows={len(chunk)} (warm {len(warm)} + new {len(new_raw)})")

            # 只對 chunk 算特徵
            chunk_feat = _compute_features_pipeline(chunk)

            # 去掉暖機部分再接到舊檔
            chunk_feat = chunk_feat[chunk_feat.index > last_ts]
            out = pd.concat([old, chunk_feat], axis=0)
            out.to_parquet(FEATURE_PARQUET)
            logging.info(f"Features appended to {FEATURE_PARQUET}. total_rows={len(out)}")
            return out
        except Exception as e:
            logging.warning(f"Failed to resume from parquet ({e}). Recomputing from scratch...")

    # 沒有舊檔或恢復失敗：全量重算
    out = _compute_features_pipeline(df_raw)
    out.to_parquet(FEATURE_PARQUET)
    logging.info(f"Features computed from scratch and saved to {FEATURE_PARQUET}. rows={len(out)}, cols={out.shape[1]}")
    return out

def _compute_features_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # 保留 future_returns（不做任何特徵工程），並從 df 中暫時移除
    future_series = None
    if "future_returns" in df.columns:
        future_series = df["future_returns"].copy()
        df = df.drop(columns=["future_returns"])

    # 先確保必要欄位型別
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # pandas_ta
    df = compute_Pandas_Ta_features(df)

    # TA-Lib
    df = compute_TA_Lib_features(df)

    # 1~6 階變化率
    df = compute_change_rate_features(df, steps=change_rate_steps)

    # 滾動縮放
    df = compute_Scaler_features(df, win=STD_WIN)

    # 收尾：無窮/NaN -> 0（或你想改成 dropna/subset）
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 把先前保留的 future_returns 放回來（保持原始值、不參與任何特徵工程）
    if future_series is not None:
        # 對齊 index 並放回 dataframe
        df["future_returns"] = future_series.reindex(df.index)

    return df


# ===== 主流程入口：產生/續算 parquet =====
def init_compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算所有特徵並存成 parquet；若 parquet 已存在且資料有新增，則恢復進度續算。
    """
    out = compute_features_all(df)
    return out

df = init_compute_features(df)
df.to_csv(os.path.join(RESULT_DIR, "features_240m.csv"))  # for debug
