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
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # 熱補丁：讓 pandas_ta 能從 numpy 匯入到 NaN

import pandas_ta as pta
import talib
from talib import abstract as tlab

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from joblib import Parallel, delayed
from tqdm.auto import tqdm
# 放在 import torch 前面
# ⬇️ 放在任何 import torch / torch._inductor / triton 之前
import os
PTXAS = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\ptxas.exe"
assert os.path.exists(PTXAS), f"ptxas not found: {PTXAS}"
os.environ["TRITON_PTXAS_PATH"] = PTXAS

# 讓搜尋順序也以 12.9 為先（保險）
cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
os.environ["CUDA_HOME"] = cuda_home
os.environ["CUDA_PATH"] = cuda_home
os.environ["PATH"] = rf"{cuda_home}\bin;{os.environ['PATH']}"

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
# ========== 參數 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result_live_bybit_ENS"); os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_PNG_DIR = os.path.join(RESULT_DIR, "png"); os.makedirs(RESULT_PNG_DIR, exist_ok=True)
FEATURE_PARQUET = os.path.join(RESULT_DIR, "features_240m.parquet")
# --------- 精簡 checkpoint（每窗一檔；只存 elites + seeds） ----------
CKPT_DIR = os.path.join(RESULT_DIR, "ga_ckpt"); os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_ELITES_MAX = 128                    # 單檔最多保存的菁英數（避免超大）
CKPT_SEEDS = 64                          # 保存的隨機種子個體數（用來恢復多樣性）

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
GENS         = 2**13     # 單次/續訓要跑的 GA 代數；初次建議設較大（下方 N_GEN_BASE 也會用到）
AT_LEAST_IMPROVE = 2    # 是否至少要有提升才存模型（避免退步）
EARLY_STOP_PATIENCE = 500  # 若超過這麼多代沒提升就提前停止（避免無謂計算）
# 特徵工程設定
STD_WIN = 2**13                 # 滾動標準化視窗 (16384)
change_rate_steps = 6           # 1~6次變化率
MIN_HISTORY = 1024+STD_WIN + 1               # 至少這麼多列才開始訓練

# GA 參數
POP_SIZE = 2**13            # 1024
N_GEN_BASE = GENS            # 初次完整訓練代數（follow 有新K時用）
N_GEN_STEP = max(20, N_GEN_BASE // 5)  # follow 沒新K時短代續訓
IMPROVE_DELTA = 1e-3
ELITE_FRAC = 0.10
MUT_RATE = 0.12
MUT_SIGMA = 0.05
MAX_LEV = 1               # 槓桿上限
FEE = 0.00055                 # 單邊費（含滑點）
SEED = 42
np.random.seed(SEED)
# === NEW: GA gene bounds (apply to W & L) ===
GA_MIN, GA_MAX = -1.0, 1.0
# --- Gene → Parameter decode (single source of truth) ---
# W 的實際取值落在 [GA_MIN, GA_MAX]（例如 [-1, 1]）
# L 的實際取值落在 [0, MAX_LEV]
W_LO, W_HI = float(GA_MIN), float(GA_MAX)
L_LO, L_HI = 0.0, float(MAX_LEV)

def decode_genes(W_gene: torch.Tensor, L_gene: torch.Tensor):
    """
    將基因（W_gene, L_gene ∈ [0,1]）線性內插到實際參數空間。
    支援張量批次形狀：
      - W_gene: (pop, D) 或 (D,)
      - L_gene: (pop,) 或 ()
    回傳對應形狀的 (W_eff, L_eff)。
    """
    W_eff = W_LO + W_gene * (W_HI - W_LO)
    L_eff = L_LO + L_gene * (L_HI - L_LO)
    return W_eff, L_eff

def decode_genes_if_needed(W_like: torch.Tensor, L_like: torch.Tensor):
    """
    向後相容：如果載入的 W 看起來已是「已解碼」（包含負值），就直接原樣返回；
    若全部在 [0,1]，判定為基因，才做 decode。
    """
    eps = 1e-6
    is_gene = (W_like.min() >= -eps) and (W_like.max() <= 1.0 + eps)
    return decode_genes(W_like, L_like) if is_gene else (W_like, L_like)

# === 新增：Mask（特徵開關）的突變率 ===
MUT_RATE_MASK = 0.05   # bit-flip 機率；可調 0.02~0.15

def gate_mask(M_gene: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    基因 ∈ [0,1] → 0/1 採樣遮罩（不可微，GA 不需要微分）
    """
    return (M_gene >= thr).to(torch.float32)



# ========= 交易品 =========
EXCHANGE = "bybit"
PRODUCT  = "perpetual"   # spot/future/perpetual
SYMBOL   = "BTCUSDT"
INTERVAL = 240           # 分鐘 (int)
ANN_FACTOR = math.sqrt(int(24*60/INTERVAL)*365)  # 240m: 一天約6根, 一年~2190根 → sqrt(年bar數)

# 模型選擇
fitness_metric = "sharpe"     # "sharpe" or "return"

def compute_fitness(rets: torch.Tensor, *, ann_factor: float = float(ANN_FACTOR)) -> torch.Tensor:
    """
    rets: (T, pop) 或 (T,) 的逐根報酬率
    回傳: (pop,) 的分數向量（每個個體一個分數）
      - sharpe:  (mean/std) * ANN_FACTOR
      - return:  年化幾何報酬 = exp( (∑log(1+r))/T * bars_per_year ) - 1
    """
    if rets.dim() == 1:
        rets = rets.unsqueeze(1)  # (T,1)

    r = rets.float()  # 統計用 float32 較穩
    metric = fitness_metric.lower()

    if metric == "sharpe":
        mu = r.mean(dim=0)
        sd = r.std(dim=0, unbiased=False).clamp_min(1e-12)
        return (mu / sd) * ann_factor  # (pop,)

    elif metric in ("return", "geo", "geometric"):
        r = torch.clamp(r, min=-0.999999)             # 保護 log1p 定義域
        log_growth = torch.log1p(r).sum(dim=0)             # (pop,)
        return log_growth

    else:
        raise ValueError(f"Unknown fitness metric: {fitness_metric}")

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
    try:
        res = psycopg2.connect(**DB_CONFIG)
        print("Connecting to database succeeded.")
        return res
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

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
    print(f"Fetching new data from DB after {last_timestamp}...")
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


print("Fetching initial data from DB...")
df = init_fetch_data_from_db(EXCHANGE, PRODUCT, SYMBOL, INTERVAL)
print(f"Initial data from DB: {df.shape}, from {df.index.min()} to {df.index.max()}")
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
            out.to_csv(os.path.join(RESULT_DIR, "features_240m.csv"))  # for debug

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

print(f"Raw data: {df.shape}, from {df.index.min()} to {df.index.max()}")
df = init_compute_features(df)
print(f"Data with features: {df.shape}, from {df.index.min()} to {df.index.max()}")

# ===================== GA 策略 (PURE CUDA, AMP/TF32, optional torch.compile) =====================
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- 硬性要求 GPU，開足馬力 ----
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# ---- Windows: 強制給 torch.inductor / triton 一個每次執行都唯一的快取目錄，避免 FileExistsError ----
import tempfile, random, shutil, glob
def _setup_unique_inductor_cache():
    stamp = f"{os.getpid()}_{int(time.time())}_{random.randint(0, 1_000_000)}"
    root = os.path.join(tempfile.gettempdir(), f"torchinductor_cache_{stamp}")
    tri = os.path.join(root, "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = root          # torch._inductor 使用
    os.environ["TRITON_CACHE_DIR"]        = tri           # triton kernel cache
    # 可選：清掉歷史的我們自己建立的 cache，避免占空間
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "torchinductor_cache_*")):
        if d != root:
            shutil.rmtree(d, ignore_errors=True)
    os.makedirs(tri, exist_ok=True)
    logging.info(f"Using unique Inductor cache: {root}")

_setup_unique_inductor_cache()

assert torch.cuda.is_available(), "❌ 需要 CUDA GPU"
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# 速度開關
USE_AMP = True
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_TORCH_COMPILE = False     # 首次會有編譯開銷，之後飆速
ALWAYS_PLOT = False          # True=每代都畫，False=只有更好時才畫（更快）

# --------- 資料準備（一次上 GPU） ----------
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    ban = {"future_returns"}
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ban]

FEATURE_COLS = select_feature_columns(df)
logging.info(f"Feature columns = {len(FEATURE_COLS)}")

X_all_t = torch.as_tensor(df[FEATURE_COLS].to_numpy(np.float32), device=device)
y_all_t = torch.as_tensor(df["future_returns"].to_numpy(np.float32), device=device)
if USE_AMP:
    X_all_t = X_all_t.to(AMP_DTYPE)
ts_idx = df.index.to_numpy()
D = X_all_t.shape[1]
logging.info(f"X={tuple(X_all_t.shape)}, y={tuple(y_all_t.shape)}, D={D}, dev={device}")

# --------- 畫圖（CPU side） ----------
# --------- 畫圖（CPU side） ----------
def _stats_from_equity(eq: np.ndarray) -> Tuple[float, float]:
    """
    從 equity 曲線還原序列報酬並計算：累積報酬率、年化 Sharpe
    回傳：(cum_return, ann_sharpe)
    """
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size < 2 or not np.isfinite(eq).all():
        return 0.0, float("nan")
    # 由 equity 還原逐期報酬：r_t = eq[t]/eq[t-1] - 1
    rets = eq[1:] / np.clip(eq[:-1], 1e-12, np.inf) - 1.0
    mu = np.nanmean(rets)
    sd = np.nanstd(rets)  # 與訓練一致：unbiased=False
    sharpe = float((mu / (sd + 1e-12)) * float(ANN_FACTOR))
    cum = float(eq[-1] - 1.0)
    return cum, sharpe

def plot_equity(eq_np: np.ndarray, bh_np: np.ndarray, title: str, save_path: str):
    # 先把可能的 Torch Tensor 全轉成 CPU numpy
    if isinstance(eq_np, torch.Tensor):
        eq_np = eq_np.detach().cpu().numpy()
    if isinstance(bh_np, torch.Tensor):
        bh_np = bh_np.detach().cpu().numpy()

    # 再計算統計
    strat_cum, strat_sh = _stats_from_equity(eq_np)
    bh_cum,    bh_sh    = _stats_from_equity(bh_np)

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(eq_np, label="Strategy")
    ax.plot(bh_np, label="Buy & Hold")
    ax.set_xlabel("Bars")
    ax.set_ylabel("Equity")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    stats_txt = (
        f"Strategy  Cum: {strat_cum*100:,.2f}%   {fitness_metric}: {strat_sh:,.2f}\n"
        f"Buy&Hold  Cum: {bh_cum*100:,.2f}%   {fitness_metric}: {bh_sh:,.2f}"
    )
    ax.text(0.01, 0.99, stats_txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", alpha=0.2, linewidth=0.5))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



@torch.no_grad()
def eval_model_range_gpu(w_vec: torch.Tensor, l_scalar: torch.Tensor, m_vec: torch.Tensor,
                         start_idx: int, end_idx: int):
    """
    用單一模型 (w,l,m) 在 [start_idx, end_idx) 做前推回測（全 GPU）
    回傳: eq_np, bh_np, sharpe
    """
    if end_idx <= start_idx:
        return np.array([]), np.array([]), float("nan")

    X = X_all_t[start_idx:end_idx]   # (T,D)
    y = y_all_t[start_idx:end_idx]   # (T,)

    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        w_eff, l_eff = decode_genes_if_needed(w_vec, l_scalar)
        m_bin = gate_mask(m_vec)                             # (D,) ∈ {0,1}
        dot = (X.float() @ (w_eff * m_bin).float())    # (T,)
        denom = torch.clamp(m_bin.sum(), min=1.0)
        pos = (dot * l_eff).to(X.dtype) / denom
    prev = torch.cat([torch.zeros(1, device=device), pos[:-1]])
    rets = pos * y - float(FEE) * (pos - prev).abs()
    rets32 = rets.float()
    eq = torch.cumsum(torch.log1p(rets32), dim=0).exp()


    rets32 = rets.float()
    score = float(compute_fitness(rets32).squeeze().item())  # 用同一套規則算分數
    eq = torch.cumsum(torch.log1p(torch.clamp(rets32, min=-0.999999)), dim=0).exp()
    bh = torch.cumsum(torch.log1p(y.float()), dim=0).exp()
    return eq.detach().cpu().numpy(), bh.detach().cpu().numpy(), score


def plot_oos_for_window(best_w_np: np.ndarray, best_l: float, best_m_np: np.ndarray,
                        train_end_idx: int, window_label: str):
    start_idx = train_end_idx + 1
    end_idx   = len(df)
    if start_idx >= end_idx:
        return

    w_vec = torch.from_numpy(best_w_np).to(device=device, dtype=torch.float32)
    l_scalar = torch.tensor(best_l, device=device, dtype=torch.float32)
    m_vec = torch.from_numpy(best_m_np).to(device=device, dtype=torch.float32)

    eq_np, bh_np, oos_score = eval_model_range_gpu(w_vec, l_scalar, m_vec, start_idx, end_idx)

    oos_dir = os.path.join(RESULT_DIR, "oos_plots"); os.makedirs(oos_dir, exist_ok=True)
    tail_ts = ts_idx[end_idx-1].strftime("%Y%m%d_%H%M")
    save_png = os.path.join(oos_dir, f"oos_{window_label}_to_{tail_ts}.png")
    title = f"[OOS {ts_idx[start_idx].strftime('%Y-%m-%d %H:%M')} → {ts_idx[end_idx-1].strftime('%Y-%m-%d %H:%M')}] {fitness_metric}={oos_score:.3f}"
    plot_equity(eq_np, bh_np, title, save_png)
    logging.info(f"OOS plot saved: {save_png}")


# ===== 背景 I/O：非同步存檔與存圖 =====
from concurrent.futures import ThreadPoolExecutor
import threading, atexit

IO_MAX_WORKERS = 2               # 同時做 2 個 I/O 任務就好
IO_MAX_OUTSTANDING = 64          # 最多排隊 64 個，超過就丟掉避免記憶體爆
_io_exec = ThreadPoolExecutor(max_workers=IO_MAX_WORKERS, thread_name_prefix="io")
_io_sem  = threading.BoundedSemaphore(IO_MAX_OUTSTANDING)

def submit_io(fn, *args, **kwargs):
    """把 I/O 任務丟到背景做；排隊太多就丟棄此任務（並 log）。"""
    acquired = _io_sem.acquire(blocking=False)
    if not acquired:
        logging.warning(f"[io] backlog full, drop task: {getattr(fn, '__name__', str(fn))}")
        return
    def _wrapper():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logging.exception(f"[io] async task error: {e}")
        finally:
            _io_sem.release()
    _io_exec.submit(_wrapper)

def _io_shutdown():
    _io_exec.shutdown(wait=True)
atexit.register(_io_shutdown)


# --------- 核心：整群評分（Sharpe, 年化） ----------
# 把「圖內」與「圖外」分開：圖內不做 .item()、不組 dict
def _evaluate_population_cuda_core(W: torch.Tensor, L: torch.Tensor, M: torch.Tensor,
                                   X: torch.Tensor, y: torch.Tensor):
    """
    回傳:
      fitness: (pop,) Tensor
      best_idx: int
      eq_best_np: numpy.ndarray (CPU)
      stats_best: dict{sharpe, mean, std, last_pos}
    """
    ctx = torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP)
    with ctx:
        # 基因就 decode，已解碼就原樣
        W_eff, L_eff = decode_genes_if_needed(W, L)
        M_bin = gate_mask(M)                                   # (pop,D) ∈ {0,1}

        # 用 float32 做點積，避免 FP16 溢位
        WT_masked = (W_eff * M_bin).t()
        dot = (X.float() @ WT_masked.float())            # (T, pop)
        denom = torch.clamp(M_bin.sum(dim=1, keepdim=True).t(), min=1.0)  # (1, pop)
        pos = (dot * L_eff).to(X.dtype) / denom

        prev = torch.zeros((1, pos.shape[1]), device=device, dtype=pos.dtype)
        delta_pos = pos - torch.cat([prev, pos[:-1, :]], dim=0)
        costs = float(FEE) * delta_pos.abs()

        # 明確廣播 y → (T,1)
        rets = pos * y.unsqueeze(1) - costs              # (T, pop)

    rets32 = rets.float()
    mu = rets32.mean(dim=0)
    sd = rets32.std(dim=0, unbiased=False).clamp_min(1e-12)
    fitness = compute_fitness(rets32)

    best_idx = int(torch.argmax(fitness).item())
    eq_best = torch.cumsum(torch.log1p(rets32[:, best_idx]), dim=0).exp()  # (T,)
    eq_best_np = eq_best.detach().cpu().numpy()

    stats_best = dict(
        sharpe=float(fitness[best_idx].item()),
        mean=float(mu[best_idx].item()),
        std=float(sd[best_idx].item()),
        last_pos=float(pos[-1, best_idx].float().item())
    )
    return fitness, best_idx, eq_best_np, stats_best




def init_population(pop_size: int, D: int):
    """
    回傳：
      W: (pop, D) 連續基因 ∈ [0,1]  → decode → [-1,1]
      L: (pop,)   連續基因 ∈ [0,1]  → decode → [0, MAX_LEV]
      M: (pop, D) 連續基因 ∈ [0,1]  → gate   → {0,1}（>0.5 視為使用）
    """
    W = torch.empty((pop_size, D), device=device, dtype=torch.float32).uniform_(0.0, 1.0)
    L = torch.empty((pop_size,),    device=device, dtype=torch.float32).uniform_(0.0, 1.0)
    # 初始 Mask：0/1 大致各半
    M = torch.empty((pop_size, D), device=device, dtype=torch.float32).uniform_(0.0, 1.0)
    return W, L, M



def tournament_select_indices(fitness: torch.Tensor, needed: int, k: int = 3) -> torch.Tensor:
    pop = fitness.shape[0]
    cand = torch.randint(0, pop, (needed, k), device=device)       # (needed, k)
    cand_fit = fitness[cand]                                                    # (needed, k)
    winners = cand.gather(1, cand_fit.argmax(dim=1, keepdim=True)).squeeze(1)   # (needed,)
    return winners

def produce_children(W: torch.Tensor, L: torch.Tensor, M: torch.Tensor,
                     fitness: torch.Tensor, elites: int):
    pop, D = W.shape
    elite_idx = torch.topk(fitness, k=elites, largest=True).indices
    W_e, L_e, M_e = W[elite_idx].clone(), L[elite_idx].clone(), M[elite_idx].clone()

    needed = pop - elites
    p1 = tournament_select_indices(fitness, needed, k=3)
    p2 = tournament_select_indices(fitness, needed, k=3)

    # 連續基因使用 BLX/線性混合
    alpha = torch.empty((needed, 1), device=device).uniform_(0.2, 0.8)
    W_c = alpha * W[p1] + (1 - alpha) * W[p2]
    L_c = alpha.squeeze(1) * L[p1] + (1 - alpha.squeeze(1)) * L[p2]

    # Mask 使用「逐特徵均勻交叉」（0.5 機率取父一或父二的位）
    take_from_p1 = (torch.rand((needed, D), device=device) < 0.5).to(torch.float32)
    M_c = take_from_p1 * M[p1] + (1 - take_from_p1) * M[p2]

    # 變異：
    # 1) W/L 加高斯噪音後 clip 至 [0,1]
    mut_mask_col = (torch.rand(needed, 1, device=device) < float(MUT_RATE))
    noise_w = torch.normal(0.0, float(MUT_SIGMA), size=W_c.shape, device=device)
    W_c = W_c + mut_mask_col.to(W_c.dtype) * noise_w
    mut_mask_1d = mut_mask_col.squeeze(1)
    noise_l = torch.normal(0.0, float(MUT_SIGMA), size=L_c.shape, device=device)
    L_c = L_c + mut_mask_1d.to(L_c.dtype) * noise_l
    W_c.clamp_(0.0, 1.0)
    L_c.clamp_(0.0, 1.0)

    # 2) M 用 bit-flip（0 ↔ 1）：先用連續空間承載，再翻轉
    flip = (torch.rand((needed, D), device=device) < float(MUT_RATE_MASK))
    M_c = torch.where(flip, 1.0 - M_c, M_c).clamp_(0.0, 1.0)

    W_new = torch.cat([W_e, W_c], dim=0)
    L_new = torch.cat([L_e, L_c], dim=0)
    M_new = torch.cat([M_e, M_c], dim=0)
    return W_new, L_new, M_new


# --- torch.compile 加速（Inductor）---
if USE_TORCH_COMPILE:
    try:
        # 可選的 inductor 調參：開啟更激進的 autotune
        try:
            import torch._inductor.config as inductor_config
            inductor_config.max_autotune_gemm = True
            inductor_config.max_autotune_pointwise = True
        except Exception as _:
            pass

        # 對計算最重的兩個函式做編譯包裹
        _evaluate_population_cuda_core = torch.compile(
            _evaluate_population_cuda_core,
            mode="max-autotune",   # GPU 場景建議值
            fullgraph=False,       # 允許圖中斷，避免遇到 Python/NumPy 步驟失敗
            dynamic=True           # 對輸入尺寸更有彈性（同窗內尺寸固定仍能吃到快取）
        )
        produce_children = torch.compile(
            produce_children,
            mode="max-autotune",
            fullgraph=False,
            dynamic=True
        )
        logging.info("torch.compile acceleration enabled for GA core.")
    except Exception as e:
        logging.warning(f"torch.compile 加速無法啟用，改回 eager：{e}")
        USE_TORCH_COMPILE = False



# --------- CUDA 版 GA 訓練單窗 ----------
def train_sparse_linear_cuda(
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    steps: int,
    window_label: str,
    *,
    lr: float = 3e-3,
    l1_gate: float = 1e-3,     # 稀疏度強度：越大越少特徵
    l2_w: float = 1e-5,        # 權重 L2 正則
    temp: float = 1.0,         # gate 的溫度（可逐步降低）
    seed_prev: dict | None = None
):
    T, D = X_t.shape
    # 參數：w_param（無界）→ w = tanh(w_param) ∈ [-1,1]
    #       l_param（無界）→ l = sigmoid(l_param)*MAX_LEV ∈ [0, MAX_LEV]
    #       m_logit（無界）→ m = sigmoid(m_logit) ∈ (0,1)
    w_param  = torch.zeros(D, device=device, dtype=torch.float32, requires_grad=True)
    l_param  = torch.zeros(1, device=device, dtype=torch.float32, requires_grad=True)
    m_logit  = torch.zeros(D, device=device, dtype=torch.float32, requires_grad=True)

    # 暖機：上一窗解
    if seed_prev and all(k in seed_prev for k in ("w","l","m")):
        with torch.no_grad():
            w_param.copy_(torch.atanh(torch.clamp(torch.as_tensor(seed_prev["w"], device=device), -0.999, 0.999)))
            l_param.copy_(torch.logit(torch.clamp(torch.tensor(seed_prev["l"], device=device), 1e-4, 1-1e-4)))
            m_init = torch.as_tensor(seed_prev["m"], device=device)
            m_logit.copy_(torch.logit(torch.clamp(m_init, 1e-3, 1-1e-3)))

    opt = torch.optim.AdamW([w_param, l_param, m_logit], lr=lr)
    best = (-1e18, None)  # (score, pack)
    bh_np = torch.cumsum(torch.log1p(y_t.float()), dim=0).exp().detach().cpu().numpy()

    for step in range(1, steps+1):
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            w = torch.tanh(w_param)                      # [-1,1]
            l = torch.sigmoid(l_param).squeeze(0) * L_HI # [0,MAX_LEV]
            m = torch.sigmoid(m_logit / temp)            # (D,)

            dot = (X_t.float() @ (w * m).float())        # (T,)
            denom = torch.clamp(m.sum(), min=1.0)
            pos = (dot * l).to(X_t.dtype) / denom

            prev = torch.cat([torch.zeros(1, device=device, dtype=pos.dtype), pos[:-1]])
            # 平滑交易成本
            costs = float(FEE) * torch.sqrt((pos - prev).float().pow(2) + 1e-8)
            rets = pos * y_t - costs                     # (T,)

            # 目標：Sharpe 或 幾何報酬
            if fitness_metric.lower() == "sharpe":
                r = rets.float()
                mu = r.mean()
                sd = r.std(unbiased=False).clamp_min(1e-12)
                score = (mu / sd) * float(ANN_FACTOR)
                loss_main = -score
            else:
                r = torch.clamp(rets.float(), min=-0.999999)
                score = torch.log1p(r).sum()
                loss_main = -score

            # 稀疏與權重正則
            loss_sparse = l1_gate * m.mean() + l2_w * (w.pow(2).mean())

            loss = loss_main + loss_sparse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        [w_param, l_param, m_logit]
        opt.step()

        # 保存最佳（評分用同一套 compute_fitness）
        with torch.no_grad():
            w_eff = (w * m).detach()
            l_eff = l.detach()
            m_bin = (m > 0.5).float()
            fit, idx, eq_best_np, stats = _evaluate_population_cuda_core(
                w_eff, l_eff, m_bin, X_t, y_t
            )
            cur = float(fit[0].item())
            if cur > best[0] + IMPROVE_DELTA:
                best = (cur, {
                    "best_weights": w.detach().cpu().numpy().astype(np.float32),
                    "best_leverage": float(l_eff.item() / L_HI),  # 注意：回傳的是 gene 空間 or 實際值；你目前下游用 gene 在 decode，保持一致的話回 0~1
                    "best_mask": m_bin.detach().cpu().numpy().astype(np.float32),
                    "best_stats": stats,
                    "history": [],
                    "last_gen": step
                })
                save_dir = os.path.join(RESULT_PNG_DIR, f"{window_label}"); os.makedirs(save_dir, exist_ok=True)
                save_png = os.path.join(save_dir, f"{window_label}_L0opt_step{step:05d}.png")
                plot_equity(eq_best_np, bh_np, f"[{window_label}][L0-Opt] step {step} | {fitness_metric}={cur:.3f}", save_png)

        # 溫度/學習率退火（可選）
        if step % 2000 == 0:
            temp = max(0.5, temp * 0.9)

    return best[1]

@torch.no_grad()
def ga_train_one_window_cuda(
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    gens: int,
    window_label: str,
    warm_start: dict | None = None,
    seed_prev: dict | None = None,   # {"w": np.ndarray, "l": float, "m": np.ndarray}
):
    T, D = X_t.shape
    pop = POP_SIZE
    elites = max(1, int(ELITE_FRAC * pop))

    resumed = load_checkpoint_slim(window_label, pop=pop, D=D)
    if resumed:
        W = resumed["W"]; L = resumed["L"]; M = resumed["M"]
        start_gen = resumed["start_gen"]
        prev_best = resumed["prev_best"]
        logging.info(f"[{window_label}] resume from slim ckpt @ gen {start_gen-1}, best={prev_best:.4f}")
    else:
        if warm_start and all(k in warm_start for k in ("W","L","M")):
            W = torch.as_tensor(warm_start["W"], device=device, dtype=torch.float32).view(pop, D).clone()
            L = torch.as_tensor(warm_start["L"], device=device, dtype=torch.float32).view(pop).clone()
            M = torch.as_tensor(warm_start["M"], device=device, dtype=torch.float32).view(pop, D).clone()
            if W.shape != (pop, D) or M.shape != (pop, D):
                W, L, M = init_population(pop, D)
        else:
            W, L, M = init_population(pop, D)
        start_gen = 1
        prev_best = -1e12

    # 跨窗種子（只有在沒有 warm-start 時才用）
    if warm_start is None and seed_prev and all(k in seed_prev for k in ("w","l","m")):
        prev_w = torch.as_tensor(seed_prev["w"], device=device, dtype=torch.float32).view(1, D)
        prev_l = torch.tensor(float(seed_prev["l"]), device=device, dtype=torch.float32).view(1)
        prev_m = torch.as_tensor(seed_prev["m"], device=device, dtype=torch.float32).view(1, D)
        k_seed = max(2, pop // 64)
        noise_w = torch.normal(0.0, 0.005, size=(k_seed, D), device=device)
        noise_l = torch.normal(0.0, 0.02,  size=(k_seed,),    device=device)
        # noise_m = torch.normal(0.0, 0.02, size=(k_seed, D), device=device)

        # M：少量 bit flip
        flip = (torch.rand((k_seed, D), device=device) < 0.02)
        W[:k_seed] = (prev_w + noise_w)
        L[:k_seed] = (prev_l + noise_l)
        M_seed = prev_m.repeat(k_seed, 1)
        M[:k_seed] = torch.where(flip, 1.0 - M_seed, M_seed)
        logging.info(f"[{window_label}] seeded {k_seed} from previous best.")

    best_hist = [] if start_gen == 1 else [prev_best]
    bh_np = torch.cumsum(torch.log1p(y_t.float()), dim=0).exp().detach().cpu().numpy()

    no_improve_gens = 0
    improved_gens = 0

    for gen in range(start_gen, gens + 1):


        if no_improve_gens >= EARLY_STOP_PATIENCE and improved_gens >= AT_LEAST_IMPROVE:
            logging.info(f"[{window_label}] early stopping at gen {gen-1} (no improvement for {no_improve_gens} gens).")
            break

        fitness, best_idx, eq_best_np, best_stats = _evaluate_population_cuda_core(W, L, M, X_t, y_t)
        cur_best = float(fitness[best_idx].item())
        best_hist.append(cur_best)

        improved = cur_best > prev_best + IMPROVE_DELTA
        if improved:
            no_improve_gens = 0
            improved_gens += 1
            save_dir = os.path.join(RESULT_PNG_DIR, f"{window_label}"); os.makedirs(save_dir, exist_ok=True)
            save_png = os.path.join(save_dir, f"Final_{window_label}.png")
            plot_equity(eq_best_np, bh_np, f"[{window_label}] Gen {gen} | {fitness_metric}={cur_best:.3f}", save_png)
        else:
            no_improve_gens += 1

        if improved:
            save_checkpoint_slim(window_label, gen, W, L, M, fitness, best_hist, pop, D)
            prev_best = cur_best
            logging.info(f"[{window_label}] Gen {gen}/{gens} | {fitness_metric}={cur_best:.3f} (improved, ckpt saved)")
        elif gen % 100 == 0:
            logging.info(f"[{window_label}] Gen {gen}/{gens} | {fitness_metric}={cur_best:.3f}")

        W, L, M = produce_children(W, L, M, fitness, elites)

    fitness, best_idx, eq_best_np, best_stats = _evaluate_population_cuda_core(W, L, M, X_t, y_t)
    best_w = W[best_idx].detach().cpu().numpy()
    best_l = float(L[best_idx].item())
    best_m = gate_mask(M[best_idx]).detach().cpu().numpy()  # 存 0/1

    save_checkpoint_slim(window_label, gens, W, L, M, fitness, best_hist, pop, D)

    return {
        "best_weights": best_w,
        "best_leverage": best_l,
        "best_mask": best_m,
        "best_stats": best_stats,
        "history": best_hist,
        "last_gen": gens,
    }



# --------- 名人堂（Archive） ----------
def load_archive() -> list:
    if not os.path.exists(ARCHIVE_PATH): return []
    try:
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_archive(items: list):
    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

def _ckpt_path(window_label: str) -> str:
    safe_lbl = window_label.replace(":", "").replace("/", "_")
    return os.path.join(CKPT_DIR, f"ckpt_{safe_lbl}.pt")

def save_checkpoint_slim(window_label: str, gen: int,
                         W: torch.Tensor, L: torch.Tensor, M: torch.Tensor,
                         fitness: torch.Tensor, best_hist: list, pop: int, D: int):
    k_elite = min(CKPT_ELITES_MAX, max(8, int(ELITE_FRAC * pop)))
    elite_idx = torch.topk(fitness, k=k_elite, largest=True).indices
    W_e = W[elite_idx].detach().to("cpu", torch.float16)
    L_e = L[elite_idx].detach().to("cpu", torch.float16)
    M_e = M[elite_idx].detach().to("cpu", torch.float16)

    W_s, L_s, M_s = init_population(CKPT_SEEDS, D)
    W_s = W_s.detach().to("cpu", torch.float16)
    L_s = L_s.detach().to("cpu", torch.float16)
    M_s = M_s.detach().to("cpu", torch.float16)

    ck = {
        "window_label": window_label,
        "gen": int(gen),
        "pop": int(pop),
        "D": int(D),
        "elite_W": W_e,
        "elite_L": L_e,
        "elite_M": M_e,
        "seed_W": W_s,
        "seed_L": L_s,
        "seed_M": M_s,
        "best_fitness": float(max(best_hist) if best_hist else -1e18),
    }
    torch.save(ck, _ckpt_path(window_label))

def load_checkpoint_slim(window_label: str, pop: int, D: int):
    p = _ckpt_path(window_label)
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu")

    W_e = ck.get("elite_W"); L_e = ck.get("elite_L"); M_e = ck.get("elite_M")
    W_s = ck.get("seed_W");  L_s = ck.get("seed_L");  M_s = ck.get("seed_M")

    if W_e is None or L_e is None:
        return None
    # 向後相容：若舊檔沒有 M，就隨機補
    if M_e is None:
        M_e = torch.rand_like(W_e)
    if W_s is None or L_s is None:
        W_s, L_s, M_s = init_population(CKPT_SEEDS, D)
        W_s = W_s.to("cpu", torch.float16); L_s = L_s.to("cpu", torch.float16); M_s = M_s.to("cpu", torch.float16)
    if M_s is None:
        M_s = torch.rand_like(W_s)

    W_e = W_e.to(torch.float32, copy=False).to(device)
    L_e = L_e.to(torch.float32, copy=False).to(device)
    M_e = M_e.to(torch.float32, copy=False).to(device)
    W_s = W_s.to(torch.float32, copy=False).to(device)
    L_s = L_s.to(torch.float32, copy=False).to(device)
    M_s = M_s.to(torch.float32, copy=False).to(device)

    k_e, k_s = W_e.shape[0], W_s.shape[0]
    W = torch.empty((pop, D), device=device, dtype=torch.float32)
    L = torch.empty((pop,),    device=device, dtype=torch.float32)
    M = torch.empty((pop, D), device=device, dtype=torch.float32)

    take_e = min(k_e, pop)
    W[:take_e] = W_e[:take_e]; L[:take_e] = L_e[:take_e]; M[:take_e] = M_e[:take_e]
    ptr = take_e
    if ptr < pop and k_s > 0:
        take_s = min(k_s, pop - ptr)
        W[ptr:ptr+take_s] = W_s[:take_s]; L[ptr:ptr+take_s] = L_s[:take_s]; M[ptr:ptr+take_s] = M_s[:take_s]
        ptr += take_s
    if ptr < pop:
        W_fill, L_fill, M_fill = init_population(pop - ptr, D)
        W[ptr:] = W_fill; L[ptr:] = L_fill; M[ptr:] = M_fill

    start_gen = int(ck.get("gen", 0)) + 1
    prev_best = float(ck.get("best_fitness", -1e18))
    return {"W": W, "L": L, "M": M, "start_gen": start_gen, "prev_best": prev_best}

def add_model_to_archive(train_end_ts, model_rec: dict):
    items = load_archive()
    items.append({
        "train_end": str(pd.Timestamp(train_end_ts).tz_convert("UTC")),
        "D": int(D),
        "weights_file": model_rec["weights_file"],  # .pt
        "leverage": float(model_rec["leverage"]),
        "train_sharpe": float(model_rec["train_sharpe"]),
        "notes": model_rec.get("notes", "")
    })
    save_archive(items)

@torch.no_grad()
def evaluate_archive_until_cuda(items: list, until_idx: int):
    if not items: return -1, {}
    valid = []
    for i, it in enumerate(items):
        st = np.searchsorted(ts_idx, pd.Timestamp(it["train_end"]), side="right")
        if st < until_idx and os.path.exists(it["weights_file"]):
            valid.append((i, st, it["weights_file"]))
    if not valid: return -1, {}

    best_score, best_idx_global, best_stat = -1e18, -1, {}
    from collections import defaultdict
    groups = defaultdict(list)
    for i, st, wf in valid:
        groups[st].append((i, wf))

    for st, lst in groups.items():
        X = X_all_t[st:until_idx]
        y = y_all_t[st:until_idx]

        Ws, Ls, Ms, idx_map = [], [], [], []
        for (i, wf) in lst:
            ck = torch.load(wf, map_location="cpu")
            w = torch.as_tensor(ck["w"] if "w" in ck else ck["W"].numpy(), dtype=torch.float32, device=device).view(1, -1)
            l = torch.as_tensor(ck["l"] if "l" in ck else ck["L"].numpy(), dtype=torch.float32, device=device).view(1)
            # 向後相容：沒有 m 就「全 1」
            if "m" in ck:
                m = torch.as_tensor(ck["m"], dtype=torch.float32, device=device).view(1, -1)
            else:
                m = torch.ones_like(w)
            Ws.append(w); Ls.append(l); Ms.append(m); idx_map.append(i)

        W_stack = torch.cat(Ws, dim=0)
        L_stack = torch.cat(Ls, dim=0).squeeze(1)
        M_stack = torch.cat(Ms, dim=0)

        fit, loc_best_idx, _, st_best = _evaluate_population_cuda_core(W_stack, L_stack, M_stack, X, y)
        score = float(fit[loc_best_idx].item())
        if score > best_score:
            best_score, best_idx_global, best_stat = score, idx_map[int(loc_best_idx)], st_best

    return best_idx_global, best_stat


# --------- 主迴圈（純 CUDA） ----------
def run_ga_live():
    items = load_archive()
    equity_live, bh_live = [], []
    pos_prev = 0.0

    prev_best_seed = None  # ★ 新增：跨窗延續的種子

    start_bar = max(MIN_HISTORY, 2)
    for t in range(start_bar, len(df)):
        train_end = t - 1
        train_start = 0  # 或者滑窗：max(0, t-1-STD_WIN)

        X_train = X_all_t[train_start:train_end+1]
        y_train = y_all_t[train_start:train_end+1]

        window_label = f"{ts_idx[train_start].strftime('%Y%m%d_%H%M')}-{ts_idx[train_end].strftime('%Y%m%d_%H%M')}"

        # === 將「上一窗 elites」當作這一窗的 warm-start ===
        warm = None
        if t - 1 > train_start:  # 確保有前一窗
            prev_end = train_end - 1
            prev_window_label = f"{ts_idx[train_start].strftime('%Y%m%d_%H%M')}-{ts_idx[prev_end].strftime('%Y%m%d_%H%M')}"
            prev_resume = load_checkpoint_slim(prev_window_label, pop=POP_SIZE, D=D)
            if prev_resume:
                # 這裡的 W/L 已由上一窗 ckpt 的 elites + seeds 重建成完整人口
                warm = {
                    "W": prev_resume["W"].detach().cpu().numpy(),
                    "L": prev_resume["L"].detach().cpu().numpy(),
                    "M": prev_resume["M"].detach().cpu().numpy(),
                }
                logging.info(f"[{window_label}] warm-start from previous window elites: {prev_window_label}")


        gens = max(N_GEN_BASE, GENS)
        # ★ 將上一窗最佳帶入這一窗
        rec = train_sparse_linear_cuda(
            X_train, y_train, steps=gens, window_label=window_label, seed_prev=prev_best_seed
        )

        best_w = rec["best_weights"]
        best_l = float(rec["best_leverage"])
        best_m = rec["best_mask"]
        best_sharpe = float(rec["best_stats"]["sharpe"])


        # ★ 每窗訓練完成就畫「從 train_end+1 到資料最新」的 OOS 表現
        plot_oos_for_window(best_w, best_l, best_m, train_end, window_label)

        # 存權重並加入名人堂
        weights_dir = os.path.join(RESULT_DIR, "weights"); os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, f"weights_{ts_idx[train_end].strftime('%Y%m%d_%H%M')}.pt")
        torch.save({"w": torch.from_numpy(best_w), "l": torch.tensor(best_l, dtype=torch.float32),
                    "m": torch.from_numpy(best_m)}, weights_file)
        add_model_to_archive(ts_idx[train_end], {
            "weights_file": weights_file,
            "leverage": best_l,
            "train_sharpe": best_sharpe,
            "notes": f"window={window_label}"
        })
        items = load_archive()

        prev_best_seed = {"w": best_w, "l": best_l, "m": best_m}


        # OOS 最佳模型（你原本 live 用的）
        # best_w is the trained weight array (numpy); convert it directly to a CUDA tensor for live inference.
        # best_w / best_l 是基因，先 decode
        use_w_gene = torch.from_numpy(best_w).to(device=device, dtype=torch.float32)
        use_l_gene = torch.tensor(best_l, device=device, dtype=torch.float32)
        use_m_gene = torch.from_numpy(best_m).to(device=device, dtype=torch.float32)

        use_w, use_l = decode_genes_if_needed(use_w_gene, use_l_gene)
        use_m = gate_mask(use_m_gene)

        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            x_t = X_all_t[t:t+1]                                 # (1,D)
            denom = torch.clamp(use_m.sum(), min=1.0)
            dot = (x_t.float() @ (use_w * use_m).view(-1,1).float()).squeeze(1)  # (1,)
            pos_t = (dot * use_l).to(x_t.dtype) / denom


        ret_t = float(pos_t.item() * y_all_t[t].item() - float(FEE) * abs(float(pos_prev) - float(pos_t.item())))
        pos_prev = float(pos_t.item())

        equity_live.append((equity_live[-1] if equity_live else 1.0) * (1.0 + ret_t))
        bh_live.append((bh_live[-1] if bh_live else 1.0) * (1.0 + float(y_all_t[t].item())))

        live_png = os.path.join(RESULT_DIR, "live_equity.png")
        plot_equity(np.array(equity_live, np.float64), np.array(bh_live, np.float64), f"Live up to {ts_idx[t]}", live_png)
        logging.info(f"[{ts_idx[t]}] pos={pos_t.item():.3f} ret={ret_t:.6f} eq={equity_live[-1]:.4f}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    logging.info("GA live run complete.")


# 入口
if __name__ == "__main__":
    import faulthandler, atexit
    faulthandler.enable()
    atexit.register(lambda: logging.info("=== Program exiting (atexit) ==="))
    try:
        run_ga_live()
    except SystemExit as e:
        logging.exception(f"SystemExit: {e}")
        raise
    except Exception as e:
        logging.exception("Fatal error in run_ga_live()")
        raise
    finally:
        logging.info("=== Program end ===")
