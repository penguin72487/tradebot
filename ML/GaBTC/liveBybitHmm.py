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

# ========= 交易品 =========
EXCHANGE = "bybit"
PRODUCT  = "perpetual"   # spot/future/perpetual
SYMBOL   = "BTCUSDT"
INTERVAL = 240           # 分鐘 (int)
ANN_FACTOR = math.sqrt(int(24*60/INTERVAL)*365)  # 240m: 一天約6根, 一年~2190根 → sqrt(年bar數)

# ========== 參數 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result_live_bybitHmm"); os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_PNG_DIR = os.path.join(RESULT_DIR, "png"); os.makedirs(RESULT_PNG_DIR, exist_ok=True)
FEATURE_PARQUET = os.path.join(RESULT_DIR, f"features_{INTERVAL}.parquet")
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
STD_WIN = 2**10                 # 滾動標準化視窗 (16384)
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

# ---- Soft penalties（預設全 0；開了就會「減分」，以軟性方式控幅）----
W_L2_PENALTY       = 1e-1   # 建議起點：1e-4 ~ 1e-3
POS_L1_PENALTY     = 0.0   # 對平均倉位絕對值做懲罰，抑制過度曝險；建議 1e-3 ~ 1e-2
TURNOVER_PENALTY   = 0.0   # 對換手(Δpos)做懲罰，抑制交易過度頻繁；建議與 FEE 同量級
SPARSITY_PENALTY   = 0.0   # 對使用特徵比例做懲罰，鼓勵稀疏；建議 1e-3

# === NEW: GA gene bounds (apply to W & L) ===
GA_MIN, GA_MAX = -1.0, 1.0
# --- Gene → Parameter decode (single source of truth) ---
# W 的實際取值落在 [GA_MIN, GA_MAX]（例如 [-1, 1]）
# L 的實際取值落在 [0, MAX_LEV]
W_LO, W_HI = float(GA_MIN), float(GA_MAX)
def decode_W(W_gene: torch.Tensor):
    """
    將 W 基因（∈[0,1]）線性內插到實際參數空間 [-1, 1]。
    支援 W_gene 形狀：(pop,D) 或 (D,)
    """
    return W_LO + W_gene * (W_HI - W_LO)

# === 新增：Mask（特徵開關）的突變率 ===
MUT_RATE_MASK = 0.05   # bit-flip 機率；可調 0.02~0.15

def gate_mask(M_gene: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    基因 ∈ [0,1] → 0/1 採樣遮罩（不可微，GA 不需要微分）
    """
    return (M_gene >= thr).to(torch.float32)




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
    呼叫 pandas_ta 指標（改為「位置索引安全」版本）：
    - 輸入給 pandas_ta 的 Series 一律 reset_index(drop=True) 成 RangeIndex
    - 計算完把輸出 DataFrame/Series 的 index 換回原本的 DatetimeIndex
    - 出錯跳過並記錄（best-effort）
    """
    logging.info("Computing pandas_ta features (pos-index safe)...")
    feats = []
    tried, ok = 0, 0

    # 保留原本索引，以便輸出時恢復
    orig_index = df.index

    # 候選參數名（不同函式可能叫不一樣）
    price_arg_alias = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low":  ["low", "l"],
        "close":["close","c"],
        "volume":["volume","vol","v"]
    }

    # 先準備可用輸入（原 df 的欄位）
    inputs = {}
    for k, aliases in price_arg_alias.items():
        for a in aliases:
            if a in df.columns:
                inputs[k] = df[a]
                break
        if k not in inputs and k in df.columns:
            inputs[k] = df[k]

    for name in dir(pta):
        if name.startswith("_"):
            continue
        func = getattr(pta, name)
        if not callable(func) or inspect.isclass(func):
            continue

        # 只嘗試接受 O/H/L/C/V 任一參數的函式
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        has_price_input = any(p in params for p in ["open","high","low","close","volume","o","h","l","c","v"])
        if not has_price_input:
            continue

        tried += 1

        # 準備 kwargs：把 Series 改成 RangeIndex（位置索引安全）
        kwargs = {}
        for std_key, aliases in price_arg_alias.items():
            for a in aliases:
                if a in params and std_key in inputs:
                    s = inputs[std_key]
                    s_pos = s.reset_index(drop=True)         # 變成 RangeIndex: 0..n-1
                    s_pos.name = s.name                      # 保留欄名
                    kwargs[a] = s_pos
                    break

        # 常見窗長參數名稱（壓到 <= STD_WIN）
        for wnd_name in ["length", "window", "timeperiod", "n", "fast", "slow", "long", "short", "lbp", "period"]:
            if wnd_name in params:
                default_len = params[wnd_name].default if params[wnd_name].default is not inspect._empty else 14
                kwargs[wnd_name] = _clip_len(default_len, STD_WIN)

        if "append" in params:
            kwargs["append"] = False

        try:
            # 抑制特定 FutureWarning（避免洗版），同時我們已用 RangeIndex 做實體修正
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Series.__getitem__ treating keys as positions is deprecated",
                    category=FutureWarning
                )
                out = func(**kwargs)

            out = _align_df(out, f"pta_{name}")
            if out is not None:
                # 把輸出的 index 換回原本的 DatetimeIndex（長度通常相同）
                if len(out) == len(orig_index):
                    out.index = orig_index
                else:
                    # 長度不一致時，做尾端對齊（大多指標與原長度一致；保險處理）
                    out.index = pd.RangeIndex(len(out))
                    out = out.tail(len(orig_index))
                    out.index = orig_index
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
            # out.to_csv(os.path.join(RESULT_DIR, f"features_{INTERVAL}.csv"))  # for debug

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
if "future_returns" not in df.columns:
    logging.warning("Feature parquet missing 'future_returns' — rebuilding from 'close'.")
    if "close" not in df.columns:
        raise KeyError("Both 'future_returns' and 'close' are missing; cannot rebuild.")
    df["future_returns"] = (
        df["close"].pct_change().shift(-1).fillna(0.0).astype(np.float32)
    )
    try:
        df.to_parquet(FEATURE_PARQUET)
        logging.info("Patched features parquet with 'future_returns'.")
    except Exception as e:
        logging.warning(f"Failed to rewrite parquet: {e}")
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
def eval_model_range_gpu(w_vec: torch.Tensor, m_vec: torch.Tensor,
                         start_idx: int, end_idx: int):
    if end_idx <= start_idx:
        return np.array([]), np.array([]), float("nan")

    X = X_all_t[start_idx:end_idx]   # (T,D)
    y = y_all_t[start_idx:end_idx]   # (T,)

    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        w_eff = decode_W(w_vec)
        m_bin = gate_mask(m_vec)
        dot = (X.float() @ (w_eff * m_bin).float())   # (T,)
        denom = m_bin.sum().clamp_min(1.0)
        pos = (dot / denom).to(X.dtype)
        # pos = pos.clamp(-100.0, 100.0)   # 保護

    prev = torch.cat([torch.zeros(1, device=device), pos[:-1]])
    rets32 = (pos * y - float(FEE) * (pos - prev).abs()).float()
    score = float(compute_fitness(rets32).squeeze().item())
    eq = torch.cumsum(torch.log1p(torch.clamp(rets32, min=-0.999999)), dim=0).exp()
    bh = torch.cumsum(torch.log1p(y.float()), dim=0).exp()
    return eq.detach().cpu().numpy(), bh.detach().cpu().numpy(), score

def plot_oos_for_window(best_w_np: np.ndarray, best_m_np: np.ndarray,
                        train_end_idx: int, window_label: str):
    start_idx = train_end_idx + 1
    end_idx   = len(df)
    if start_idx >= end_idx:
        return
    w_vec = torch.from_numpy(best_w_np).to(device=device, dtype=torch.float32)
    m_vec = torch.from_numpy(best_m_np).to(device=device, dtype=torch.float32)
    eq_np, bh_np, oos_score = eval_model_range_gpu(w_vec, m_vec, start_idx, end_idx)
    oos_dir = os.path.join(RESULT_DIR, "oos_plots"); os.makedirs(oos_dir, exist_ok=True)
    tail_ts = ts_idx[end_idx-1].strftime("%Y%m%d_%H%M")
    save_png = os.path.join(oos_dir, f"oos_{window_label}_to_{tail_ts}.png")
    title = f"[OOS {ts_idx[start_idx].strftime('%Y-%m-%d %H:%M')} → {ts_idx[end_idx-1].strftime('%Y-%m-%d %H:%M')}] {fitness_metric}={oos_score:.3f}"
    plot_equity(eq_np, bh_np, title, save_png)
    logging.info(f"OOS plot saved: {save_png}")





# 入口
if __name__ == "__main__":
    import faulthandler, atexit
    faulthandler.enable()
    atexit.register(lambda: logging.info("=== Program exiting (atexit) ==="))
    try:
        run_hmm_ga_live()   # ⬅ 改呼叫新的 HMM→GA(states) 管線
    except SystemExit as e:
        logging.exception(f"SystemExit: {e}")
        raise
    except Exception as e:
        logging.exception("Fatal error in run_hmm_ga_live()")
        raise
    finally:
        logging.info("=== Program end ===")
