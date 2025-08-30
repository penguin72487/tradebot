# -*- coding: utf-8 -*-
"""
DB -> OHLCV -> pandas_ta + TA-Lib -> 1..6 階變化率 -> 多種標準化(rolling, block 平行) -> CUDA GA 回測
- 若 features parquet 存在：直接讀；否則重新計算並快取
- pandas_ta 關閉 multiprocessing，排除 ha/hilo，再用向量化 Heikin-Ashi 補回
- TA-Lib 用 threads 平行
- 全部 rolling transform 僅用「當下以前」資料，避免展望偏誤
"""
import os, re, sys, time, math, random, warnings, logging
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

# ========= 目錄與檔名 =========
BASE_DIR     = os.path.dirname(__file__)

RESULT_DIR   = os.path.join(BASE_DIR, "result_bybit"); os.makedirs(RESULT_DIR, exist_ok=True)
def R(*parts): 
    return os.path.join(RESULT_DIR, *parts)

LOG_DIR      = os.path.join(BASE_DIR, "log");    os.makedirs(LOG_DIR, exist_ok=True)
JOBLIB_TMP   = os.path.join(BASE_DIR, "joblib_tmp"); os.makedirs(JOBLIB_TMP, exist_ok=True)
LOG_DIR      = R("log")
JOBLIB_TMP   = R("joblib_tmp")
TMP_DIR      = R("tmp")
MPLCFG_DIR   = R("mplconfig")
FIG_DIR      = R("figs")
CKPT_DIR     = R("checkpoints")

for d in [RESULT_DIR, LOG_DIR, JOBLIB_TMP, TMP_DIR, MPLCFG_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# 把所有臨時與繪圖設定也綁到 RESULT_DIR，務必在 import matplotlib 前
os.environ.setdefault("TMPDIR", TMP_DIR)
os.environ.setdefault("TEMP",   TMP_DIR)
os.environ.setdefault("TMP",    TMP_DIR)
os.environ.setdefault("MPLCONFIGDIR", MPLCFG_DIR)

# ========= 交易商品設定 =========
EXCHANGE = "bybit"
PRODUCT  = "perpetual"       # spot/future/perpetual
SYMBOL   = "BTCUSDT"
INTERVAL = "15"             # 15m = 15分鐘
DB_TABLE = "price"           # 你的資料表名稱
FEATURES_PARQUET = os.path.join(RESULT_DIR, f"features_{EXCHANGE}_{PRODUCT}_{SYMBOL}_{INTERVAL}.parquet")

# ========= 環境 =========
from dotenv import load_dotenv
load_dotenv()
DB_CONFIG = {
    "dbname":   os.getenv("DBNAME"),
    "user":     os.getenv("USER"),
    "password": os.getenv("PASSWORD"),
    "host":     os.getenv("HOST", "127.0.0.1"),
    "port":     os.getenv("PORT", "5432"),
}

# ========= 日誌 =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "gaBybit.log")), logging.StreamHandler()]

)
logging.info("==== Pipeline start ====")

# ========= Pandas 與 GPU 環境小設定 =========
pd.options.mode.copy_on_write = False  # 避免 pandas 3.x CoW 把一些賦值視為錯誤
os.environ.setdefault("GABTC_DEBUG_CPU", "0")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ========= 依賴 =========
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas_ta as pta
import talib
from talib import abstract
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import torch
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and os.getenv("GABTC_DEBUG_CPU") != "1") else 'cpu')
logging.info(f"Device: {DEVICE}")

# ========= 工具 =========
def to_utc_index(ts: pd.Series) -> pd.DatetimeIndex:
    v0 = ts.iloc[0]
    if isinstance(v0, (int, np.integer, float, np.floating)):
        unit = "ms" if float(v0) > 1e11 else "s"
        idx = pd.to_datetime(ts.astype(np.int64), unit=unit, utc=True)
    else:
        parsed = pd.to_datetime(ts, errors="coerce", utc=False)
        if getattr(parsed.dt, "tz", None) is None:
            # assume Asia/Taipei then convert to UTC
            idx = parsed.dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")
        else:
            idx = parsed.dt.tz_convert("UTC")
    return pd.DatetimeIndex(idx)

def sanitize_cols(cols: List[str], prefix: str) -> List[str]:
    out = []
    for c in cols:
        cc = re.sub(r"[^\w\-.]+", "_", c)
        out.append(f"{prefix}{cc}")
    return out

# ========= 1) 從 DB 讀 OHLCV =========
def fetch_ohlcv_from_db() -> pd.DataFrame:
    sql = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM {DB_TABLE}
        WHERE exchange=%s AND product=%s AND symbol=%s AND interval=%s
        ORDER BY timestamp ASC
    """
    with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (EXCHANGE, PRODUCT, SYMBOL, INTERVAL))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError("資料庫沒有符合條件的資料（price 為空或條件不符）")
    df = pd.DataFrame(rows)
    df.index = to_utc_index(df["timestamp"])
    df = df.drop(columns=["timestamp"]).rename(columns=str.lower)
    # 型別清理
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

# ========= 2) pandas-ta 指標（關閉 multiprocessing + 排除 ha/hilo） =========
# ========= 抽象化：要算哪些指標 =========
PTA_SPEC = {
    "candles": [
        "cdl_doji", "cdl_inside", "ha"   # ha 走我們的向量化版本
    ],
    "cycles": [
        "ebsw"
    ],
    "momentum": [
        "ao","apo","bias","bop","brar","cci","cfo","cg","cmo","coppock","er","eri",
        "fisher","inertia","kdj","kst","macd","mom","pgo","ppo","psl","pvo","qqe",
        "roc","rsi","rsx","rvgi","slope","smi","squeeze","stoch","stochrsi",
        # "td_seq","trix","tsi","uo","willr"
    ],
    "overlap": [
        "alma","dema","ema","fwma","hilo","hl2","hlc3","hma","hwma","ichimoku",
        "kama","linreg","mcgd","midpoint","midprice","ohlc4","pwma","rma","sinwma",
        "sma","ssf","supertrend","swma","t3","tema","trima","vidya","vwap","vwma",
        "wcp","wma","zlma"
    ],
}

# ========= 特例：需要客製的函式（ha/ichimoku 等） =========
import inspect

def heikin_ashi_safe(df):
    o,h,l,c = [df[k].to_numpy(float) for k in ("open","high","low","close")]
    ha_c = (o+h+l+c)/4.0
    ha_o = ha_c.copy()
    ha_o[0] = (o[0]+c[0])/2.0
    for i in range(1,len(ha_o)):
        ha_o[i] = 0.5*(ha_o[i-1] + ha_c[i-1])
    ha_h = np.maximum.reduce([h,ha_o,ha_c])
    ha_l = np.minimum.reduce([l,ha_o,ha_c])
    out = pd.DataFrame({
        "pta_ha_open": ha_o, "pta_ha_high": ha_h,
        "pta_ha_low": ha_l,  "pta_ha_close": ha_c
    }, index=df.index)
    return out

SPECIAL_FUNCS = {
    "ha": heikin_ashi_safe,
    # ichimoku 在 pandas_ta 會回兩個 DataFrame，且第一個最後一欄是 Chikou（潛在洩漏），我們丟掉它
    "ichimoku": "special_ichimoku",
}

def _pta_call(fn_name, ohlcv):
    fn = getattr(pta, fn_name)
    sig = inspect.signature(fn)
    avail = {
        "open":   ohlcv["open"],
        "high":   ohlcv["high"],
        "low":    ohlcv["low"],
        "close":  ohlcv["close"],
        "volume": ohlcv["volume"] if "volume" in ohlcv.columns else None,
    }
    kwargs = {k: v for k, v in avail.items() if (v is not None and k in sig.parameters)}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*getitem.*positions is deprecated.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*incompatible dtype.*")
        warnings.filterwarnings("ignore", category=UserWarning,    message=".*PeriodArray.*timezone.*")
        res = fn(**kwargs)

    if res is None:
        return None
    if isinstance(res, pd.Series):
        res = res.to_frame(name=fn_name)

    # 統一為 float，避免後續拼接時的 dtype 警告
    res = res.apply(pd.to_numeric, errors="coerce").astype(float)

    res.index = ohlcv.index
    if res.shape[1] == 1 and res.columns[0] == fn_name:
        res.columns = sanitize_cols([fn_name], "pta_")
    else:
        res.columns = sanitize_cols([f"{fn_name}_{c}" for c in res.columns], "pta_")
    return res


def _run_one_indicator(name, ohlcv):
    try:
        # 特例：Heikin-Ashi
        if name in SPECIAL_FUNCS and SPECIAL_FUNCS[name] == heikin_ashi_safe:
            return heikin_ashi_safe(ohlcv)

        # 特例：Ichimoku（回兩份表 + 去掉 Chikou）
        if name == "ichimoku":
            fn = getattr(pta, "ichimoku")
            # 僅注入函式需要的參數
            sig = inspect.signature(fn)
            base = {"high": ohlcv["high"], "low": ohlcv["low"], "close": ohlcv["close"]}
            kwargs = {k:v for k,v in base.items() if k in sig.parameters}
            df1, df2 = fn(**kwargs)
            if isinstance(df1, pd.DataFrame) and df1.shape[1] >= 1:
                # 丟掉最後一欄（Chikou Span）
                df1 = df1.iloc[:, :-1]
            res = pd.concat([df1, df2], axis=1)
            if isinstance(res, pd.Series): res = res.to_frame()
            res.index = ohlcv.index
            res.columns = sanitize_cols([f"ichimoku_{c}" for c in res.columns], "pta_")
            return res

        # 一般指標：直接動態呼叫
        if hasattr(pta, name):
            return _pta_call(name, ohlcv)
        else:
            # 有些燭型在 pandas_ta 是 cdl_*；我們已經用 cdl_ 前綴了，若不存在就跳過
            return None
    except Exception as e:
        logging.debug(f"[pta:{name}] skipped ({e})")
        return None

# ========= 平行執行：threads 循環呼叫 =========
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm.auto import tqdm

def compute_pandas_ta_by_spec(ohlcv: pd.DataFrame, spec=PTA_SPEC, n_jobs=-1) -> pd.DataFrame:
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("ohlcv.index 需要是 DatetimeIndex 才能計算 vwap/部分指標")

    # 先留住原 index（tz-aware），計算時用 UTC naive 避免 PeriodArray 警告
    orig_index = ohlcv.index
    ohlcv_ = ohlcv.copy()
    if isinstance(ohlcv_.index, pd.DatetimeIndex) and ohlcv_.index.tz is not None:
        ohlcv_.index = ohlcv_.index.tz_convert("UTC").tz_localize(None)

    task_list = []
    for _, names in spec.items():
        task_list.extend(names)
    task_list = list(dict.fromkeys(task_list))  # 去重

    with tqdm_joblib(tqdm(total=len(task_list), desc="pandas_ta tasks", leave=False)):
        parts = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_run_one_indicator)(name, ohlcv_) for name in task_list
        )

    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(index=orig_index)

    out = pd.concat(parts, axis=1)
    out.index = orig_index  # 還原成原本 tz-aware index，方便後面 join
    out = out.loc[:, ~out.columns.duplicated()]
    return out



# ========= 3) TA-Lib 全函式（threads 平行） =========
def _run_talib(fn_name: str, ohlcv: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        func = abstract.Function(fn_name)
        res = func(ohlcv)
        if res is None:
            return None
        if isinstance(res, pd.Series):
            res = res.to_frame(name=fn_name)
        # 命名
        if res.shape[1] == 1 and res.columns[0] == fn_name:
            res.columns = sanitize_cols(res.columns.tolist(), "talib_")
        else:
            res.columns = sanitize_cols([f"{fn_name}_{c}" for c in res.columns], "talib_")
        res.index = ohlcv.index
        return res
    except Exception:
        return None

def compute_talib_all(ohlcv: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    fns = talib.get_functions()
    with tqdm_joblib(tqdm(total=len(fns), desc="TA-Lib functions", leave=False)):
        parts = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_run_talib)(fn, ohlcv) for fn in fns
        )
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts: return pd.DataFrame(index=ohlcv.index)
    out = parts[0]
    for i in range(1, len(parts)):
        out = out.join(parts[i], how="outer")
    return out

# ========= 4) 6 階變化率 =========
def add_pct_changes(df: pd.DataFrame, cols: List[str], steps=6) -> pd.DataFrame:
    frames=[df]
    for k in range(1, steps+1):
        chg = df[cols].pct_change(periods=k).replace([np.inf,-np.inf], np.nan).fillna(0)
        chg.columns = [f"{c}_chg{k}" for c in cols]
        frames.append(chg)
    return pd.concat(frames, axis=1)

# ========= 5) 多種標準化（rolling fit，分塊平行） =========
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
    Normalizer, PowerTransformer, QuantileTransformer, FunctionTransformer
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import clone

def pca_whitening(X):
    X = SimpleImputer(strategy="median").fit_transform(X)
    return PCA(whiten=True).fit_transform(X)

def rank_scaler(X):
    df = pd.DataFrame(X)
    return df.rank(method="average", pct=True).to_numpy()

def unit_vector_featurewise(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    d = np.linalg.norm(X, axis=0)
    d = np.where(d < eps, 1.0, d)
    return X / d

def tanh_estimator_scaling(X):
    X = np.asarray(X)
    mu, sd = np.mean(X, axis=0), np.std(X, axis=0) + 1e-12
    return 0.5 * (np.tanh(0.01 * (X - mu) / sd) + 1)

def zca_whitening(X):
    X = SimpleImputer(strategy="median").fit_transform(X)
    sigma = np.cov(X, rowvar=False)
    U, S, _ = np.linalg.svd(sigma)
    eps = 1e-5
    Z = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return (X - np.mean(X, axis=0)) @ Z

def row_maxabs_scaling(X):
    X = np.asarray(X)
    m = np.max(np.abs(X), axis=1).reshape(-1,1)
    m[m==0] = 1.0
    return X / m

SCALERS = {
    'z': StandardScaler(),
    'minmax': MinMaxScaler(),
    'maxabs': MaxAbsScaler(),
    'robust': RobustScaler(),
    'row_maxabs': FunctionTransformer(row_maxabs_scaling, validate=False),
    'rank': FunctionTransformer(rank_scaler, validate=False),
    'unit_vector': FunctionTransformer(unit_vector_featurewise, validate=False),
    'tanh': FunctionTransformer(tanh_estimator_scaling, validate=False),
    'zca': FunctionTransformer(zca_whitening, validate=False),
}

def _process_block(scaler_obj, X_full: np.ndarray, n_cols: int, start: int, end: int):
    scaler = clone(scaler_obj)
    out = np.empty((end-start, n_cols), dtype=np.float64)
    imp = SimpleImputer(strategy="median")
    for local, i in enumerate(range(start, end)):
        X = X_full[:i+1]
        X = imp.fit_transform(X)
        X = np.clip(X, -1e6, 1e6)
        # QuantileTransformer 動態調 n_quantiles
        if isinstance(scaler, QuantileTransformer):
            scaler.set_params(n_quantiles=min(getattr(scaler, "n_quantiles", 100), X.shape[0]))
        try:
            Xt = scaler.fit_transform(X)
        except Exception:
            Xt = X
        out[local,:] = Xt[-1,:n_cols]
    return start, out

def rolling_scalers_block(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X_full = df[feature_cols].to_numpy(np.float64)
    n_rows, n_cols = X_full.shape
    out_frames = [df]
    n_jobs_rows = min(os.cpu_count() or 1, 16)
    from math import ceil
    n_blocks = max(1, min(n_rows, n_jobs_rows*64))
    step = ceil(n_rows / n_blocks)
    blocks = [(s, min(s+step, n_rows)) for s in range(0, n_rows, step)]

    for name, scaler in tqdm(list(SCALERS.items()), desc=f"Scalers (n={len(SCALERS)}) | rows={n_rows}"):
        with tqdm_joblib(tqdm(total=len(blocks), desc=f"{name} blocks", leave=False)):
            blk = Parallel(
                n_jobs=n_jobs_rows, backend="loky", max_nbytes="100M", temp_folder=JOBLIB_TMP
            )([delayed(_process_block)(scaler, X_full, n_cols, s, e) for (s,e) in blocks])
        blk.sort(key=lambda x: x[0])
        arr = np.empty((n_rows, n_cols), dtype=np.float64)
        for s, part in blk:
            arr[s:s+part.shape[0], :] = part
        out_frames.append(pd.DataFrame(arr, columns=[f"{c}_{name}" for c in feature_cols], index=df.index))
        logging.info(f"Scaler done: {name} (+{n_cols} cols)")
    return pd.concat(out_frames, axis=1)

# ========= 6) 產生/讀取特徵 =========
def build_or_load_features() -> pd.DataFrame:
    if os.path.exists(FEATURES_PARQUET):
        try:
            df = pd.read_parquet(FEATURES_PARQUET, engine="pyarrow")
            logging.info(f"Loaded features parquet: {FEATURES_PARQUET} shape={df.shape}")
            return df
        except Exception as e:
            logging.warning(f"Read parquet failed: {e}. Will recompute.")

    logging.info("Reading OHLCV from DB ...")
    ohlcv = fetch_ohlcv_from_db()
    logging.info(f"OHLCV rows={len(ohlcv)}")

    logging.info("pandas_ta AllStrategy (exclude ha/hilo) ...")
    pta_feat = compute_pandas_ta_by_spec(ohlcv, spec=PTA_SPEC, n_jobs=1)
    logging.info(f"pandas_ta features: {pta_feat.shape}")

    logging.info("TA-Lib (threaded) ...")
    talib_feat = compute_talib_all(ohlcv, n_jobs=-1)
    logging.info(f"TA-Lib features: {talib_feat.shape}")

    # 合併原始 + 指標
    feats = ohlcv.join(pta_feat, how="left").join(talib_feat, how="left")

    # 清掉全 NaN 欄，再填基礎缺失（之後變化率才不會爆）
    feats = feats.dropna(axis=1, how="all")
    num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    feats[num_cols] = feats[num_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    # 6 階變化率
    logging.info("Adding pct_change(1..6) ...")
    feats = add_pct_changes(feats, num_cols, steps=6)

    # rolling 標準化（block 平行）
    logging.info("Rolling scalers (block parallel) ...")
    feats = rolling_scalers_block(feats, num_cols)

    feats.to_parquet(FEATURES_PARQUET, engine="pyarrow", compression="snappy", index=True)
    logging.info(f"Wrote features parquet: {FEATURES_PARQUET} shape={feats.shape}")
    return feats

df = build_or_load_features()