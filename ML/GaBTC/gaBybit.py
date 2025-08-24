
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
# 並行工具
from joblib import Parallel, delayed
from joblib import parallel_backend
# 進度
# ====== 1) imports：放在檔案前面，tqdm_joblib 可選 ======

from tqdm.auto import tqdm, trange
from tqdm_joblib import tqdm_joblib   # pip install tqdm-joblib


# 1) 讓 CUDA 同步回報錯誤（得到正確堆疊）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 2) cuBLAS 決定性工作空間（也常順便避開一些 internal error）
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 3) 出問題時改走 CPU（臨時開關：想用才設成 '1'）
os.environ.setdefault("GABTC_DEBUG_CPU", "0")


# 讀取檔案
file_path = os.path.join(os.path.dirname(__file__), 'BITSTAMP_BTCUSD, 240.csv')
base_dir = os.path.dirname(file_path)
result_path = os.path.join(base_dir, 'result_bybit')
os.makedirs(result_path, exist_ok=True)
log_path = os.path.join(base_dir, 'log')
os.makedirs(log_path, exist_ok=True)

# 日誌設定
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'gaBTC.log')),
        logging.StreamHandler()
    ]
)
logging.info(f"Starting gaBTC with file: {file_path}")

scaled_parquet = os.path.join(result_path, 'features_bybit_perpetual_BTCUSDT_240.parquet')
scaled_csv     = os.path.join(result_path, 'features_bybit_perpetual_BTCUSDT_240.csv')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False

# 預設需要重算，除非我們成功載入任一種「已縮放後」的快取
recompute = True

if os.path.exists(scaled_parquet):
    # 1) 直接讀 parquet
    try:
        df = pd.read_parquet(scaled_parquet, engine='pyarrow')
        logging.info(f"Loaded existing scaled parquet: {scaled_parquet} "
                     f"(rows={len(df)}, cols={len(df.columns)})")
        recompute = False
    except Exception as e:
        logging.warning(f"Read parquet failed ({scaled_parquet}): {e}. "
                        f"Will try CSV or recompute.")
        # 1a) parquet 壞掉就移除，避免之後誤用
        try:
            os.remove(scaled_parquet)
        except Exception:
            pass
        # 1b) 若 CSV 在就轉檔；否則才回到原始 CSV 重算
        if os.path.exists(scaled_csv):
            try:
                df = pd.read_csv(scaled_csv)
                logging.info(f"Loaded scaled CSV: {scaled_csv} "
                             f"(rows={len(df)}, cols={len(df.columns)})")
                # 轉存為 parquet（若沒裝 pyarrow 會失敗，但不影響後續）
                try:
                    df.to_parquet(scaled_parquet, engine='pyarrow', index=False)
                    logging.info(f"Converted CSV → Parquet: {scaled_parquet}")
                except Exception as e2:
                    logging.warning(f"CSV→Parquet convert failed: {e2}. "
                                    f"(Tip: pip install pyarrow)")
                recompute = False
            except Exception as e_csv:
                logging.warning(f"Read scaled CSV failed: {e_csv}. Will recompute.")
                df = pd.read_csv(file_path)  # 原始資料
                recompute = True
        else:
            df = pd.read_csv(file_path)      # 原始資料
            recompute = True

else:
    # 2) parquet 不在 → 檢查是否有舊的 scaled CSV
    if os.path.exists(scaled_csv):
        try:
            df = pd.read_csv(scaled_csv)
            logging.info(f"Loaded scaled CSV: {scaled_csv} "
                         f"(rows={len(df)}, cols={len(df.columns)})")
            # 轉存為 parquet（非必要，但之後就能直接讀 parquet）
            try:
                df.to_parquet(scaled_parquet, engine='pyarrow', index=False)
                logging.info(f"Converted CSV → Parquet: {scaled_parquet}")
            except Exception as e2:
                logging.warning(f"CSV→Parquet convert failed: {e2}. "
                                f"(Tip: pip install pyarrow)")
            recompute = False
        except Exception as e:
            logging.warning(f"Read scaled CSV failed: {e}. Will recompute.")
            df = pd.read_csv(file_path)      # 原始資料
            recompute = True
    else:
        # 3) 兩個快取都沒有 → 從原始 CSV 讀，等會兒跑完流程會輸出 parquet
        df = pd.read_csv(file_path)
        logging.info("No scaled parquet/csv found. Will recompute features.")
        recompute = True

if recompute:
    # --- 資料清洗 ---
    # drop first 676 rows
    df = df.iloc[676:].reset_index(drop=True)
    logging.info(f"Dropped first 676 rows. New row count: {len(df)}")

    # 預處理
    # 統計每欄與每列的 NaN 數量
    col_nan_counts = df.isna().sum(axis=0)
    row_nan_counts = df.isna().sum(axis=1)

    # 簡短日誌摘要
    logging.info(f"Total NaNs in dataframe: {int(df.isna().sum().sum())}")
    logging.info(f"NaNs per column: {col_nan_counts.to_dict()}")

    df = df.dropna(how='any', axis=1)
    df = df.dropna(how='any', axis=0)

    logging.info(f"import features: {df.columns.tolist()}")
    logging.info(f"Number of available features: {len(df.columns)}")
    logging.info(f"Number of rows after NaN removal: {len(df)}")

    # 選數值欄，但排除時間戳等不該做 pct_change 的欄位
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_like = {'time', 'timestamp'}
    numeric_cols = [c for c in numeric_cols if c.lower() not in exclude_like]

    chg_frames = []
    for k in range(1, 7):
        chg = df[numeric_cols].pct_change(periods=k)
        chg = chg.replace([np.inf, -np.inf], np.nan).fillna(0)
        chg.columns = [f"{c}_chg{k}" for c in numeric_cols]
        chg_frames.append(chg)

    # 一次性合併，避免反覆插欄
    df = pd.concat([df] + chg_frames, axis=1)
    logging.info(f"Added change-rate features (vectorized). New column count: {df.shape[1]}")

    # 轉換相關函式與 scalers 定義
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
        Normalizer, PowerTransformer, QuantileTransformer
    )
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA

    def pca_whitening(X):
        X = np.asarray(X)
        X = SimpleImputer(strategy="median").fit_transform(X)
        return PCA(whiten=True).fit_transform(X)

    log_trans = lambda x: np.log1p(x)
    def rank_scaler(X):
        if isinstance(X, pd.DataFrame):
            return X.rank(method="average", pct=True).to_numpy()
        else:
            return pd.DataFrame(X).rank(method="average", pct=True).to_numpy()

    from sklearn.preprocessing import Normalizer
    l1norm = Normalizer(norm='l1')
    l2norm = Normalizer(norm='l2')
    def unit_vector_featurewise(X, eps=1e-12):
        X = np.asarray(X, dtype=np.float64)
        denom = np.linalg.norm(X, axis=0)
        denom = np.where(denom < eps, 1.0, denom)
        return X / denom

    def sigmoid_scaling(x):
        X = np.asarray(x, dtype=np.float64)
        X = np.clip(X, -500, 500)
        return 1.0 / (1.0 + np.exp(-X))

    def tanh_estimator_scaling(X):
        X = np.asarray(X)
        return 0.5 * (np.tanh(0.01 * (X - np.mean(X)) / np.std(X)) + 1)

    def zca_whitening(X):
        X = np.asarray(X)
        X = SimpleImputer(strategy="median").fit_transform(X)
        sigma = np.cov(X, rowvar=False)
        U, S, _ = np.linalg.svd(sigma)
        epsilon = 1e-5
        ZCAMatrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
        return (X - np.mean(X, axis=0)) @ ZCAMatrix

    def row_maxabs_scaling(X):
        X = np.asarray(X)
        max_per_row = np.max(np.abs(X), axis=1).reshape(-1, 1)
        return X / max_per_row

    def mean_centering(x):
        return x - np.mean(x, axis=0)

    from sklearn.preprocessing import FunctionTransformer
    scalers = {
        'z': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'row_maxabs': FunctionTransformer(row_maxabs_scaling, validate=False),
        'mean_centering': FunctionTransformer(mean_centering, validate=False),
        'rank': FunctionTransformer(rank_scaler, validate=False),
        'unit_vector': FunctionTransformer(unit_vector_featurewise, validate=False),
        'tanh': FunctionTransformer(tanh_estimator_scaling, validate=False),
        'zca': FunctionTransformer(zca_whitening, validate=False),
    }

    from sklearn.impute import SimpleImputer
    from sklearn.base import clone
    from joblib import Parallel, delayed
    import numpy as np

    n_rows = len(df)
    df_values = df[numeric_cols].to_numpy(dtype=np.float64)
    # ====== 並行設定：一個 scaler 的所有時間點用區塊平行，再往下個 scaler ======
    from math import ceil

    def _process_block(scaler_obj, df_values, n_cols, start, end):
        """計算單一 scaler 在 [start, end) 這段列的結果，回傳 (start, block_array)。"""
        scaler = clone(scaler_obj)
        out_blk = np.empty((end - start, n_cols), dtype=np.float64)

        # 逐 i 計算，彼此獨立（每次都用前 i+1 筆資料）
        for local_idx, i in enumerate(range(start, end)):
            X = df_values[: i + 1].copy()
            X = SimpleImputer(strategy="median").fit_transform(X)
            X = np.clip(X, -1e6, 1e6)

            # 有些轉換（如 QuantileTransformer）需要依樣本數動態調整參數
            try:
                from sklearn.preprocessing import QuantileTransformer
                if isinstance(scaler, QuantileTransformer):
                    n_q = getattr(scaler, "n_quantiles", 100)
                    scaler.set_params(n_quantiles=min(int(n_q), X.shape[0]))
            except Exception:
                pass

            try:
                Xt = scaler.fit_transform(X)
            except Exception:
                Xt = X  # 保底

            out_blk[local_idx, :] = Xt[-1, :n_cols]

        return start, out_blk

    n_rows = len(df)
    n_cols = len(numeric_cols)
    df_values = df[numeric_cols].to_numpy(dtype=np.float64)

    # 每個 scaler 內部用多少平行工作
    n_jobs_rows = min(os.cpu_count() or 1, 16)        # 可自行調整上限
    # 將時間點切成多少區塊（多一點區塊，進度條更平滑、調度更彈性）
    n_blocks = max(1, min(n_rows, n_jobs_rows* 64))

    # 產生區塊邊界
    block_idxs = []
    step = ceil(n_rows / n_blocks)
    for s in range(0, n_rows, step):
        e = min(s + step, n_rows)
        block_idxs.append((s, e))

    parallel_results = []
    # 外層：依序處理每個 scaler（顯示「所有 scaler 的進度」）
    for sname, scaler in tqdm(list(scalers.items()),
                            desc=f"Scalers (n={len(scalers)}) | rows={n_rows}"):
        # 內層：平行把該 scaler 的所有時間區塊算完（顯示該 scaler 的區塊進度）
        desc = f"{sname} blocks ({len(block_idxs)})"
        with tqdm_joblib(tqdm(total=len(block_idxs), desc=desc, leave=False)):
            blk_results = Parallel(
                n_jobs=n_jobs_rows,
                backend='loky',
                max_nbytes="100M",
                temp_folder=os.path.join(base_dir, 'joblib_tmp')
            )(
                delayed(_process_block)(scaler, df_values, n_cols, start, end)
                for (start, end) in block_idxs
            )

        # 按起點排序並回填到完整陣列
        blk_results.sort(key=lambda x: x[0])
        result = np.empty((n_rows, n_cols), dtype=np.float64)
        offset = 0
        for start, arr in blk_results:
            end = start + arr.shape[0]
            result[start:end, :] = arr

        parallel_results.append((sname, result))
        logging.info(f"Completed rolling transform for scaler: {sname} (+{result.shape[1]} cols)")

    # 合併結果回 df（一次性 concat，避免碎片化）
    to_concat = [df]
    for sname, result in parallel_results:
        add = pd.DataFrame(result, columns=[f"{c}_{sname}" for c in numeric_cols], index=df.index)
        to_concat.append(add)

    df = pd.concat(to_concat, axis=1)
    logging.info(f"number of features after scaling: {df.shape[1]}")
    df.to_parquet(scaled_parquet, engine='pyarrow', index=False)
    logging.info(f"Wrote scaled results to: {scaled_parquet}")



# ===================== GA with CUDA (PyTorch) =====================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"GA device = {device}")

def _ensure_cuda_f32_contig(x, device):
    if x.device.type != 'cuda':
        x = x.to(device, non_blocking=True)
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return x


@torch.no_grad()
def mutate(children, mut_sigma, mask=None):
    # 避免對奇怪的 view 做 in-place，先複製成連續區塊
    if not children.is_contiguous():
        children = children.contiguous()
    # 產生雜訊用 float32 on-device
    noise = torch.randn_like(children, dtype=torch.float32, device=children.device)
    if mask is not None:
        # mask 可是 broadcastable 的同型狀或 [1, D]；確保 dtype
        if mask.dtype != torch.float32:
            mask = mask.float()
        noise = noise * mask
    mutated = children + mut_sigma * noise
    # 清掉非有限值，避免把 NaN/Inf 帶進下一輪
    mutated = torch.nan_to_num(mutated, nan=0.0, posinf=1e6, neginf=-1e6)
    return mutated
def _check_finite(name, x):
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values: {x.dtype}, {x.shape}")
def safe_eval_fitness(weights: torch.Tensor) -> torch.Tensor:
    try:
        return eval_fitness(weights)
    except RuntimeError as e:
        msg = str(e).lower()
        # 這些通常表示 CUDA context 已經髒掉，繼續跑只會一直炸
        if ("illegal memory access" in msg or 
            "cublas_status" in msg or 
            "out of memory" in msg):
            # 先把當前種群保存成 panic checkpoint，避免回溯太多
            try:
                save_ckpt(gen if 'gen' in locals() else -1, pop,
                          best_score_so_far, best_w_so_far)
            except Exception as _:
                logging.warning("panic checkpoint failed; continuing restart")
            # _hard_restart_from_ckpt(reason=f"CUDA runtime error: {e}")
        # 其它錯誤照舊丟出
        raise
# 例：在選擇/交配/變異後、eval_fitness 前後都插
# 1) 構建特徵矩陣 X 與報酬向量 ret（避免展望偏誤：用 ret_{t+1}）
#    排除明顯價格欄；其他數值欄都視為特徵（包含你前面生成的 chg/scaler 欄位）
exclude_cols = {'time','timestamp','date','datetime'}
feature_cols = [c for c in df.columns
                if (isinstance(df[c].dtype, np.dtype) and np.issubdtype(df[c].dtype, np.number))
                and (c.strip().lower() not in exclude_cols)]

if 'close' not in df.columns:
    raise ValueError("找不到 close 欄位，請確認原始 CSV 有 close。")

# y = 下一根的 close 報酬；position_t 乘以 ret_{t+1}
ret_np = df['close'].pct_change().shift(-1).fillna(0).to_numpy(np.float32)
X_np   = df[feature_cols].to_numpy(np.float32)

# 對齊長度（去掉最後一列，因為 ret 已 shift(-1)）
X_np   = X_np[:-1]
ret_np = ret_np[:-1]

T, D = X_np.shape
logging.info(f"GA data ready | T={T}, D={D}, features={len(feature_cols)}")

X   = torch.from_numpy(X_np).contiguous().to(device)
ret = torch.from_numpy(ret_np).contiguous().to(device)


# 2) GA 參數（可依機器調整）
pop_size       = 2**13          # 族群大小（越大越穩、也越耗）
generations    = 2**30          # 代數
elite_frac     = 0.05         # 菁英比例
mut_rate       = 0.10         # 突變機率
mut_sigma      = 0.05         # 突變幅度（高斯雜訊）
crossover_rate = 0.70         # 交配比率（uniform）
tcost          = 0.00       # 單位交易成本（例如 50bps）
l2_penalty     = 1e-4         # 權重 L2 正則
bound_with_tanh = True        # 用 tanh 把部位限制在 [-1,1]

# 3) 初始化族群（權重向量）
pop = torch.randn(pop_size, D, device=device) * 0.1

import math

# ---- 年化係數（4H K線 -> 1天6根）----
BAR_HOURS = 4
periods_per_day  = int(24 / BAR_HOURS)       # = 6
periods_per_year = 365 * periods_per_day     # = 2190

# ---- 選擇你要最大化的目標 ----
# 可選: 'cumlog'｜'sharpe'｜'sortino'｜'calmar'｜'omega'
# fitness_mode = 'cumlog'  # 選擇適存函式
fitness_mode = 'sharpe'  # 選擇適存函式
# fitness_mode = 'sortino'  # 選擇適存函式
# fitness_mode = 'calmar'  # 選擇適存函式
# fitness_mode = 'omega'  # 選擇適存函式


def make_fitness_fn(mode: str,
                    periods_per_year: int,
                    tcost: float,
                    l2_penalty: float,
                    bound_with_tanh: bool):
    """回傳 eval_fitness(weights) -> [pop] 的可切換適存函式"""
    sqrt_apy = torch.sqrt(torch.tensor(float(periods_per_year),
                                       device=device, dtype=torch.float32))
    eps = 1e-12

    def eval_fitness(weights: torch.Tensor) -> torch.Tensor:
        # weights: [pop, D]
        # ---- 安全 matmul：確保 dtype/裝置/連續 + 分塊 ----
        X_dev = _ensure_cuda_f32_contig(X, X.device)
        W_dev = _ensure_cuda_f32_contig(weights, X.device)

        m, k = X_dev.shape
        n, k2 = W_dev.shape
        assert k == k2, f"Shape mismatch: X={tuple(X_dev.shape)}, W={tuple(W_dev.shape)}"

        if (torch.isnan(X_dev).any() or torch.isnan(W_dev).any() or
            torch.isinf(X_dev).any() or torch.isinf(W_dev).any()):
            raise ValueError("NaN/Inf detected in X or W before matmul")

        if k == 0 or n == 0 or m == 0:
            raw = torch.zeros((m, n), device=X_dev.device, dtype=torch.float32)
        else:
            CHUNK = 512  # 需要時可調 256/1024
            outs = []
            for i in range(0, n, CHUNK):
                subW = W_dev[i:i+CHUNK]
                if not subW.is_contiguous():
                    subW = subW.contiguous()
                outs.append(X_dev @ subW.t())  # [T, chunk]
            raw = torch.cat(outs, dim=1)       # [T, pop]

        if not torch.isfinite(raw).all():
            raise ValueError("Non-finite detected in raw after matmul")

        pos = torch.tanh(raw) if bound_with_tanh else raw  # 部位界限

        # turnover 與 交易成本
        pos_prev = torch.zeros_like(pos)
        pos_prev[1:] = pos[:-1]
        turnover = (pos - pos_prev).abs()
        pnl = pos * ret.unsqueeze(1) - tcost * turnover    # [T, pop]

        if mode == 'cumlog':
            # 最大化總報酬：exp(sum(log1p(pnl))) - 1
            logret = torch.log1p(pnl.clamp(min=-0.9999))
            gross  = logret.sum(dim=0).exp() - 1.0
            metric = gross

        elif mode == 'sharpe':
            mean = pnl.mean(dim=0)
            std  = pnl.std(dim=0).clamp_min(eps)
            metric = (mean / std) * sqrt_apy

        elif mode == 'sortino':
            downside = torch.clamp(-pnl, min=0.0)
            dstd = downside.std(dim=0).clamp_min(eps)
            mean = pnl.mean(dim=0)
            metric = (mean / dstd) * sqrt_apy

        elif mode == 'calmar':
            # CAGR / |MaxDD|
            logret = torch.log1p(pnl.clamp(min=-0.9999))
            cumlog = torch.cumsum(logret, dim=0)           # [T, pop]
            equity = torch.exp(cumlog)                      # 起始1的權益曲線
            runmax, _ = torch.cummax(equity, dim=0)
            dd = (equity / runmax - 1.0).min(dim=0).values.abs().clamp_min(eps)

            total_log = cumlog[-1, :]                       # ln(1+總報酬)
            cagr = torch.exp(total_log * (periods_per_year / float(T))) - 1.0
            metric = cagr / dd

        elif mode == 'omega':
            # Omega(0) ≈ Σ正pnl / Σ負pnl
            gains  = torch.clamp(pnl,  min=0.0).sum(dim=0)
            losses = torch.clamp(-pnl, min=0.0).sum(dim=0).clamp_min(eps)
            metric = gains / losses

        else:
            raise ValueError(f"Unknown fitness mode: {mode}")

        # L2 懲罰（越大越扣分）
        reg = l2_penalty * (weights**2).sum(dim=1)
        score = metric - reg

        # 安全處理數值
        score = torch.nan_to_num(score, nan=-1e9, posinf=-1e9, neginf=-1e9)
        return score

    return eval_fitness

# 建立可切換的適存函式
eval_fitness = make_fitness_fn(
    mode=fitness_mode,
    periods_per_year=periods_per_year,
    tcost=tcost,
    l2_penalty=l2_penalty,
    bound_with_tanh=bound_with_tanh
)


# ===== GA Checkpoint =====
import random

checkpoint_dir = os.path.join(result_path, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f'ga_{fitness_mode}_pop{pop_size}.pt')

# 每隔多少代存一次；依你的 generations 調整
checkpoint_every = 500
resume_from_ckpt = True  # True: 若檔案存在則續跑

# 方便記錄/檢查參數（可自行加上你想追蹤的超參數）
_ga_params = dict(
    pop_size=pop_size, generations=generations, elite_frac=elite_frac,
    mut_rate=mut_rate, mut_sigma=mut_sigma, crossover_rate=crossover_rate,
    tcost=tcost, l2_penalty=l2_penalty, bound_with_tanh=bound_with_tanh,
    fitness_mode=fitness_mode, D=D, T=T
)

def save_ckpt(gen:int, pop:torch.Tensor, best_score:float=None, best_w:np.ndarray=None):
    """原子性保存 checkpoint（先寫 tmp 再替換）"""
    state = {
        'gen': gen,
        'ga_params': _ga_params,
        'pop': pop.detach().cpu(),          # 存 CPU，避免裝置不一致
        'rng_py': random.getstate(),
        'rng_np': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'best_score': float(best_score) if best_score is not None else None,
        'best_w': best_w if best_w is not None else None,
    }
    tmp = checkpoint_path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, checkpoint_path)
    logging.info(f"[CKPT] Saved checkpoint at gen={gen} → {checkpoint_path}")

def try_load_ckpt():
    """若存在 checkpoint，回傳 (start_gen, pop_loaded, state)；否則 (0, None, None)"""
    if not (resume_from_ckpt and os.path.exists(checkpoint_path)):
        return 0, None, None
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pop_loaded = state['pop'].to(device)
    # 還原 RNG 狀態（非必要，但可重現性更好）
    try:
        random.setstate(state['rng_py'])
        np.random.set_state(state['rng_np'])
        torch.set_rng_state(state['rng_torch'])
        if device == 'cuda' and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    except Exception as e:
        logging.warning(f"[CKPT] RNG restore failed: {e}")

    start_gen = int(state.get('gen', -1)) + 1
    logging.info(f"[CKPT] Resuming from gen={start_gen} using {checkpoint_path}")
    return start_gen, pop_loaded, state

# 嘗試從 checkpoint 續跑
start_gen, pop_ckpt, ckpt_state = try_load_ckpt()
if pop_ckpt is not None:
    if pop_ckpt.shape == pop.shape:
        pop = pop_ckpt
    else:
        logging.warning(f"[CKPT] Pop shape mismatch {pop_ckpt.shape} vs {pop.shape}; starting fresh.")
        start_gen, ckpt_state = 0, None

best_score_so_far = ckpt_state.get('best_score') if ckpt_state else None
best_w_so_far     = ckpt_state.get('best_w')     if ckpt_state else None

import sys
import math
import csv

# 你可以改成別的鍵；Windows 支援單鍵即時偵測，Unix 以 Ctrl+C 為主（或要按 Enter）
STOP_KEYS = {b'q', b'Q', b'\x1b'}  # q / Q / ESC

# 5) GA 主迴圈（全在 GPU 上向量化）
# --- CUDA 出錯時的硬重啟工具 ---
def _hard_restart_from_ckpt(reason: str = ""):
    logging.error(f"[RESTART] {reason}  → 保存現況並重啟程序，從 checkpoint 續跑。")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    # 重新 exec 自己（Windows/Linux 都可），啟動後會自動 try_load_ckpt() 續跑
    os.execv(sys.executable, [sys.executable] + sys.argv)




try:
    for gen in range(start_gen, generations):
        _check_finite("pop_before_eval", pop)
        fitness = safe_eval_fitness(pop)

        elite_n  = max(1, int(pop_size * elite_frac))
        elite_idx = torch.topk(fitness, elite_n).indices
        elite     = pop[elite_idx]

        # 輪盤選擇（避免負值問題）
        probs = (fitness - fitness.min() + 1e-12)
        probs = probs / probs.sum()

        # 抽父母
        need = pop_size - elite_n
        idx1 = torch.multinomial(probs, need, replacement=True)
        idx2 = torch.multinomial(probs, need, replacement=True)
        p1, p2 = pop[idx1], pop[idx2]

        # uniform crossover
        if crossover_rate > 0:
            mask = (torch.rand_like(p1) < 0.5)
            children = torch.where(mask, p1, p2)
        else:
            children = p1.clone()

        # mutation
        # mutation（使用安全版，避免 in-place/view 汙染）
        if mut_rate > 0:
            mut_mask = (torch.rand_like(children) < mut_rate).float()
            children = mutate(children, mut_sigma, mask=mut_mask)


        # 下一代
        pop = torch.cat([elite, children], dim=0)

        if (gen+1) % 10 == 0 or gen == 1:
            best = fitness[elite_idx[0]].item()
            mean = fitness.mean().item()
            logging.info(f"Gen {gen+1}/{generations} | best={best:.6f} | mean={mean:.6f}")

        # ---- 定期寫 checkpoint ----
        if (gen+1) % checkpoint_every == 0 or (gen+1) == generations:
            with torch.no_grad():
                bidx = torch.argmax(fitness)
                best_score_so_far = float(fitness[bidx].item())
                best_w_so_far = pop[bidx].detach().cpu().numpy()
            save_ckpt(gen, pop, best_score_so_far, best_w_so_far)
except KeyboardInterrupt:
    logging.info("[STOP] Caught KeyboardInterrupt. Saving artifacts...")
except Exception as e:
    logging.error(f"[STOP] Caught unexpected exception: {e}")

# 6) 輸出最佳解，回測績效 & 存檔
fitness = safe_eval_fitness(pop)  # 最後一次評估
best_idx = torch.argmax(fitness)
best_w = pop[best_idx].detach().cpu().numpy()
# 最佳權重的績效（用 numpy 做最後的圖）
pos = np.tanh(X_np @ best_w) if bound_with_tanh else (X_np @ best_w)
turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
pnl = pos * ret_np - tcost * turnover
equity = np.cumprod(1 + pnl)
cum_ret = equity[-1] - 1.0
periods_per_day  = int(24/4)       # 4H bar -> 6
periods_per_year = 365 * periods_per_day
sharpe = (pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(periods_per_year)
logging.info(f"GA done | cum_ret={cum_ret:.4f} | sharpe={sharpe:.3f}")

# === Buy & Hold（常態滿倉，含初始建倉一次成本）===
pos_bh = np.ones_like(ret_np)
turnover_bh = np.zeros_like(ret_np); turnover_bh[0] = 1.0  # 初始從 0 -> 1
pnl_bh = pos_bh * ret_np - tcost * turnover_bh
equity_bh = np.cumprod(1 + pnl_bh)
cum_ret_bh = equity_bh[-1] - 1.0
sharpe_bh = (pnl_bh.mean() / (pnl_bh.std() + 1e-12)) * np.sqrt(periods_per_year)
logging.info(f"Buy&Hold | cum_ret={cum_ret_bh:.4f} | sharpe={sharpe_bh:.3f}")

# 存權重與圖
weights_csv = os.path.join(result_path, 'ga_best_weights.csv')
pd.Series(best_w, index=feature_cols, name='weight').to_csv(weights_csv)
np.save(os.path.join(result_path, 'ga_best_weights.npy'), best_w)

# 另存兩條 equity（方便之後比對/畫圖）
np.save(os.path.join(result_path, 'equity_ga.npy'), equity)
np.save(os.path.join(result_path, 'equity_bh.npy'), equity_bh)

plt.figure()
plt.plot(equity, label='GA best')
plt.plot(equity_bh, label='Buy & Hold')
plt.title('Equity Curve: GA vs Buy & Hold')
plt.grid(True)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_path, f'ga_vs_bh_equity_{fitness_mode}.png'), dpi=150)
plt.close()

logging.info(f"Saved weights to: {weights_csv}")
logging.info(f"Saved equity curves to: {os.path.join(result_path, f'ga_vs_bh_equity_{fitness_mode}.png')}")
