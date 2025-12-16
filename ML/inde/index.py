#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IndicatorNet on BITSTAMP_BTCUSD,240more.csv
- PyTorch + (auto) CUDA
- TimeSeries cross-validation
- Sharpe ratio as training objective
- Plot strategy vs Buy&Hold equity curve
"""

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt


# ====== 工具：Sharpe / 週期估算 ======
def _estimate_periods_per_year_from_ts(timestamps):
    try:
        ts = np.asarray(timestamps)
        if ts.size < 2:
            return np.nan
        # 若為數字型（unix seconds or ms）
        if np.issubdtype(ts.dtype, np.integer) or np.issubdtype(ts.dtype, np.floating):
            diffs = np.diff(ts.astype(np.int64))
            median_diff = float(np.median(diffs))
            # 若數值看起來像毫秒 (大於 1e10)，轉成秒
            if median_diff > 1e10:
                median_diff = median_diff / 1000.0
        else:
            # datetime64 -> 轉成 seconds
            secs = ts.astype('datetime64[s]').astype(np.int64)
            diffs = np.diff(secs)
            median_diff = float(np.median(diffs))
        if median_diff <= 0:
            return np.nan
        periods_per_year = 365.0 * 24.0 * 3600.0 / median_diff
        return periods_per_year
    except Exception:
        return np.nan


def compute_sharpes(returns: np.ndarray, timestamps=None):
    """回傳 (per_period_sharpe, annualized_sharpe)
    若 timestamps 提供，會估算每年期間數並回傳年化 Sharpe。
    """
    r = np.asarray(returns)
    if r.size == 0:
        return float('nan'), float('nan')
    per_sharpe = float(r.mean() / (r.std(ddof=0) + 1e-6))
    if timestamps is None:
        return per_sharpe, float('nan')
    ppy = _estimate_periods_per_year_from_ts(timestamps)
    if np.isnan(ppy) or ppy <= 0:
        return per_sharpe, float('nan')
    ann_sharpe = per_sharpe * np.sqrt(ppy)
    return per_sharpe, ann_sharpe


# ===================== 工具：固定 random seed =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== Model：IndicatorNet =====================
# ===================== Model：General TA IndicatorNet =====================

class IndicatorNet(nn.Module):
    """
    General TA-style IndicatorNet：
    - 線性平滑（Conv → 類 MA/EMA/WMA）
    - 視窗統計：mean / std / max / min / range
    - 動量：最後一根 - 第一根
    - Stoch/RSI-like：在 high-low / 漲跌分解上做 normalization
    - OBV-like：sign(Δclose) * volume 累積

    Input:  x ∈ R^{batch, L, d}   (window OHLCV)
    Output: alpha ∈ R^{batch} ∈ [-1, 1]   (position / signal)
    """

    def __init__(self,
                 in_dim: int,
                 window_length: int,
                 num_kernels: int = 16,
                 hidden_dim: int = 64,
                 idx_open: int = 0,
                 idx_high: int = 1,
                 idx_low: int = 2,
                 idx_close: int = 3,
                 idx_vol: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.window_length = window_length
        self.num_kernels = num_kernels

        # OHLCV index（對應 load_ohlcv_csv 裡的順序）
        self.idx_open = idx_open
        self.idx_high = idx_high
        self.idx_low = idx_low
        self.idx_close = idx_close
        self.idx_vol = idx_vol

        # ① 線性平滑：Conv kernel_size = window_length → 每個 kernel 看整個視窗
        #    這可以學出一群「泛 MA / EMA / WMA」filter
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=num_kernels,
            kernel_size=window_length,
            bias=False
        )

        # 下面這些 feature：
        # - smooth_feat: num_kernels
        # - mean_feat:   in_dim
        # - std_feat:    in_dim
        # - max_feat:    in_dim
        # - min_feat:    in_dim
        # - last_feat:   in_dim
        # - momentum:    in_dim
        # - stoch_close: 1
        # - range_hl:    1
        # - rsi_like:    1
        # - obv_like:    1
        # => 總維度 = num_kernels + 6*in_dim + 4
        feat_dim = num_kernels + 6 * in_dim + 4

        # ② MLP：把這一整包「TA-style feature」組起來 → alpha_t
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, L, in_dim)
        """
        B, L, D = x.shape
        assert L == self.window_length, "window_length 不一致"

        # ----- ① 線性平滑（MA/EMA family）：Conv over time -----
        # x: (B, in_dim, L) → conv → (B, num_kernels, 1)
        x_ch = x.transpose(1, 2)
        smooth_feat = self.conv(x_ch).squeeze(-1)   # (B, K)

        # ----- ② 視窗統計：mean / std / max / min -----
        mean_feat = x.mean(dim=1)                   # (B, D)
        std_feat = x.std(dim=1, unbiased=False)     # (B, D)
        max_feat, _ = x.max(dim=1)                  # (B, D)
        min_feat, _ = x.min(dim=1)                  # (B, D)

        # ----- ③ 動量：最後一根 - 第一根 -----
        first_feat = x[:, 0, :]                     # (B, D)
        last_feat = x[:, -1, :]                     # (B, D)
        momentum = last_feat - first_feat           # (B, D)

        # ----- ④ Oscillator / Band 類結構 -----
        close_series = x[:, :, self.idx_close]      # (B, L)
        high_series  = x[:, :, self.idx_high]       # (B, L)
        low_series   = x[:, :, self.idx_low]        # (B, L)
        vol_series   = x[:, :, self.idx_vol]        # (B, L)

        # 4-1 Stoch-style：視窗內 close 在 [minC, maxC] 的位置
        max_close, _ = close_series.max(dim=1, keepdim=True)   # (B, 1)
        min_close, _ = close_series.min(dim=1, keepdim=True)   # (B, 1)
        last_close = close_series[:, -1:].clone()              # (B, 1)
        stoch_close = (last_close - min_close) / (max_close - min_close + 1e-6)  # (B,1), 0~1

        # 4-2 Band-range：視窗內 high-low 區間
        max_high, _ = high_series.max(dim=1, keepdim=True)     # (B, 1)
        min_low, _ = low_series.min(dim=1, keepdim=True)       # (B, 1)
        range_hl = max_high - min_low                          # (B, 1)

        # 4-3 RSI-style：把 Δclose 分成 up / down，做一個 0~1 的強弱值
        diff_close = close_series[:, 1:] - close_series[:, :-1]   # (B, L-1)
        up = torch.relu(diff_close)
        down = torch.relu(-diff_close)
        up_sum = up.sum(dim=1, keepdim=True)                      # (B,1)
        down_sum = down.sum(dim=1, keepdim=True)                  # (B,1)
        rsi_like = up_sum / (up_sum + down_sum + 1e-6)            # (B,1), 0~1

        # ----- ⑤ OBV-style：累積 sign(Δclose) * volume -----
        vol_mid = vol_series[:, 1:]                               # (B, L-1)
        direction = torch.sign(diff_close)                        # (B, L-1)
        obv_like = (direction * vol_mid).sum(dim=1, keepdim=True) # (B,1)

        # ----- ⑥ 拼成一個大 feature 向量，交給 MLP -----
        feat_list = [
            smooth_feat,
            mean_feat,
            std_feat,
            max_feat,
            min_feat,
            last_feat,
            momentum,
            stoch_close,
            range_hl,
            rsi_like,
            obv_like,
        ]
        feat = torch.cat(feat_list, dim=1)        # (B, feat_dim)

        out = self.mlp(feat).squeeze(-1)          # (B,)
        # 壓到 [-1, 1] 當作可連續 position
        alpha = torch.tanh(out)
        return alpha


# ===================== Dataset =====================

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, L, d)  float32
        y: (N,)       float32  (future log return)
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===================== Data Prep =====================

def load_ohlcv_csv(path: str):
    """
    讀 BITSTAMP_BTCUSD,240more.csv
    自動找:
      - 時間欄: 包含 "time" 或 "date"
      - 價格欄: open, high, low, close
      - 量: volume / vol
    回傳:
      timestamps: (T,) datetime64
      features:   (T, d)  (O,H,L,C,V)
      close:      (T,)
    """
    df = pd.read_csv(path)

    cols_lower = {c.lower(): c for c in df.columns}

    def find_col(candidates):
        for name in candidates:
            for key, col in cols_lower.items():
                if name in key:
                    return col
        raise ValueError(f"找不到欄位，候選名字: {candidates}, 請手動改程式中的欄位名稱")

    ts_col = find_col(["time", "date"])
    close_col = find_col(["close"])
    open_col = find_col(["open"])
    high_col = find_col(["high"])
    low_col = find_col(["low"])
    vol_col = find_col(["volume", "vol"])

    # 兼容時間欄為 unix timestamp (秒 or 毫秒) 或字串時間
    raw_ts = df[ts_col]
    if np.issubdtype(raw_ts.dtype, np.integer) or np.issubdtype(raw_ts.dtype, np.floating):
        # 數字 timestamp，判斷是秒還是毫秒（若值 > 1e12 則視為毫秒）
        sample = int(raw_ts.iloc[0])
        if sample > 1e12:
            df[ts_col] = pd.to_datetime(raw_ts, unit='ms')
        else:
            df[ts_col] = pd.to_datetime(raw_ts, unit='s')
    else:
        df[ts_col] = pd.to_datetime(raw_ts)
    df = df.sort_values(ts_col).reset_index(drop=True)

    features = df[[open_col, high_col, low_col, close_col, vol_col]].values.astype(np.float64)
    timestamps = df[ts_col].values
    close = df[close_col].values.astype(np.float64)

    return timestamps, features, close


def build_windows(features: np.ndarray,
                  close: np.ndarray,
                  timestamps: np.ndarray,
                  window_length: int = 64,
                  horizon: int = 1):
    """
    將 OHLCV 做成 sliding windows + 未來 log return label

    features: (T, d)
    close:    (T,)
    timestamps: (T,)
    window_length: L
    horizon: H  (預測 t+H 的 log return)

    回傳:
      X:  (N, L, d)
      y:  (N,)          (log return over [t, t+H])
      ts: (N,)          (對應 label 的時間點)
    """
    T = len(close)
    assert features.shape[0] == T
    assert T > window_length + horizon

    # log return: r_t = log(C_t / C_{t-1})
    logret = np.zeros(T, dtype=np.float64)
    logret[1:] = np.log(close[1:] / close[:-1])

    N = T - window_length - horizon + 1

    X = np.zeros((N, window_length, features.shape[1]), dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    ts = np.zeros(N, dtype="datetime64[ns]")

    for i in range(N):
        start = i
        end = i + window_length  # not inclusive
        X[i] = features[start:end]

        # window 尾端 index = end - 1 = t
        # label 看 horizon 之後那一根的 log return：logret[t + horizon]
        target_index = end - 1 + horizon
        y[i] = logret[target_index]
        ts[i] = timestamps[target_index]

    return X, y, ts


# ===================== 訓練一個 fold =====================
# ===================== 訓練一個 fold =====================

def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_length: int,
    device: torch.device,
    num_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    mse_weight: float = 0.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-4,
    ts_train: np.ndarray = None,
    ts_test: np.ndarray = None,
    fee_rate: float = 0.005,
    fee_lambda: float = 1.0,
):
    """
    用「相對市場的 alpha / beta」當目標：
      - 市場報酬 Rm = y (buy & hold 的 log return)
      - 策略報酬 Rs = position * y
      - beta  = Cov(Rs, Rm) / Var(Rm)
      - alpha = E[Rs] - beta * E[Rm]
      - loss  = - alpha / std(residual) （類 Information Ratio）

    回傳：
      - model
      - strategy log returns on test
      - buy&hold log returns on test
    """
    in_dim = X_train.shape[2]

    model = IndicatorNet(
        in_dim=in_dim,
        window_length=window_length,
        num_kernels=16,
        hidden_dim=64,   # 比 32 稍微大一點，因為特徵變多
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = WindowDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, drop_last=False
    )

    eps = 1e-6

    # 把測試集轉為 tensor，供 epoch-level 評估（early stopping）使用
    X_test_t_full = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test_t_full = torch.from_numpy(y_test.astype(np.float32)).to(device)

    best_sharpe = None
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        batch_alphas = []
        batch_betas = []
        batch_irs = []

        for xb, yb in train_loader:
            xb = xb.to(device)        # (batch, L, d)
            yb = yb.to(device)        # (batch,)  → Rm

            optimizer.zero_grad()

            # position / signal ∈ [-1, 1]
            pos = model(xb)           # (batch,)
            Rs = pos * yb             # 策略 log-return
            Rm = yb                   # 市場 log-return（buy & hold）

            # ---- 計算 alpha / beta ----
            mu_m = Rm.mean()
            mu_s = Rs.mean()

            cov_sm = ((Rs - mu_s) * (Rm - mu_m)).mean()
            var_m = (Rm - mu_m).pow(2).mean() + eps

            beta = cov_sm / var_m
            alpha = mu_s - beta * mu_m

            # 殘差 ε = Rs - (α + β Rm)
            residual = Rs - (alpha + beta * Rm)
            sigma_eps = residual.std(unbiased=False) + eps

            # Information Ratio 風格（保留計算供紀錄用）
            info_ratio = alpha / sigma_eps

            # 訓練目標改為 Sharpe：mean(Rs) / std(Rs)
            mu_s_batch = mu_s
            sigma_s_batch = Rs.std(unbiased=False) + eps
            sharpe_batch = mu_s_batch / sigma_s_batch

            # 要最大化 Sharpe → 最小化負的
            loss = -sharpe_batch

            # 如有需要可以混一點回歸 loss，避免梯度太 noisy
            if mse_weight > 0.0:
                loss = loss + mse_weight * F.mse_loss(pos, torch.zeros_like(pos))

            # fee regularization penalty (approximate entry cost from zero)
            if fee_lambda > 0.0 and fee_rate > 0.0:
                fee_penalty = fee_lambda * fee_rate * pos.abs().mean()
                loss = loss + fee_penalty

            loss.backward()
            optimizer.step()

            batch_alphas.append(alpha.detach().item())
            batch_betas.append(beta.detach().item())
            batch_irs.append(info_ratio.detach().item())

            # 每個 batch 印出該 batch 的累積報酬（基於該 batch 的 log-return sum）
            try:
                Rs_np = Rs.detach().cpu().numpy()
                Rm_np = Rm.detach().cpu().numpy()
                pos_np = pos.detach().cpu().numpy()
                batch_strat_cum = float(np.exp(Rs_np.sum()) - 1.0)
                batch_bh_cum = float(np.exp(Rm_np.sum()) - 1.0)
                # 估計 batch 手續費 (簡單假設由 0 -> pos 的進場成本)
                batch_fee = fee_rate * float(np.abs(pos_np).sum()) if fee_rate > 0 else 0.0
                batch_strat_cum_net = float(np.exp(Rs_np.sum() - batch_fee) - 1.0)
                print(f"    Epoch {epoch+1}/{num_epochs} sharpe={sharpe_batch.item():.4f}  batch: strat={batch_strat_cum:.4f}, strat_net={batch_strat_cum_net:.4f}, bh={batch_bh_cum:.4f}, alpha={alpha.item():.6f}, beta={beta.item():.3f}, IR={info_ratio.item():.4f}")
            except Exception:
                pass

        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            avg_alpha = float(np.mean(batch_alphas)) if batch_alphas else float("nan")
            avg_beta = float(np.mean(batch_betas)) if batch_betas else float("nan")
            avg_ir = float(np.mean(batch_irs)) if batch_irs else float("nan")
            print(
                f"  Epoch {epoch+1:4d}/{num_epochs}, "
                f"alpha≈{avg_alpha:.6f}, beta≈{avg_beta:.3f}, IR≈{avg_ir:.4f}"
            )

        # 每個 epoch 在測試集上快速評估，用來做 early stopping
        model.eval()
        with torch.no_grad():
            pos_test_epoch = model(X_test_t_full).cpu().numpy()
        strat_logret_test_epoch = pos_test_epoch * y_test
        sharpe_strat_epoch = strat_logret_test_epoch.mean() / (strat_logret_test_epoch.std(ddof=0) + 1e-6)

        if best_sharpe is None or sharpe_strat_epoch > best_sharpe + early_stop_min_delta:
            best_sharpe = sharpe_strat_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1} (best sharpe={best_sharpe:.6f})")
            # 載入最佳權重
            try:
                model.load_state_dict(best_state)
            except Exception:
                pass
            break

    # ===== 最終 test 評估（載入最佳權重後） =====
    try:
        if best_state is not None:
            model.load_state_dict(best_state)
    except Exception:
        pass

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_test_t = torch.from_numpy(y_test.astype(np.float32)).to(device)
        pos = model(X_test_t).cpu().numpy()    # (N_test,)

    strat_logret = pos * y_test               # 策略 log-return
    bh_logret = y_test                        # buy & hold log-return

    # 扣除手續費：使用簡化假設，手續費按交易量收取，cost_t = fee_rate * |pos_t - pos_{t-1}|，初始假設從 0 建倉
    if fee_rate > 0.0:
        pos_prev = np.concatenate(([0.0], pos[:-1]))
        costs = fee_rate * np.abs(pos - pos_prev)
        strat_logret = strat_logret - costs

    # 使用 timestamps 計算每期與年化 Sharpe（若 ts_test 提供）
    per_sh_test, ann_sh_test = compute_sharpes(strat_logret, timestamps=ts_test)
    per_sh_bh_test, ann_sh_bh_test = compute_sharpes(bh_logret, timestamps=ts_test)

    print(f"  Test Sharpe (per-period): strategy = {per_sh_test:.4f}, buy&hold = {per_sh_bh_test:.4f}")
    if not np.isnan(ann_sh_test):
        print(f"  Test Sharpe (annualized): strategy = {ann_sh_test:.4f}, buy&hold = {ann_sh_bh_test:.4f}")
    # ===== 在訓練集上也做一次評估，方便產生訓練報表 =====
    with torch.no_grad():
        X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
        pos_train = model(X_train_t).cpu().numpy()

    strat_logret_train = pos_train * y_train
    bh_logret_train = y_train
    if fee_rate > 0.0:
        pos_prev_train = np.concatenate(([0.0], pos_train[:-1]))
        costs_train = fee_rate * np.abs(pos_train - pos_prev_train)
        strat_logret_train = strat_logret_train - costs_train

    per_sh_train, ann_sh_train = compute_sharpes(strat_logret_train, timestamps=ts_train)
    per_sh_bh_train, ann_sh_bh_train = compute_sharpes(bh_logret_train, timestamps=ts_train)
    print(f"  Train Sharpe (per-period): strategy = {per_sh_train:.4f}, buy&hold = {per_sh_bh_train:.4f}")
    if not np.isnan(ann_sh_train):
        print(f"  Train Sharpe (annualized): strategy = {ann_sh_train:.4f}, buy&hold = {ann_sh_bh_train:.4f}")

    return model, strat_logret, bh_logret, pos, strat_logret_train, bh_logret_train, pos_train



# ===================== Main =====================

def main():
    # ==== 基本設定 ====
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "BITSTAMP_BTCUSD,240more.csv")  # 檔名如有不同記得改
    # csv_path = os.path.join(base_dir, "BYBIT_BTCUSDT15.csv")  # 檔名如有不同記得改
    window_length = 2**12
    horizon = 1          # 預測下一根 log return
    n_splits = 5         # TimeSeries cross-validation 折數
    num_epochs = 100000
    batch_size = 2**12
    lr = 1e-3
    mse_weight = 0.01     # 如訓練不穩可以改成 0.01 試試看
    early_stop_patience = 200
    early_stop_min_delta = 1e-4

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==== 讀資料 ====
    print("Loading data...")
    timestamps, features, close = load_ohlcv_csv(csv_path)
    print(f"Total bars: {len(close)}")

    # ==== 建 window ====
    print("Building windows...")
    X_all, y_all, ts_all = build_windows(
        features,
        close,
        timestamps,
        window_length=window_length,
        horizon=horizon
    )
    N, L, d = X_all.shape
    print(f"Num samples: {N}, window_length={L}, features={d}")

    # ==== TimeSeries Cross-Validation ====
    tscv = TimeSeriesSplit(n_splits=n_splits)

    all_strat_logret = np.full(N, np.nan, dtype=np.float64)
    all_bh_logret = np.full(N, np.nan, dtype=np.float64)

    fold_id = 0
    for train_idx, test_idx in tscv.split(X_all):
        fold_id += 1
        print(f"\n===== Fold {fold_id} =====")
        X_train_raw = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test_raw = X_all[test_idx]
        y_test = y_all[test_idx]

        # 每個 fold 各自用 train 的統計量做標準化，避免資訊洩漏
        d = X_train_raw.shape[2]
        train_flat = X_train_raw.reshape(-1, d)
        mean = train_flat.mean(axis=0)
        std = train_flat.std(axis=0) + 1e-6

        def scale(x):
            return (x - mean) / std

        X_train = scale(X_train_raw)
        X_test = scale(X_test_raw)

        model, strat_logret, bh_logret, alpha_test, strat_logret_train, bh_logret_train, alpha_train = train_one_fold(
            X_train, y_train, X_test, y_test,
            window_length=window_length,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            mse_weight=mse_weight,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            ts_train=ts_all[train_idx],
            ts_test=ts_all[test_idx],
        )

        # 存放 fold 的回傳結果
        all_strat_logret[test_idx] = strat_logret
        all_bh_logret[test_idx] = bh_logret

        # 繪製並儲存本 fold 的 equity curve（從訓練集到測試集的連續曲線）
        ts_train = ts_all[train_idx]
        ts_fold = ts_all[test_idx]

        strat_cum_train = np.exp(np.cumsum(strat_logret_train))
        bh_cum_train = np.exp(np.cumsum(bh_logret_train))

        strat_cum_fold = np.exp(np.cumsum(strat_logret))
        bh_cum_fold = np.exp(np.cumsum(bh_logret))

        # 連續序列
        ts_combined = np.concatenate([ts_train, ts_fold])
        strat_combined = np.concatenate([strat_logret_train, strat_logret])
        bh_combined = np.concatenate([bh_logret_train, bh_logret])
        strat_cum_combined = np.exp(np.cumsum(strat_combined))
        bh_cum_combined = np.exp(np.cumsum(bh_combined))

        plt.figure(figsize=(12, 5))
        plt.plot(ts_combined, strat_cum_combined, label="Strategy")
        plt.plot(ts_combined, bh_cum_combined, label="Buy & Hold", linestyle="--")
        # 在訓練/測試分界處畫垂直線
        if len(ts_train) > 0:
            bound_time = ts_train[-1]
            plt.axvline(bound_time, color='gray', linestyle=':', linewidth=1)
            plt.text(bound_time, plt.ylim()[1], ' train|test ', rotation=90, va='top', ha='center')

        plt.title(f"Fold {fold_id} Equity (train → test)")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Return")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        fold_plot_path = os.path.join(base_dir, f"fold_{fold_id}_equity.png")
        plt.tight_layout()
        plt.savefig(fold_plot_path, dpi=150)
        plt.close()
        print(f"Saved fold equity (train→test) to {fold_plot_path}")

        # 儲存本 fold 的簡易報表（包含 train & test）
        strat_total = float(strat_cum_fold[-1] - 1.0)
        bh_total = float(bh_cum_fold[-1] - 1.0)
        # 訓練集表現
        strat_cum_train = np.exp(np.cumsum(strat_logret_train))
        bh_cum_train = np.exp(np.cumsum(bh_logret_train))
        strat_total_train = float(strat_cum_train[-1] - 1.0)
        bh_total_train = float(bh_cum_train[-1] - 1.0)
        # 使用回歸後的 alpha（alpha = mean(Rs) - beta * mean(Rm)）作為 alpha_excess
        mu_s_f = float(strat_logret.mean())
        mu_m_f = float(bh_logret.mean())
        cov_sm_f = float(((strat_logret - mu_s_f) * (bh_logret - mu_m_f)).mean())
        var_m_f = float(((bh_logret - mu_m_f) ** 2).mean()) + 1e-12
        beta_f = cov_sm_f / var_m_f
        alpha_fold_reg = mu_s_f - beta_f * mu_m_f
        # test set Sharpe (per-period & annualized)
        per_sh_test, ann_sh_test = compute_sharpes(strat_logret, timestamps=ts_fold)
        per_sh_bh_test, ann_sh_bh_test = compute_sharpes(bh_logret, timestamps=ts_fold)

        # training set metrics
        mu_s_train = float(strat_logret_train.mean())
        mu_m_train = float(bh_logret_train.mean())
        cov_sm_train = float(((strat_logret_train - mu_s_train) * (bh_logret_train - mu_m_train)).mean())
        var_m_train = float(((bh_logret_train - mu_m_train) ** 2).mean()) + 1e-12
        beta_train = cov_sm_train / var_m_train
        alpha_train_reg = mu_s_train - beta_train * mu_m_train
        # train set Sharpe (per-period & annualized)
        per_sh_train, ann_sh_train = compute_sharpes(strat_logret_train, timestamps=ts_train)
        per_sh_bh_train, ann_sh_bh_train = compute_sharpes(bh_logret_train, timestamps=ts_train)

        fold_report = pd.DataFrame({
            'fold': [fold_id],
            'start_time': [ts_fold[0]],
            'end_time': [ts_fold[-1]],
            'strategy_total_return': [strat_total],
            'buyhold_total_return': [bh_total],
            'alpha_excess_regression': [alpha_fold_reg],
            'beta_regression': [beta_f],
            'strategy_sharpe': [per_sh_test],
            'buyhold_sharpe': [per_sh_bh_test],
            'strategy_sharpe_annualized': [ann_sh_test],
            'buyhold_sharpe_annualized': [ann_sh_bh_test],
            'strategy_total_return_train': [strat_total_train],
            'buyhold_total_return_train': [bh_total_train],
            'alpha_excess_regression_train': [alpha_train_reg],
            'beta_regression_train': [beta_train],
            'strategy_sharpe_train': [per_sh_train],
            'buyhold_sharpe_train': [per_sh_bh_train],
            'strategy_sharpe_train_annualized': [ann_sh_train],
            'buyhold_sharpe_train_annualized': [ann_sh_bh_train],
            'mean_alpha_test': [float(np.mean(alpha_test))],
            'mean_alpha_train': [float(np.mean(alpha_train))],
            'num_samples': [len(strat_logret)]
        })
        fold_report_path = os.path.join(base_dir, f"fold_{fold_id}_report.csv")
        fold_report.to_csv(fold_report_path, index=False)
        print(f"Saved fold report to {fold_report_path}")

    # ==== 整體 CV 結果 ====
    valid_mask = ~np.isnan(all_strat_logret)
    ts_valid = ts_all[valid_mask]
    strat_logret_valid = all_strat_logret[valid_mask]
    bh_logret_valid = all_bh_logret[valid_mask]

    per_sh_overall, ann_sh_overall = compute_sharpes(strat_logret_valid, timestamps=ts_valid)
    per_sh_bh_overall, ann_sh_bh_overall = compute_sharpes(bh_logret_valid, timestamps=ts_valid)

    print("\n===== Overall CV result =====")
    print(f"Overall strategy Sharpe (per-period): {per_sh_overall:.4f}")
    if not np.isnan(ann_sh_overall):
        print(f"Overall strategy Sharpe (annualized): {ann_sh_overall:.4f}")
    print(f"Overall buy&hold Sharpe (per-period): {per_sh_bh_overall:.4f}")
    if not np.isnan(ann_sh_bh_overall):
        print(f"Overall buy&hold Sharpe (annualized): {ann_sh_bh_overall:.4f}")

    # keep variable names for backward compatibility
    strat_sharpe = per_sh_overall
    bh_sharpe = per_sh_bh_overall
    strat_sharpe_annualized = ann_sh_overall
    bh_sharpe_annualized = ann_sh_bh_overall

    # ==== 畫整體 equity curve（log-return → equity） ====
    strat_cum = np.exp(np.cumsum(strat_logret_valid))
    bh_cum = np.exp(np.cumsum(bh_logret_valid))

    # 計算總報酬（從序列最後一個累積值減 1）
    strat_total_return = float(strat_cum[-1] - 1.0)
    bh_total_return = float(bh_cum[-1] - 1.0)  # 這裡把 buy&hold 視為市場回報

    # 使用回歸後的 alpha（alpha = mean(Rs) - beta * mean(Rm)）作為最終的 alpha_excess
    mu_s = float(strat_logret_valid.mean())
    mu_m = float(bh_logret_valid.mean())
    cov_sm = float(((strat_logret_valid - mu_s) * (bh_logret_valid - mu_m)).mean())
    var_m = float(((bh_logret_valid - mu_m) ** 2).mean()) + 1e-12
    beta_overall = cov_sm / var_m
    alpha_excess = mu_s - beta_overall * mu_m

    print("\n===== Performance Summary =====")
    print(f"Period: {ts_valid[0]} to {ts_valid[-1]}")
    print(f"Strategy total return: {strat_total_return:.4f} ({strat_total_return*100:.2f} %)")
    print(f"Buy&Hold (beta) total return: {bh_total_return:.4f} ({bh_total_return*100:.2f} %)")
    print(f"Alpha (regression excess): {alpha_excess:.6f} ({alpha_excess*100:.4f} % per period)")
    print(f"Overall beta (regression): {beta_overall:.6f}")
    print(f"Overall strategy Sharpe: {strat_sharpe:.4f}")
    print(f"Overall buy&hold Sharpe: {bh_sharpe:.4f}")

    # 輸出報表為 CSV（會覆寫舊檔）
    report = pd.DataFrame({
        'start_time': [ts_valid[0]],
        'end_time': [ts_valid[-1]],
        'strategy_total_return': [strat_total_return],
        'buyhold_total_return': [bh_total_return],
        'alpha_excess_regression': [alpha_excess],
        'beta_regression': [beta_overall],
        'strategy_sharpe': [strat_sharpe],
        'buyhold_sharpe': [bh_sharpe],
        'strategy_sharpe_annualized': [strat_sharpe_annualized],
        'buyhold_sharpe_annualized': [bh_sharpe_annualized],
        'num_samples': [len(strat_logret_valid)],
        # 最終評分標準：Sharpe
        'final_metric_name': ['sharpe'],
        'final_metric_value': [strat_sharpe]
    })

    report_path = os.path.join(base_dir, 'performance_report.csv')
    report.to_csv(report_path, index=False)
    print(f"Saved performance report to {report_path}")

    # 繪圖保持不變
    plt.figure(figsize=(12, 6))
    plt.plot(ts_valid, strat_cum, label="Strategy (IndicatorNet)")
    plt.plot(ts_valid, bh_cum, label="Buy & Hold", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return (exp of sum log-returns)")
    plt.yscale("log")
    plt.title("IndicatorNet Strategy vs Buy & Hold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "equity_curve.png"), dpi=150)
    plt.show()

    print("Saved equity curve to equity_curve.png")


if __name__ == "__main__":
    main()
