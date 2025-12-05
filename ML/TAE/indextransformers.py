#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OHLCV -> IndicatorNet (大量 TA-style 特徵) -> Transformer Encoder
→ position alpha_t ∈ [-1, 1]

訓練目標：
- 讓策略相對 B&H 有高 Alpha、低 Beta
- 同時算出 Test 集上的：
  - 累積報酬曲線
  - 年化報酬
  - Sharpe / Sortino
  - Alpha / Beta
  - Max Drawdown
"""

import os
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

    # PyTorch 2.0+: new torch.amp API
from torch.amp import autocast, GradScaler



# ===================== 檔案路徑設定 =====================

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'BITSTAMP_BTCUSD,240more.csv')  # 這是你給的路徑

# ===================== 超參數設定 =====================

# ===================== 超參數設定 =====================
SEQ_LEN = 2**8
HORIZON = 1
BATCH_SIZE = 2**9
EPOCHS = 1000
LR = 3e-4   # 比 1e-2 溫和很多

D_FEAT = 64
D_MODEL = 128
N_HEADS = 16
N_LAYERS = 2

# ---- 新的風控 / 報酬權重設定 ----
TARGET_MDD      = 0.10     # 希望策略最終 MDD ≈ 10%
MAX_LEVERAGE    = 100.0      # 槓桿上限，避免誇張放大
LAMBDA_RETURN   = 1.0      # 目前只在 compute_trading_loss 裡當權重用（其實可以寫死）
LAMBDA_MDD      = 100.0    # 這組已經不用了，可以保留但不再用到
LAMBDA_POS_L2   = 1e-4
LAMBDA_MEAN_POS = 0.0
LAMBDA_FORECAST = 0.0      # 先照妳說的簡單版，不加 forecast
LAMBDA_RECON    = 0.0


BAR_MINUTES = 240      # 240 分 K
BARS_PER_YEAR = int(60 * 24 * 365 / BAR_MINUTES)  # 4 小時一根，一年約 2190 根

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
USE_AMP = USE_CUDA          # 有 CUDA 就開 AMP
USE_COMPILE = USE_CUDA      # 有 CUDA 就試著用 torch.compile
USE_COMPILE = USE_CUDA and (os.name != "nt")
# DataLoader 相關
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = USE_CUDA       # 有 GPU 就啟用 pinned memory

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ===================== 工具：固定 random seed =====================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== 讀取 OHLCV CSV =====================

def load_ohlcv_csv(path: str):
    """
    嘗試從 CSV 裡找出 open/high/low/close/volume 欄位。
    如果欄位名不同，你可以自己改下面的對應。
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(candidates):
        for name in candidates:
            key = name.lower()
            if key in cols_lower:
                return cols_lower[key]
        # 再試試看包含關鍵字的欄位
        for key, real in cols_lower.items():
            for name in candidates:
                if name.lower() in key:
                    return real
        raise KeyError(f"找不到欄位: {candidates}，請檢查 CSV 欄位名稱")

    col_open = pick(["open", "o"])
    col_high = pick(["high", "h"])
    col_low = pick(["low", "l"])
    col_close = pick(["close", "c"])
    col_volume = pick(["volume", "vol", "volume btc", "volume usd"])

    o = df[col_open].astype(np.float32).values
    h = df[col_high].astype(np.float32).values
    l = df[col_low].astype(np.float32).values
    c = df[col_close].astype(np.float32).values
    v = df[col_volume].astype(np.float32).values

    x = np.stack([o, h, l, c, v], axis=-1)  # (N, 5)
    return x, df


# ===================== Dataset: sliding window =====================

class OhlcvDataset(Dataset):
    def __init__(self, ohlcv_array: np.ndarray, seq_len: int, horizon: int):
        """
        ohlcv_array: shape (N, 5) [O, H, L, C, V]
        每個 sample:
          x_seq: 過去 seq_len 根 K 線
          r_b:   這根之後 horizon 的 B&H log return
        """
        self.x = torch.from_numpy(ohlcv_array)  # (N, 5)
        self.seq_len = seq_len
        self.horizon = horizon
        self.N = self.x.shape[0]

        # 最後一個 sample 的起點 index = N - seq_len - horizon
        self.num_samples = self.N - seq_len - horizon + 1
        assert self.num_samples > 0, "資料太短，無法生成樣本"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        x_seq = self.x[start:end]  # (L, 5)

        # B&H log return：從最後一根到 horizon 之後
        close_t = self.x[end - 1, 3]
        close_future = self.x[end - 1 + self.horizon, 3]
        r_b = torch.log(close_future / close_t)

        return x_seq.float(), r_b.float()


# ===================== 一些 rolling 工具 =====================

def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    x: (B, L)
    回傳 same-length rolling mean（window 不足時用第一個 mean 補）
    """
    if window <= 1:
        return x
    B, L = x.shape
    if L < window:
        mean = x.mean(dim=1, keepdim=True).expand(-1, L)
        return mean
    unfolded = x.unfold(dimension=1, size=window, step=1)  # (B, L-window+1, window)
    mean = unfolded.mean(dim=-1)  # (B, L-window+1)
    pad_left = mean[:, 0:1].expand(-1, window - 1)
    out = torch.cat([pad_left, mean], dim=1)
    return out


def rolling_std(x: torch.Tensor, window: int) -> torch.Tensor:
    mean = rolling_mean(x, window)
    var = rolling_mean((x - mean) ** 2, window)
    return torch.sqrt(var + 1e-8)


def rolling_min(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    B, L = x.shape
    if L < window:
        m = x.min(dim=1, keepdim=True).values.expand(-1, L)
        return m
    unfolded = x.unfold(1, window, 1)  # (B, L-window+1, window)
    m = unfolded.min(dim=-1).values
    pad_left = m[:, 0:1].expand(-1, window - 1)
    return torch.cat([pad_left, m], dim=1)


def rolling_max(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    B, L = x.shape
    if L < window:
        m = x.max(dim=1, keepdim=True).values.expand(-1, L)
        return m
    unfolded = x.unfold(1, window, 1)
    m = unfolded.max(dim=-1).values
    pad_left = m[:, 0:1].expand(-1, window - 1)
    return torch.cat([pad_left, m], dim=1)


def compute_rsi(close: torch.Tensor, window: int = 14) -> torch.Tensor:
    """
    close: (B, L)
    return: RSI-like in [0, 1]
    """
    eps = 1e-6
    diff = torch.zeros_like(close)
    diff[:, 1:] = close[:, 1:] - close[:, :-1]
    up = torch.clamp(diff, min=0.0)
    down = torch.clamp(-diff, min=0.0)

    avg_gain = rolling_mean(up, window)
    avg_loss = rolling_mean(down, window)

    rs = avg_gain / (avg_loss + eps)
    rsi = 1.0 - 1.0 / (1.0 + rs)  # 0~1
    return rsi


def compute_stoch_k(close: torch.Tensor,
                    high: torch.Tensor,
                    low: torch.Tensor,
                    window: int = 14) -> torch.Tensor:
    eps = 1e-6
    lowest_low = rolling_min(low, window)
    highest_high = rolling_max(high, window)
    k = (close - lowest_low) / (highest_high - lowest_low + eps)
    return k


def compute_atr(open_: torch.Tensor,
                high: torch.Tensor,
                low: torch.Tensor,
                close: torch.Tensor,
                window: int = 14) -> torch.Tensor:
    """
    ATR: True Range 的 rolling mean
    """
    eps = 1e-6
    prev_close = torch.zeros_like(close)
    prev_close[:, 1:] = close[:, :-1]

    high_low = high - low
    high_pc = torch.abs(high - prev_close)
    low_pc = torch.abs(low - prev_close)
    tr = torch.max(high_low, torch.max(high_pc, low_pc))
    atr = rolling_mean(tr, window)
    return atr


def compute_obv(close: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """
    On-Balance Volume
    """
    diff = torch.zeros_like(close)
    diff[:, 1:] = close[:, 1:] - close[:, :-1]
    direction = torch.sign(diff)  # 漲=+1 跌=-1 平=0
    obv = torch.cumsum(direction * volume, dim=1)
    return obv


# ===================== Positional Encoding =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]

# ===================== 純 OHLCV 版 Transformer QuantModel =====================

class QuantModel(nn.Module):
    """
    New architecture:
    OHLCV (B, L, 5)
      -> 簡單縮放 / log 轉換
      -> Linear input projection 到 d_model
      -> PositionalEncoding
      -> TransformerEncoder（含時間因果 mask，不能看未來）
      -> 取最後一根 K 棒的 hidden state 做 head：
          - head_ret: 未來 B&H return 預測 (r_b_hat)
          - head_vol: 未來波動預測 (sigma_hat)
          - head_pos: position alpha_t ∈ [-1, 1]

    為了沿用你的 compute_trading_loss，
    這裡仍然輸出：
      - recon:   (B, L, in_dim)   # 重建輸出（但 loss 權重現在是 0）
      - feat_target: (B, L, in_dim)   # AE target，這裡用縮放後的 OHLCV
      - r_b_hat, sigma_hat, alpha
    """

    def __init__(
        self,
        in_dim:   int = 5,
        d_feat:   int = D_FEAT,   # 只是為了相容舊參數，這裡不再使用
        d_model:  int = D_MODEL,
        n_heads:  int = N_HEADS,
        n_layers: int = N_LAYERS,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model

        # OHLCV -> d_model 的線性投影
        self.input_proj = nn.Linear(in_dim, d_model)

        # 位置編碼：沿用你原本的 PositionalEncoding
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer Encoder，本身支援 batch_first
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Reconstruction head：把 hidden 再打回 OHLCV 維度
        self.recon_head = nn.Linear(d_model, in_dim)

        # Forecast / risk / position heads
        self.head_ret = nn.Linear(d_model, 1)
        self.head_vol = nn.Linear(d_model, 1)
        self.head_pos = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh()     # position ∈ [-1, 1]
        )

    def _preprocess_ohlcv(self, x: torch.Tensor) -> torch.Tensor:
        """
        對 OHLCV 做一點簡單縮放，讓 transformer 比較好學：
        - 價格 O/H/L/C 以第一根 close 當基準做 scale
        - Volume 做 log 變換

        x: (B, L, 5)
        return: x_scaled (B, L, 5)
        """
        eps = 1e-6
        B, L, D = x.shape
        assert D == self.in_dim

        O = x[:, :, 0:1]  # (B, L, 1)
        H = x[:, :, 1:2]
        L_ = x[:, :, 2:3]
        C = x[:, :, 3:4]
        V = x[:, :, 4:5]

        # 以每個 window 的第一根收盤價當基準
        price_ref = C[:, 0:1, :].clamp_min(eps)   # (B, 1, 1)

        O_s = O / price_ref
        H_s = H / price_ref
        L_s = L_ / price_ref
        C_s = C / price_ref

        # Volume 做 log
        V_log = torch.log(V + eps)

        x_scaled = torch.cat([O_s, H_s, L_s, C_s, V_log], dim=-1)  # (B, L, 5)
        return x_scaled

    def _build_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        建立因果 mask，確保時間 t 不能看到未來 t' > t：

        mask 形狀：(L, L)，上三角（不含對角線）為 -inf，其餘為 0。
        """
        # 上三角（不含對角）為 1，其餘為 0
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        # 轉成 attention 使用的 -inf / 0 mask
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0.0)
        return mask

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, 5)  (原始 OHLCV)
        """
        # 1) 縮放 OHLCV
        x_scaled = self._preprocess_ohlcv(x)          # (B, L, 5)

        # 2) 線性投影 + 非線性
        h = self.input_proj(x_scaled)                 # (B, L, d_model)
        h = F.gelu(h)

        # 3) 加 positional encoding
        h = self.pos_enc(h)                           # (B, L, d_model)

        # 4) Causal mask（避免看未來）
        B, L, _ = h.shape
        src_mask = self._build_causal_mask(L, h.device)

        # 5) Transformer Encoder
        h = self.encoder(h, mask=src_mask)           # (B, L, d_model)

        # 6) Reconstruction head：重建縮放後的 OHLCV（當 AE target）
        recon = self.recon_head(h)                   # (B, L, in_dim)

        # 7) 取最後一根 K 棒的 hidden state 當整個 window summary
        h_last = h[:, -1, :]                         # (B, d_model)

        # 8) 各種 head
        r_b_hat = self.head_ret(h_last)              # (B, 1)
        sigma_hat = F.softplus(self.head_vol(h_last))  # (B, 1), > 0
        alpha = self.head_pos(h_last)                # (B, 1), ∈ [-1, 1]

        return {
            "recon": recon,               # (B, L, in_dim)
            "r_b_hat": r_b_hat,           # (B, 1)
            "sigma_hat": sigma_hat,       # (B, 1)
            "alpha": alpha,               # (B, 1)
            "feat_target": x_scaled       # AE target: 縮放後 OHLCV
        }



# ===================== Trading Loss（Alpha / Beta / AE） =====================
# ===================== Trading Loss（Max Return with MDD Constraint） =====================
# ===================== Trading Loss（原始 MDD → 槓桿 → 槓桿報酬） =====================

def compute_trading_loss(
    outputs,
    r_b_batch,
    target_mdd: float = TARGET_MDD,
    use_annualized: bool = True,
):
    """
    流程：
    1) 用 alpha * r_b 得到「未開槓桿」的策略 log return 序列 r_s_raw
    2) 用 r_s_raw 算出原始 MDD_raw
    3) 槓桿 = target_mdd / |MDD_raw|，並限制在 [0, MAX_LEVERAGE]
    4) 槓桿後的 log return = leverage * r_s_raw
    5) Loss = - 槓桿後的平均報酬（或年化報酬）

    這樣就對應妳說的：
      leverage = 10% / 原始MDD
      loss = 這個 leverage 回測出來的報酬率（我們這裡用 log 年化報酬表示）
    """
    alpha = outputs["alpha"].squeeze(-1)   # (B,)
    r_b   = r_b_batch                      # (B,)

    # 1) 未開槓桿策略 log return
    r_s_raw = alpha * r_b                  # (B,)

    # 2) 用這一個 batch 的序列近似原始 MDD
    cum_log = r_s_raw.cumsum(dim=0)        # (B,)
    equity  = torch.exp(cum_log)           # (B,)

    running_max, _ = torch.cummax(equity, dim=0)
    drawdown = equity / (running_max + 1e-8) - 1.0
    mdd_raw = drawdown.min()               # 大多數時候是負數，例如 -0.23

    # |MDD_raw|
    mdd_abs = torch.clamp(-mdd_raw, min=1e-6)

    # 3) 槓桿 = 目標 MDD / 原始 MDD，並上限到 MAX_LEVERAGE
    leverage = target_mdd / mdd_abs
    if MAX_LEVERAGE is not None:
        leverage = torch.clamp(leverage, max=MAX_LEVERAGE)

    # 4) 槓桿後 log return
    r_s_lev = leverage * r_s_raw
    mu_lev = r_s_lev.mean()

    # 5) 定義 loss：最大化槓桿後的報酬
    if use_annualized:
        ann_ret_lev = torch.exp(mu_lev * BARS_PER_YEAR) - 1.0
        loss = -ann_ret_lev
    else:
        loss = -mu_lev

    # 一些方便看 log 的數值
    with torch.no_grad():
        mu_raw = r_s_raw.mean().item()
        ann_ret_raw = math.exp(mu_raw * BARS_PER_YEAR) - 1.0

    aux = {
        "loss":        loss.item(),
        "leverage":    leverage.item(),
        "mdd_raw":     mdd_raw.item(),
        "mu_raw":      mu_raw,
        "ann_ret_raw": ann_ret_raw,
        "mu_lev":      mu_lev.item(),
    }
    return loss, aux



def collect_returns(loader, model):
    """
    把整個 loader 跑一遍，收集：
      - r_s_raw: 策略未槓桿 log return 序列 (T,)
      - r_b:     B&H log return 序列 (T,)
    """
    model.eval()
    all_r_b = []
    all_alpha = []
    device_type = "cuda" if USE_CUDA else "cpu"

    with torch.no_grad():
        for x_seq, r_b in loader:
            x_seq = x_seq.to(DEVICE, non_blocking=True)
            r_b   = r_b.to(DEVICE, non_blocking=True)
            with autocast(device_type=device_type, enabled=USE_AMP):
                out = model(x_seq)
                alpha = out["alpha"].squeeze(-1)
            all_r_b.append(r_b)
            all_alpha.append(alpha)

    r_b_all = torch.cat(all_r_b, dim=0)
    alpha_all = torch.cat(all_alpha, dim=0)
    r_s_all = alpha_all * r_b_all
    return r_s_all, r_b_all


# ===================== 評估指標（Sharpe, Sortino, Alpha, Beta, MDD） =====================

def compute_metrics(r_s: torch.Tensor,
                    r_b: torch.Tensor,
                    bar_minutes: int = BAR_MINUTES):
    """
    r_s: 策略 log return 時間序列 (T,)
    r_b: B&H log return 時間序列 (T,)
    """
    with torch.no_grad():
        r_s = r_s.detach().cpu()
        r_b = r_b.detach().cpu()

        bars_per_year = int(60 * 24 * 365 / bar_minutes)

        mu_s = r_s.mean().item()
        mu_b = r_b.mean().item()
        sigma_s = r_s.std(unbiased=False).item() + 1e-12
        sigma_b = r_b.std(unbiased=False).item() + 1e-12

        cov_sb = ((r_s - mu_s) * (r_b - mu_b)).mean().item()
        var_b = (r_b - mu_b).pow(2).mean().item() + 1e-12
        beta = cov_sb / var_b
        alpha_capm = mu_s - beta * mu_b
        alpha_adj = alpha_capm / sigma_s

        # 年化報酬（log 轉簡單報酬）
        ann_ret_s = math.exp(mu_s * bars_per_year) - 1.0
        ann_ret_b = math.exp(mu_b * bars_per_year) - 1.0

        # Sharpe
        sharpe = (mu_s / sigma_s) * math.sqrt(bars_per_year)

        # Sortino
        neg = r_s[r_s < 0]
        if neg.numel() > 0:
            downside_std = (neg.pow(2).mean().sqrt().item() + 1e-12)
            sortino = (mu_s / downside_std) * math.sqrt(bars_per_year)
        else:
            sortino = float('inf')

        # Equity curve & MDD
        cum_log_s = r_s.cumsum(dim=0)
        cum_log_b = r_b.cumsum(dim=0)
        equity_s = torch.exp(cum_log_s)
        equity_b = torch.exp(cum_log_b)

        running_max = equity_s.cummax(dim=0).values
        drawdown = equity_s / running_max - 1.0
        mdd = drawdown.min().item()

    metrics = {
        "mu_s": mu_s,
        "mu_b": mu_b,
        "sigma_s": sigma_s,
        "sigma_b": sigma_b,
        "ann_ret_s": ann_ret_s,
        "ann_ret_b": ann_ret_b,
        "sharpe": sharpe,
        "sortino": sortino,
        "alpha_capm": alpha_capm,
        "alpha_adj": alpha_adj,
        "beta": beta,
        "mdd": mdd,
        "equity_s": equity_s.numpy(),
        "equity_b": equity_b.numpy(),
        "r_s": r_s.numpy(),
        "r_b": r_b.numpy()
    }
    return metrics

def eval_epoch(loader, model):
    """
    驗證邏輯：
    1) 先計算「未槓桿」策略報酬 r_s_all
    2) 用 r_s_all 算出原始 MDD_raw
    3) 槓桿 k_val = TARGET_MDD / |MDD_raw|（上限 MAX_LEVERAGE）
    4) 用 k_val 槓桿後的年化報酬當作 ValScore
    """
    # 1) 收集未槓桿策略報酬
    r_s_all, r_b_all = collect_returns(loader, model)

    # 2) 未槓桿 metrics
    metrics_raw = compute_metrics(r_s_all, r_b_all, BAR_MINUTES)

    # 3) 根據 val 原始 MDD 算出對應槓桿
    mdd_raw = metrics_raw["mdd"]        # 負數
    mdd_abs = max(-mdd_raw, 1e-6)
    k_val = TARGET_MDD / mdd_abs
    if MAX_LEVERAGE is not None:
        k_val = min(k_val, MAX_LEVERAGE)

    # 4) 槓桿後的 metrics
    r_s_lev_all = r_s_all * k_val
    metrics_lev = compute_metrics(r_s_lev_all, r_b_all, BAR_MINUTES)

    # 把幾個重要值塞回去方便 log
    metrics_raw["k_val"]          = k_val
    metrics_raw["ann_ret_s_lev"]  = metrics_lev["ann_ret_s"]
    metrics_raw["mdd_lev"]        = metrics_lev["mdd"]
    metrics_raw["sharpe_lev"]     = metrics_lev["sharpe"]

    # ValScore：用「槓桿後的年化報酬」
    val_score = metrics_lev["ann_ret_s"]
    return metrics_raw, val_score



# ===================== 畫圖 =====================

def plot_equity(equity_s, equity_b, title="Equity Curve", savepath: str = None):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_s, label="Strategy")
    plt.plot(equity_b, label="Buy & Hold")
    plt.xlabel("Bars")
    plt.yscale("log")
    plt.ylabel("Equity (start = 1)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savepath is not None:
        try:
            plt.savefig(savepath)
            print(f"[indextransformers] saved equity plot to {savepath}")
        except Exception as e:
            print(f"[indextransformers] failed to save plot: {e}")
    else:
        plt.show()


# ===================== 主程式 =====================

def main():
    set_seed(42)

    print(f"讀取資料檔案: {file_path}")
    ohlcv, df = load_ohlcv_csv(file_path)
    print(f"資料筆數: {len(ohlcv)}")

    dataset = OhlcvDataset(ohlcv, SEQ_LEN, HORIZON)
    num_samples = len(dataset)
    print(f"可用樣本數: {num_samples}")

    # 時間切割：70% Train, 15% Val, 15% Test
    train_end = int(num_samples * 0.5)
    val_end = int(num_samples * 0.75)

    indices = np.arange(num_samples)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"Train samples: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = QuantModel(in_dim=5,
                       d_feat=D_FEAT,
                       d_model=D_MODEL,
                       n_heads=N_HEADS,
                       n_layers=N_LAYERS).to(DEVICE)

    # torch.compile 在 Windows 上需要 Triton，通常不可用
    # 禁用以避免 TritonMissing 錯誤
    # if USE_COMPILE and hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model)
    #         print("[indextransformers] using torch.compile 加速模型")
    #     except Exception as e:
    #         print(f"[indextransformers] torch.compile 失敗，改用原始模型: {e}")

    # fused AdamW（新版本 PyTorch 在 CUDA 上會更快）
    optim_kwargs = {"lr": LR}
    if USE_CUDA:
        try:
            optim_kwargs["fused"] = True
            print("[indextransformers] AdamW 使用 fused=True")
        except TypeError:
            # 舊版 PyTorch 不支援 fused 參數
            pass

    optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    # AMP scaler（只有 CUDA 時開啟）
    # 使用新的 torch.amp API 以避免棄用警告
    if USE_CUDA:
        scaler = GradScaler(device="cuda", enabled=USE_AMP)
    else:
        scaler = GradScaler(device="cpu", enabled=USE_AMP)

    # Dynamic LR scheduler: reduce LR on plateau of validation score
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    # torch.optim.lr_scheduler.CosineAnnealingLR as alternative
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_state = None
    best_val_score = -1e9
    # Early stopping configuration
    patience_es = 100
    no_improve = 0

    # ========== Training Loop ==========
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for x_seq, r_b in train_loader:
            # non_blocking 要配合 pin_memory=True 才有用
            x_seq = x_seq.to(DEVICE, non_blocking=True)
            r_b = r_b.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # ===== AMP 前向與 loss =====
            device_type = "cuda" if USE_CUDA else "cpu"
            with autocast(device_type=device_type, enabled=USE_AMP):
                out = model(x_seq)
                loss, aux = compute_trading_loss(out, r_b)

            # ===== AMP 反向傳播與更新 =====
            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1


        avg_train_loss = total_loss / max(total_batches, 1)

        # 驗證
        val_metrics, val_score = eval_epoch(val_loader, model)

        print(f"[Epoch {epoch:03d}] "
              f"TrainLoss={avg_train_loss:.6f} | "
              f"ValScore(lev_ret)={val_score:.6f} | "
              f"Val k={val_metrics['k_val']:.3f} | "
              f"Val AnnRet_S(raw)={val_metrics['ann_ret_s']:.4%} | "
              f"Val AnnRet_S(lev)={val_metrics['ann_ret_s_lev']:.4%} | "
              f"Val MDD(raw)={val_metrics['mdd']:.4%} | "
              f"Val MDD(lev)={val_metrics['mdd_lev']:.4%}")


        # update best model and early-stopping / scheduler
        if val_score > best_val_score:
            best_val_score = val_score
            # store CPU copy of state_dict for safe saving/loading
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            # save canonical best model file plus a timestamped copy with epoch/score
            try:
                best_path = os.path.join(base_dir, 'best_model.pth')
                torch.save(best_state, best_path)
                backup_path = os.path.join(base_dir, f'best_model_epoch{epoch:04d}_score{best_val_score:.4f}.pth')
                torch.save(best_state, backup_path)
                print(f"  -> saved best model to {best_path} and backup {backup_path}")
            except Exception as e:
                print(f"[indextransformers] failed to save best model: {e}")
            print(f"  -> 新最佳模型，ValScore 提升為 {best_val_score:.6f}")
            no_improve = 0
            # Immediately save a validation equity plot for this new best model
            try:
                # val_metrics contains 'equity_s' and 'equity_b' as numpy arrays
                savepath = os.path.join(base_dir, f"val_equity_epoch{epoch:04d}_score{best_val_score:.4f}.png")
                plot_equity(val_metrics["equity_s"], val_metrics["equity_b"], title=f"Val Equity (epoch {epoch})", savepath=savepath)
            except Exception as e:
                print(f"[indextransformers] failed to save val equity plot: {e}")
        else:
            no_improve += 1

        # step scheduler (CosineAnnealingLR: call step() once per epoch, no args)
        try:
            scheduler.step()
        except Exception:
            pass

        # early stopping check
        if no_improve >= patience_es:
            print(f"Early stopping triggered at epoch {epoch} (no_improve={no_improve}, best_val_score={best_val_score:.6f})")
            break

    # ========== 測試 ==========
    # ========== 測試 ==========
    if best_state is not None:
        model.load_state_dict(best_state)

        # ========= 先用 Validation 算出最終 leverage =========
        r_s_val, r_b_val = collect_returns(val_loader, model)
        val_metrics_raw = compute_metrics(r_s_val, r_b_val, BAR_MINUTES)

        mdd_val_raw = val_metrics_raw["mdd"]
        mdd_abs = max(-mdd_val_raw, 1e-6)
        k_val = TARGET_MDD / mdd_abs
        if MAX_LEVERAGE is not None:
            k_val = min(k_val, MAX_LEVERAGE)

        val_metrics_lev = compute_metrics(r_s_val * k_val, r_b_val, BAR_MINUTES)

        print("\n===== Final Val Metrics (no leverage) =====")
        print(f"年化報酬 (策略): {val_metrics_raw['ann_ret_s']:.4%}")
        print(f"年化報酬 (B&H):  {val_metrics_raw['ann_ret_b']:.4%}")
        print(f"Sharpe:          {val_metrics_raw['sharpe']:.4f}")
        print(f"Max Drawdown:    {val_metrics_raw['mdd']:.4%}")

        print(f"\n===== Final Val Metrics (with leverage k={k_val:.3f}, target MDD={TARGET_MDD:.0%}) =====")
        print(f"年化報酬 (策略): {val_metrics_lev['ann_ret_s']:.4%}")
        print(f"Sharpe:          {val_metrics_lev['sharpe']:.4f}")
        print(f"Max Drawdown:    {val_metrics_lev['mdd']:.4%}")

        # ========= 再用這個 k_val 去跑 Test =========
        r_s_test, r_b_test = collect_returns(test_loader, model)
        test_metrics_raw = compute_metrics(r_s_test, r_b_test, BAR_MINUTES)
        test_metrics_lev = compute_metrics(r_s_test * k_val, r_b_test, BAR_MINUTES)

        print("\n===== Test Metrics (no leverage) =====")
        print(f"年化報酬 (策略): {test_metrics_raw['ann_ret_s']:.4%}")
        print(f"年化報酬 (B&H):  {test_metrics_raw['ann_ret_b']:.4%}")
        print(f"Sharpe:          {test_metrics_raw['sharpe']:.4f}")
        print(f"Sortino:         {test_metrics_raw['sortino']:.4f}")
        print(f"Alpha (CAPM):    {test_metrics_raw['alpha_capm']:.6f}")
        print(f"Alpha_adj:       {test_metrics_raw['alpha_adj']:.6f}")
        print(f"Beta:            {test_metrics_raw['beta']:.4f}")
        print(f"Max Drawdown:    {test_metrics_raw['mdd']:.4%}")

        print(f"\n===== Test Metrics (with leverage k={k_val:.3f}, from Val) =====")
        print(f"年化報酬 (策略): {test_metrics_lev['ann_ret_s']:.4%}")
        print(f"Sharpe:          {test_metrics_lev['sharpe']:.4f}")
        print(f"Max Drawdown:    {test_metrics_lev['mdd']:.4%}")

        # 畫 equity curve：分別畫無槓桿 / 有槓桿
        plot_equity(test_metrics_raw["equity_s"],
                    test_metrics_raw["equity_b"],
                    title="Strategy vs Buy & Hold (Test, no leverage)")

        plot_equity(test_metrics_lev["equity_s"],
                    test_metrics_lev["equity_b"],
                    title=f"Strategy vs Buy & Hold (Test, with leverage k={k_val:.3f})")
    else:
        print("No best model was recorded during training; skipping test evaluation & plot.")


if __name__ == "__main__":
    main()
