# -*- coding: utf-8 -*-
"""
Fourier 多峰週期偵測 + 80/20 外推預測（PyTorch，支援 CUDA）
- 直接在下方 CONFIG 填好路徑與參數就能跑
- 預設用 log-returns 做 FFT（較穩定），在 80% 訓練集找前 K 個峰
- 用多正弦（sin/cos）線性回歸擬合 returns，外推 20% 並還原到價格
- 頻段預設 20~80 天（可改）
"""

import os
import math
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt

# ================================
#            CONFIG
# ================================
base_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(base_dir, "price.csv")  # <- 在這裡改路徑
os.chdir(base_dir)
CONFIG = {
    # 必填：CSV 路徑
    "CSV_PATH": CSV_PATH,
    # 欄位名（留空字串會自動偵測）
    "TIME_COL": "time",
    "PRICE_COL": "close",
    # 切分比例與峰數、頻段（天）
    "TRAIN_RATIO": 0.8,
    "PEAKS": 10,
    "MIN_DAYS": 20.0,
    "MAX_DAYS": 80.0,
    # 使用序列： "returns" 或 "detrended-log" 或 "log-price"
    "SERIES": "returns",
    # 裝置："auto"（預設，有 GPU 就用 CUDA）或 "cuda" 或 "cpu"
    "DEVICE": "auto",
    # 是否顯示圖
    "PLOT": True,
    # 輸出資料夾（會存 summary.csv 與圖）
    "SAVE_DIR": "fft_out",
    "SAVE_CSV": True,
}

# ================================
#           Utilities
# ================================
def pick_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("要求使用 CUDA，但環境沒有可用 GPU。")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_datetime_auto(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        med = float(pd.Series(s).median())
        if med >= 1e17:
            unit = "ns"
        elif med >= 1e14:
            unit = "us"
        elif med >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    else:
        return pd.to_datetime(s, utc=True, errors="coerce")

def auto_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = df.columns
    t_cands = [c for c in cols if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    p_cands = [c for c in cols if c.lower() in ["close", "c", "price", "adj close", "adj_close"]]
    time_col = t_cands[0] if t_cands else cols[0]
    price_col = p_cands[0] if p_cands else ( "close" if "close" in df.columns.str.lower().tolist() else cols[-1] )
    return time_col, price_col

def make_even_sampling(df: pd.DataFrame, time_col: str) -> float:
    med = df[time_col].diff().median()
    if pd.isna(med) or med == pd.Timedelta(0):
        raise ValueError("時間欄位重複或不足以推斷取樣間隔。")
    return med / pd.Timedelta(days=1)

# ...existing code...
def build_series(df: pd.DataFrame, price_col: str, kind: str) -> pd.Series:
    price_values = df[price_col].astype(float).values
    logp = np.log(price_values)
    
    if kind == "returns":
        r = np.diff(logp, prepend=logp[0])
        return pd.Series(r, index=df.index)
    elif kind == "log-price":  # 新增選項
        return pd.Series(logp, index=df.index)  # 直接返回 log 價格
    elif kind == "detrended-log":
        # detrended log price
        t = np.arange(len(logp), dtype=float)
        slope, intercept = np.polyfit(t, logp, deg=1)
        trend = intercept + slope * t
        x = logp - trend
        x = x - x.mean()
        return pd.Series(x, index=df.index)
    else:
        raise ValueError(f"未知的 SERIES 類型: {kind}")


def topk_peaks(power: Tensor, freq: Tensor, min_period_smp: int, max_period_smp: int, k: int) -> List[int]:
    with torch.no_grad():
        mask = (freq > 0)
        period_smp = torch.where(freq > 0, 1.0 / freq, torch.inf)
        mask = mask & (period_smp >= float(min_period_smp)) & (period_smp <= float(max_period_smp))
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            power_no0 = power.clone()
            power_no0[0] = -torch.inf
            top = torch.topk(power_no0, k=min(k, power_no0.numel())).indices
            return top.tolist()
        p_sel = power[idx]
        k_sel = min(k, p_sel.numel())
        top_local = torch.topk(p_sel, k=k_sel).indices
        top_global = idx[top_local]
        return top_global.tolist()

def fit_harmonics_torch(y: Tensor, freqs: Tensor, device: torch.device) -> Tuple[Tensor, Tensor]:
    N = y.shape[0]
    t = torch.arange(N, dtype=torch.float64, device=device)
    cols = []
    two_pi = 2.0 * math.pi
    for f in freqs.tolist():
        omega = two_pi * f
        cols.append(torch.sin(omega * t))
        cols.append(torch.cos(omega * t))
    X = torch.stack(cols, dim=1) if cols else torch.zeros((N,0), dtype=torch.float64, device=device)
    X = torch.cat([X, torch.ones((N,1), dtype=torch.float64, device=device)], dim=1)  # + 截距
    beta = torch.linalg.lstsq(X, y).solution
    return beta, X

def forecast_with_harmonics(beta: Tensor, freqs: Tensor, n_future: int, start_idx: int, device: torch.device) -> Tensor:
    two_pi = 2.0 * math.pi
    h = torch.arange(start_idx, start_idx + n_future, dtype=torch.float64, device=device)
    cols = []
    for f in freqs.tolist():
        omega = two_pi * f
        cols.append(torch.sin(omega * h))
        cols.append(torch.cos(omega * h))
    X2 = torch.stack(cols, dim=1) if cols else torch.zeros((n_future, 0), dtype=torch.float64, device=device)
    X2 = torch.cat([X2, torch.ones((n_future,1), dtype=torch.float64, device=device)], dim=1)
    return X2 @ beta

def metrics(y: np.ndarray, yhat: np.ndarray):
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mape = float(np.mean(np.abs((yhat - y) / (np.abs(y) + 1e-9))) * 100.0)
    dy = np.sign(pd.Series(y).pct_change().fillna(0).values)
    dh = np.sign(pd.Series(yhat).pct_change().fillna(0).values)
    acc = float(np.mean(dy == dh))
    return rmse, mape, acc

# ================================
#              Main
# ================================
def main():
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    device = pick_device(CONFIG["DEVICE"])
    torch.set_default_dtype(torch.float64)
    print(f"[DEVICE] {device}")

    df = pd.read_csv(CONFIG["CSV_PATH"])

    tcol = CONFIG["TIME_COL"] or ""
    pcol = CONFIG["PRICE_COL"] or ""
    if not tcol or not pcol:
        auto_t, auto_p = auto_columns(df)
        tcol = tcol or auto_t
        pcol = pcol or auto_p

    df[tcol] = parse_datetime_auto(df[tcol])
    df = df.dropna(subset=[tcol, pcol]).sort_values(tcol).drop_duplicates(subset=[tcol], keep="last").reset_index(drop=True)

    sample_days = make_even_sampling(df, tcol)
    print(f"[INFO] samples={len(df)}  interval≈{sample_days*24:.2f} hours")

    # 準備序列
    y_series = build_series(df, pcol, CONFIG["SERIES"])
    N = len(y_series)
    cut = int(N * CONFIG["TRAIN_RATIO"])
    train = y_series.iloc[:cut].astype(float).values
    test  = y_series.iloc[cut:].astype(float).values

    # FFT（Hann 視窗）
    win = np.hanning(len(train)) if len(train) > 1 else np.ones_like(train)
    xw = (train - train.mean()) * win
    xw_t = torch.tensor(xw, device=device)

    fft = torch.fft.rfft(xw_t)
    power = (fft.abs() ** 2) / max(len(xw), 1)
    freqs = torch.fft.rfftfreq(n=len(xw), d=1.0, device=device, dtype=torch.float64)

    # 頻段限制（天→樣本）
    min_period_smp = max(2, int(np.floor(CONFIG["MIN_DAYS"] / sample_days)))
    max_period_smp = int(np.ceil(CONFIG["MAX_DAYS"] / sample_days))

    # 取前 K 峰
    peak_idx = topk_peaks(power, freqs, min_period_smp, max_period_smp, CONFIG["PEAKS"])
    sel_freqs = freqs[peak_idx]
    sel_period_days = (1.0 / sel_freqs.detach().cpu().numpy()) * sample_days
    print("[PEAKS] periods(days) =", ", ".join(f"{d:.2f}" for d in sel_period_days))

    # 擬合（訓練）
    y_train_t = torch.tensor(train, device=device)
    beta, X = fit_harmonics_torch(y_train_t, sel_freqs, device)
    yhat_train = (X @ beta).detach().cpu().numpy()

    # 外推（測試）
    yhat_test = forecast_with_harmonics(beta, sel_freqs, n_future=len(test), start_idx=len(train), device=device).detach().cpu().numpy()

    # 還原到價格
    close = df[pcol].astype(float).values
    if CONFIG["SERIES"] == "returns":
        logp0 = np.log(close[0])
        logp_pred_train = logp0 + np.cumsum(yhat_train)
        logp_pred_test  = logp_pred_train[-1] + np.cumsum(yhat_test)
        pred_train = np.exp(logp_pred_train)
        pred_test  = np.exp(logp_pred_test)
        actual_train = close[:cut]
        actual_test  = close[cut:]
    else:
        t = np.arange(N, dtype=float)
        slope, intercept = np.polyfit(t[:cut], np.log(close[:cut]), deg=1)
        trend_train = intercept + slope * t[:cut]
        trend_test  = intercept + slope * t[cut:]
        logp_pred_train = yhat_train + trend_train
        logp_pred_test  = yhat_test  + trend_test
        pred_train = np.exp(logp_pred_train)
        pred_test  = np.exp(logp_pred_test)
        actual_train = close[:cut]
        actual_test  = close[cut:]

    # 指標
    rmse_tr, mape_tr, acc_tr = metrics(actual_train, pred_train)
    rmse_te, mape_te, acc_te = metrics(actual_test,  pred_test)
    print(f"[TRAIN] RMSE={rmse_tr:.4f}  MAPE={mape_tr:.2f}%  DirAcc={acc_tr:.3f}")
    print(f"[TEST ] RMSE={rmse_te:.4f}  MAPE={mape_te:.2f}%  DirAcc={acc_te:.3f}")

    # 存 summary
    if CONFIG["SAVE_CSV"]:
        pd.DataFrame({
            "metric": ["RMSE", "MAPE(%)", "DirectionAcc"],
            "train": [rmse_tr, mape_tr, acc_tr],
            "test":  [rmse_te, mape_te, acc_te]
        }).to_csv(os.path.join(CONFIG["SAVE_DIR"], "summary.csv"), index=False)

    # 繪圖
    if CONFIG["PLOT"]:
        # 功率譜（freq→period days）
        with torch.no_grad():
            mask = freqs > 0
            period_days_all = torch.where(mask, (1.0 / freqs) * sample_days, torch.nan).detach().cpu().numpy()
            power_all = power.detach().cpu().numpy()
        plt.figure()
        plt.plot(period_days_all[~np.isnan(period_days_all)], power_all[mask.cpu().numpy()])
        for d in sel_period_days:
            plt.axvline(d, linestyle="--")
        plt.xscale("log")
        plt.xlabel("Period (days)")
        plt.ylabel("Power")
        plt.title("Power Spectrum (returns / Hann)")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "power_spectrum.png"), dpi=160)
        plt.show()

        # 價格：訓練+測試
        time = df[tcol].values
        full_pred = np.concatenate([pred_train, pred_test])
        plt.figure()
        plt.plot(time, close, label="Actual")
        plt.plot(time, full_pred, label="Pred")
        plt.axvline(time[cut-1], linestyle="--")
        plt.legend()
        plt.title(f"Actual vs Pred | peaks(days)={','.join(f'{d:.1f}' for d in sel_period_days)}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["SAVE_DIR"], "actual_vs_pred.png"), dpi=160)
        plt.show()

if __name__ == "__main__":
    main()
