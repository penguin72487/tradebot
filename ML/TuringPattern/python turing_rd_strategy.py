#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Turing / Reaction-Diffusion style activator-inhibitor trading strategy (BTCUSD 240m)

CSV: BITSTAMP_BTCUSD,240more.csv
Required columns: time (unix), open, high, low, close

Outputs:
- Equity curve (log scale)
- Position size curve
- Metrics: Sharpe, Sortino, Max Drawdown (MDD)

Run:
  python turing_rd_strategy.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ✅ 你只需要改這裡：全是常數（不要任何環境參數 / CLI 參數）
# ============================================================

MODEL = "fhn"         # "fhn" | "gm" | "gs"

FEE_BPS = 10.0        # 交易成本：每次換倉 turnover 的 bps（10 = 0.10%）
MAX_ABS_POS = 1.0     # 最大槓桿/部位（[-1,1] 代表全空到全多）
DEADBAND = 0.06       # 訊號死區：小訊號直接當 0，降低頻繁交易

# bar frequency: 240m = 4hr = about 6 bars/day
BARS_PER_YEAR = 6 * 365

FORCE_WIN = 24        # forcing = fast mean(return) window
VOL_WIN   = 48        # vol window (risk scaling)
Z_WIN     = 240       # z-score normalization window

# ODE integration
DT = 0.18
STEPS_PER_BAR = 4
SEED_U = 0.02
SEED_V = 0.02

# Model params (你也可以直接改這些)
FHN_A = 0.7
FHN_B = 0.8
FHN_EPS = 0.08
FHN_I_SCALE = 1.0

GM_ALPHA = 1.4
GM_BETA  = 1.0
GM_GAMMA = 0.25
GM_DELTA = 0.6
GM_S_SCALE = 0.6

GS_F = 0.035
GS_K = 0.062
GS_S_SCALE = 0.8

# Signal mix
RAW_MIX_V = 0.8       # raw = u - RAW_MIX_V * v


# =========================
# Metrics
# =========================
def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / np.clip(peak, 1e-12, None))
    return float(np.max(dd))

def sharpe_sortino(returns: np.ndarray, bars_per_year: int) -> tuple[float, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return float("nan"), float("nan")

    mu = r.mean()
    sd = r.std(ddof=1)
    sharpe = (mu / (sd + 1e-12)) * math.sqrt(bars_per_year)

    downside = r[r < 0]
    if len(downside) < 5:
        sortino = float("nan")
    else:
        dd = downside.std(ddof=1)
        sortino = (mu / (dd + 1e-12)) * math.sqrt(bars_per_year)

    return float(sharpe), float(sortino)


# =========================
# Data loading
# =========================
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    cols = {c.lower(): c for c in df.columns}
    def pick(name: str):
        if name in cols:
            return cols[name]
        raise ValueError(f"CSV missing column: {name}. Found: {list(df.columns)}")

    tcol = pick("time")
    ocol = pick("open")
    hcol = pick("high")
    lcol = pick("low")
    ccol = pick("close")

    out = pd.DataFrame({
        "time": df[tcol].astype(np.int64),
        "open": pd.to_numeric(df[ocol], errors="coerce"),
        "high": pd.to_numeric(df[hcol], errors="coerce"),
        "low": pd.to_numeric(df[lcol], errors="coerce"),
        "close": pd.to_numeric(df[ccol], errors="coerce"),
    }).dropna().reset_index(drop=True)

    out = out.sort_values("time").reset_index(drop=True)
    return out


# =========================
# Activator/Inhibitor ODE simulation
# =========================
def simulate_uv(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    u = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)

    u0, v0 = SEED_U, SEED_V

    def clip_uv(uu, vv):
        uu = float(np.clip(uu, -5.0, 5.0))
        vv = float(np.clip(vv, -5.0, 5.0))
        return uu, vv

    for i in range(n):
        forcing = x[i]
        uu, vv = u0, v0

        for _ in range(STEPS_PER_BAR):
            if MODEL == "fhn":
                I = FHN_I_SCALE * forcing
                du = (uu - (uu**3)/3.0 - vv + I)
                dv = FHN_EPS * (uu + FHN_A - FHN_B * vv)

            elif MODEL == "gm":
                s = GM_S_SCALE * forcing
                denom = 1.0 + max(vv, -0.9)  # avoid blow-up
                du = GM_ALPHA * (uu*uu) / denom - GM_BETA * uu + s
                dv = GM_GAMMA * (uu*uu) - GM_DELTA * vv

            elif MODEL == "gs":
                s = GS_S_SCALE * forcing
                uvv2 = uu * (vv*vv)
                du = -uvv2 + GS_F * (1.0 - uu) + s
                dv =  uvv2 - (GS_F + GS_K) * vv

            else:
                raise ValueError("MODEL must be one of: 'fhn', 'gm', 'gs'")

            uu += DT * du
            vv += DT * dv
            uu, vv = clip_uv(uu, vv)

        u[i] = uu
        v[i] = vv
        u0, v0 = uu, vv

    return u, v


# =========================
# Backtest
# =========================
def backtest(df: pd.DataFrame):
    close = df["close"].to_numpy(dtype=float)

    # log returns
    logret = np.zeros_like(close)
    logret[1:] = np.log(np.clip(close[1:] / close[:-1], 1e-12, None))

    # forcing: fast mean of returns then z-score
    r_fast = pd.Series(logret).rolling(FORCE_WIN, min_periods=FORCE_WIN).mean().to_numpy()
    r_mu = pd.Series(r_fast).rolling(Z_WIN, min_periods=Z_WIN).mean().to_numpy()
    r_sd = pd.Series(r_fast).rolling(Z_WIN, min_periods=Z_WIN).std(ddof=1).to_numpy()
    forcing = (r_fast - r_mu) / (r_sd + 1e-12)
    forcing = np.nan_to_num(forcing, nan=0.0, posinf=0.0, neginf=0.0)

    # vol stress for risk scaling
    vol = pd.Series(logret).rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=1).to_numpy()
    vol_mu = pd.Series(vol).rolling(Z_WIN, min_periods=Z_WIN).mean().to_numpy()
    vol_sd = pd.Series(vol).rolling(Z_WIN, min_periods=Z_WIN).std(ddof=1).to_numpy()
    vol_z = (vol - vol_mu) / (vol_sd + 1e-12)
    vol_z = np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=0.0)

    # ODE -> u,v
    u, v = simulate_uv(forcing)

    # signal: activator - inhibitor
    raw = u - RAW_MIX_V * v
    sig = np.tanh(raw)

    # deadband
    sig[np.abs(sig) < DEADBAND] = 0.0

    # risk scaling (high vol -> smaller)
    scale = 1.0 / (1.0 + np.exp(vol_z))  # (0,1)
    pos = sig * scale
    pos = np.clip(pos, -MAX_ABS_POS, MAX_ABS_POS)

    # execute with 1-bar delay (no lookahead)
    pos_exec = np.zeros_like(pos)
    pos_exec[1:] = pos[:-1]

    # costs on turnover
    turnover = np.abs(pos_exec - np.roll(pos_exec, 1))
    turnover[0] = 0.0
    fee = FEE_BPS * 1e-4
    cost = turnover * fee

    strat_ret = pos_exec * logret - cost
    equity = np.exp(np.cumsum(strat_ret))

    sharpe, sortino = sharpe_sortino(strat_ret[1:], bars_per_year=BARS_PER_YEAR)
    mdd = max_drawdown(equity)

    out = df.copy()
    out["logret"] = logret
    out["forcing"] = forcing
    out["u"] = u
    out["v"] = v
    out["pos"] = pos_exec
    out["strat_logret"] = strat_ret
    out["equity"] = equity

    metrics = {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD": mdd,
        "FinalEquity": float(equity[-1]),
    }
    return out, metrics


def plot_results(df_bt: pd.DataFrame, metrics: dict):
    dt = pd.to_datetime(df_bt["time"].to_numpy(), unit="s")

    equity = df_bt["equity"].to_numpy(dtype=float)
    pos = df_bt["pos"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(dt, equity)
    plt.yscale("log")
    plt.title(
        f"Turing Activator-Inhibitor Strategy ({MODEL.upper()})\n"
        f"Sharpe={metrics['Sharpe']:.3f}  Sortino={metrics['Sortino']:.3f}  "
        f"MDD={metrics['MDD']*100:.2f}%  Final={metrics['FinalEquity']:.2f}"
    )
    plt.xlabel("Time")
    plt.ylabel("Equity (log scale)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(dt, pos)
    plt.title("Position Size")
    plt.xlabel("Time")
    plt.ylabel("Position ([-1,1])")
    plt.tight_layout()
    plt.show()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "BITSTAMP_BTCUSD,240more.csv")

    df = load_csv(csv_path)
    df_bt, metrics = backtest(df)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    plot_results(df_bt, metrics)


if __name__ == "__main__":
    main()
