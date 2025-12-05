# transformer_position_trader.py
# -*- coding: utf-8 -*-
"""
OHLCV -> Transformer Encoder -> position in [-1, 1]

訓練目標：
- 用自訂 backtest-based loss：
  1) 先用 position 做回測，得到原始 MDD
  2) leverage = 0.1 / MDD  (有下限防止除以 0)
  3) 用 leverage 做一遍槓桿回測，loss = - 槓桿後最終報酬

資料切割：
- 50% Train, 25% Val, 25% Test（依時間順序切）

模型選擇：
- 在 Val 上，用「上面那個槓桿後最終報酬」當指標
- 每次有新 best model：
  - 存權重 best_model.pt
  - 畫出 Val 的 equity 曲線（不加槓桿＆加槓桿）+ 指標輸出

最後：
- 在 Test 上輸出兩張圖：
  1) 不加槓桿
  2) 加槓桿（leverage 用 Test 無槓桿 MDD 算）
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 檔案路徑設定 =====================

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'BITSTAMP_BTCUSD,240more.csv')  # 你給的路徑


# ===================== 工具函數：指標計算 =====================

def compute_mdd_torch(equity: torch.Tensor) -> torch.Tensor:
    """
    equity: (T,) 單調不一定遞增
    回傳最大回撤（正值比例，如 0.25 代表 25%）
    """
    # running max
    running_max, _ = torch.cummax(equity, dim=0)
    drawdown = (running_max - equity) / running_max.clamp(min=1e-8)
    mdd = drawdown.max()
    return mdd


def compute_mdd_np(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / np.maximum(running_max, 1e-8)
    return float(np.max(drawdown))


def compute_perf_stats(returns: np.ndarray,
                       bars_per_year: int = 365 * 6,
                       rf: float = 0.0):
    """
    returns: 單期報酬（例如 log return 或 simple return 都可，一致就好）
    bars_per_year: 一年大約幾個 bar（240min -> 約 6*365 = 2190）
    rf: 風險率（這裡簡化用 0）
    回傳: total_ret, ann_ret, sharpe, sortino, mdd
    """
    # 用 simple return 來算 equity
    equity = np.cumprod(1.0 + returns)
    total_ret = equity[-1] - 1.0

    # Excess returns
    excess = returns - rf / bars_per_year

    mean_ret = np.mean(excess)
    std_ret = np.std(excess)
    downside = excess[excess < 0]
    std_down = np.std(downside) if downside.size > 0 else np.nan

    ann_ret = (1.0 + total_ret) ** (bars_per_year / len(returns)) - 1.0 if len(returns) > 0 else np.nan

    sharpe = np.nan
    if std_ret > 0:
        sharpe = math.sqrt(bars_per_year) * mean_ret / std_ret

    sortino = np.nan
    if std_down > 0:
        sortino = math.sqrt(bars_per_year) * mean_ret / std_down

    mdd = compute_mdd_np(equity)

    return total_ret, ann_ret, sharpe, sortino, mdd, equity


def plot_equity(dates,
                equity,
                bh_equity=None,
                title="Equity Curve",
                filename="equity.png",
                stats_text=None):
    """
    畫出策略 equity 曲線（可選擇畫 B&H），並在圖上加上指標文字
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates, equity, label="Strategy")
    if bh_equity is not None:
        plt.plot(dates, bh_equity, linestyle="--", label="Buy & Hold")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()

    if stats_text is not None:
        # 把文字塞在圖的左上角
        plt.gcf().text(0.02, 0.95, stats_text, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle="round", alpha=0.1))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ===================== Transformer 模型 =====================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # Self-Attention block + 殘差
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        # FFN block + 殘差
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x


class PositionTransformer(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 seq_len: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(d_in, d_model)
        # learned positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output head：用最後一根時間步的 hidden state 當 summary
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 限制在 [-1, 1]
        )

        # 初始化位置 embedding
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (B, T, d_in)
        回傳: position (B,)
        """
        h = self.input_proj(x)  # (B, T, d_model)
        # 加上 positional embedding
        if h.size(1) != self.seq_len:
            raise ValueError(f"seq_len mismatch: got {h.size(1)}, expected {self.seq_len}")
        h = h + self.pos_embedding

        for layer in self.layers:
            h = layer(h)

        # 取最後一個 time step 的 hidden 作為 summary
        last_h = h[:, -1, :]  # (B, d_model)
        pos = self.head(last_h).squeeze(-1)  # (B,)
        return pos  # 已經在 [-1,1]


# ===================== Data 準備 =====================

def load_ohlcv_csv(path: str):
    """
    讀取 CSV，假設至少有: timestamp/date, open, high, low, close, volume
    如果欄位名稱不同，你自己改這裡的欄位對應即可。
    """
    df = pd.read_csv(path)

    # 嘗試找日期欄位
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        dates = df['date'].values
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df = df.sort_values('Timestamp')
        dates = df['Timestamp'].values
    else:
        # 如果沒有，直接用 index 當日期
        df = df.sort_index()
        dates = df.index.values

    # 欄位名猜測，可依你的實際 CSV 調整
    col_map = {
        'open': None,
        'high': None,
        'low': None,
        'close': None,
        'volume': None
    }

    # 自動匹配（小寫比較）
    lower_cols = {c.lower(): c for c in df.columns}
    for key in col_map.keys():
        if key in lower_cols:
            col_map[key] = lower_cols[key]

    # 檢查
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(f"找不到必要欄位: {missing}，請對照你的 CSV 修改 load_ohlcv_csv 中的欄位對應。")

    o = df[col_map['open']].astype(float).values
    h = df[col_map['high']].astype(float).values
    l = df[col_map['low']].astype(float).values
    c = df[col_map['close']].astype(float).values
    v = df[col_map['volume']].astype(float).values

    return dates, o, h, l, c, v


def build_sequence_data(o, h, l, c, v, seq_len: int = 64):
    """
    根據 OHLCV 建立序列資料：
    - features: [open, high, low, close, volume, log_return, hl_range]
    - returns: 使用 log-return 作為策略回測用的 single-period return
    """
    c = np.asarray(c, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # log return
    logret = np.zeros_like(c)
    logret[1:] = np.log(c[1:] / c[:-1].clip(min=1e-12))

    hl_range = (h - l) / h.clip(min=1e-12)

    feats = np.stack([o, h, l, c, v, logret, hl_range], axis=-1)  # (N, d_in)
    N = feats.shape[0]

    X_list = []
    r_list = []
    for i in range(seq_len, N):
        X_list.append(feats[i-seq_len:i])
        r_list.append(logret[i])  # 這一根 bar 的 log return

    X = np.stack(X_list, axis=0)  # (M, seq_len, d_in)
    R = np.asarray(r_list, dtype=np.float64)  # (M,)
    return X, R


# ===================== 自訂 Loss: backtest + leverage =====================

def backtest_loss(model,
                  X: torch.Tensor,
                  R: torch.Tensor,
                  device: torch.device):
    """
    X: (N, T, d_in)  單一整段（例如整個 train 或整個 val）
    R: (N,)          對應的 log return 或 simple return
    回傳:
        loss (要 minimize),
        final_ret (槓桿後),
        mdd (無槓桿),
        leverage,
        equity (無槓桿),
        equity_levered
    """
    model.train()  # 這個 function 會在訓練中被呼叫

    X = X.to(device)
    R = R.to(device)

    # 產生 position
    pos = model(X)  # (N,), 已經在 [-1,1]
    strat_ret = pos * R  # (N,) 策略單期報酬（不加槓桿）

    # 無槓桿 equity
    equity = torch.cumprod(1.0 + strat_ret, dim=0)
    mdd = compute_mdd_torch(equity)
    mdd_clamped = torch.clamp(mdd, min=1e-4)  # 防止除以 0

    leverage = 0.1 / mdd_clamped
    leverage = torch.clamp(leverage, min=1.0, max=100.0)  # 可選上限

    # 槓桿後策略
    levered_ret = leverage * strat_ret
    equity_levered = torch.cumprod(1.0 + levered_ret, dim=0)

    final_ret = equity_levered[-1] - 1.0  # 槓桿後總報酬

    loss = -final_ret  # 想 maximize final_ret -> minimize -final_ret

    # detach 指標，避免外面再 backward
    return loss, final_ret.detach(), mdd.detach(), leverage.detach(), equity.detach(), equity_levered.detach()


@torch.no_grad()
def eval_backtest(model,
                  X: torch.Tensor,
                  R: torch.Tensor,
                  device: torch.device):
    """
    跟 backtest_loss 類似，但不做 backward，用於 Val / Test。
    """
    model.eval()
    X = X.to(device)
    R = R.to(device)

    pos = model(X)  # (N,)
    strat_ret = pos * R
    equity = torch.cumprod(1.0 + strat_ret, dim=0)
    mdd = compute_mdd_torch(equity)
    mdd_clamped = torch.clamp(mdd, min=1e-4)
    leverage = 0.1 / mdd_clamped
    leverage = torch.clamp(leverage, min=1.0, max=100.0)  # 可選上限
    levered_ret = leverage * strat_ret
    equity_levered = torch.cumprod(1.0 + levered_ret, dim=0)
    final_ret = equity_levered[-1] - 1.0

    return final_ret.cpu(), mdd.cpu(), leverage.cpu(), equity.cpu(), equity_levered.cpu(), strat_ret.cpu()


# ===================== 主訓練流程 =====================

def main():
    # --------- Hyperparams ----------
    seq_len = 2**6
    d_model = 64
    nhead = 8
    num_layers = 3
    dim_ff = 128
    dropout = 0.1

    max_epochs = 10000
    lr = 1e-3
    weight_decay = 1e-5
    patience = 100  # early stopping
    min_delta = 0.0

    bars_per_year = 365 * 6  # 240min bar 粗估

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # --------- Data 讀取與處理 ----------
    print("讀取 CSV ...")
    dates_raw, o, h, l, c, v = load_ohlcv_csv(file_path)
    print(f"資料長度: {len(c)} bars")

    print("建立序列資料 ...")
    X, R = build_sequence_data(o, h, l, c, v, seq_len=seq_len)
    dates_seq = dates_raw[seq_len:]  # 對齊 X, R

    N = X.shape[0]
    print(f"序列樣本數: {N}")

    # 時序切割：50% / 25% / 25%
    n_train = int(N * 0.5)
    n_val = int(N * 0.25)
    n_test = N - n_train - n_val

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, N)

    X_train = X[train_slice]
    R_train = R[train_slice]
    dates_train = dates_seq[train_slice]

    X_val = X[val_slice]
    R_val = R[val_slice]
    dates_val = dates_seq[val_slice]

    X_test = X[test_slice]
    R_test = R[test_slice]
    dates_test = dates_seq[test_slice]

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # 轉 tensor
    X_train_t = torch.from_numpy(X_train).float()
    R_train_t = torch.from_numpy(R_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    R_val_t = torch.from_numpy(R_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    R_test_t = torch.from_numpy(R_test).float()

    d_in = X.shape[-1]

    # --------- 模型與 optimizer ----------
    model = PositionTransformer(
        d_in=d_in,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        seq_len=seq_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_score = -np.inf
    best_epoch = -1
    epochs_no_improve = 0

    # --------- 訓練 loop ----------
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        loss, train_final_ret, train_mdd, train_lev, train_equity, train_equity_lev = backtest_loss(
            model, X_train_t, R_train_t, device
        )

        loss.backward()
        optimizer.step()

        # Val 評估（不更新參數）
        val_final_ret, val_mdd, val_lev, val_equity, val_equity_lev, val_strat_ret = eval_backtest(
            model, X_val_t, R_val_t, device
        )

        val_final_ret = float(val_final_ret)
        val_mdd = float(val_mdd)
        val_lev = float(val_lev)

        print(f"[Epoch {epoch:03d}] "
              f"Train loss={loss.item():.6f}, "
              f"Train leveredRet={float(train_final_ret):.4f}, "
              f"Val leveredRet={val_final_ret:.4f}, "
              f"Val MDD={val_mdd:.4f}, "
              f"Val leverage={val_lev:.2f}")

        # early stopping & best model
        improved = (val_final_ret - best_val_score) > min_delta
        if improved:
            best_val_score = val_final_ret
            best_epoch = epoch
            epochs_no_improve = 0

            # 存模型
            best_model_path = os.path.join(base_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 新 best model (epoch={epoch})，已存至 {best_model_path}")

            # 同時畫出 Val 的 equity 圖（無槓桿 & 槓桿）
            val_returns_np = val_strat_ret.numpy()
            total_ret, ann_ret, sharpe, sortino, mdd_np, equity_np = compute_perf_stats(
                val_returns_np, bars_per_year=bars_per_year
            )

            # B&H（Val 區間）
            bh_ret = R_val  # log return -> 直接當 simple 也可以，但這裡為一致，用簡單 approx
            bh_equity = np.cumprod(1.0 + bh_ret)

            stats_text = (
                f"Strategy (no lev)\n"
                f"Total: {total_ret:.2%}\n"
                f"Ann:   {ann_ret:.2%}\n"
                f"Sharpe:{sharpe:.2f}\n"
                f"Sortino:{sortino:.2f}\n"
                f"MDD:   {mdd_np:.2%}\n"
                f"\nLeveraged\n"
                f"Lev:   {val_lev:.2f}\n"
                f"Final: {(float(val_equity_lev[-1]) - 1.0):.2%}"
            )

            val_equity_np = equity_np
            filename = os.path.join(base_dir, f"val_equity_epoch_{epoch:03d}.png")
            plot_equity(
                dates_val,
                val_equity_np,
                bh_equity=bh_equity,
                title=f"Validation Equity (Epoch {epoch})",
                filename=filename,
                stats_text=stats_text
            )
            print(f"  -> 已輸出驗證集報酬圖: {filename}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"連續 {patience} 個 epoch 沒有進步，提前停止訓練。")
            break

    print(f"訓練結束。best_epoch = {best_epoch}, best_val_leveredRet = {best_val_score:.4f}")

    # --------- 使用 best model 在 Test 上評估 ----------
    # 重新載入 best model
    best_model_path = os.path.join(base_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"已載入 best model: {best_model_path}")
    else:
        print("找不到 best_model.pt，只用目前 model 做評估。")

    # Test 評估（無槓桿）
    test_final_ret, test_mdd_t, test_lev_t, test_equity_t, test_equity_lev_t, test_strat_ret_t = eval_backtest(
        model, X_test_t, R_test_t, device
    )

    test_strat_ret_np = test_strat_ret_t.numpy()
    total_ret, ann_ret, sharpe, sortino, mdd_np, equity_np = compute_perf_stats(
        test_strat_ret_np, bars_per_year=bars_per_year
    )

    print("\n===== Test (no leverage) =====")
    print(f"Total return: {total_ret:.2%}")
    print(f"Annualized:   {ann_ret:.2%}")
    print(f"Sharpe:       {sharpe:.4f}")
    print(f"Sortino:      {sortino:.4f}")
    print(f"MDD:          {mdd_np:.2%}")

    # Test B&H
    bh_ret_test = R_test
    bh_equity_test = np.cumprod(1.0 + bh_ret_test)

    # 圖1：Test 無槓桿
    stats_text1 = (
        f"Strategy (no leverage)\n"
        f"Total: {total_ret:.2%}\n"
        f"Ann:   {ann_ret:.2%}\n"
        f"Sharpe:{sharpe:.2f}\n"
        f"Sortino:{sortino:.2f}\n"
        f"MDD:   {mdd_np:.2%}"
    )

    filename1 = os.path.join(base_dir, "test_equity_no_leverage.png")
    plot_equity(
        dates_test,
        equity_np,
        bh_equity=bh_equity_test,
        title="Test Equity (No Leverage)",
        filename=filename1,
        stats_text=stats_text1
    )
    print(f"已輸出 Test 無槓桿報酬圖: {filename1}")

    # 圖2：Test 加槓桿（照你的定義：先拿無槓桿 MDD 算 leverage）
    mdd_for_lev = max(mdd_np, 1e-4)
    leverage_test = 0.1 / mdd_for_lev
    leverage_test = np.clip(leverage_test, 1.0, 100.0)  # 可選上限
    levered_ret_np = leverage_test * test_strat_ret_np
    _, _, sharpe_l, sortino_l, mdd_l, equity_l = compute_perf_stats(
        levered_ret_np, bars_per_year=bars_per_year
    )

    print("\n===== Test (with leverage) =====")
    print(f"Leverage:     {leverage_test:.2f}")
    print(f"Total return: {equity_l[-1] - 1.0:.2%}")
    print(f"Sharpe:       {sharpe_l:.4f}")
    print(f"Sortino:      {sortino_l:.4f}")
    print(f"MDD:          {mdd_l:.2%}")

    stats_text2 = (
        f"Strategy (with leverage)\n"
        f"Lev:   {leverage_test:.2f}\n"
        f"Total: {equity_l[-1] - 1.0:.2%}\n"
        f"Sharpe:{sharpe_l:.2f}\n"
        f"Sortino:{sortino_l:.2f}\n"
        f"MDD:   {mdd_l:.2%}"
    )
    filename2 = os.path.join(base_dir, "test_equity_with_leverage.png")
    plot_equity(
        dates_test,
        equity_l,
        bh_equity=bh_equity_test,
        title="Test Equity (With Leverage)",
        filename=filename2,
        stats_text=stats_text2
    )
    print(f"已輸出 Test 加槓桿報酬圖: {filename2}")


if __name__ == "__main__":
    main()
