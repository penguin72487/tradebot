"""
transformer_ae_alpha.py

Transformer-based AutoEncoder for K-line sequences with:
- Encoder: Transformer encoder (self-attention)
- Decoder: MLP decoder to reconstruct the input sequence (autoencoder)
- Alpha head: directly outputs position/size (alpha) in [-1, 1]
- Optional return head: predicts future return

You can plug this into your own training loop or use the example at bottom.
"""

import math
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from typing import Tuple


# ===================== Positional Encoding =====================


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding: adds position info to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ===================== Transformer AE + Alpha Head =====================


class TransformerAEAlpha(nn.Module):
    """
    Transformer-based AutoEncoder with:
    - Transformer encoder over sequence
    - Latent vector z (global representation of sequence)
    - MLP decoder to reconstruct input
    - Alpha head to output position in [-1, 1]
    - Optional return head to predict future return

    Args:
        seq_len: number of time steps per sample
        feature_dim: number of features per time step (e.g. OHLCV -> 5)
        d_model: Transformer hidden dimension
        nhead: number of attention heads
        num_layers: number of Transformer encoder layers
        latent_dim: dimension of latent vector z
        use_return_head: whether to add a head to predict future return
    """

    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        latent_dim: int = 32,
        use_return_head: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.use_return_head = use_return_head
        self.dropout = dropout

        # Project per-timestep features into Transformer dimension
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding + Transformer encoder
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,  # (B, T, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # From sequence representation -> global latent z
        # Here we use mean pooling over time, then a linear to latent_dim
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity(),
        )

        # Decoder: from latent z back to flattened sequence (seq_len * feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity(),
            nn.Linear(d_model, seq_len * feature_dim),
        )

        # Alpha head: from z to continuous position in [-1, 1]
        self.alpha_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout and dropout > 0.0 else nn.Identity(),
            nn.Linear(latent_dim, 1),
            nn.Tanh(),  # ensures output in [-1, 1]
        )

        # Optional future return head
        if use_return_head:
            self.return_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1),
            )
        else:
            self.return_head = None

    # -------- Encoding & decoding helpers -------- #

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode sequence into latent representation.

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            {
              "z": (batch, latent_dim),
              "h_seq": (batch, seq_len, d_model)  # encoded per-timestep reps
            }
        """
        # project features
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)  # (B, T, d_model)

        # mean pooling over time
        h_mean = h.mean(dim=1)  # (B, d_model)
        z = self.to_latent(h_mean)  # (B, latent_dim)
        return {"z": z, "h_seq": h}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z back to sequence.

        Args:
            z: (batch, latent_dim)

        Returns:
            x_recon: (batch, seq_len, feature_dim)
        """
        x_flat = self.decoder(z)  # (B, seq_len * feature_dim)
        x_recon = x_flat.view(-1, self.seq_len, self.feature_dim)
        return x_recon

    # -------- Forward -------- #

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, feature_dim), already normalized/scaled

        Returns:
            {
              "x_recon": (B, seq_len, feature_dim),
              "z": (B, latent_dim),
              "alpha": (B, 1),
              "ret_pred": (B, 1) or None
            }
        """
        enc = self.encode(x)
        z = enc["z"]
        x_recon = self.decode(z)

        alpha = self.alpha_head(z)  # (B, 1)

        if self.return_head is not None:
            ret_pred = self.return_head(z)  # (B, 1), no activation
        else:
            ret_pred = None

        return {
            "x_recon": x_recon,
            "z": z,
            "alpha": alpha,
            "ret_pred": ret_pred,
        }


# ===================== Loss helper =====================


def compute_objective(
    model_outputs: Dict[str, torch.Tensor],
    x_true: torch.Tensor,
    alpha_target: Optional[torch.Tensor] = None,
    ret_target: Optional[torch.Tensor] = None,
    lambda_recon: float = 0.0,
    lambda_alpha: float = 0.0,
    lambda_ret: float = 0.0,
    lambda_pnl: float = 0.0,
    lambda_alpha_reg: float = 0.0,
    annualize: float = 1.0,
    lambda_sharpe_bh: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-task loss:
      total = lambda_recon * recon_loss + lambda_alpha * alpha_loss + lambda_ret * ret_loss

    Args:
        model_outputs: dict from model(...)
        x_true: (B, seq_len, feature_dim), original normalized input sequence
        alpha_target: (B, 1), desired position/size in [-1, 1] (can be derived from future returns)
        ret_target: (B, 1), future k-step return (regression target)
        lambda_*: weights for each loss term

    Returns:
        {
          "loss": total_loss,
          "recon_loss": recon_loss,
          "alpha_loss": alpha_loss or zero,
          "ret_loss": ret_loss or zero
        }
    """
    x_recon = model_outputs["x_recon"]
    alpha_pred = model_outputs["alpha"]
    ret_pred = model_outputs["ret_pred"]

    # Reconstruction loss (sequence MSE)
    recon_loss = torch.mean((x_recon - x_true) ** 2)

    # Alpha loss (regression MSE by default)
    if alpha_target is not None:
        alpha_loss = torch.mean((alpha_pred - alpha_target) ** 2)
    else:
        alpha_loss = torch.zeros((), device=x_true.device)

    # Return prediction loss
    if ret_target is not None and ret_pred is not None:
        ret_loss = torch.mean((ret_pred - ret_target) ** 2)
    else:
        ret_loss = torch.zeros((), device=x_true.device)

    # PnL surrogate: encourage alpha to align with realized return
    # We maximize mean(alpha * ret) -> include as negative loss term
    if lambda_pnl is not None and lambda_pnl != 0.0 and ret_target is not None:
        # alpha_pred: (B,1), ret_target: (B,1) -> squeeze
        pnl_term = torch.mean((alpha_pred.view(-1) * ret_target.view(-1)))
        pnl_loss = -pnl_term
    else:
        pnl_loss = torch.zeros((), device=x_true.device)

    # Alpha magnitude regularization (L2 on alpha) to discourage extreme positions
    if lambda_alpha_reg is not None and lambda_alpha_reg != 0.0:
        alpha_reg = torch.mean(alpha_pred ** 2)
        alpha_reg_loss = lambda_alpha_reg * alpha_reg
    else:
        alpha_reg_loss = torch.zeros((), device=x_true.device)



    # Compute Sharpe-BH metric on the batch (if ret_target provided).
    # Treat buy-and-hold mean return as a baseline (risk-free for this metric).
    sharpe_bh = torch.tensor(0.0, device=x_true.device)
    if ret_target is not None:
        try:
            # alpha_pred: (B,1), ret_target: (B,1)
            strat_ret = (alpha_pred.view(-1) * ret_target.view(-1))
            bh_ret = ret_target.view(-1)
            mu = torch.mean(strat_ret)
            sigma = torch.std(strat_ret)
            mu_bh = torch.mean(bh_ret)
            sharpe_bh = (mu - mu_bh) / (sigma + 1e-12)
            # annualize if requested
            if annualize is not None and annualize > 0:
                sharpe_bh = sharpe_bh * math.sqrt(annualize)
        except Exception:
            sharpe_bh = torch.tensor(0.0, device=x_true.device)

    # Loss = only Sharpe-BH (negative because optimizer minimizes)
    # We set the optimisation objective to maximize Sharpe-BH by minimizing -sharpe_bh.
    # Note: if `ret_target` is None then sharpe_bh==0 and loss provides no gradient.
    total_loss = -sharpe_bh
    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "alpha_loss": alpha_loss,
        "ret_loss": ret_loss,
        "sharpe_bh": sharpe_bh,
    }

# backward-compatible alias: some modules call compute_loss()
compute_loss = compute_objective


# ===================== Simple Dataset / Training Sketch =====================


class KlineSeqDataset(Dataset):
    """
    Generic dataset for windowed K-line sequences.

    Expected precomputed arrays:
        X:          (N, seq_len, feature_dim), already normalized (e.g. z-score)
        y_alpha:    (N,), target positions in [-1, 1]
        y_ret:      (N,), future returns (optional, can be None or not used)

    You are free to compute y_alpha from y_ret however you like:
        e.g. y_alpha = clip(y_ret / scale, -1, 1)
    """

    def __init__(
        self,
        X: np.ndarray,
        y_alpha: Optional[np.ndarray] = None,
        y_ret: Optional[np.ndarray] = None,
    ):
        super().__init__()
        assert X.ndim == 3, "X must be (N, seq_len, feature_dim)"
        self.X = torch.from_numpy(X.astype(np.float32))

        if y_alpha is not None:
            assert y_alpha.shape[0] == X.shape[0]
            self.y_alpha = torch.from_numpy(y_alpha.astype(np.float32)).view(-1, 1)
        else:
            self.y_alpha = None

        if y_ret is not None:
            assert y_ret.shape[0] == X.shape[0]
            self.y_ret = torch.from_numpy(y_ret.astype(np.float32)).view(-1, 1)
        else:
            self.y_ret = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y_alpha = None if self.y_alpha is None else self.y_alpha[idx]
        y_ret = None if self.y_ret is None else self.y_ret[idx]
        return x, y_alpha, y_ret


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_recon: float = 1.0,
    lambda_alpha: float = 1.0,
    lambda_ret: float = 1.0,
    lambda_pnl: float = 0.0,
    lambda_alpha_reg: float = 0.0,
    lambda_sharpe_bh: float = 0.0,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_alpha = 0.0
    total_ret = 0.0
    n_samples = 0
    total_sharpe_bh = 0.0

    for batch in dataloader:
        x, y_alpha, y_ret = batch
        x = x.to(device)
        y_alpha = None if y_alpha is None else y_alpha.to(device)
        y_ret = None if y_ret is None else y_ret.to(device)

        outputs = model(x)
        losses = compute_objective(
            outputs,
            x_true=x,
            alpha_target=y_alpha,
            ret_target=y_ret,
            lambda_recon=lambda_recon,
            lambda_alpha=lambda_alpha,
            lambda_ret=lambda_ret,
            lambda_pnl=lambda_pnl,
            lambda_alpha_reg=lambda_alpha_reg,
            lambda_sharpe_bh=lambda_sharpe_bh,
        )

        loss = losses["loss"]
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping to stabilize training
        if grad_clip is not None and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        n_samples += batch_size
        total_loss += loss.item() * batch_size
        total_recon += losses["recon_loss"].item() * batch_size
        total_alpha += losses["alpha_loss"].item() * batch_size
        total_ret += losses["ret_loss"].item() * batch_size
        # accumulate sharpe_bh (tensor -> float)
        if "sharpe_bh" in losses:
            total_sharpe_bh += float(losses["sharpe_bh"].item()) * batch_size

    out = {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "alpha_loss": total_alpha / n_samples,
        "ret_loss": total_ret / n_samples,
    }
    # attach avg sharpe_bh if accumulated
    out['sharpe_bh'] = (total_sharpe_bh / n_samples) if n_samples > 0 else 0.0
    return out


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lambda_recon: float = 1.0,
    lambda_alpha: float = 1.0,
    lambda_ret: float = 1.0,
    lambda_pnl: float = 0.0,
    lambda_alpha_reg: float = 0.0,
    lambda_sharpe_bh: float = 0.0,
):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_alpha = 0.0
    total_ret = 0.0
    n_samples = 0
    total_sharpe_bh = 0.0

    for batch in dataloader:
        x, y_alpha, y_ret = batch
        x = x.to(device)
        y_alpha = None if y_alpha is None else y_alpha.to(device)
        y_ret = None if y_ret is None else y_ret.to(device)

        outputs = model(x)
        losses = compute_objective(
            outputs,
            x_true=x,
            alpha_target=y_alpha,
            ret_target=y_ret,
            lambda_recon=lambda_recon,
            lambda_alpha=lambda_alpha,
            lambda_ret=lambda_ret,
            lambda_pnl=lambda_pnl,
            lambda_alpha_reg=lambda_alpha_reg,
            lambda_sharpe_bh=lambda_sharpe_bh,
        )

        batch_size = x.size(0)
        n_samples += batch_size
        total_loss += losses["loss"].item() * batch_size
        total_recon += losses["recon_loss"].item() * batch_size
        total_alpha += losses["alpha_loss"].item() * batch_size
        total_ret += losses["ret_loss"].item() * batch_size
        if "sharpe_bh" in losses:
            total_sharpe_bh += float(losses["sharpe_bh"].item()) * batch_size

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "alpha_loss": total_alpha / n_samples,
        "ret_loss": total_ret / n_samples,
        "sharpe_bh": (total_sharpe_bh / n_samples) if n_samples > 0 else 0.0,
    }


# ===================== Usage sketch (you adapt to your data) =====================

if __name__ == "__main__":
    """
    This is just a minimal example of how to instantiate and train the model.
    You should replace the random data with your real K-line windows and targets.
    """
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Default data file (you can edit this path)
    file_path = os.path.join(base_dir, 'BITSTAMP_BTCUSD,240more.csv')
    # file_path = os.path.join(base_dir, 'BYBIT_BTCUSDT15.csv')
    # Hyper-parameters (adjust to your case)
    seq_len = 2** 7  # e.g. 128 time steps
    # feature columns to use (order matters). 'time' will be converted to numeric seconds.
    # keep a simple, safe default list here. If you want more indicators
    # include them as quoted strings (column names) separated by commas.
    feature_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    # feature_cols = [
    #     'time','open','high',low,close,Buy-Side Liquidation,Sell-Side Liquidation,Negative Delta Up Candle,Positive Delta Down Candle,Plot,Plot,Plot,Plot,Plot,PlotCandle (Open),PlotCandle (High,PlotCandle (Low),PlotCandle (Close),Up Trend,UpTrend Begins,Buy,Down Trend,DownTrend Begins,Sell,,Open line,Close line,High line,Low line,Highest Body line,Lowest Body line,Plot,Plot,Plot,Plot,Plot,Plot,Plot,Conversion Line,Base Line,Lagging Span,Leading Span A,Leading Span B,MHULL,SHULL,KD,J,RSI,ATR,Plot,Plot,Plot,Plot,Shapes,Shapes,Shapes,Shapes,Shapes,Plot,Shapes,Bullish Component,Bearish Component,Signal,MACD,Signal Line,Histogram,Cross,Plot,Plot,Plot,Plot,Plot,Shapes,Shapes,Histogram,MACD,Signal,Secondary QQE Trend Line,Secondary RSI Histogram,QQE Up Signal,QQE Down Signal,MOM,SQZ,STC,Plot,Plot,RSI HA Candles (Open),RSI HA Candles (High,RSI HA Candles (Low),RSI HA Candles (Close),RSI Line,MA

    # ]
    feature_dim = len(feature_cols)
    latent_dim = 32
    d_model = 64
    nhead = 8
    num_layers = 2

    # Try to load real K-line CSV and build windows; if it fails, fallback to random data.
    def parse_time_series_column(s: pd.Series) -> np.ndarray:
        """Convert a time-like pandas Series to numeric seconds since epoch."""
        try:
            dt = pd.to_datetime(s, unit='s', errors='coerce')
            if dt.isna().all():
                # try parsing without unit
                dt = pd.to_datetime(s, errors='coerce')
        except Exception:
            dt = pd.to_datetime(s, errors='coerce')
        # convert to int64 seconds; for NaT we set 0
        ts = dt.astype('int64') // 10 ** 9
        ts = ts.fillna(0).astype(np.int64).to_numpy()
        return ts

    def load_kline_csv(path: str, seq_len: int, feature_cols=None, future_horizon: int = 1, alpha_scale: float = 0.02):
        """
        Load K-line CSV using pandas and produce windows X, future returns and alpha targets.

        - Per-window normalization: for each window, compute mean/std over time for each feature.
        - feature_cols: ordered list of column names to include. 'time' will be converted to numeric seconds.
        """
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']

        if not os.path.exists(path):
            return None, None, None

        try:
            df = pd.read_csv(path)
            # debug: show what columns were read
            print(f"[transformerAutoEncoder] debug: read {path}, columns={list(df.columns)[:50]} (total {len(df.columns)})")
        except Exception as e:
            print(f"[transformerAutoEncoder] debug: pd.read_csv failed for {path}: {e!r}")
            return None, None, None

        # build a mapping from lowercase stripped column name -> actual column name
        col_map = {c.strip().lower(): c for c in df.columns}
        wanted = []
        # Allow missing optional features: collect only those present and warn for missing ones.
        for c in feature_cols:
            key = c.strip().lower()
            if key in col_map:
                wanted.append(col_map[key])
            else:
                print(f"[transformerAutoEncoder] debug: WARNING - requested feature '{c}' (key='{key}') not found in CSV columns; skipping it.")

        if len(wanted) == 0:
            print(f"[transformerAutoEncoder] debug: no requested features found in CSV; aborting load.")
            print(f"[transformerAutoEncoder] debug: requested feature_cols: {feature_cols}")
            print(f"[transformerAutoEncoder] debug: available columns sample: {list(df.columns)[:50]}")
            return None, None, None

        # start with selected columns
        df_sel = df[wanted].copy()

        # If any column contains at least one NaN, drop that entire feature (user requirement)
        na_mask = df_sel.isna().any()
        na_cols = na_mask[na_mask].index.tolist()
        if len(na_cols) > 0:
            # remove them from df_sel and update wanted
            for c in na_cols:
                print(f"[transformerAutoEncoder] dropping feature '{c}' because it contains NaN values")
                if c in df_sel.columns:
                    df_sel.drop(columns=[c], inplace=True)
                if c in wanted:
                    wanted.remove(c)

        # convert time if present (only if time column still present)
        if 'time' in [c.strip().lower() for c in feature_cols]:
            # find the actual col name for time in current col_map
            if 'time' in col_map:
                time_col_name = col_map['time']
                if time_col_name in df_sel.columns:
                    ts = parse_time_series_column(df_sel[time_col_name])
                    df_sel[time_col_name] = ts

        # ensure numeric dtype where possible (coerce remaining non-numeric to NaN then fill 0)
        for c in df_sel.columns:
            if not np.issubdtype(df_sel[c].dtype, np.number):
                df_sel[c] = pd.to_numeric(df_sel[c], errors='coerce')
        # After coercion, if any column now has NaN (unexpected), drop it as well
        na_mask2 = df_sel.isna().any()
        na_cols2 = na_mask2[na_mask2].index.tolist()
        if len(na_cols2) > 0:
            for c in na_cols2:
                print(f"[transformerAutoEncoder] dropping feature '{c}' after coercion because it contains NaN values")
                if c in df_sel.columns:
                    df_sel.drop(columns=[c], inplace=True)
                if c in wanted:
                    wanted.remove(c)
        # finally fill any remaining NA (should be none) with 0.0
        df_sel = df_sel.fillna(0.0)

        N_rows = len(df_sel)
        min_needed = seq_len + future_horizon
        if N_rows < min_needed + 1:
            return None, None, None

        # compute closes for future return; find which column corresponds to 'close'
        close_key = None
        for orig in feature_cols:
            if orig.strip().lower() == 'close':
                close_key = col_map['close'] if 'close' in col_map else None
        if close_key is None:
            return None, None, None

        closes = df_sel[close_key].to_numpy(dtype=np.float64)

        X_windows = []
        future_rets = []
        eps = 1e-9
        for i in range(0, N_rows - seq_len - future_horizon + 1):
            win_df = df_sel.iloc[i : i + seq_len]
            win = win_df.to_numpy(dtype=np.float64)  # (seq_len, F)

            # per-window normalize (z-score) over time axis for each feature
            mean = win.mean(axis=0, keepdims=True)
            std = win.std(axis=0, keepdims=True) + eps
            win_norm = ((win - mean) / std).astype(np.float32)

            close_now = closes[i + seq_len - 1]
            close_future = closes[i + seq_len + future_horizon - 1]
            fut_ret = (close_future / (close_now + 1e-12)) - 1.0

            X_windows.append(win_norm)
            future_rets.append(fut_ret)

        X = np.stack(X_windows, axis=0).astype(np.float32)
        future_rets = np.array(future_rets, dtype=np.float32)
        alpha_target = np.clip(future_rets / alpha_scale, -1.0, 1.0).astype(np.float32)

        return X, future_rets, alpha_target

    # Attempt load
    X, future_ret, alpha_target = load_kline_csv(file_path, seq_len, feature_cols=feature_cols)
    if X is None:
        # Dummy data example: N random samples
        print(f"[transformerAutoEncoder] failed to load '{file_path}', falling back to random data.")
        N = 1000
        X = np.random.randn(N, seq_len, feature_dim).astype(np.float32)
        future_ret = np.random.randn(N).astype(np.float32) * 0.01
        alpha_target = np.clip(future_ret / 0.02, -1.0, 1.0).astype(np.float32)

    # If X loaded, ensure feature_dim matches actual data (some features may have been dropped)
    if X is not None:
        feature_dim_actual = int(X.shape[2])
        if feature_dim_actual != feature_dim:
            print(f"[transformerAutoEncoder] feature_dim changed from {feature_dim} to {feature_dim_actual} after dropping NaN-containing features")
            feature_dim = feature_dim_actual

    def run_splits():
        # We'll run a series of experiments where training fraction increases
        # from 10% to 80% (step 10). For each experiment we reserve a fixed 10%
        # of the dataset as validation (immediately after the training block),
        # and the remainder is used as test. We keep the model that achieves
        # the best validation Sharpe and evaluate that on the test set.
        N = X.shape[0]
        indices = np.arange(N)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core hyperparams for per-split training
        split_epochs = 1000
        batch_size = 64
        lambda_pnl = 0.1

        # annualization (example default: 15-min bars -> 96*365). Keep as before.
        bar_hours = 4.0
        annualize = (24.0 / bar_hours) * 365.0

        print(f"[transformerAutoEncoder] running incremental-train experiments: N={N} samples")
        # train_pct runs: 10,20,...,80
        for train_pct in range(10, 90, 10):
            train_frac = train_pct / 100.0
            n_train = max(1, int(N * train_frac))
            n_val = max(1, int(N * 0.10))  # fixed 10% as validation

            if n_train + n_val >= N:
                print(f"[split {train_pct}%] not enough data for validation+test; skipping")
                continue

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            X_train = X[train_idx]
            X_val = X[val_idx]
            X_test = X[test_idx]
            fut_train = future_ret[train_idx]
            fut_val = future_ret[val_idx]
            fut_test = future_ret[test_idx]
            alpha_train = alpha_target[train_idx]
            alpha_val = alpha_target[val_idx]
            alpha_test = alpha_target[test_idx]

            train_dataset = KlineSeqDataset(X_train, y_alpha=alpha_train, y_ret=fut_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # model capacity and regularization scaling
            if train_frac < 0.3:
                dropout_local = 0.25
            else:
                dropout_local = 0.1

            lambda_alpha_reg = 1e-3
            latent_dim_local = max(8, int(latent_dim * max(0.5, train_frac)))
            d_model_local = max(16, int(d_model * max(0.5, train_frac)))

            best_val_sharpe = float("-inf")
            patience = 100
            no_improve = 0
            best_state = None

            model = TransformerAEAlpha(
                seq_len=seq_len,
                feature_dim=feature_dim,
                d_model=d_model_local,
                nhead=nhead,
                num_layers=num_layers,
                latent_dim=latent_dim_local,
                use_return_head=True,
                dropout=dropout_local,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

            print(f"[split {train_pct}%] train={n_train}, val={n_val}, test={len(test_idx)}, model(d={d_model_local}, z={latent_dim_local})")

            # training loop with early stopping on validation Sharpe
            for epoch in range(1, split_epochs + 1):
                stats = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    lambda_pnl=lambda_pnl,
                    lambda_alpha_reg=lambda_alpha_reg,
                    grad_clip=1.0,
                )

                # evaluate on validation and compute sharpe_bh (use this as selection metric)
                try:
                    val_wealth, val_sharpe, val_alphas = evaluate_strategy(model, X_val, fut_val, annualize=annualize, return_alphas=True)
                    bh_val = np.cumprod(1.0 + fut_val)
                    val_metrics = compute_performance_metrics(val_wealth, bh_val, annualize)
                    val_sharpe_bh = val_metrics.get('sharpe_bh', float('-inf'))
                except Exception as e:
                    print(f"[split {train_pct}%][epoch {epoch}] val evaluation failed: {e}")
                    val_sharpe_bh = float("-inf")

                print(f"[split {train_pct}%][epoch {epoch}] val_sharpe_bh={val_sharpe_bh:.4f} | loss={stats['loss']:.6f}")

                if val_sharpe_bh > best_val_sharpe + 1e-8:
                    best_val_sharpe = val_sharpe_bh
                    no_improve = 0
                    # save best model state
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"[split {train_pct}%] early stopping at epoch {epoch} (no_improve={no_improve}, best_val_sharpe={best_val_sharpe:.4f})")
                    break

            # load best model state (if any)
            if best_state is not None:
                model.load_state_dict(best_state)
            else:
                print(f"[split {train_pct}%] no best model recorded; using last epoch model")

            # final test evaluation using best model, compute metrics and plot
            try:
                test_wealth, test_sharpe, test_alphas = evaluate_strategy(model, X_test, fut_test, annualize=annualize, return_alphas=True)
            except Exception as e:
                print(f"[split {train_pct}%] final test evaluation failed: {e}")
                continue

            # compute buy-and-hold wealth for same test period
            bh_wealth = np.cumprod(1.0 + fut_test)

            # compute metrics using common helper (includes sharp-BH)
            strat_metrics = compute_performance_metrics(test_wealth, bh_wealth, annualize)
            bh_metrics = compute_performance_metrics(bh_wealth, bh_wealth, annualize)

            print(f"[SPLIT {train_pct}%] TEST results: strategy final_net={strat_metrics['final_net']:.6f}, sharpe={strat_metrics['sharpe']:.4f}, sortino={strat_metrics['sortino']:.4f}, sharp-bh={strat_metrics['sharpe_bh']:.4f}, mdd={strat_metrics['mdd']:.4f}")
            print(f"[SPLIT {train_pct}%] B&H     results: final_net={bh_metrics['final_net']:.6f}, sharpe={bh_metrics['sharpe']:.4f}, sortino={bh_metrics['sortino']:.4f}, mdd={bh_metrics['mdd']:.4f}")

            # save plot for this split (annotate metrics)
            savepath = os.path.join(base_dir, f"performance_split_{train_pct}pct.png")
            plot_performance(test_wealth, benchmark_wealth=bh_wealth, positions=test_alphas, title=f"Split {train_pct}% Train", savepath=savepath, metrics_strategy=strat_metrics, metrics_bh=bh_metrics)

    # ---------------- evaluation & plotting ----------------
    def evaluate_strategy(
        model: nn.Module,
        X: np.ndarray,
        future_rets: np.ndarray,
        annualize: float = 96.0 * 365.0,
        batch_size: int = 128,
        fee_rate: float = 0.0005,
        return_alphas: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute strategy returns and Sharpe ratio.

        - model: trained model
        - X: (N, seq_len, F) numpy
        - future_rets: (N,) numpy, next-horizon returns (same as used for alpha target)
        - annualize: annualization factor (default set for 15-minute bars: 96*365).
        - fee_rate: transaction fee rate applied to absolute change in position (e.g. 0.0005 for 0.05%).

        Returns: cumulative_wealth (N,), sharpe (float)
        """
        model.eval()
        model_dev = next(model.parameters()).device

        alphas_list = []
        N = X.shape[0]
        with torch.no_grad():
            # try batching on model device; if CUDA OOM, fallback to CPU
            use_device = model_dev
            try_on_cuda = model_dev.type == 'cuda'
            if not try_on_cuda:
                use_device = torch.device('cpu')

            if use_device.type == 'cuda':
                # run in small batches on GPU
                for i in range(0, N, batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(use_device)
                    outs = model(xb)
                    alphas_list.append(outs['alpha'].cpu().numpy())
            else:
                # run on CPU to avoid GPU OOM
                for i in range(0, N, batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to('cpu')
                    outs = model(xb)
                    alphas_list.append(outs['alpha'].cpu().numpy())

        alphas = np.concatenate([a.reshape(-1) for a in alphas_list], axis=0)

        # transaction cost: fee_rate * abs(change in position)
        prev = np.concatenate(([0.0], alphas[:-1]))
        tx_cost = fee_rate * np.abs(alphas - prev)

        # strategy returns net of transaction costs
        strat_ret = alphas * future_rets - tx_cost

        # cumulative wealth (start at 1)
        wealth = np.cumprod(1.0 + strat_ret)

        # Sharpe: mean/std * sqrt(annualize)
        mean = strat_ret.mean()
        std = strat_ret.std()
        sharpe = float((mean / (std + 1e-12)) * np.sqrt(annualize))
        if return_alphas:
            return wealth, sharpe, alphas
        return wealth, sharpe


    def plot_performance(
        wealth: np.ndarray,
        benchmark_wealth: np.ndarray = None,
        positions: np.ndarray = None,
        title: str = "Strategy Equity Curve",
        savepath: str = None,
        metrics_strategy: Dict[str, float] = None,
        metrics_bh: Dict[str, float] = None,
    ):
        """
        Plot equity curve and optional position size (alphas) under it.
        """
        if HAS_MPL:
            # two-row figure: top wealth (log), bottom positions
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            axes[0].plot(wealth, label='strategy')
            if benchmark_wealth is not None:
                axes[0].plot(benchmark_wealth, label='benchmark')
            axes[0].set_ylabel('Wealth')
            axes[0].set_yscale('log')
            axes[0].set_title(title)
            axes[0].legend()
            axes[0].grid(True)

            if positions is not None:
                axes[1].plot(positions, color='tab:orange', label='position (alpha)')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Alpha')
                axes[1].grid(True)
                axes[1].legend()
            else:
                axes[1].set_visible(False)

            plt.tight_layout()
            # annotate metrics if provided
            if metrics_strategy is not None or metrics_bh is not None:
                txt_lines = []
                if metrics_strategy is not None:
                    ms = metrics_strategy
                    txt_lines.append(f"STRAT: net={ms['final_net']:.4f}, MDD={ms['mdd']:.4f}")
                    txt_lines.append(f"       Sharpe={ms['sharpe']:.3f}, Sortino={ms['sortino']:.3f}, Sharpe-BH={ms.get('sharpe_bh',0.0):.3f}")
                if metrics_bh is not None:
                    mb = metrics_bh
                    txt_lines.append(f"B&H:   net={mb['final_net']:.4f}, MDD={mb['mdd']:.4f}")
                    txt_lines.append(f"       Sharpe={mb['sharpe']:.3f}, Sortino={mb['sortino']:.3f}")
                txt = "\n".join(txt_lines)
                # place text box on upper left
                axes[0].text(0.01, 0.95, txt, transform=axes[0].transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

            if savepath is not None:
                plt.savefig(savepath)
                print(f"[transformerAutoEncoder] saved performance plot to {savepath}")
            else:
                plt.show()
        else:
            # matplotlib not available: save arrays to disk for later plotting
            out = {
                'wealth': wealth,
                'benchmark_wealth': benchmark_wealth,
                'positions': positions,
            }
            out_path = os.path.join(base_dir, 'performance_arrays.npz')
            np.savez(out_path, **out)
            print(f"[transformerAutoEncoder] matplotlib not installed; saved arrays to {out_path}")

    def compute_mdd(wealth: np.ndarray) -> float:
        """
        Compute Maximum Drawdown (MDD) of a wealth curve.

        Returns fractional drawdown (e.g. 0.2 == 20% max drawdown).
        """
        if wealth is None or len(wealth) == 0:
            return 0.0
        peak = np.maximum.accumulate(wealth)
        # avoid division by zero
        dd = (peak - wealth) / (peak + 1e-12)
        mdd = float(np.max(dd))
        return mdd


    def compute_performance_metrics(
        wealth: np.ndarray, bh_wealth: np.ndarray, annualize_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute final_net, MDD, Sharpe, Sortino and Sharp-BH for a strategy and buy-and-hold.

        Returns a dict with keys: final_net, mdd, sharpe, sortino, sharpe_bh
        """
        out = {"final_net": 0.0, "mdd": 0.0, "sharpe": 0.0, "sortino": 0.0, "sharpe_bh": 0.0}
        if wealth is None or len(wealth) < 2:
            return out

        eps = 1e-12
        rets = (wealth[1:] / wealth[:-1]) - 1.0
        bh_rets = (bh_wealth[1:] / bh_wealth[:-1]) - 1.0 if bh_wealth is not None and len(bh_wealth) > 1 else np.zeros_like(rets)

        mu = rets.mean()
        sigma = rets.std()
        sharpe = float((mu / (sigma + eps)) * np.sqrt(annualize_factor))

        # Sortino: downside deviation
        downside = rets[rets < 0.0]
        if downside.size == 0:
            dd = 0.0
        else:
            dd = float(np.sqrt(np.mean(downside ** 2)))
        sortino = float((mu / (dd + eps)) * np.sqrt(annualize_factor))

        mean_bh = float(bh_rets.mean()) if bh_rets.size > 0 else 0.0
        sharpe_bh = float(((mu - mean_bh) / (sigma + eps)) * np.sqrt(annualize_factor))

        mdd_val = compute_mdd(wealth)
        final_net = float(wealth[-1] - 1.0)

        out.update({"final_net": final_net, "mdd": mdd_val, "sharpe": sharpe, "sortino": sortino, "sharpe_bh": sharpe_bh})
        return out


    def evaluate_and_plot_test(
        model: nn.Module,
        X_test: np.ndarray,
        fut_test: np.ndarray,
        annualize: float = 96.0 * 365.0,
        batch_size: int = 128,
        fee_rate: float = 0.0005,
        savepath: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate `model` on test set and plot strategy vs buy-and-hold.

        Prints: final net return, Sharpe, and MDD for both strategy and buy-and-hold.

        Returns: (wealth_strategy, wealth_bh, alphas)
        """
        # get strategy wealth, sharpe and alphas
        wealth, sharpe, alphas = evaluate_strategy(
            model,
            X_test,
            fut_test,
            annualize=annualize,
            batch_size=batch_size,
            fee_rate=fee_rate,
            return_alphas=True,
        )

        # buy-and-hold wealth (baseline)
        bh_wealth = np.cumprod(1.0 + fut_test)

        # compute MDD
        mdd_strat = compute_mdd(wealth)
        mdd_bh = compute_mdd(bh_wealth)

        final_net = float(wealth[-1] - 1.0) if len(wealth) > 0 else 0.0
        final_net_bh = float(bh_wealth[-1] - 1.0) if len(bh_wealth) > 0 else 0.0

        # compute full metrics (including sharp-bh and sortino)
        strat_metrics = compute_performance_metrics(wealth, bh_wealth, annualize)
        bh_metrics = compute_performance_metrics(bh_wealth, bh_wealth, annualize)

        print("[evaluate_and_plot_test] Results:")
        print(f"  Strategy: final_net={strat_metrics['final_net']:.6f}, sharpe={strat_metrics['sharpe']:.4f}, sortino={strat_metrics['sortino']:.4f}, sharp-bh={strat_metrics['sharpe_bh']:.4f}, mdd={strat_metrics['mdd']:.4f}")
        print(f"  Buy&Hold: final_net={bh_metrics['final_net']:.6f}, sharpe={bh_metrics['sharpe']:.4f}, sortino={bh_metrics['sortino']:.4f}, mdd={bh_metrics['mdd']:.4f}")

        # plotting with annotations
        title = "Strategy vs Buy&Hold"
        plot_performance(wealth, benchmark_wealth=bh_wealth, positions=alphas, title=title, savepath=savepath, metrics_strategy=strat_metrics, metrics_bh=bh_metrics)

        return wealth, bh_wealth, alphas

    # Run split experiments (train/test sweeps). This will train fresh models per split
    # and print train/test net return + Sharpe for each split.
    try:
        run_splits()
    except Exception as e:
        print(f"[transformerAutoEncoder] split experiments failed: {e}")

    # After training:
    # For live trading, you would:
    #   1. Take last seq_len bars -> build x (1, seq_len, feature_dim), normalized
    #   2. outputs = model(x)
    #   3. alpha = outputs["alpha"].item()  # position in [-1, 1]
    #   4. Map it to real position size / leverage according to your risk rules.
