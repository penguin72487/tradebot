#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_positions.py

工具：
- 從 `indextransformers.py` 載入模型與資料處理函式
- 在測試集上取得 model 的 position（alpha），允許以 `--leverage` 放大或縮放整體曝險
- 輸出 positions 為 CSV（index, alpha, alpha_scaled, r_b, r_s_log, equity）
- 畫出 equity curve 並儲存圖檔

用法範例：
python analyze_positions.py --model best_model.pth --leverage 2.0 --save_csv test_positions.csv --save_plot test_equity_leverage2.png
"""
import os
import argparse
import numpy as np
import torch

try:
    # import helpers from indextransformers in same folder
    from indextransformers import (
        QuantModel,
        load_ohlcv_csv,
        OhlcvDataset,
        DEVICE,
        SEQ_LEN,
        HORIZON,
        plot_equity,
        compute_metrics,
        base_dir,
        file_path,
    )
except Exception as e:
    print(f"Failed to import from indextransformers: {e}")
    raise


def run(model_path: str, leverage: float = 1.0, max_leverage: float = 3.0, batch_size: int = 256, save_csv: str = None, save_plot: str = None):
    # load data (same splitting as indextransformers)
    X, df = load_ohlcv_csv(file_path)
    if X is None:
        raise RuntimeError(f"Failed to load data from {file_path}")

    dataset = OhlcvDataset(X, SEQ_LEN, HORIZON)
    num = len(dataset)
    train_end = int(num * 0.5)
    val_end = int(num * 0.75)
    test_idx = np.arange(val_end, num)

    # build test arrays
    X_test = []
    r_b_test = []
    for i in test_idx:
        x_seq, r_b = dataset[i]
        X_test.append(x_seq.numpy())
        r_b_test.append(r_b.item())

    X_test = np.stack(X_test, axis=0)
    r_b_test = np.array(r_b_test, dtype=np.float32)

    # load model
    model = QuantModel(in_dim=5).to(DEVICE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = torch.load(model_path, map_location=DEVICE)
    # state might be a state_dict or wrapped dict; try to load appropriately
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        # assume state_dict
        model.load_state_dict(state)
    else:
        # fallback: try to find 'model_state' key
        if 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            raise RuntimeError("Unrecognized checkpoint format; expected state_dict or {'model_state': ...}")

    model.eval()

    # batch inference
    alphas = []
    N = X_test.shape[0]
    bs = batch_size
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.from_numpy(X_test[i:i+bs].astype(np.float32)).to(DEVICE)
            out = model(xb)
            a = out['alpha'].cpu().numpy().reshape(-1)
            alphas.append(a)
    alphas = np.concatenate(alphas, axis=0)

    # apply leverage and clip
    alpha_scaled = np.clip(alphas * leverage, -max_leverage, max_leverage)

    # compute log strategy returns and equity
    # r_b_test is log return per sample (as in indextransformers)
    r_s_log = alpha_scaled * r_b_test
    cum_log = np.cumsum(r_s_log)
    equity = np.exp(cum_log)

    # compute metrics via compute_metrics (expects torch tensors)
    import torch as _torch
    metrics = compute_metrics(_torch.from_numpy(r_s_log), _torch.from_numpy(r_b_test))

    print("=== Test results with leverage=%.3f ===" % leverage)
    print(f"AnnRet (strategy): {metrics['ann_ret_s']:.4%}")
    print(f"AnnRet (B&H):      {metrics['ann_ret_b']:.4%}")
    print(f"Sharpe:            {metrics['sharpe']:.4f}")
    print(f"Sortino:           {metrics['sortino']:.4f}")
    print(f"Alpha_adj:         {metrics['alpha_adj']:.6f}")
    print(f"Beta:              {metrics['beta']:.4f}")
    print(f"Max Drawdown:      {metrics['mdd']:.4%}")

    # save positions CSV
    if save_csv is None:
        save_csv = os.path.join(base_dir, f"test_positions_leverage{leverage:.2f}.csv")
    import csv
    with open(save_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'alpha', 'alpha_scaled', 'r_b_log', 'r_s_log', 'equity'])
        for i in range(N):
            writer.writerow([int(test_idx[i]), float(alphas[i]), float(alpha_scaled[i]), float(r_b_test[i]), float(r_s_log[i]), float(equity[i])])
    print(f"Saved positions to {save_csv}")

    # save plot
    if save_plot is None:
        save_plot = os.path.join(base_dir, f"test_equity_leverage{leverage:.2f}.png")
    plot_equity(equity, metrics['equity_b'], title=f"Test Equity (leverage={leverage:.2f})", savepath=save_plot)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default="C:\\gitproject\\tradebot\\ML\\TAE\\best_model.pth", help='Path to model state_dict (.pth)')
    p.add_argument('--leverage', '-l', type=float, default=10.0, help='Global leverage multiplier applied to alphas')
    p.add_argument('--max-leverage', type=float, default=100, help='Clip scaled alpha to +/- this value')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--save-csv', type=str, default=None)
    p.add_argument('--save-plot', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.model, leverage=args.leverage, max_leverage=args.max_leverage, batch_size=args.batch_size, save_csv=args.save_csv, save_plot=args.save_plot)
