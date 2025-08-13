# -*- coding: utf-8 -*-
"""
逐年走勢 + 只用過去資料 GA 調參 + 當年與往後整段評估（含平行處理 & 進度顯示）
- GA 評分（population fitness）與最終評分：joblib 並行 + tqdm 進度
- 每個訓練窗對未來各年（t~end）的推論：可選擇並行 + tqdm 進度
- 詳細日誌：results_ga_roll_year/run.log，並輸出 training_windows.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# 並行工具
from joblib import Parallel, delayed

# 進度條
from tqdm.auto import tqdm

# === 模型們（先預設用 BayesianRidge；想換可往下加）===
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# 使用相對路徑讀取 CSV
file_path = os.path.join(os.path.dirname(__file__), 'top200_cleaned_noname.csv')
base_dir = os.path.dirname(file_path)
result_dir = os.path.join(base_dir, 'results_ga_roll_year')
os.makedirs(result_dir, exist_ok=True)  # 確保資料夾存在

# ---- 簡單日誌器（同時寫檔/印出）----
import logging
log_path = os.path.join(result_dir, "run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("ga_walk_forward")

df = pd.read_csv(file_path)

# ========= 基礎前處理 =========
df['year_month'] = df['year_month'].astype(str)
df['year'] = df['year_month'].str[:4].astype(int)
df['return'] = df['return'] / 100.0
df['current_return'] = df['return'].shift(1)
df['current_return_label'] = (df['current_return'] > 0).astype(int)

# 數值欄位基底
base_features = df.drop(columns=['year_month','year','return','return_label'], errors='ignore')\
                  .select_dtypes(include=[np.number]).columns.tolist()

# 差分比例變化
chg_blocks = []
for col in base_features:
    prevs = {k: df[col].shift(k).replace(0, np.nan) for k in range(1, 7)}
    chg_df = pd.DataFrame({
        f"{col}_chg{k}": ((df[col] - prevs[k]) / prevs[k]).fillna(0.0)
        for k in range(1, 7)
    })
    chg_blocks.append(chg_df)

df = pd.concat([df] + chg_blocks, axis=1, copy=False)


changed_features = df.drop(columns=['year_month','year','return','return_label'], errors='ignore')\
                     .select_dtypes(include=[np.number]).columns.tolist()

# ========= 轉換 =========
def pca_whitening(X):
    X = np.asarray(X)
    X = SimpleImputer(strategy="median").fit_transform(X)
    return PCA(whiten=True, svd_solver="auto").fit_transform(X)

scalers = {
    'z'  : StandardScaler(),
    'pca': FunctionTransformer(pca_whitening, validate=False),
}

df = df.sort_values('year').reset_index(drop=True)
years_all = sorted(df['year'].unique())

# 逐年滾動轉換
for name, scaler in scalers.items():
    transformed_parts = []
    for y in years_all:
        fit_data = df[df['year'] <= y][changed_features]
        cur_data = df[df['year'] == y][changed_features]
        try:
            sc = clone(scaler)
            sc.fit(fit_data)
            cur_tr = sc.transform(cur_data)
            cur_tr = pd.DataFrame(cur_tr, index=cur_data.index,
                                  columns=[f'{c}_{name}' for c in changed_features])
            transformed_parts.append(cur_tr)
        except Exception as e:
            log.warning(f'[{name}] year={y} 轉換失敗: {e}')
    if transformed_parts:
        transformed_all = pd.concat(transformed_parts).sort_index()
        df = pd.concat([df, transformed_all], axis=1)

# 可用特徵
exclude_cols = ['year_month','year','return','return_label','current_return','current_return_label']
all_features = df.drop(columns=[c for c in exclude_cols if c in df.columns])\
                 .select_dtypes(include=[np.number]).columns.tolist()

# 去掉首尾年
years = years_all[1:-1]
df = df[df['year'].isin(years)].copy()
years = sorted(df['year'].unique())

log.info('每年樣本數（股票數）：\n' + str(df.groupby('year')['stock_id'].nunique()))

# ========= 策略計算 =========
TOP_NS = [10, 20, 30, 200]
KINDs  = ['long','short','long_short']

def year_strategy_returns(test_df, pred_col='predicted_return'):
    res = {n: {} for n in TOP_NS}
    for n in TOP_NS:
        n_eff = min(n, len(test_df))
        if n_eff <= 0:
            for kind in KINDs:
                res[n][kind] = np.nan
            continue
        top_n    = test_df.nlargest(n_eff, pred_col)
        bottom_n = test_df.nsmallest(n_eff, pred_col)

        long_ret  = top_n['true_return'].mean()
        short_ret = - bottom_n['true_return'].mean()
        long_short = (long_ret + short_ret) / 2.0

        res[n]['long'] = float(long_ret)
        res[n]['short'] = float(short_ret)
        res[n]['long_short'] = float(long_short)
    return res

def fit_predict_one_year(model, train_df, test_df, feat_cols):
    Xtr, ytr = train_df[feat_cols], train_df['return']
    Xte, yte = test_df[feat_cols], test_df['return']

    m = clone(model)
    try:
        m.fit(Xtr, ytr)
        preds = m.predict(Xte)
    except Exception:
        return None, None
    out = test_df[['year']].copy()
    out['true_return'] = yte.values
    out['predicted_return'] = preds
    return m, out

def cumulative_score(series_list):
    vals = [x for x in series_list if pd.notna(x)]
    if not vals:
        return 0.0
    cum = np.cumprod([1.0 + v for v in vals])[-1]
    return float(cum)

# ========= 可切換模型與參數空間 =========
MODELS = {
    'BayesianRidge': BayesianRidge
}

PARAM_SPACES = {
    'BayesianRidge': {
        'max_iter'     : (100, 500),
        'tol'          : (1e-6, 1e-2),
        'alpha_1'      : (1e-7, 1e1),
        'alpha_2'      : (1e-7, 1e1),
        'lambda_1'     : (1e-7, 1e1),
        'lambda_2'     : (1e-7, 1e1),
        'fit_intercept': [True, False],
        'compute_score': [True, False],
        'copy_X'       : [True, False],
    },
}

def init_model_by_name(name, params):
    if name == 'BayesianRidge':
        return BayesianRidge(**params)
    elif name == 'Ridge':
        return Ridge(**params)
    elif name == 'SVR':
        return SVR(**params)
    elif name == 'RandomForest':
        return RandomForestRegressor(**params)
    elif name == 'ExtraTrees':
        return ExtraTreesRegressor(**params)
    elif name == 'XGB':
        return xgb.XGBRegressor(**params)
    elif name == 'CatBoost':
        return cb.CatBoostRegressor(verbose=0, **params)
    elif name == 'Linear':
        return LinearRegression(**params)
    else:
        raise ValueError(f'未知模型: {name}')

# ========= GA（含平行 & 進度）=========
rng = np.random.default_rng(42)

def decode_params(space, gene):  # gene: [0,1]^D
    decoded = {}
    i = 0
    for k, v in space.items():
        if isinstance(v, tuple):
            lo, hi = v
            g = float(gene[i])
            decoded[k] = lo + g * (hi - lo)
            if isinstance(lo, int) and isinstance(hi, int):
                decoded[k] = int(round(decoded[k]))
            i += 1
        elif isinstance(v, list):
            g = float(gene[i])
            idx = int(np.floor(g * len(v))) % len(v)
            decoded[k] = v[idx]
            i += 1
        else:
            raise ValueError('param space 僅支援 tuple/list')
    return decoded

def build_gene_dim(space):
    return sum(1 for _ in space.items())

# --- checkpoint utils ---
import pickle

def save_checkpoint(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# === 新增：GA 內部可視化（畫 inner_years 的累積曲線）===
def _plot_ga_inner_curves(traj, inner_years, out_dir, tag, viz_kinds=('long','short','long_short')):
    """
    traj: {n:{kind:[r_t,...]}} 來自 evaluate_gene_traj
    會各別輸出 Top10/20/30/200 的圖片；每張圖含 long/short/long_short 三條線
    """
    os.makedirs(out_dir, exist_ok=True)
    years_axis = inner_years[1:]  # 第一個年是用來訓練，所以序列從第二個年開始
    for n in sorted(traj.keys()):
        plt.figure(figsize=(12, 7))
        best_end = -np.inf
        best_label = ""
        for kind in viz_kinds:
            seq = [x for x in traj[n][kind] if pd.notna(x)]
            cum = np.cumprod([1.0 + (v if pd.notna(v) else 0.0) for v in seq]) if seq else np.array([])
            label = f'{kind.capitalize()} Top{n}'
            if len(cum) > 0:
                plt.plot(years_axis[:len(cum)], cum, marker='o', linestyle='-' if kind=='long' else ('--' if kind=='short' else ':'), label=label)
                if cum[-1] > best_end:
                    best_end = float(cum[-1]); best_label = label

        plt.title(f'GA {tag} – Top{n} (Best: {best_label} {best_end:.2f}×)' if best_end>0 else f'GA {tag} – Top{n}')
        plt.xlabel('Year'); plt.ylabel('Cumulative (×)')
        plt.yscale('log'); plt.grid(True); plt.legend()
        fname = os.path.join(out_dir, f'GA_{tag}_Top{n}.png')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()


def _plot_window_all_in_one(
    train_span, test_year, inner_years,
    inner_traj,                     # dict: {TopN:{kind:[r_t,...]}}
    future_years,                   # list: [t, t+1, ...]
    future_returns_seq,             # dict: {TopN:{kind:[r_t, r_{t+1}, ...]}}
    out_dir,                        # 輸出目錄
    fname_prefix="window",
    kinds=('long','short','long_short'),
    top_ns=(10,20,30,200)
):
    """
    產出 1 張 3x3 圖：
      每行=kind；每列分別是 [Inner 累積, Forward 累積, Forward 年化(起點 t 與 t+1)]
    - inner_traj: 來自 evaluate_gene_traj 的回傳
    - future_returns_seq: 你在主流程組好的 future_returns（把 NaN 換 0 再累乘）
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), constrained_layout=True)
    # 把 kinds 固定順序對應到 row
    kind_rows = {k:i for i,k in enumerate(kinds)}
    col_titles = ["Inner cumulative (×)", f"Forward cumulative from {test_year} (×)",
                  f"Annualized: {test_year}→end & {test_year+1}→end"]

    # 1) Inner cumulative
    x_inner = inner_years[1:]  # inner fitness 從第二個年開始
    for k in kinds:
        r = kind_rows[k]
        ax = axes[r, 0]
        for n in top_ns:
            seq = inner_traj.get(n, {}).get(k, [])
            if len(seq) == 0:
                continue
            cum = np.cumprod([1.0 + (v if pd.notna(v) else 0.0) for v in seq])
            ax.plot(x_inner[:len(cum)], cum, marker='o', label=f"Top{n}")
        ax.set_title(f"{k} – {col_titles[0]}")
        ax.set_yscale('log')
        ax.grid(True)
        if r == len(kinds)-1:
            ax.set_xlabel("Year")
        ax.set_ylabel("×")
        ax.legend()

    # 2) Forward cumulative (固定模型 from test_year)
    x_fwd = future_years
    for k in kinds:
        r = kind_rows[k]
        ax = axes[r, 1]
        for n in top_ns:
            seq = future_returns_seq.get(n, {}).get(k, [])
            if len(seq) == 0:
                continue
            cum = np.cumprod([1.0 + (v if pd.notna(v) else 0.0) for v in seq])
            ax.plot(x_fwd[:len(cum)], cum, marker='o', label=f"Top{n}")
        ax.set_title(f"{k} – {col_titles[1]}")
        ax.set_yscale('log')
        ax.grid(True)
        if r == len(kinds)-1:
            ax.set_xlabel("Year")
        ax.set_ylabel("×")
        ax.legend()

    # 3) Annualized curves（僅起點 t）
    for k in kinds:
        r = kind_rows[k]
        ax = axes[r, 2]
        for n in top_ns:
            seq = future_returns_seq.get(n, {}).get(k, [])
            seq = [0.0 if not pd.notna(v) else v for v in seq]
            if len(seq) == 0:
                continue
            # 起點 t
            cum = np.cumprod([1.0 + v for v in seq])
            years_span = np.arange(1, len(seq)+1)
            ann_t = np.power(cum, 1.0/years_span) - 1.0
            ax.plot(x_fwd[:len(ann_t)], ann_t, label=f"Top{n} (start {test_year})")

        ax.set_title(f"{k} – Annualized: {test_year}→end")
        ax.grid(True)
        if r == len(kinds)-1:
            ax.set_xlabel("Year")
        ax.set_ylabel("annualized")
        ax.legend()

    fig.suptitle(f"Train {train_span}  |  Test from {test_year}", fontsize=14)
    out_path = os.path.join(out_dir, f"{fname_prefix}_{train_span}_T{test_year}.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path




def ga_optimize_params(
    model_name,
    train_df,
    feat_cols,
    inner_years,
    population=128,
    generations=200,
    cx_prob=0.85,
    init_mut_prob=0.10,
    improve_delta=1e-4,
    early_stop_patience=10,
    enable_feature_selection=True,
    init_density=0.05,
    n_jobs=-1,
    backend='loky',
    checkpoint_dir=None,
    resume=True,
    checkpoint_every=5,
    progress_desc="GA",
    log=print,
    # === 新增三個 ===
    viz_on_improve=True,
    viz_every=1,
    viz_dir=None,                        # 若 None 則用 checkpoint_dir 或 result_dir 下的 "ga_viz"
    viz_kinds=('long','short','long_short'),
     # === 新增：每代探查下一年 OOS（日誌用，不參與選擇）===
    next_year_df=None,           # 外層傳進來的 test_df（下一年）
    oos_probe=True,              # 是否啟用每代 OOS 探查
    oos_every=1,                 # 每幾代做一次（1=每代，省時間可設 5/10）
    oos_metric=('10', 'long_short'),  # 要打印哪個指標（TopN, kind）
):

    """
    以 inner_years 做 expanding walk-forward 當 fitness。
    若 enable_feature_selection=True:
        基因 = [ feature_bits (len=F), param_genes (len=P) ]
      其中 feature_bits ∈ {0,1}，param_genes ∈ [0,1]
    否則（只調參數）:
        基因 = [ param_genes (len=P) ]
    動態突變率：
        依 no_improvement_count 調整，參考你給的規則，並區分
        - 特徵位元：用 bit-flip 概率
        - 參數位元：加高斯噪音並 clip 到 [0,1]
    """
    space = PARAM_SPACES[model_name]

    # ---- 基因長度 ----
    param_dim = sum(1 for _ in space.items())
    F = len(feat_cols) if enable_feature_selection else 0
    P = param_dim
    D = F + P

    # ---- 基因初始化 ----
    rng_local = np.random.default_rng(42)  # 固定這層的隨機性，便於重現
    def init_population(n):
        if enable_feature_selection:
            # 特徵位元：依密度產生 0/1
            feature_bits = (rng_local.random((n, F)) < init_density).astype(np.float64)
            # 參數位元：[0,1] 均勻
            param_genes = rng_local.random((n, P))
            return np.concatenate([feature_bits, param_genes], axis=1)
        else:
            return rng_local.random((n, P))

    # ---- 基因解碼 ----
    def decode_params(space, gene_param_part):
        decoded = {}
        i = 0
        for k, v in space.items():
            g = float(gene_param_part[i])
            if isinstance(v, tuple):
                lo, hi = v
                val = lo + g * (hi - lo)
                if isinstance(lo, int) and isinstance(hi, int):
                    val = int(round(val))
                decoded[k] = val
            elif isinstance(v, list):
                idx = int(np.floor(g * len(v))) % len(v)
                decoded[k] = v[idx]
            else:
                raise ValueError('param space 僅支援 tuple/list')
            i += 1
        return decoded

    # ---- fitness ----
    def fitness_one(gene):
        # 解析特徵
        if enable_feature_selection:
            bits = gene[:F] >= 0.5
            sel_cols = [c for c, b in zip(feat_cols, bits) if b]
            # 至少保底 1 個特徵，否則直接 0 分
            if len(sel_cols) == 0:
                return 0.0, None, None
        else:
            sel_cols = feat_cols

        # 解析參數
        params = decode_params(space, gene[-P:])
        model  = init_model_by_name(model_name, params)
        traj = {n:{k:[] for k in KINDs} for n in TOP_NS}

        # expanding walk-forward
        for i in range(1, len(inner_years)):
            tr_years = inner_years[:i]
            te_year  = inner_years[i]
            tr_df = train_df[train_df['year'].isin(tr_years)]
            te_df = train_df[train_df['year'] == te_year]
            if te_df.empty or tr_df.empty:
                continue

            m, pred = fit_predict_one_year(model, tr_df, te_df, sel_cols)
            if m is None:
                continue
            one = year_strategy_returns(pred, 'predicted_return')
            for n in TOP_NS:
                for kind in KINDs:
                    traj[n][kind].append(one[n][kind])

        # 分數 = 最佳 (n, kind) 的累乘終值
        best = 0.0
        for n in TOP_NS:
            for kind in KINDs:
                seq = [x for x in traj[n][kind] if pd.notna(x)]
                if seq:
                    score = np.prod([1.0 + v for v in seq])
                    if score > best:
                        best = score
        return best, sel_cols, params
    
        # --- 針對某個基因，回算 inner_years 的完整軌跡（用於出圖） ---
    def evaluate_gene_traj(gene):
        # 解析特徵
        if enable_feature_selection:
            bits = gene[:F] >= 0.5
            sel_cols = [c for c, b in zip(feat_cols, bits) if b]
            if len(sel_cols) == 0:
                return {n:{k:[] for k in KINDs} for n in TOP_NS}
        else:
            sel_cols = feat_cols

        params = decode_params(space, gene[-P:])
        model  = init_model_by_name(model_name, params)
        traj = {n:{k:[] for k in KINDs} for n in TOP_NS}
        for i in range(1, len(inner_years)):
            tr_years = inner_years[:i]
            te_year  = inner_years[i]
            tr_df = train_df[train_df['year'].isin(tr_years)]
            te_df = train_df[train_df['year'] == te_year]
            if te_df.empty or tr_df.empty:
                for n in TOP_NS:
                    for k in KINDs:
                        traj[n][k].append(np.nan)
                continue
            m, pred = fit_predict_one_year(model, tr_df, te_df, sel_cols)
            if m is None:
                for n in TOP_NS:
                    for k in KINDs:
                        traj[n][k].append(np.nan)
                continue
            one = year_strategy_returns(pred, 'predicted_return')
            for n in TOP_NS:
                for k in KINDs:
                    traj[n][k].append(one[n][k])
        return traj


    # ---- checkpoint 恢復 ----
    start_gen = 0
    best_score = -np.inf
    best_cols = None
    best_params = None
    no_improve = 0

    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pkl")
    else:
        ckpt_path = None

    if ckpt_path and resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path)
        pop = ckpt["population"]
        start_gen = ckpt["generation"]
        best_score = ckpt["best_score"]
        best_cols = ckpt["best_cols"]
        best_params = ckpt["best_params"]
        no_improve = ckpt.get("no_improve", 0)
        log(f"🔄 恢復 GA checkpoint: gen={start_gen}, best={best_score:.4f}")
    else:
        pop = init_population(population)

    # ---- 主要循環 ----
    pbar = tqdm(range(start_gen, generations), desc=progress_desc, leave=False, ncols=100)
    for gen in pbar:
        # 並行計分
        evals = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
            delayed(fitness_one)(ind) for ind in pop
        )
        scores = np.array([e[0] for e in evals], dtype=float)

        # 這代最優
        gen_best_idx = int(np.nanargmax(scores)) if len(scores) else 0
        gen_best = float(scores[gen_best_idx]) if len(scores) else 0.0
        pbar.set_postfix(best=f"{gen_best:.4f}", no_improve=no_improve)

        # 檢查是否進步
        if (gen_best - best_score) <= improve_delta:
            no_improve += 1
        else:
            no_improve = 0
            best_score = gen_best
            best_cols = evals[gen_best_idx][1]
            best_params = evals[gen_best_idx][2]

        # 早停
        if no_improve >= early_stop_patience:
            log(f"⏹️ 早停於第 {gen+1} 代（連續 {no_improve} 代無提升）")
            # 保存停下來時的 checkpoint
            if ckpt_path:
                save_checkpoint(ckpt_path, {
                    "population": pop,
                    "generation": gen + 1,
                    "best_score": best_score,
                    "best_cols": best_cols,
                    "best_params": best_params,
                    "no_improve": no_improve
                })
            break

        # --- 選擇（輪盤）---
        s = scores.copy()
        s[~np.isfinite(s)] = 0.0
        total = s.sum()
        if total <= 0:
            probs = np.ones_like(s) / len(s)
        else:
            probs = s / total
        idx = rng.choice(len(pop), size=len(pop), replace=True, p=probs)
        mates = pop[idx]

        # --- 交配 ---
        next_pop = np.empty_like(pop)
        for i in range(0, len(mates), 2):
            p1 = mates[i]
            p2 = mates[(i+1) % len(mates)]
            if rng.random() < cx_prob:
                cp = rng.integers(1, D)
                c1 = np.concatenate([p1[:cp], p2[cp:]])
                c2 = np.concatenate([p2[:cp], p1[cp:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            next_pop[i] = c1
            next_pop[(i+1) % len(mates)] = c2

        # --- 動態突變率（依 no_improve）---
        # 參考你提供的規則，延伸到 param/feature 兩段
        if no_improve >= 100:
            mut_intensity = 1 + no_improve
        elif no_improve >= 20:
            mut_intensity = 1 + no_improve / 20.0
        elif no_improve >= 5:
            mut_intensity = init_mut_prob + ((0.8 - init_mut_prob) * (1 - no_improve / 20.0))
        else:
            mut_intensity = init_mut_prob

        # 限制到合理區間（避免爆掉）
        mut_intensity = float(np.clip(mut_intensity, 0.0, 1.0))

        # 為了讓「整體變異比例」與維度無關，採 per-gene 機率 = mut_intensity / D
        per_gene_p = mut_intensity / max(D, 1)

        # --- 突變：特徵位元 flip；參數位元加噪音 ---
        for i in range(len(next_pop)):
            if enable_feature_selection and F > 0:
                # feature bits: Bernoulli flip
                flip_mask = rng.random(F) < per_gene_p
                next_pop[i, :F][flip_mask] = 1.0 - next_pop[i, :F][flip_mask]
            # param genes: add gaussian noise then clip
            noise_mask = rng.random(P) < per_gene_p
            noise = rng.normal(0.0, 0.10, size=P)   # 可視需要把 0.10 做成參數
            vec = next_pop[i, -P:]
            vec[noise_mask] = np.clip(vec[noise_mask] + noise[noise_mask], 0.0, 1.0)
            next_pop[i, -P:] = vec

        pop = next_pop

        # --- checkpoint（每隔 checkpoint_every 代 & 最後一代）---
        if ckpt_path and ((gen + 1) % checkpoint_every == 0):
            save_checkpoint(ckpt_path, {
                "population": pop,
                "generation": gen + 1,
                "best_score": best_score,
                "best_cols": best_cols,
                "best_params": best_params,
                "no_improve": no_improve
            })
        
        # ---（新增）探查下一年 OOS：只做日誌，不影響 GA 決策 ---
        
        oos_msg = ""
        if oos_probe and (next_year_df is not None) and ((gen + 1) % max(1, oos_every) == 0):
            try:
                # 以該代最佳個體的特徵選擇
                if enable_feature_selection and F > 0:
                    bits = pop[gen_best_idx][:F] >= 0.5
                    sel_cols_probe = [c for c, b in zip(feat_cols, bits) if b]
                    # 保底，避免 0 特徵
                    if len(sel_cols_probe) == 0:
                        sel_cols_probe = feat_cols
                else:
                    sel_cols_probe = feat_cols

                # 以當前最佳參數重建模型，使用完整 inner_years 的 train_df 來 fit
                params_probe = decode_params(space, pop[gen_best_idx][-P:])
                model_probe  = init_model_by_name(model_name, params_probe)
                m_probe, pred_next = fit_predict_one_year(model_probe, train_df, next_year_df, sel_cols_probe)

                if (m_probe is not None) and (pred_next is not None):
                    r_next = year_strategy_returns(pred_next)
                    tn, kd = oos_metric  # e.g. ('10','long_short')
                    oos_val = r_next[int(tn)][kd]
                    oos_msg = f" | nextY_OOS(Top{tn}/{kd})={oos_val:+.4f}"
                else:
                    oos_msg = " | nextY_OOS=NaN"
            except Exception:
                oos_msg = " | nextY_OOS=ERR"


        # 顯示細節
        if enable_feature_selection:
            # 取出目前最佳個體的特徵位元
            bits = pop[gen_best_idx][:F] >= 0.5
            sel_cols_current = [c for c, b in zip(feat_cols, bits) if b]
            sel_count = len(sel_cols_current)

            preview = ", ".join(sel_cols_current)

            log(
                f"Gen {gen+1:04d} | best={best_score:.4f} | no_imp={no_improve:02d} | "
                f"{oos_msg or ' | nextY_OOS=N/A'}"
                f"mut={mut_intensity:.4f} | sel_features={sel_count} "
                f"[{preview}] | best_params={best_params}"
            )
        else:
            log(
                f"Gen {gen+1:04d} | best={best_score:.4f} | no_imp={no_improve:02d} | "
                f"mut={mut_intensity:.4f} | sel_features={sel_count} "
            )
        # （可選）把每代最佳特徵名單存檔，便於追蹤
        try:
            if checkpoint_dir:
                path = os.path.join(checkpoint_dir, f"sel_features_gen{gen+1:04d}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(sel_cols_current))
        except Exception:
            pass

    pbar.close()

    # 收尾：確保最後 checkpoint
    if ckpt_path:
        save_checkpoint(ckpt_path, {
            "population": pop,
            "generation": min(generations, (gen if 'gen' in locals() else 0) + 1),
            "best_score": best_score,
            "best_cols": best_cols,
            "best_params": best_params,
            "no_improve": no_improve
        })

    # === 每代固定出圖（依 viz_every 控制頻率）===
    if (gen + 1) % max(1, viz_every) == 0:
        tag = f"{progress_desc.replace(' ', '_')}_gen{gen+1:04d}"
        out_dir = (viz_dir or checkpoint_dir or os.path.join(os.path.dirname(__file__), 'results_ga_roll_year', 'ga_viz'))
        try:
            # 以該代最佳個體繪圖
            traj = evaluate_gene_traj(pop[gen_best_idx])
            _plot_ga_inner_curves(traj, inner_years, out_dir, tag, viz_kinds=viz_kinds)
            log(f"📈 [viz] 已輸出：{out_dir}/GA_{tag}_Top(10|20|30|200).png")
        except Exception as e:
            log(f"⚠️ [viz] 生成失敗（gen {gen+1}）：{e}")

    # 在外層，拿到 best_cols 後（某一個訓練窗的最終結果）
    if best_cols is not None and ckpt_dir:
        out_feats = os.path.join(ckpt_dir, "best_selected_features.txt")
        with open(out_feats, "w", encoding="utf-8") as f:
            f.write("\n".join(best_cols))

    # 回傳最佳
    if enable_feature_selection:
        return best_params, best_score, (best_cols if best_cols is not None else feat_cols)
    else:
        return best_params, best_score, None


# ========= 主流程：逐年 =========
MODEL_NAME = 'BayesianRidge'
FEATURES   = all_features

# 並行選項
GA_N_JOBS = -1          # GA fitness 並行核數
FWD_N_JOBS = -1         # 針對未來各年的推論是否並行（-1=全部；設 1 則關閉）
FWD_BACKEND = 'loky'    # 多進程

# 1) 逐年 OOS
oos_traj = {n:{k:[] for k in KINDs} for n in TOP_NS}
oos_years = []

# 2) 訓練窗→整段年化
forward_summary_rows = []

min_years_needed = 3

# 用於彙總每個訓練窗的重點，最後另存 CSV
training_rows = []

outer_pbar = tqdm(range(1, len(years)), desc="Walk-forward years", ncols=100)
for i in outer_pbar:
    train_years = years[:i]      # <= t-1
    test_year   = years[i]       # t
    if len(train_years) < min_years_needed:
        continue

    train_df = df[df['year'].isin(train_years)]
    test_df  = df[df['year'] == test_year]

    # === GA 超參數 ===
    # === GA 超參數（有 checkpoint/早停/初始密度/動態突變率/特徵選擇）===
    train_start, train_end = train_years[0], train_years[-1]
    ckpt_dir = os.path.join(result_dir, "ckpt", f"{train_start}-{train_end}")
    next_year_df = df[df['year'] == years[i+1]] if (i + 1) < len(years) else None

    desc = f"GA {train_start}-{train_end}"
    best_params, best_fit, best_cols = ga_optimize_params(
        model_name=MODEL_NAME,
        train_df=train_df,
        feat_cols=FEATURES,            # 若要同時做特徵選擇，可在下方把 enable_feature_selection=True
        inner_years=train_years,
        # --- GA 規模/代數 ---
        population=1024,
        generations=1000,
        # --- 交配/初始突變率 ---
        cx_prob=0.85,
        init_mut_prob=0.12,            # 你原本的 mut_prob 改名為 init_mut_prob
        # --- 早停 ---
        improve_delta=1e-4,
        early_stop_patience=50,  # 連續 20 代無提升就早停
        # --- 特徵選擇（想開就 True）---
        enable_feature_selection=True,  # 是否同時做特徵選擇
        init_density=0.05,             # 特徵 bit 初始 1 的密度
        # --- 並行 ---
        n_jobs=GA_N_JOBS,
        backend='loky',
        # --- checkpoint ---
        checkpoint_dir=ckpt_dir,
        resume=True,                   # 有 ckpt 就續跑
        checkpoint_every=5,            # 每幾代存一次
        # --- 顯示 ---
        progress_desc=desc,
        log=log.info,
        next_year_df=next_year_df,
        oos_probe=True,
        oos_every=1,   
    )
    log.info(
        f'[GA] Train {train_start}-{train_end} -> best_score={best_fit:.4f}, params={best_params}'
        + (f', selected_features={len(best_cols)}' if best_cols is not None else '')
    )

    log.info(f'[GA] Train {train_years[0]}-{train_years[-1]} -> best_score={best_fit:.4f}, params={best_params}')

    # === 用最佳參數訓練固定模型 ===
    base_model = init_model_by_name(MODEL_NAME, best_params)
    cols_for_this_window = (best_cols if best_cols is not None else FEATURES)
    m, pred_t = fit_predict_one_year(base_model, train_df, test_df, cols_for_this_window)

    if (m is None) or (pred_t is None):
        log.warning(f"[SKIP] {test_year} 由於模型訓練/預測失敗")
        continue

    # (A) OOS：當年 t
    one_t = year_strategy_returns(pred_t)
    for n in TOP_NS:
        for kind in KINDs:
            oos_traj[n][kind].append(one_t[n][kind])
    oos_years.append(test_year)

    # 目前 OOS 累積（以 Top10/long_short 做主指標）
    oos_seq = [x if pd.notna(x) else 0.0 for x in oos_traj[10]['long_short']]
    oos_cum = float(np.prod([1.0 + v for v in oos_seq]))
    outer_pbar.set_postfix(year=test_year, oos_top10_ls=f"{oos_cum:.2f}x")

    # (B) Forward：t ~ end
    future_years = years[i:]  # 從 t 到最後

    def infer_one_future_year(fy):
        fy_df = df[df['year'] == fy]
        Xte, yte = fy_df[cols_for_this_window], fy_df['return']
        preds = m.predict(Xte)  # 固定模型，不重訓
        te = fy_df[['year']].copy()
        te['true_return'] = yte.values
        te['predicted_return'] = preds
        return year_strategy_returns(te)

    # 針對未來各年推論的進度列
    fut_iter = future_years if FWD_N_JOBS == 1 else tqdm(future_years, desc=f"Forward {test_year}→end", leave=False, ncols=100)

    if FWD_N_JOBS == 1:
        future_returns_list = [infer_one_future_year(fy) for fy in fut_iter]
    else:
        # 先為了顯示進度，把年份展開，並行 + 手動更新進度
        def _wrap(fy):
            res = infer_one_future_year(fy)
            return (fy, res)

        # 用 Parallel 並行後，再按原順序聚合
        future_returns_list = Parallel(n_jobs=FWD_N_JOBS, backend=FWD_BACKEND, prefer="processes")(
            delayed(infer_one_future_year)(fy) for fy in future_years
        )
        if isinstance(fut_iter, tqdm):
            fut_iter.close()

    # 聚合
    future_returns = {n:{k:[] for k in KINDs} for n in TOP_NS}
    for yr_res in future_returns_list:
        for n in TOP_NS:
            for kind in KINDs:
                future_returns[n][kind].append(yr_res[n][kind])

    # 算整段累積 & 年化
    span = len(future_years)
    row = {
        'TrainYears' : f'{train_years[0]}-{train_years[-1]}',
        'TestSpan'   : f'{future_years[0]}-{future_years[-1]}',
        'Params'     : repr(best_params)
    }
    for n in TOP_NS:
        for kind in KINDs:
            seq = [x for x in future_returns[n][kind] if pd.notna(x)]
            if len(seq) == 0:
                row[f'Top{n}_{kind}_Cumulative'] = np.nan
                row[f'Top{n}_{kind}_Annual'] = np.nan
            else:
                cum = np.prod([1.0 + v for v in seq]) - 1.0
                ann = ( (1.0 + cum)**(1.0 / span) - 1.0 ) if (1.0 + cum) > 0 else np.nan
                row[f'Top{n}_{kind}_Cumulative'] = round(float(cum), 6)
                row[f'Top{n}_{kind}_Annual']     = round(float(ann), 6)
    forward_summary_rows.append(row)

    # ---- 當輪摘要（關鍵數字）----
    key_yr_ret = one_t[10]['long_short']  # 當年 Top10 long_short
    key_forward_ann = row.get('Top10_long_short_Annual', np.nan)
    log.info(
        f"[{train_years[0]}-{train_years[-1]} -> {test_year}] "
        f"OOS_t(Top10/LS)={key_yr_ret:+.4f} | OOS_cum(Top10/LS)={oos_cum:.2f}x | "
        f"Forward_ann(Top10/LS)={key_forward_ann if pd.notna(key_forward_ann) else 'NaN'}"
    )

        # === 建一張「訓練窗總覽圖」：Inner + Forward 累積 + 年化（Top10/20/30/200 疊在一起） ===
    # 1) 取得 inner_traj（用當代 GA 最佳個體對應的「最終最佳參數/特徵」重新回算 inner_years）
    #    若你在 ga_optimize_params 內部已經有 evaluate_gene_traj(gene) 可呼叫，
    #    這裡直接「用最佳模型重算」一遍就好（等價）：
    inner_traj = {n:{k:[] for k in KINDs} for n in TOP_NS}
    # 用最佳參數 + 此窗的特徵 cols_for_this_window，重算 inner_years 的 expanding fitness
    _model_viz = init_model_by_name(MODEL_NAME, best_params)
    for j in range(1, len(train_years)):
        tr_yrs = train_years[:j]
        te_yr  = train_years[j]
        tr_df2 = df[df['year'].isin(tr_yrs)]
        te_df2 = df[df['year'] == te_yr]
        _m, _pred = fit_predict_one_year(_model_viz, tr_df2, te_df2, cols_for_this_window)
        if (_m is None) or (_pred is None):
            for n in TOP_NS:
                for k in KINDs:
                    inner_traj[n][k].append(np.nan)
            continue
        _yr_res = year_strategy_returns(_pred)
        for n in TOP_NS:
            for k in KINDs:
                inner_traj[n][k].append(_yr_res[n][k])

    # 2) forward_returns 已是 {TopN:{kind:[r_t, r_{t+1}, ...]}}
    #    直接送去畫綜合圖
    win_viz_dir = os.path.join(result_dir, "window_viz")
    plot_path = _plot_window_all_in_one(
        train_span=f"{train_years[0]}-{train_years[-1]}",
        test_year=test_year,
        inner_years=train_years,
        inner_traj=inner_traj,
        future_years=future_years,
        future_returns_seq=future_returns,
        out_dir=win_viz_dir,
        fname_prefix="WF",
        kinds=tuple(KINDs),
        top_ns=tuple(TOP_NS),
    )
    log.info(f"🖼️ Window figure saved: {plot_path}")


    training_rows.append({
        'TrainStart': train_years[0],
        'TrainEnd'  : train_years[-1],
        'TestYear'  : test_year,
        'OOS_Top10_LongShort' : round(float(key_yr_ret), 6) if pd.notna(key_yr_ret) else np.nan,
        'OOS_Cum_Top10_LongShort' : round(float(oos_cum), 6),
        'Forward_Annual_Top10_LongShort' : key_forward_ann,
        'BestParams' : repr(best_params),
        'Selected_Feature_Count': (len(best_cols) if best_cols is not None else np.nan),
        'GA_Fitness' : round(float(best_fit), 6)
    })

outer_pbar.close()

# ========= 輸出 =========
# (1) OOS 串接曲線
plt.figure(figsize=(14,10))
markers = {"long": 'o', "short": 'v', "long_short": 's'}
ls = {"long": '-', "short": '--', "long_short": ':'}
color = {'10':'tab:orange','20':'tab:green','30':'tab:red','200':'tab:purple'}

oos_df = pd.DataFrame({'Year': oos_years})
best_label, best_end = '', -np.inf
for n in TOP_NS:
    for kind in KINDs:
        seq = oos_traj[n][kind]
        cum = np.cumprod([1.0 + (x if pd.notna(x) else 0.0) for x in seq])
        label = f'{kind.capitalize()} Top{n}'
        plt.plot(oos_years, cum, marker=markers[kind], linestyle=ls[kind], color=color[str(n)], label=label)
        oos_df[label] = cum
        if len(cum)>0 and cum[-1] > best_end:
            best_end = float(cum[-1]); best_label = label

plt.title(f'OOS (stitched by year) – Best: {best_label} ({best_end:.2f})')
plt.xlabel('Year'); plt.ylabel('Cumulative (×)')
plt.yscale('log'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, f'{MODEL_NAME}_OOS_stitched.png'))
plt.close()
oos_df.to_csv(os.path.join(result_dir, f'{MODEL_NAME}_OOS_stitched.csv'), index=False)

# (2) Forward 年化彙總表
forward_df = pd.DataFrame(forward_summary_rows)
forward_csv = os.path.join(result_dir, f'{MODEL_NAME}_forward_span_annual.csv')
forward_df.to_csv(forward_csv, index=False)

# (3) 當輪摘要逐年表
trainwin_csv = os.path.join(result_dir, f'{MODEL_NAME}_training_windows.csv')
pd.DataFrame(training_rows).to_csv(trainwin_csv, index=False)

# (4) 畫 forward 年化（以 long_short/Top10 為例）
key_col = 'Top10_long_short_Annual'
if key_col in forward_df.columns:
    plt.figure(figsize=(14,6))
    plt.plot(forward_df['TestSpan'], forward_df[key_col], marker='o')
    plt.title(f'Forward Annualized ({key_col}) per training window')
    plt.xlabel('Forward Span'); plt.ylabel('Annualized Return')
    plt.grid(True); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'{MODEL_NAME}_forward_annual_{key_col}.png'))
    plt.close()

log.info(f'✅ 成品輸出：{result_dir}')
log.info(f'📄 日誌：{log_path}')
