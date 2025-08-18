
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 並行工具
from joblib import Parallel, delayed
# 進度條
from tqdm.auto import tqdm


# 讀取檔案
file_path = os.path.join(os.path.dirname(__file__), 'BITSTAMP_BTCUSD, 240.csv')
base_dir = os.path.dirname(file_path)
result_path = os.path.join(base_dir, 'result')
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


df = pd.read_csv(file_path)
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
# logging.info(f"Row NaN summary: {row_nan_counts.to_dict()}")

# 將每列的 NaN 數加入原始 df（便於後續觀察）


df = df.dropna(how='any',axis=1)
df = df.dropna(how='any',axis=0)

logging.info(f"import features: {df.columns.tolist()}")
logging.info(f"Number of available features: {len(df.columns)}")
logging.info(f"Number of rows after NaN removal: {len(df)}")

# 逐時間滾動算，每次資料量都會+1，命名規則是 {col}_{name}
# 每個特徵變化率 1~6
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    for k in range(1, 7):
        new_col = f"{col}_chg{k}"
        # 相對變化率 (pct_change)，處理無限值與 NaN -> 填 0，保留原始欄位
        df[new_col] = df[col].pct_change(periods=k)
        df[new_col].replace([np.inf, -np.inf], np.nan, inplace=True)
        df[new_col].fillna(0, inplace=True)

logging.info(f"Added change-rate features for cols: {numeric_cols} with k=1..6. New column count: {len(df.columns)}")

#轉換
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


# 建立標準化版本
log_trans = lambda x: np.log1p(x)  # log(x+1) to avoid log(0)
def rank_scaler(X):
    if isinstance(X, pd.DataFrame):
        return X.rank(method="average", pct=True).to_numpy()
    else:
        # 如果是 numpy array，也做一樣的事（保險）
        return pd.DataFrame(X).rank(method="average", pct=True).to_numpy()
from sklearn.preprocessing import Normalizer
l1norm = Normalizer(norm='l1')
l2norm = Normalizer(norm='l2')
def unit_vector_featurewise(X):
    return X / np.linalg.norm(X, axis=0)

def sigmoid_scaling(x):
    X = np.asarray(X)
    X = np.clip(X, -500, 500)  # 限制範圍
    return 1 / (1 + np.exp(-X))

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
    X = np.asarray(X)  # 保證是 numpy array
    max_per_row = np.max(np.abs(X), axis=1).reshape(-1, 1)  # 等效於 keepdims=True
    return X / max_per_row
def mean_centering(x):
    return x - np.mean(x, axis=0)

from sklearn.preprocessing import FunctionTransformer
scalers = {
    'z': StandardScaler(),
    'minmax': MinMaxScaler(),
    'maxabs': MaxAbsScaler(),
    'robust': RobustScaler(),
    # 'row_maxabs': FunctionTransformer(row_maxabs_scaling, validate=False),
    # 'mean_centering': FunctionTransformer(mean_centering, validate=False),
    # 'rank': FunctionTransformer(rank_scaler, validate=False),
    # 'unit_vector': FunctionTransformer(unit_vector_featurewise, validate=False),
    # 'tanh': FunctionTransformer(tanh_estimator_scaling, validate=False),
    # 'pca': FunctionTransformer(pca_whitening, validate=False),
    # 'zca': FunctionTransformer(zca_whitening, validate=False),
    'power': PowerTransformer(method='yeo-johnson'),
    'quantile': QuantileTransformer(output_distribution='normal', n_quantiles=100)
}
# 逐時間滾動算，每次資料量都會+1
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from joblib import Parallel, delayed
import numpy as np
n_rows = len(df)
df_values = df[numeric_cols].to_numpy()


def _process_scaler(sname, scaler_obj, df_values, numeric_cols, n_rows):
    scaler = clone(scaler_obj)
    n_cols = len(numeric_cols)
    result = np.zeros((n_rows, n_cols), dtype=float)

    for i in range(n_rows):
        X = df_values[: i + 1].copy()
        try:
            if hasattr(scaler, "fit_transform"):
                Xt = scaler.fit_transform(X)
            else:
                scaler.fit(X)
                Xt = scaler.transform(X)
        except Exception:
            if X.shape[0] == 1:
                try:
                    Xt_tmp = scaler.fit_transform(np.vstack([X, X]))
                    Xt = Xt_tmp[:1]
                except Exception:
                    Xt = X.copy()
            else:
                try:
                    Ximp = SimpleImputer(strategy="median").fit_transform(X)
                    if hasattr(scaler, "fit_transform"):
                        Xt = scaler.fit_transform(Ximp)
                    else:
                        scaler.fit(Ximp)
                        Xt = scaler.transform(Ximp)
                except Exception:
                    Xt = X.copy()

        Xt = np.atleast_2d(Xt)
        if Xt.shape[1] >= n_cols:
            last_row = Xt[-1, : n_cols]
        else:
            pad = np.zeros(n_cols - Xt.shape[1], dtype=float)
            last_row = np.concatenate([Xt[-1, :], pad])

        result[i, :] = last_row

    return sname, result

# parallel over scalers (each scaler handled in its own process)
n_jobs = min(len(scalers), os.cpu_count() or 1)
logging.info(f"Starting parallel rolling transforms with n_jobs={n_jobs}")
parallel_results = Parallel(n_jobs=n_jobs, backend='loky')(
    delayed(_process_scaler)(sname, scaler, df_values, numeric_cols, n_rows)
    for sname, scaler in scalers.items()
)

# collect results back into df
for sname, result in parallel_results:
    new_col_names = [f"{col}_{sname}" for col in numeric_cols]
    for j, col in enumerate(numeric_cols):
        df[new_col_names[j]] = result[:, j]
    logging.info(f"Completed rolling transform for scaler: {sname}. Added cols: {new_col_names}")

print(f"number of features after scaling: {len(df.columns)}")
