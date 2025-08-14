# -*- coding: utf-8 -*-
import os
import re
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
# === DB 連線設定 ===
DB_CONFIG = {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("USER"),
    "password": os.getenv("PASSWORD"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT")
}
print(f"🔗 連線到資料庫 {DB_CONFIG['dbname']} 在 {DB_CONFIG['host']}:{DB_CONFIG['port']}")
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

# === Step 1: 讀取 StockList*.csv → 寫入 stocks / annual_financials ===
def import_financials_from_csv(file_dir):
    # 1) 收集長表資料
    file_paths = sorted(glob(os.path.join(file_dir, 'StockList*.csv')))
    long_data = []
    year_metric_pat = re.compile(r"(\d{4})(?:年度)?(.+)")

    for fp in file_paths:
        df = pd.read_csv(fp, encoding='utf-8-sig')
        if '排名' in df.columns:
            df = df.drop(columns=['排名'])
        if not {'代號','名稱'}.issubset(df.columns):
            raise ValueError(f'檔案 {os.path.basename(fp)} 缺少必要欄位：代號/名稱')

        id_name_cols = ['代號','名稱']
        data_cols = [c for c in df.columns if c not in id_name_cols]

        # 將每個「YYYY 指標」拆成年與指標
        for col in data_cols:
            m = year_metric_pat.match(col)
            if not m:
                continue
            year = int(m.group(1))
            metric = m.group(2).strip()
            for _, row in df.iterrows():
                raw_code = row['代號']
                # 兼容 ="2330" / 2330 / '2330'
                if isinstance(raw_code, str) and raw_code.startswith('="'):
                    code = raw_code[2:-1]
                else:
                    code = str(raw_code)
                code = re.sub(r'\D', '', code).zfill(4)

                long_data.append({
                    '代號': code,
                    '名稱': row['名稱'],
                    '年': year,
                    '指標': metric,
                    '值': row[col]
                })

    if not long_data:
        raise ValueError(f'❌ 沒有從 {file_dir} 的 StockList*.csv 解析到任何財報欄位（請檢查欄位是否為「2022 營收(億)」這種格式）')

    # 2) 長→寬（保留缺值，後面寫 DB 會變 NULL）
    long_df = pd.DataFrame(long_data)
    # 數字清洗：移除逗號，空字串→NA，括號負數 (123) → -123
    def _to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace(',', '')
        if s == '':
            return np.nan
        m = re.match(r'^\((\-?\d+(?:\.\d+)?)\)$', s)  # (123.4)
        if m:
            return -float(m.group(1))
        try:
            return float(s)
        except:
            return np.nan
    long_df['值'] = long_df['值'].map(_to_float)

    pivot_df = long_df.pivot_table(
        index=['代號','名稱','年'],
        columns='指標',
        values='值',
        aggfunc='first'
    ).reset_index()
    pivot_df.dropna(how='any', inplace=True)  # 移除有空值的列
    print(f"📊 CSV 檔 {len(file_paths)} 個 → 長表 {len(long_df):,} 筆 → 寬表 {len(pivot_df):,} 列")

    # 3) 先 upsert stocks（以四碼代號當主鍵，market=TW）
    with get_conn() as conn, conn.cursor() as cur:
        stocks = (
            pivot_df[['代號','名稱']].drop_duplicates()
            .assign(
                market='TW',
                yf_symbol=lambda x: x['代號'].astype(str).str.zfill(4) + '.TW'
            )
            .rename(columns={'代號':'stock_id','名稱':'name'})
        )
        # stock_id 轉純數字 BIGINT
        stocks['stock_id'] = stocks['stock_id'].astype(str).str.extract(r'(\d+)')[0].astype('int64')

        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO stocks(stock_id, market, name, yf_symbol)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (stock_id, market) DO UPDATE SET
                name = EXCLUDED.name,
                yf_symbol = EXCLUDED.yf_symbol
            """,
            list(stocks[['stock_id','market','name','yf_symbol']].itertuples(index=False, name=None))
        )

        # 建立 '四碼字串' -> int 的 map
        cur.execute("SELECT stock_id FROM stocks WHERE market='TW'")
        code2id = {str(sid).zfill(4): int(sid) for (sid,) in cur.fetchall()}

        # 4) 準備 annual_financials 的欄位對照（CSV→DB 欄名）
        rename_map = {
            '營收(億)': '營收_億',
            '營益(億)': '營益_億',
            '淨利率(%)': '淨利率_pct',
            '淨利增減(%)': '淨利增減_pct',
            'ROA(%)': 'roa_pct',
            '發行量(萬張)': '發行量_萬張',
            '股東權益總額(億)': '股東權益總額_億',
            '負債總額(億)': '負債總額_億',
            '流動資產對流動負債(%)': '流動資產對流動負債_pct',
            '速動資產對流動負債(%)': '速動資產對流動負債_pct',
            '存貨週轉率': '存貨週轉率',
            '應收帳款週轉率': '應收帳款週轉率',
        }

        fin = pivot_df.rename(columns=rename_map).copy()
        fin.rename(columns={'年':'year'}, inplace=True)
        fin['stock_id'] = fin['代號'].astype(str).str.zfill(4).map(code2id).astype('Int64')

        missing_codes = fin[fin['stock_id'].isna()]['代號'].unique().tolist()
        if missing_codes:
            raise ValueError(f"❌ 下列代號在 stocks 找不到對應 stock_id：{sorted(missing_codes)}")

        # 取 DB 真實欄位，確保只插存在的欄位
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='annual_financials'
            ORDER BY ordinal_position
        """)
        db_cols = [r[0] for r in cur.fetchall()]

        # 我們要寫入的欄位：DB 有、且 fin 有資料的欄位
        candidate_cols = ['stock_id','year'] + [c for c in rename_map.values()]
        insert_cols = [c for c in db_cols if c in candidate_cols]

        # 額外統計：CSV 有但 DB 沒有的指標（僅提示）
        csv_metrics = set(col for col in pivot_df.columns if col not in ['代號','名稱','年'])
        mapped_metrics = set(rename_map.keys())
        not_in_db = sorted((csv_metrics - mapped_metrics))
        if not_in_db:
            print(f"ℹ️ 這些 CSV 指標目前未寫入（DB 未建或未映射）：{not_in_db[:12]}{' ...' if len(not_in_db)>12 else ''}")

        # 組資料（NaN→None）
        fin = fin.dropna(how='any')
        rows = [tuple(int(val) if isinstance(val, np.int64) else val for val in fin.loc[i, insert_cols].values) for i in fin.index]

        if not rows:
            raise ValueError("❌ annual_financials 沒有可寫入的資料列")

        # 動態 upsert
        cols_sql = ','.join(f'"{c}"' for c in insert_cols)
        placeholders = ','.join(['%s']*len(insert_cols))
        update_sql = ','.join(f'"{c}"=EXCLUDED."{c}"' for c in insert_cols if c not in ('stock_id','year'))

        psycopg2.extras.execute_batch(
            cur,
            f"""
            INSERT INTO annual_financials({cols_sql})
            VALUES ({placeholders})
            ON CONFLICT (stock_id, year) DO UPDATE SET
            {update_sql}
            """,
            rows,
            page_size=500
        )

    print(f"✅ 匯入年度財報完成：{len(pivot_df):,} 列、寫入欄位 {len(insert_cols)} 個")
    return code2id



# === Step 2: 抓日價 → daily_prices ===
PRICE_DIR = os.path.join(os.path.dirname(__file__), 'price_csv')
CHUNK_ROWS = 50_000   # 累積到這麼多 rows 就批次寫入一次（可調）

def _to_float_or_none(x):
    if pd.isna(x): 
        return None
    s = str(x).strip().replace(',', '')
    if s == '': 
        return None
    try:
        return float(s)
    except:
        return None

def load_price_csv(stock_id: int):
    """
    讀取單一股票的價格 CSV，轉成可直接寫 DB 的 rows list。
    僅做 I/O+清洗，不進行任何 DB 操作。
    回傳: List[Tuple(stock_id, dt, open, high, low, close, adj_close, volume)]
    """
    # 支援 2330 與 02330 兩種命名（視你實際檔名情況）
    candidates = [
        os.path.join(PRICE_DIR, f'Price_{stock_id}.csv'),
    ]
    price_file = next((p for p in candidates if os.path.exists(p)), None)
    if not price_file:
        return [], f"❌ Price_{stock_id}.csv 不存在"

    try:
        df = pd.read_csv(price_file, dtype=str)
        # 必要欄位檢查
        need_cols = {'Date','Open','High','Low','Close','Volume'}
        if not need_cols.issubset(df.columns):
            return [], f"⚠️ {os.path.basename(price_file)} 欄位缺失：{need_cols - set(df.columns)}"

        # 轉型
        df['dt'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df = df.dropna(subset=['dt'])

        open_p   = df['Open'].map(_to_float_or_none)
        high_p   = df['High'].map(_to_float_or_none)
        low_p    = df['Low'].map(_to_float_or_none)
        close_p  = df['Close'].map(_to_float_or_none)
        adj_close= df['Adj Close'].map(_to_float_or_none) if 'Adj Close' in df.columns else close_p
        # Volume 可能有小數/空字串，轉 int 或 None
        vol = []
        for v in df['Volume'].fillna(''):
            vv = str(v).replace(',', '').strip()
            if vv == '':
                vol.append(None)
            else:
                try:
                    vol.append(int(float(vv)))
                except:
                    vol.append(None)

        # 保留最後一次出現的該日（若檔案內有重複日期）
        out = {}
        for d, o, h, l, c, a, v in zip(df['dt'], open_p, high_p, low_p, close_p, adj_close, vol):
            out[d] = (int(stock_id), d, o, h, l, c, a, v)

        rows = list(out.values())
        return rows, f"✅ {stock_id} 讀入 {len(rows)} 列"
    except Exception as e:
        return [], f"⚠️ {stock_id} 讀檔失敗：{e}"

def write_prices_to_db(rows):
    """
    單執行緒批次寫 DB。使用 execute_values 一次塞多列＋ON CONFLICT UPSERT。
    """
    if not rows:
        return 0
    sql = """
        INSERT INTO daily_prices
            (stock_id, dt, open_p, high_p, low_p, close_p, adj_close, volume)
        VALUES %s
        ON CONFLICT (stock_id, dt) DO UPDATE SET
            open_p = EXCLUDED.open_p,
            high_p = EXCLUDED.high_p,
            low_p  = EXCLUDED.low_p,
            close_p= EXCLUDED.close_p,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume
    """
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=10_000)
    return len(rows)

def parallel_import_prices(code2id: dict, max_workers=16, chunk_rows=CHUNK_ROWS):
    """
    多執行緒讀 CSV -> 主執行緒累積到一定行數就批次寫入 DB。
    code2id: {'2330': 2330, ...} （key不使用，主要是 value 為整數 stock_id）
    """
    # 只取整數 stock_id 列表
    stock_ids = list(set(int(v) for v in code2id.values()))
    rows_buffer = []
    ok, miss = 0, 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(load_price_csv, sid) for sid in stock_ids]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rows, msg = fut.result()
            print(msg)
            if rows:
                rows_buffer.extend(rows)
                # 累積夠大就寫一次
                if len(rows_buffer) >= chunk_rows:
                    n = write_prices_to_db(rows_buffer)
                    ok += n
                    rows_buffer.clear()
            else:
                miss += 1

    # 收尾
    if rows_buffer:
        n = write_prices_to_db(rows_buffer)
        ok += n

    print(f"✅ daily_prices 匯入完成：寫入 {ok:,} 列，缺檔/失敗 {miss} 檔")


# === Step 3: 計算特徵 → financial_features ===
def compute_and_store_features():
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT a.*, v.closing_price_year
            FROM annual_financials a
            LEFT JOIN v_year_end_close v
            ON a.stock_id = v.stock_id AND a.year = v.year
            WHERE a.year >= 2000
            ORDER BY a.stock_id, a.year
        """, conn)

    df['market_cap_mil'] = df['closing_price_year'] * df['發行量(萬張)'] * 1000
    df['pb_ratio'] = df['closing_price_year'] / (df['股東權益總額(億)'] * 100 / df['發行量(萬張)'])
    df['ps_ratio'] = df['market_cap_mil'] / (df['營收(億)'] * 100)
    df['roe_m'] = df['淨利率(%)']/100 * df['營收(億)'] / df['股東權益總額(億)']
    df['opm'] = df['營益(億)'] / df['營收(億)']
    df['npm'] = df['淨利率(%)']/100
    df['roa'] = df['ROA(%)']/100
    df['debt_to_equity'] = df['負債總額(億)'] / df['股東權益總額(億)']
    df['current_ratio_m'] = df['流動資產對流動負債(%)']/100
    df['quick_ratio_m'] = df['速動資產對流動負債(%)']/100
    df['inventory_turnover_m'] = df['存貨週轉率']
    df['ar_turnover_m'] = df['應收帳款週轉率']

    df = df.sort_values(['stock_id', 'year'])
    df['op_growth_m'] = df.groupby('stock_id')['營益(億)'].pct_change()
    df['net_income_growth_m'] = df.groupby('stock_id')['淨利率(%)'].pct_change()
    df['future_price'] = df.groupby('stock_id')['closing_price_year'].shift(-1)
    df['return_1y'] = (df['future_price'] / df['closing_price_year']) - 1
    df['return_label'] = (df['return_1y'] > 0).astype(int)

    cols = [
        'stock_id','year','closing_price_year','market_cap_mil','pb_ratio','ps_ratio',
        'roe_m','opm','npm','roa','debt_to_equity','current_ratio_m','quick_ratio_m',
        'inventory_turnover_m','ar_turnover_m','op_growth_m','net_income_growth_m',
        'return_1y','return_label'
    ]
    out = df[cols].dropna(subset=['closing_price_year'])

    with get_conn() as conn, conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, f"""
            INSERT INTO financial_features({','.join(cols)})
            VALUES ({','.join(['%s']*len(cols))})
            ON CONFLICT (stock_id, year) DO UPDATE SET
            {','.join([f"{c}=EXCLUDED.{c}" for c in cols[2:]])}
        """, list(out.itertuples(index=False, name=None)))
    print("✅ financial_features 更新完成")

# === Main ===
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    print(f"📂 使用目錄：{base_dir}")
    code2id = import_financials_from_csv(base_dir)
    parallel_import_prices(code2id, max_workers=16, chunk_rows=50_000)
    # compute_and_store_features()
    print("🎉 buildDB 全流程完成喵♡")
