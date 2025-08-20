import pandas as pd
import yfinance as yf
import os
# import json  # Removed as it is unused
from tqdm import tqdm
from datetime import datetime  # Ensure datetime is used correctly (used in get_last_day and other parts)
import math  # Import math module for mathematical operations
# from threading import Lock  # Removed as it is unused
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from dotenv import load_dotenv  # Ensure the 'python-dotenv' package is installed and .env file is properly configured

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

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ 資料庫連線成功")
except Exception as e:
    print(f"❌ 資料庫連線失敗：{e}")

# # === 路徑設定 ===
# base_dir = 'AIFinTech/Final/gooddata'  # Ensure this path exists if uncommented and replace 'gooddata' with an appropriate directory name if needed
# csv_path = os.path.join(base_dir, 'merged_all_metrics.csv')
# cache_path = os.path.join(base_dir, 'yahoo_price_cache.json')
# output_path = os.path.join(base_dir, 'financial_features_all.csv')
# price_dir = os.path.join(base_dir, 'price_csv')
# os.makedirs(price_dir, exist_ok=True)

# === 讀取資料 ===
cur = conn.cursor()
cur.execute("ALTER TABLE daily_prices DISABLE TRIGGER trg_refresh_fin_features;")
cur.execute("""
    SELECT s.stock_id,
           MIN(dp.dt) AS first_date,
           MAX(dp.dt) AS last_date
    FROM stocks s
    LEFT JOIN daily_prices dp ON s.stock_id = dp.stock_id
    GROUP BY s.stock_id
""")
rows = cur.fetchall()
df = pd.DataFrame(rows, columns=['代號', '第一交易日', '最後交易日'])
df['代號'] = df['代號'].astype(int)

# 轉成 Python date，若為 NULL 則填入 2000-01-01（方便後續比較與 year 屬性）
default_First_Date = pd.to_datetime('2000-01-01').date()
default_Last_Date = pd.to_datetime('2025-12-31').date()
df['第一交易日'] = pd.to_datetime(df['第一交易日'], errors='coerce').dt.date.fillna(default_Last_Date)
df['最後交易日'] = pd.to_datetime(df['最後交易日'], errors='coerce').dt.date.fillna(default_First_Date)

# === 股票清單 ===
all_stock_ids = df['代號'].unique()

def get_last_day(stock_id):
    try:
        cur.execute("SELECT MAX(dt) FROM daily_prices WHERE stock_id = %s", (int(stock_id),))
    except psycopg2.Error as e:
        conn.rollback()
        print(f"⚠️ Database error while fetching last day for stock {stock_id}: {e}")
        return None
    result = cur.fetchone()
    return result[0] if result else None


print("📦 抓取每檔股票 2000～2024加上2025YTD 所有收盤價...")
# 轉成原生型別 tuple：[(stock_id, dt, open, high, low, close, adj, vol), ...]
cached_rows = []

def fetch_and_process_stock(stock_id):

    yf_id = f"{stock_id}.TW"
    hist = None
    # success variable is unused, removing it

    # if os.path.exists(price_csv_path):
    #     try:
    #         hist = pd.read_csv(price_csv_path, parse_dates=['Date'], index_col='Date')
    #     except Exception:
    #         os.remove(price_csv_path)

    # hist_cache variable is unused, removing it

    if hist is None:

        first_year = 2000
        last_year = 2025
        last_date = df.loc[df['代號'] == stock_id, '最後交易日'].values[0]
        first_year = max(first_year, last_date.year if last_date else first_year)
        today = datetime.now()
        # print(f"last_date: {last_date}, today: {today.date()}")
        if last_date == today.date():
            print(f"✅ {stock_id} 今天已經有最新資料，跳過抓取")
            return f"✅ {stock_id} 資料處理完成"
        for end_year in range(last_year, first_year - 1, -1):
            for start_year in range(first_year, end_year + 1):
                start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if last_date else f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"

                try:
                    hist = yf.download(
                        yf_id,
                        start=start_date,
                        end=end_date,
                        auto_adjust=False,
                        progress=False,
                        timeout=20
                    )
                    if hist is not None and not hist.empty:
                        hist.reset_index(inplace=True)  # Ensure 'Date' is a regular column
                        hist = hist.loc[:, ~hist.columns.duplicated()]

                        print(f"✅ {stock_id} 從 {start_date} 到 {end_date} 抓取成功")
                        break
                except Exception as e:
                    print(f"⚠️ {stock_id} 從 {start_date} 到 {end_date} 抓取失敗：{e}")
                    continue
            def _to_date_str(val):
                # row['Date'] 若因重複欄名等狀況變成 Series，先取第一個
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                ts = pd.to_datetime(val, errors="coerce")
                if pd.isna(ts):
                    return None
                return ts.strftime("%Y-%m-%d")
            def _first_scalar(x):
                # 如果是 Series（代表有重複欄名），拿第一個；否則原值
                val = x.iloc[0] if isinstance(x, pd.Series) else x
                # 如果值是 NaT 或無效，返回 None
                if pd.isna(val) or isinstance(val, pd._libs.tslibs.nattype.NaTType):
                    return -1
                return val


            if hist is not None and not hist.empty:
                for _, row in hist.iterrows():
                    stock_id = int(stock_id)
                    dt = _to_date_str(row['Date'])
                    if dt is None:
                        continue
                    op = float(_first_scalar(row['Open']))
                    hi = float(_first_scalar(row['High']))
                    lo = float(_first_scalar(row['Low']))
                    cl = float(_first_scalar(row['Close']))
                    ac = float(_first_scalar(row['Adj Close']))
                    vol_raw = _first_scalar(row['Volume'])
                    vol = int(vol_raw) if (vol_raw is not None and not pd.isna(vol_raw) and not math.isnan(float(vol_raw))) else 0
                    if op==-1 or hi==-1 or lo==-1 or cl==-1 or ac==-1:
                        continue
                    cached_rows.append((stock_id, dt, op, hi, lo, cl, ac, vol))
            else:
                print(f"⚠️ {stock_id}: 無法抓取任何資料，跳過此股票")

            if hist is not None and not hist.empty:
                hist_end_date = pd.to_datetime(hist['Date'].iloc[-1]).strftime('%Y-%m-%d')
                print(f"✅ {stock_id} 從 {start_date} 到 {hist_end_date} 抓取成功")
            else:
                print(f"⚠️ {stock_id}: 無法抓取任何資料，跳過此股票")
            break

    if hist is None or hist.empty:
        return f"❌ {stock_id} 完全抓不到任何資料"


    return f"✅ {stock_id} 資料處理完成"


def fetch_and_process_stock_early(stock_id):
    yf_id = f"{stock_id}.TW"
    hist = None

    # 取出資料庫紀錄的最早交易日（第一交易日）
    first_date = df.loc[df['代號'] == stock_id, '第一交易日'].values[0]
    if first_date is None:
        return f"❌ {stock_id} 無法取得資料庫最早交易日"

    # 如果已經有 2000 年或更早的資料，則不需要再抓更早的價格
    if first_date <= pd.to_datetime('2000-01-04').date():
        print(f"🔁 {stock_id} 已有 2000 年或更早的資料，跳過抓取")
        return f"✅ {stock_id} 資料處理完成（無需抓更早）"

    # 我們要抓到 first_date 之前一天為止的歷史資料（從 2000-01-01 開始）
    fetch_start = "2000-01-01"
    fetch_end_date = (pd.to_datetime(first_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_year = 2000
    end_year = pd.to_datetime(fetch_end_date).year

    # 嘗試一次抓完整區間；若失敗再用逐年縮短的方式重試
    for try_end_year in range(end_year, start_year - 1, -1):
        if try_end_year == end_year:
            start = fetch_start
            end = fetch_end_date
        else:
            start = fetch_start
            end = f"{try_end_year}-12-31"

        try:
            hist = yf.download(
                yf_id,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                timeout=20
            )
            if hist is not None and not hist.empty:
                hist.reset_index(inplace=True)  # Ensure 'Date' is a regular column
                hist = hist.loc[:, ~hist.columns.duplicated()]
                print(f"✅ {stock_id} 從 {start} 到 {end} 抓取成功（用於補抓舊資料）")
                break
        except Exception as e:
            print(f"⚠️ {stock_id} 從 {start} 到 {end} 抓取失敗：{e}")
            continue

    def _to_date_str(val):
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        ts = pd.to_datetime(val, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")

    def _first_scalar(x):
        val = x.iloc[0] if isinstance(x, pd.Series) else x
        if pd.isna(val) or isinstance(val, pd._libs.tslibs.nattype.NaTType):
            return -1
        return val

    if hist is None or hist.empty:
        print(f"⚠️ {stock_id}: 完全抓不到任何舊資料（從 {fetch_start} 到 {fetch_end_date}）")
        return f"❌ {stock_id} 完全抓不到任何資料"

    # 將抓到的舊資料逐列加入 cached_rows（注意不要覆寫外層的 stock_id）
    for _, row in hist.iterrows():
        sid = int(stock_id)
        dt = _to_date_str(row['Date'])
        if dt is None:
            continue
        op = float(_first_scalar(row['Open']))
        hi = float(_first_scalar(row['High']))
        lo = float(_first_scalar(row['Low']))
        cl = float(_first_scalar(row['Close']))
        ac = float(_first_scalar(row['Adj Close']))
        vol_raw = _first_scalar(row['Volume'])
        vol = int(vol_raw) if (vol_raw is not None and not pd.isna(vol_raw) and not math.isnan(float(vol_raw))) else 0
        if op == -1 or hi == -1 or lo == -1 or cl == -1 or ac == -1:
            continue
        cached_rows.append((sid, dt, op, hi, lo, cl, ac, vol))

    hist_end_date = pd.to_datetime(hist['Date'].iloc[-1]).strftime('%Y-%m-%d')
    print(f"✅ {stock_id} 補抓到最舊資料，區間 {fetch_start} ~ {hist_end_date}")

    return f"✅ {stock_id} 資料處理完成（已補抓舊資料）"


# 🎯 多執行緒跑起來
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(fetch_and_process_stock, int(sid)): int(sid) for sid in all_stock_ids}
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        print(result)

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(fetch_and_process_stock_early, int(sid)): int(sid) for sid in all_stock_ids}
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        print(result)
    
    




# ✅ 到這裡就可以接下來做指標運算囉～
print("🎉 所有股票歷史收盤價抓取完成，可通知資料庫開始特徵運算了喵♡")
# ✅ 到這裡就可以接下來做指標運算囉～
print("🎉 所有股票歷史收盤價抓取完成，可通知資料庫開始特徵運算了喵♡")

#暫存cache to .csv
# cached_rows_df = pd.DataFrame(cached_rows, columns=['stock_id', 'dt', 'open_p', 'high_p', 'low_p', 'close_p', 'adj_close', 'volume'])
# cached_rows_df.to_csv("cached_stock_data.csv", index=False)

import io

if cached_rows:
    try:
        # 0) 建 staging 表（只會在不存在時建立；UNLOGGED 較快）
        cur.execute("""
        CREATE UNLOGGED TABLE IF NOT EXISTS daily_prices_staging
        (
          stock_id  BIGINT,
          dt        DATE,
          open_p    NUMERIC,
          high_p    NUMERIC,
          low_p     NUMERIC,
          close_p   NUMERIC,
          adj_close NUMERIC,
          volume    BIGINT
        );
        """)

        # 1) 準備 COPY 的緩衝
        buf = io.StringIO()
        for r in cached_rows:
            # (stock_id, dt, open, high, low, close, adj_close, volume)
            buf.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}\t{r[7]}\n")
        buf.seek(0)

        # 2) 清空 staging，將本次批次 COPY 進去（注意：不是清空正表）
        cur.execute("TRUNCATE TABLE daily_prices_staging")
        cur.copy_from(buf, 'daily_prices_staging',
                      columns=['stock_id','dt','open_p','high_p','low_p','close_p','adj_close','volume'])

        # 3) 從 staging 合併（UPSERT）回正表
        cur.execute("""
            INSERT INTO daily_prices AS t
              (stock_id, dt, open_p, high_p, low_p, close_p, adj_close, volume)
            SELECT stock_id, dt, open_p, high_p, low_p, close_p, adj_close, volume
            FROM daily_prices_staging
            ON CONFLICT (stock_id, dt) DO UPDATE SET
              open_p    = EXCLUDED.open_p,
              high_p    = EXCLUDED.high_p,
              low_p     = EXCLUDED.low_p,
              close_p   = EXCLUDED.close_p,
              adj_close = EXCLUDED.adj_close,
              volume    = EXCLUDED.volume;
        """)

        # 4) 合併完成就把 staging 刪掉（你要的是刪除）
        cur.execute("DROP TABLE IF EXISTS daily_prices_staging")

        conn.commit()
        print(f"✅ COPY→UPSERT 完成，筆數約 {len(cached_rows)}")

    except psycopg2.Error as e:
        conn.rollback()
        print(f"❌ 寫入資料庫失敗：{e}")
    finally:
        # 不管成功與否都把觸發器打開（避免忘了）
        try:
            print("🔄 開啟觸發器和物化檢視...")
            cur.execute("ALTER TABLE daily_prices ENABLE TRIGGER trg_refresh_fin_features;")
            # 如需刷新物化檢視，再執行（可視情況移到 try 裡）
            print("🔄 刷新物化檢視 financial_features...")
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY financial_features;")
            conn.commit()
        except Exception as e2:
            print(f"⚠️ 觸發器/物化檢視收尾時出錯：{e2}")
else:
    print("⚠️ 沒有資料需要寫入資料庫")
