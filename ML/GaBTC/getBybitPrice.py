# -*- coding: utf-8 -*-
"""
Bybit BTCUSDT (linear/perpetual, 240m) → Postgres.Price
- 若 DB 已有資料：只抓「最新之後」的 K 線
- 若無資料：從最早一路抓到現在
環境變數：DBNAME, USER, PASSWORD, HOST, PORT
"""
import os
import time
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# ========= 設定 =========
EXCHANGE = "bybit"
PRODUCT  = "perpetual"   # 你的表用語：spot/future/perpetual
SYMBOL   = "BTCUSDT"
INTERVAL = "240"          # Bybit 的 1 = 1 minute

def interval_to_ms(interval: str) -> int:
    s = interval.upper()
    if s.isdigit():
        return int(s) * 60 * 1000
    if s == "D":
        return 24 * 60 * 60 * 1000
    if s == "W":
        return 7 * 24 * 60 * 60 * 1000
    if s == "M":
        return 30 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")

INTERVAL_MS = interval_to_ms(INTERVAL)

# ========= DB 連線 =========
load_dotenv()
DB_CONFIG = {
    "dbname":   os.getenv("DBNAME"),
    "user":     os.getenv("USER"),
    "password": os.getenv("PASSWORD"),
    "host":     os.getenv("HOST", "127.0.0.1"),
    "port":     os.getenv("PORT", "5432"),
}

def connect_db():
    conn = psycopg2.connect(**DB_CONFIG)
    print("Connected to the database successfully.")
    return conn

def get_latest_ts_ms(conn, exchange, product, symbol, interval):
    """回傳該鍵在 Price 的最新 timestamp（ms, UTC）; 若無資料回傳 None"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(timestamp) FROM Price
            WHERE exchange=%s AND product=%s AND symbol=%s AND interval=%s
        """, (exchange, product, symbol, interval))
        row = cur.fetchone()
        if row and row[0]:
            dt = row[0].astimezone(timezone.utc)
            return int(dt.timestamp() * 1000)
    return None

def upsert_prices(conn, rows):
    """
    rows: list of tuple(exchange, product, symbol, interval,
                        open, high, low, close, volume, timestamp_utc)
    """
    if not rows:
        return 0
    with conn.cursor() as cur:
        sql = """
        INSERT INTO Price (
            exchange, product, symbol, interval,
            open, high, low, close, volume, timestamp
        )
        VALUES %s
        ON CONFLICT (exchange, product, symbol, interval, timestamp)
        DO UPDATE SET
            open  = EXCLUDED.open,
            high  = EXCLUDED.high,
            low   = EXCLUDED.low,
            close = EXCLUDED.close,
            volume= EXCLUDED.volume;
        """
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)

# ========= Bybit 取 K 線（往過去翻頁）=========
def fetch_bybit_backward(category="linear", symbol="BTCUSDT", interval="240",
                         limit=1000, sleep_sec=0.03, stop_after_ms=None):
    """
    由現在往過去翻頁；若 stop_after_ms 指定，當抓到的最早一根 <= stop_after_ms 即停止
    回傳：「舊→新」排序之 list（元素是原始 kline 陣列）
    Bybit /v5/market/kline 回傳： [start, open, high, low, close, volume, turnover]
    """
    session = HTTP(testnet=False)
    out = []
    end_cursor = int(time.time() * 1000)  # 以現在為終點
    page = 0
    while True:
        page += 1
        resp = session.get_kline(
            category=category, symbol=symbol, interval=str(interval),
            end=end_cursor, limit=limit
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {resp}")
        lst = resp["result"]["list"]
        if not lst:
            print(f"Done. total={len(out)}")
            break

        out.extend(lst)
        earliest = int(lst[-1][0])  # 此頁最早一根（ms, UTC）
        ts = datetime.fromtimestamp(earliest/1000, tz=timezone.utc)
        print(f"[page {page:03d}] got={len(lst)} earliest={ts.isoformat()} total={len(out)}")

        # 若設定了 stop_after_ms（增量模式），到界線就停
        if stop_after_ms is not None and earliest <= stop_after_ms:
            break

        end_cursor = earliest - 1
        time.sleep(sleep_sec)

    # 去掉界線以前的資料（僅保留 > stop_after_ms）
    if stop_after_ms is not None:
        out = [k for k in out if int(k[0]) > stop_after_ms]

    # 轉為「舊→新」
    out.sort(key=lambda x: int(x[0]))
    return out

def to_price_rows(klines,
                  exchange=EXCHANGE, product=PRODUCT,
                  symbol=SYMBOL, interval=INTERVAL):
    rows = []
    for k in klines:
        # k: [start, open, high, low, close, volume, turnover]（字串）
        start_ms = int(k[0])
        o, h, l, c = map(float, k[1:5])
        vol = float(k[5]) if k[5] is not None else 0.0
        rows.append((
            exchange, product, symbol, interval,
            round(o, 8), round(h, 8), round(l, 8), round(c, 8),
            round(vol, 8),
            datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)  # UTC timestamptz
        ))
    return rows

# ========= 主流程 =========
if __name__ == "__main__":
    conn = connect_db()
    try:
        latest_ms = get_latest_ts_ms(conn, EXCHANGE, PRODUCT, SYMBOL, INTERVAL)

        if latest_ms is None:
            print("No existing rows → full backfill from earliest.")
            # 全量回補：一路往過去抓到最早
            klines = fetch_bybit_backward(
                category="linear", symbol=SYMBOL, interval=INTERVAL,
                limit=1000, sleep_sec=0.03, stop_after_ms=None
            )
        else:
            # 增量：從「最新一根的下一根」開始抓
            since_ms = latest_ms + INTERVAL_MS
            print(
                f"Latest in DB = {datetime.fromtimestamp(latest_ms/1000, tz=timezone.utc).isoformat()} "
                f"→ fetch > {datetime.fromtimestamp(latest_ms/1000, tz=timezone.utc).isoformat()} "
                f"(start from {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).isoformat()})"
            )
            klines = fetch_bybit_backward(
                category="linear", symbol=SYMBOL, interval=INTERVAL,
                limit=1000, sleep_sec=0.03, stop_after_ms=since_ms - 1
            )

        rows = to_price_rows(klines)
        n = upsert_prices(conn, rows)
        print(f"Upsert done, rows={n}")
    finally:
        conn.close()
