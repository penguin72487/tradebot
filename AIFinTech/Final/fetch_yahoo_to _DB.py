import pandas as pd
import yfinance as yf
import os
# import json  # Removed as it is unused
from tqdm import tqdm
from datetime import datetime  # Ensure datetime is used correctly
# from threading import Lock  # Removed as it is unused
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from dotenv import load_dotenv  # Ensure .env file is present and properly configured

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
# base_dir = 'AIFinTech/Final/gooddata'  # Ensure this path exists if uncommented
# csv_path = os.path.join(base_dir, 'merged_all_metrics.csv')
# cache_path = os.path.join(base_dir, 'yahoo_price_cache.json')
# output_path = os.path.join(base_dir, 'financial_features_all.csv')
# price_dir = os.path.join(base_dir, 'price_csv')
# os.makedirs(price_dir, exist_ok=True)

# === 讀取資料 ===
cur = conn.cursor()
cur.execute("SELECT stock_id FROM stocks")
rows = cur.fetchall()
df = pd.DataFrame(rows, columns=['代號'])
df['代號'] = df['代號'].astype(int)  # Ensure stock_id is of type int

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
def fetch_and_process_stock(stock_id):

    yf_id = f"{stock_id}.TW"
    hist = None
    success = False

    # if os.path.exists(price_csv_path):
    #     try:
    #         hist = pd.read_csv(price_csv_path, parse_dates=['Date'], index_col='Date')
    #     except Exception:
    #         os.remove(price_csv_path)

    hist_cache = None  # Initialize hist_cache to None

    if hist is None:

        first_year = 2000
        last_year = 2025
        last_date = get_last_day(stock_id)
        first_year = max(first_year, last_date.year if last_date else first_year)
        for end_year in range(last_year, first_year - 1, -1):
            for start_year in range(first_year, end_year + 1):
                start_date = last_date.strftime('%Y-%m-%d') if last_date else f"{start_year}-01-01"
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
                        success = True
                        print(f"✅ {stock_id} 從 {start_date} 到 {end_date} 抓取成功")
                        break
                except Exception as e:
                    print(f"⚠️ {stock_id} 從 {start_date} 到 {end_date} 抓取失敗：{e}")
                    continue
            if success:

                hist_end_date = hist.index[-1].strftime('%Y-%m-%d')

                print(f"✅ {stock_id} 從 {start_date} 到 {hist_end_date} 抓取成功")
                break

    if hist is None or hist.empty:
        return f"❌ {stock_id} 完全抓不到任何資料"


    return f"✅ {stock_id} 資料處理完成"

# 🎯 多執行緒跑起來
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(fetch_and_process_stock, int(sid)): int(sid) for sid in all_stock_ids}
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        print(result)



# ✅ 到這裡就可以接下來做指標運算囉～
print("🎉 所有股票歷史收盤價抓取完成，可通知資料庫開始特徵運算了喵♡")
# 💡 只保留有成功抓到收盤價的資料（防止後續爆錯）
