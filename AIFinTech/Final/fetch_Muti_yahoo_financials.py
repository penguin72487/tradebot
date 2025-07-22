import pandas as pd
import yfinance as yf
import os
import json
from tqdm import tqdm
from datetime import datetime
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed


# === 路徑設定 ===
base_dir = 'AIFinTech/Final/gooddata'
csv_path = os.path.join(base_dir, 'merged_all_metrics.csv')
cache_path = os.path.join(base_dir, 'yahoo_price_cache.json')
output_path = os.path.join(base_dir, 'financial_features_all.csv')
price_dir = os.path.join(base_dir, 'price_csv')
os.makedirs(price_dir, exist_ok=True)

# === 讀取資料 ===
df = pd.read_csv(csv_path)
df['year'] = df['年'].astype(int)
df['代號'] = df['代號'].astype(str).str.zfill(4)
df = df[df['year'] >= 2000]

# === 快取記憶體 ===
if os.path.exists(cache_path):
    with open(cache_path, 'r', encoding='utf-8') as f:
        price_cache = json.load(f)
else:
    price_cache = {}

# === 股票清單 ===
all_stock_ids = df['代號'].unique()
# price_cache = dict()
today_success_set = set()       # 今天成功抓到的股票
previous_success_set = set()    # 從昨天快取中讀到的所有股票代號

# 開頭這段才是對的
if os.path.exists(cache_path):
    with open(cache_path, 'r', encoding='utf-8') as f:
        price_cache = json.load(f)
    previous_success_set = set(k.split("_")[0] for k, v in price_cache.items() if v is not None)
else:
    price_cache = {}
    previous_success_set = set()


price_cache_lock = Lock()  # 🔒 保護共享資源

print("📦 抓取每檔股票 2000～2024加上2025YTD 所有收盤價...")
def fetch_and_process_stock(stock_id):
    global price_cache

    yf_id = f"{stock_id}.TW"
    price_csv_path = os.path.join(price_dir, f'Price_{stock_id}.csv')
    hist = None
    success = False

    # if os.path.exists(price_csv_path):
    #     try:
    #         hist = pd.read_csv(price_csv_path, parse_dates=['Date'], index_col='Date')
    #     except Exception:
    #         os.remove(price_csv_path)

    if hist is None:
        stock_years = df[df['代號'] == stock_id]['year']
        if stock_years.empty:
            return f"⚠️ {stock_id} 沒有財報資料，跳過"

        first_year = 2000
        last_year = 2025
        for end_year in range(last_year, first_year - 1, -1):
            for start_year in range(first_year, end_year + 1):
                start_date = f"{start_year}-01-01"
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
                        hist.reset_index().to_csv(price_csv_path, index=False)
                        success = True
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

    try:
        hist = hist if isinstance(hist, pd.DataFrame) else pd.read_csv(price_csv_path, parse_dates=['Date'], index_col='Date')
        available_years = hist.index.year.unique()
        earliest_year = int(available_years.min())
        if pd.isna(earliest_year):
            raise ValueError("No valid year")
    except Exception as e:
        return f"⚠️ {stock_id} 的 price_csv 無法判斷年份：{e}"

    local_cache = {}
    for year in range(earliest_year, 2025):
        key = f"{stock_id}_{year}"
        try:
            year_data = hist[str(year)]
            if not year_data.empty:
                last_close = year_data['Close'].iloc[-1].item()
                local_cache[key] = last_close
        except Exception:
            local_cache[key] = None

    with price_cache_lock:
        price_cache.update(local_cache)
        today_success_set.add(stock_id)  # ✅ 成功就加入今天成功的集合


    return f"✅ {stock_id} 資料處理完成"

# 🎯 多執行緒跑起來
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(fetch_and_process_stock, sid): sid for sid in all_stock_ids}
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        print(result)

# === 儲存快取（可續跑）===
with open(cache_path, 'w', encoding='utf-8') as f:
    json.dump(price_cache, f, ensure_ascii=False, indent=2)

# === 產生成功與失敗報告 ===
newly_fetched = today_success_set - previous_success_set
missing_today = previous_success_set - today_success_set

print(f"\n🆕 今天新增成功的股票（{len(newly_fetched)} 檔）: {sorted(newly_fetched)}")
print(f"⚠️ 有抓過但今天沒抓到的股票（{len(missing_today)} 檔）: {sorted(missing_today)}")
print(f"📊 總共抓取了 {len(price_cache)} 檔股票的收盤價")
# === 加入收盤價欄位到 df ===
df['closing_price_year'] = df.apply(
    lambda x: price_cache.get(f"{x['代號']}_{x['year']}"), axis=1
)

# ✅ 到這裡就可以接下來做指標運算囉～
print("🎉 所有股票歷史收盤價抓取完成，可進行特徵運算了喵♡")
# 💡 只保留有成功抓到收盤價的資料（防止後續爆錯）
df = df[df['closing_price_year'].notna()]


# === 計算特徵 ===
df['market_cap_mil'] = df['closing_price_year'] * df['發行量(萬張)'] * 1000
df['pb_ratio'] = df['closing_price_year'] / (df['股東權益總額(億)'] * 100 / df['發行量(萬張)'])
market_cap_mil = df['market_cap_mil']
revenue_mil = df['營收(億)'] * 100
df['ps_ratio'] = df['market_cap_mil'] / (df['營收(億)'] * 100)
df['roe_m'] = df['淨利率(%)'] / 100 * df['營收(億)'] / df['股東權益總額(億)']
df['opm'] = df['營益(億)'] / df['營收(億)']
df['npm'] = df['淨利率(%)'] / 100
df['roa'] = df['ROA(%)'] / 100
df['debt_to_equity'] = df['負債總額(億)'] / df['股東權益總額(億)']
df['current_ratio_m'] = df['流動資產對流動負債(%)'] / 100
df['quick_ratio_m'] = df['速動資產對流動負債(%)'] / 100
df['inventory_turnover_m'] = df['存貨週轉率']
df['ar_turnover_m'] = df['應收帳款週轉率']
df = df.sort_values(by=['代號', 'year'])
df['op_growth_m'] = df.groupby('代號')['營益(億)'].pct_change()
df['net_income_growth_m'] = df.groupby('代號')['淨利率(%)'].pct_change()
df['future_price'] = df.groupby('代號')['closing_price_year'].shift(-1)
df['return'] = (df['future_price'] / df['closing_price_year']) - 1
df['return_label'] = (df['return'] > 0).astype(int)
df.drop(columns=['future_price'], inplace=True)

# === 輸出結果 ===
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n🎉 收盤價計算完成：{output_path}，還自動快取了喵♡")
print("接下來前往merge_and_compute_features合併特徵工程的部分喵♡")
