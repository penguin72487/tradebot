import ccxt
import time
import csv
import os

# CSV 文件的名稱
CSV_FILE = "okx_prices.csv"

# 初始化 CSV 文件（如果文件不存在，則創建並添加標題）
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Spot Bid", "Spot Ask", "Perp Bid", "Perp Ask", "Bid Diff (%)", "Ask Diff (%)"])

def fetch_okx_prices():
    exchange = ccxt.okx()
    symbol_spot = 'BTC/USDT'
    symbol_perpetual = 'BTC/USDT:USDT'
    
    try:
        # 獲取現貨訂單簿
        order_book_spot = exchange.fetch_order_book(symbol_spot)
        spot_bid = order_book_spot['bids'][0][0]  # 現貨的買價
        spot_ask = order_book_spot['asks'][0][0]  # 現貨的賣價

        # 獲取永續合約訂單簿
        order_book_perpetual = exchange.fetch_order_book(symbol_perpetual)
        perp_bid = order_book_perpetual['bids'][0][0]  # 永續合約的買價
        perp_ask = order_book_perpetual['asks'][0][0]  # 永續合約的賣價

        # 計算價差百分比
        sell_diff_percent = (spot_ask - perp_bid) / spot_ask * 100 # 賣價差
        buy_diff_percent = (perp_ask - spot_bid) / perp_ask * 100 # 買價差
        sell_diff_percent = abs(sell_diff_percent)
        buy_diff_percent = abs(buy_diff_percent)

        # 紀錄數據到 CSV
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),  # 當前時間
                spot_bid,
                spot_ask,
                perp_bid,
                perp_ask,
                round(buy_diff_percent, 6),
                round(sell_diff_percent, 6)
            ])

        # 打印到控制台
        print(f"現貨買價: {spot_bid}, 永續賣價: {perp_bid}, 差異: {buy_diff_percent:.6f}%")
        print(f"現貨賣價: {spot_ask}, 永續買價: {perp_ask}, 差異: {sell_diff_percent:.6f}%\n")
    
    except Exception as e:
        print(f"獲取價格時發生錯誤: {e}")

print("開始執行，按 Ctrl+C 暫停程序。")

# 每秒執行 10 次（每 0.1 秒執行一次）
try:
    while True:
        fetch_okx_prices()
        # time.sleep(0.1)  # 暫停 0.1 秒
except KeyboardInterrupt:
    print("\n程序已暫停，隨時可重新啟動以繼續記錄。")
