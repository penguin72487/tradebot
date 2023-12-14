import requests
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
import os
import time

# Binance API設置
# API_URL = "https://api.binance.com/fapi/v1/klines"
API_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"  # 你可以更改為你想要的交易對
INTERVAL = "1m"     # K線間隔
load_dotenv()

# MySQL連接設置
db = mysql.connector.connect(
    host="127.0.0.1",
    user=os.getenv('MYSQL_USER'),
    password=os.getenv('MYSQL_PASSWORD'),
    database="binanceKline"
)
cursor = db.cursor()

def fetch_klines(symbol, interval, limit=1500):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1500  # 可以根據需要更改
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def fetch_klines(symbol, interval, start_time, end_time, limit=1500):
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,  # 指定開始時間（毫秒）
        'endTime': end_time,      # 指定結束時間（毫秒）
        'limit': limit             # 可以根據需要更改
    }
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def insert_kline(kline):
    sql = (
        "INSERT INTO binanceKline_By_Symbol "
        "(symbol, openTime, openPrice, highPrice, lowPrice, closePrice, volume, closeTime, "
        "quoteAssetVolume, numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, bin_Ignore) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    cursor.execute(sql, kline)
    db.commit()

def insert_customer_index(symbol, openTime, ATR210, volume15, avgVolume15):
    sql = (
        "INSERT INTO customer_Index "
        "(symbol, openTime, ATR210, volume15, avgVolume15) "
        "VALUES (%s, %s, %s, %s, %s)"
    )
    cursor.execute(sql, (symbol, openTime, ATR210, volume15, avgVolume15))
    db.commit()

def insert_binance(symbol, startTime, endTime):
    sql = (
        "INSERT INTO binanceKline "
        "(symbol, startTime, endTime) "
        "VALUES (%s, %s, %s)"
    )
    cursor.execute(sql, (symbol, startTime, endTime))
    db.commit()

def modify_binance(symbol, startTime, endTime):
    sql = (
        "UPDATE binanceKline "
        "SET endTime = %s "
        "WHERE symbol = %s AND startTime = %s"
    )
    cursor.execute(sql, (endTime, symbol, startTime))
    db.commit()
     

def main():
    # 資料庫連接...

    begin_time = datetime(2023, 1, 1)  # 設定起始時間
    cursor.execute("SELECT MAX(startTime) FROM binanceKline WHERE symbol = %s", (SYMBOL,))
    start_time = cursor.fetchone()[0]  # 從資料庫獲取最新的K線時間
    if start_time is None:
        # 如果資料庫中沒有K線，則從預設的開始時間開始獲取
        start_time = int(begin_time.timestamp() * 1000)
        insert_binance(SYMBOL, start_time, start_time)
        while start_time < int(datetime.now().timestamp() * 1000):
            end_time = start_time + 60 * 1000 * 1500
            klines = fetch_klines(SYMBOL, INTERVAL, start_time, end_time)
            if klines is None:
                print("無法獲取K線")
                break

            for kline in klines:
                # print("API data:", kline)
                insert_data = (SYMBOL, kline[0], kline[1], kline[2], kline[3], kline[4], kline[5], kline[6], kline[7], kline[8], kline[9], kline[10], kline[11])
                # print("Insert data:", insert_data)
                insert_kline(insert_data)

            modify_binance(SYMBOL, start_time, end_time)
            start_time = klines[-1][6]
            print("已獲取", datetime.fromtimestamp(start_time / 1000))
            cursor.execute("SELECT MAX(endTime) FROM binanceKline WHERE symbol = %s", (SYMBOL,))
            end_time = cursor.fetchone()[0]
            print("最新K線時間", datetime.fromtimestamp(end_time / 1000))
            time.sleep(0.5)
    else:
        # 如果資料庫中已經有K線，則從最新的K線時間開始獲取
        start_time += 60 * 1000
        while start_time < int(datetime.now().timestamp() * 1000):
            end_time = start_time + 60 * 1000 * 1500
            klines = fetch_klines(SYMBOL, INTERVAL, start_time, end_time)
            if klines is None:
                print("無法獲取K線")
                break
            for kline in klines:
                # print("API data:", kline)
                insert_data = (SYMBOL, kline[0], kline[1], kline[2], kline[3], kline[4], kline[5], kline[6], kline[7], kline[8], kline[9], kline[10], kline[11])
                # print("Insert data:", insert_data)
                insert_kline(insert_data)
                modify_binance(SYMBOL, start_time, kline[6])
            start_time = klines[-1][6]
            print("已獲取", datetime.fromtimestamp(start_time / 1000))
            time.sleep(0.5)
    # 計算指標並更新 customer_Index 表
    # calculate_indicators(cursor)
    # 資料庫斷開連接...

    
    


    
        
    # 持續更新
    # while True:
    #     update_latest_klines(cursor, SYMBOL)
    #     calculate_indicators(cursor)  # 計算指標並更新 customer_Index 表
    #     time.sleep(60)  # 等待60秒

    cursor.close()
    db.close()

if __name__ == "__main__":
    main()