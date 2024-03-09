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

def calculate_atr(klines, period=210):
    tr_list = []
    for i in range(1, len(klines)):
        high = float(klines[i][2])
        low = float(klines[i][3])
        close_prev = float(klines[i - 1][4])
        tr = max(high - low, abs(high - close_prev), abs(close_prev - low))
        tr_list.append(tr)
    
    atr = np.mean(tr_list[-period:]) if len(tr_list) >= period else np.nan
    return atr

def calculate_volume_stats(klines, period=15):
    volume_list = [float(kline[5]) for kline in klines[-period:]]
    volume15 = sum(volume_list)
    avgVolume15 = np.mean(volume_list) if volume_list else 0
    return volume15, avgVolume15

     

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

            # modify_binance(SYMBOL, start_time, end_time)
            start_time = klines[-1][6]
            print("已獲取", datetime.fromtimestamp(start_time / 1000))
            cursor.execute("SELECT MAX(endTime) FROM binanceKline WHERE symbol = %s", (SYMBOL,))
            end_time = cursor.fetchone()[0]
            print("最新K線時間", datetime.fromtimestamp(end_time / 1000))
            time.sleep(0.5)
    else:
        # 如果資料庫中已經有K線，則從最新的K線時間開始獲取
        while start_time < int(begin_time.timestamp() * 1000):
            end_time = start_time + 60 * 1000 * 1500
            klines = fetch_klines(SYMBOL, INTERVAL, start_time, end_time)
            if klines is None:
                print("已經獲取到最新的K線")
                break
            for kline in klines:
                # print("API data:", kline)
                insert_data = (SYMBOL, kline[0], kline[1], kline[2], kline[3], kline[4], kline[5], kline[6], kline[7], kline[8], kline[9], kline[10], kline[11])
                # print("Insert data:", insert_data)
                insert_kline(insert_data)
                # modify_binance(SYMBOL, start_time, kline[6])
            start_time = klines[-1][6]
            print("已獲取", datetime.fromtimestamp(start_time / 1000))
            time.sleep(0.5)

        cursor.execute("SELECT MAX(endTime) FROM binanceKline WHERE symbol = %s", (SYMBOL,))
        last_time = cursor.fetchone()[0]

        while last_time < int(datetime.now().timestamp() * 1000):
            end_time = last_time + 60 * 1000 * 1500
            limit = 1500
            if end_time > int(datetime.now().timestamp() * 1000):
                end_time = int(datetime.now().timestamp() * 1000)
                limit = int((end_time - last_time) / (60 * 1000))
            klines = fetch_klines(SYMBOL, INTERVAL, last_time, end_time, limit)
            if klines is None:
                print("無法獲取K線")
                break
            for kline in klines:
                # print("API data:", kline)
                insert_data = (SYMBOL, kline[0], kline[1], kline[2], kline[3], kline[4], kline[5], kline[6], kline[7], kline[8], kline[9], kline[10], kline[11])
                # print("Insert data:", insert_data)
                insert_kline(insert_data)
                # modify_binance(SYMBOL, start_time, kline[6])
            last_time = klines[-1][6]
            print("已獲取", datetime.fromtimestamp(last_time / 1000))
            time.sleep(0.5)

    # couculate index
    cursor.execute("SELECT * FROM binanceKline_By_Symbol WHERE symbol = %s", (SYMBOL,))
    klines = cursor.fetchall()
    for kline in klines:
        # print("API data:", kline)
        insert_data = (SYMBOL, kline[1], kline[2], kline[3], kline[4], kline[5], kline[6], kline[7], kline[8], kline[9], kline[10], kline[11], kline[12])
        # print("Insert data:", insert_data)
        insert_customer_index(insert_data)
        # modify_binance(SYMBOL, start_time, kline[6])
    print("已獲取", datetime.fromtimestamp(klines[-1][1] / 1000))
    time.sleep(0.5)


    #       
    while True:
        now = datetime.now()
        if now.second == 0:
            cursor.execute("SELECT MAX(openTime) FROM customer_Index WHERE symbol = %s", (SYMBOL,))
            last_time = cursor.fetchone()[0]
            if last_time is None:
                last_time = int(begin_time.timestamp() * 1000)
            cursor.execute("SELECT MAX(openTime) FROM binanceKline_By_Symbol WHERE symbol = %s", (SYMBOL,))
            end_time = cursor.fetchone()[0]
            if end_time is None:
                end_time = int(begin_time.timestamp() * 1000)
            if last_time < end_time:
                cursor.execute("SELECT * FROM binanceKline_By_Symbol WHERE symbol = %s AND openTime > %s", (SYMBOL, last_time))
                klines = cursor.fetchall()
                # print(klines)
                for kline in klines:
                    # print("API data:", kline)
                    insert_data = (SYMBOL, kline[1], kline[2], kline[3], kline[4], kline[5], kline[6], kline[7], kline[8], kline[9], kline[10], kline[11], kline[12])
                    # print("Insert data:", insert_data)
                    insert_customer_index(insert_data)
                    # modify_binance(SYMBOL, start_time, kline[6])
                print("已獲取", datetime.fromtimestamp(klines[-1][1] / 1000))
                time.sleep(0.5)
            else:
                print("無需更新")
        

    cursor.close()
    db.close()

if __name__ == "__main__":
    main()