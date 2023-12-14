from BinanceAPI import BinanceAPI
import pandas as pd
import time
def calculate_moving_averages(df, short_window, long_window):
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    return df

def check_golden_cross(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    golden_cross = latest['short_ma'] > latest['long_ma'] and previous['short_ma'] < previous['long_ma']
    death_cross = latest['short_ma'] < latest['long_ma'] and previous['short_ma'] > previous['long_ma']
    
    return golden_cross, death_cross

def execute_trade(api, symbol, golden_cross, death_cross):
    if golden_cross:
        print("黃金交叉發生，執行買入")
        # api.create_order(symbol, 'BUY', 'LIMIT', quantity, price)
    elif death_cross:
        print("死亡交叉發生，執行賣出")
        # api.create_order(symbol, 'SELL', 'LIMIT', quantity, price)
    else:
        print("沒有交叉事件，不執行操作")

def main():
    api = BinanceAPI()
    symbol = 'BTCUSDT'
    df = api.get_historical_data(symbol, '1d', 300) # 調整為合適的歷史數據範圍
    df = pd.DataFrame(df)
    df = calculate_moving_averages(df, 50, 200)

    golden_cross, death_cross = check_golden_cross(df)
    execute_trade(api, symbol, golden_cross, death_cross)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(1)


