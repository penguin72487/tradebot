import BinanceAPI
import time
import numpy as np
import talib

class TradingStrategy:
    def __init__(self, api, symbol, short_term_interval, long_term_interval):
        self.api = api
        self.funds = self.api.get_futures_balance()
        
        self.tradePercent = 5 / 100
        self.fibHighLength = 75
        self.fibLowLength = 25
        self.forceStopLossPercent = 0.5 / 100
        self.forceTakeProfitPercent = 1.5 / 100
        self.leverage = 1
        self.atrLength = 14
        self.atrFilter = 0
        self.pre_trade_Volume = 0
        self.pre_trade_Volume_Fillter = 5/100 # 50% of pre_trade_Volume
        data = self.get_historical_data(symbol, long_term_interval, self.fibHighLength)
        self.pre_fibStatus = self.get_fib_status(data)# 0 <=0 1 0~0.236 2 0.236~0.382 3 0.382~0.5 4 0.5~0.618 5 0.618~0.786 6 0.786~0.886 7 0.886~1 8 >=1
        self.tradeStatus = 0 # 0 空手 1 買入 -1 賣出
        self.avgVolume = self.get_avgVolume(self.fibHighLength)
        self.avgVolume_Fillter = 50/100 # 50% of avgVolume

    def get_historical_data(self, symbol, interval, lookback):
        # 根據不同的時間間隔獲取歷史數據
        data = self.api.get_historical_data(symbol, interval, lookback)
        return np.array(data)

    def calculate_atr(self, data):
        # 計算 ATR
        high = data[:, 2]
        low = data[:, 3]
        close = data[:, 4]
        atr = talib.ATR(high, low, close, timeperiod=self.atrLength)
        return atr

    def calculate_fib_levels(self, data):
        # 計算 Fibonacci 水平
        high = np.max(data[:, 2][-self.fibHighLength:])
        low = np.min(data[:, 3][-self.fibLowLength:])
        fib_range = high - low
        fib_levels = {
            'fib0': low,
            'fib236': high - fib_range * 0.236,
            'fib382': high - fib_range * 0.382,
            'fib5': high - fib_range * 0.5,
            'fib618': high - fib_range * 0.618,
            'fib786': high - fib_range * 0.786,
            'fib886': high - fib_range * 0.886,
            'fib1': high
        }
        return fib_levels
    
    def calculate_fib_Price(self, data):
        # 計算 Fibonacci 水平
        high = np.max(data[:, 2][-self.fibHighLength:])
        low = np.min(data[:, 3][-self.fibLowLength:])
        fib_range = high - low
        fib_Price = []
        fib_Price.append(low)// 0
        fib_Price.append(high - fib_range * 0.236) 
        fib_Price.append(high - fib_range * 0.382)
        fib_Price.append(high - fib_range * 0.5)
        fib_Price.append(high - fib_range * 0.618)
        fib_Price.append(high - fib_range * 0.786)
        fib_Price.append(high)
        return fib_Price

    def get_fib_status(self, data):
        # 獲取 Fibonacci 狀態
        current_price = data[-1, 4]
        fib_levels = self.calculate_fib_levels(data)
        fib_status = 0
        if current_price >= fib_levels['fib1']:
            fib_status = 8
        elif current_price >= fib_levels['fib786']:
            fib_status = 7
        elif current_price >= fib_levels['fib618']:
            fib_status = 6
        elif current_price >= fib_levels['fib5']:
            fib_status = 5
        elif current_price >= fib_levels['fib382']:
            fib_status = 4
        elif current_price >= fib_levels['fib236']:
            fib_status = 3
        elif current_price >= fib_levels['fib236']:
            fib_status = 2
        elif current_price >= fib_levels['fib0']:
            fib_status = 1
        else: 
            fib_status = 0

        return fib_status

    def get_avgVolume(self, lookback):
        # 獲取平均成交量
        data = self.get_historical_data('BTCUSDT', '15m', lookback)
        volume = data[:, 5]
        avgVolume = np.mean(volume)
        return avgVolume


    def check_trade_conditions(self, data, atr, fib_levels):
        # 定義交易條件
        current_price = data[-1, 4]
        current_volume = data[-1, 5]
        now_fibStatus = self.get_fib_status(data)
        fibsStatus_V= now_fibStatus - self.pre_fibStatus

        # 判斷是否有交易機會
        if self.tradeStatus == 0:
            if fibsStatus_V>0 and current_volume > self.avgVolume * self.avgVolume_Fillter:
                self.tradeStatus = 1
                self.pre_trade_Volume = current_volume
                return 'BUY'
            elif fibsStatus_V<0 and current_volume > self.avgVolume * self.avgVolume_Fillter:
                self.tradeStatus = -1
                self.pre_trade_Volume = current_volume
                return 'SELL'
            else:
                return None
            
        elif self.tradeStatus == 1:
            if fibsStatus_V > 0 and current_volume >=self.pre_trade_Volume*self.pre_trade_Volume_Fillter:
                self.pre_trade_Volume = current_volume
                return 'BUY'
            elif fibsStatus_V >0 and current_volume <self.pre_trade_Volume*self.pre_trade_Volume_Fillter:
                
                return 'stop'
                # return 'trailingStop'
        
        elif self.tradeStatus == -1:
            if fibsStatus_V < 0 and current_volume >=self.pre_trade_Volume*self.pre_trade_Volume_Fillter:
                self.pre_trade_Volume = current_volume
                return 'SELL'
            elif fibsStatus_V <0 and current_volume <self.pre_trade_Volume*self.pre_trade_Volume_Fillter:
                
                return 'stop'
                # return 'trailingStop'
                
