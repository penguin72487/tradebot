# trading_strategy.py
import requests
import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

def get_historical_prices(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    prices = [float(kline[4]) for kline in data]  # Use the closing price
    return prices

def golden_cross_strategy(prices, short_window, long_window):
    short_sma = calculate_sma(prices, short_window)
    long_sma = calculate_sma(prices, long_window)

    if len(short_sma) > len(long_sma):
        short_sma = short_sma[-len(long_sma):]

    if short_sma[-1] > long_sma[-1] and short_sma[-2] < long_sma[-2]:
        return "BUY"
    elif short_sma[-1] < long_sma[-1] and short_sma[-2] > long_sma[-2]:
        return "SELL"
    else:
        return "HOLD"
