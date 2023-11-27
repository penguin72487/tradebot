from dotenv import load_dotenv
from binance.spot import Spot as Client
import pandas as pd
import ta
import os

# 加载 .env 文件
load_dotenv()

# 从 .env 文件中获取环境变量
api_key = os.getenv('API_KEY')
api_secret = os.getenv('API_SECRET')

# 使用测试网络
base_url = 'https://testnet.binance.vision'  # 测试网络
# base_url = 'https://api3.binance.com'  # 实际网络

client = Client(api_key, api_secret, base_url=base_url)
try:
    account_info = client.account()
except Exception as e:
    print(f"Error occurred: {e}")
# 获取比特币的历史数据
klines = client.klines("BTCUSDT", "15m", limit=1000)
# account_info = request_client.get_account_information()
account_info = client.account()



# 創建 DataFrame
columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
df = pd.DataFrame(klines, columns=columns)
df['close'] = pd.to_numeric(df['close'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['volume'] = pd.to_numeric(df['volume'])

# 計算 EMA 和 ATR
df['ema'] = ta.trend.ema_indicator(df['close'], window=25)
df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

# 計算 Fibonacci 回撤級別
def calculate_fib_levels(high, low):
    levels = {
        'fib236': high - (high - low) * 0.236,
        'fib382': high - (high - low) * 0.382,
        'fib500': high - (high - low) * 0.500,
        'fib618': high - (high - low) * 0.618,
        'fib786': high - (high - low) * 0.786,
        'fib1': high,
        'fib1618': high + (high - low) * 0.618,
        'fib2618': high + (high - low) * 1.618
    }
    return levels

high = df['high'].rolling(window=75).max()
low = df['low'].rolling(window=25).min()
fib_levels = calculate_fib_levels(high, low)
for level in fib_levels:
    df[level] = fib_levels[level]

# 創建交易條件的布林序列
long_condition_series = ((df['close'] > df['fib236']) | (df['close'] > df['fib382']) | (df['close'] > df['fib500']) | (df['close'] > df['fib618']) | (df['close'] > df['fib786']) | (df['close'] > df['fib1']) | (df['close'] > df['fib1618']) | (df['close'] > df['fib2618'])) & (df['volume'] > df['volume'].rolling(window=75).mean() * 1.5) & (df['close'] > df['ema'])

short_condition_series = ((df['close'] < df['fib236']) | (df['close'] < df['fib382']) | (df['close'] < df['fib500']) | (df['close'] < df['fib618']) | (df['close'] < df['fib786']) | (df['close'] < df['fib1']) | (df['close'] < df['fib1618']) | (df['close'] < df['fib2618'])) & (df['volume'] > df['volume'].rolling(window=75).mean() * 1.5) & (df['close'] < df['ema'])

leverage = 50
inital_equity = initial_equity = float(account_info.totalWalletBalance)*0.01

# 設定交易數量


# 執行交易邏輯
for index, row in df.iterrows():
    trade_quantity = initial_equity * (0.5 / 100) * leverage / df['atr'][index]
    if long_condition_series[index]:
        # 执行买入操作
        result = client.new_order(symbol="BTCUSDT", side="BUY", type="MARKET", quantity=trade_quantity)
    elif short_condition_series[index]:
        # 执行卖出操作
        result = client.new_order(symbol="BTCUSDT", side="SELL", type="MARKET", quantity=trade_quantity)