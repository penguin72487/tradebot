import requests
import time
import hmac
import hashlib
from dotenv import load_dotenv
import pandas as pd
import os

class BinanceAPI:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        self.base_url = 'https://testnet.binancefuture.com'  # Change to futures testnet URL

    def __sign_request(self, params):
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature

    def __get_headers(self):
        return {
            'X-MBX-APIKEY': self.api_key,
        }
    
    def ping(self):
        response = requests.get(f'{self.base_url}/fapi/v1/ping')  # Change to futures endpoint
        return response.json()
      
    def __get_server_time(self):
        response = requests.get(f'{self.base_url}/fapi/v1/time')  # Change to futures endpoint
        return response.json()
    
    def __get_Klines(self, symbol, interval):
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.get(f'{self.base_url}/fapi/v1/klines', headers=headers, params=params)  # Change to futures endpoint
        return response.json()
    
    def __get_historical_data(self, symbol, interval, limit):
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.get(f'{self.base_url}/fapi/v1/klines', headers=headers, params=params)  # Change to futures endpoint
        return response.json()
    
    def __get_account_info(self):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.get(f'{self.base_url}/fapi/v2/account', headers=headers, params=params)  # Change to futures endpoint
        return response.json()
    
    def __get_open_orders(self, symbol):
        params = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.get(f'{self.base_url}/fapi/v1/openOrders', headers=headers, params=params)
        return response.json()
    
    def __get_current_positions(self):
        account_info = self.__get_account_info()
        return account_info['positions']
    
    def __create_order(self, symbol, side, order_type, quantity, price):
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.post(f'{self.base_url}/fapi/v1/order', headers=headers, params=params)
        return response.json()
    
    def __cancel_order(self, symbol, order_id):
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.delete(f'{self.base_url}/fapi/v1/order', headers=headers, params=params)
        return response.json()
    
    def __get_leverage(self, symbol):
        params = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.get(f'{self.base_url}/fapi/v1/leverageBracket', headers=headers, params=params)
        return response.json()


    def __change_leverage(self, symbol, leverage):
        params = {
            'symbol': symbol,
            'leverage': leverage,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.post(f'{self.base_url}/fapi/v1/leverage', headers=headers, params=params)
        return response.json()
        
    def create_order_with_stop(self, symbol, side, order_type, quantity, price, stopPrice):
        responsec = self.__create_order(symbol, side, order_type, quantity, price)
        responses = self.__create_order(symbol, side,'LIMIT' , quantity, stopPrice)
        return responsec, responses
    
    def cancel_All_open_orders(self, symbol):
        orders = self.__get_open_orders(symbol)
        for order in orders:
            self.__cancel_order(symbol, order['orderId'])
    
    def get_historical_data(api, symbol, interval, limit):
        data = api.__get_historical_data(symbol, interval, limit)
        df = pd.DataFrame(data)
        df.columns = ['open_time',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume',
                    'close_time',
                    'quote_asset_volume',
                    'number_of_trades',
                    'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume',
                    'ignore']
        # 轉換日期時間列
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        # 只轉換數值列為float
        for col in ['open', 'high', 'low', 'close', 'volume', 
                    'quote_asset_volume', 'number_of_trades', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
            df[col] = df[col].astype(float)
        return df
    
    def get_account_info(self):
        account_info = self.__get_account_info()
        formatted_info = f"Account Information:\n"
        for key, value in account_info.items():
            formatted_info += f"{key}: {value}\n"
        return formatted_info
    
    def get_futures_balance(self, asset='USDT'):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self.__sign_request(params)
        headers = self.__get_headers()
        response = requests.get(f'{self.base_url}/fapi/v2/balance', headers=headers, params=params)
        balances = response.json()
        for balance in balances:
            if balance['asset'] == asset:
                return balance['balance']
        return None

    def get_open_orders(self, symbol):
        open_orders = self.__get_open_orders(symbol)
        formatted_orders = f"Open Orders for {symbol}:\n"
        for order in open_orders:
            formatted_orders += f"Order ID: {order['orderId']}, Symbol: {order['symbol']}, Status: {order['status']}\n"
        return formatted_orders
        


    def get_current_positions(self):
        positions = self.__get_current_positions()
        formatted_positions = "Current Positions:\n"
        for position in positions:
            if float(position['free']) > 0.0 or float(position['locked']) > 0.0:
                formatted_positions += f"Asset: {position['asset']}, Free: {position['free']}, Locked: {position['locked']}\n"
        return formatted_positions
    
    def get_current_positions_dict(self):
        positions = self.__get_current_positions()
        formatted_positions = {}
        for position in positions:
            if float(position['entryPrice']) > 0.0:
                formatted_positions[position['symbol']] = position['positionAmt']
        return formatted_positions
    
    def get_leverage(self, symbol):
        leverage = self.__get_leverage(symbol)
        return leverage
    
    def change_leverage(self, symbol, leverage):
        leverage = self.__change_leverage(symbol, leverage)
        return leverage
    
    def get_maxNotionalCap(self, symbol):
        notionalCap = self.__get_leverage(symbol)
        return notionalCap

    
   