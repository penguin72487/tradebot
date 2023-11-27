import requests
import time
import hmac
import hashlib
from dotenv import load_dotenv
import os

class BinanceAPI:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        self.base_url = 'https://testnet.binance.vision/api'

    def create_order(self, symbol, side, order_type, quantity, price):
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }

        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature

        headers = {
            'X-MBX-APIKEY': self.api_key,
        }

        response = requests.post(f'{self.base_url}/v3/order', headers=headers, params=params)
        return response.json()
    
    def get_account_info(self):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self._sign_request(params)
        response = requests.get(f'{self.base_url}/v3/account', headers=self._get_headers(), params=params)
        return response.json()

    def get_open_orders(self, symbol=None):
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        if symbol:
            params['symbol'] = symbol
        self._sign_request(params)
        response = requests.get(f'{self.base_url}/v3/openOrders', headers=self._get_headers(), params=params)
        return response.json()

    def _sign_request(self, params):
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature

    def _get_headers(self):
        return {
            'X-MBX-APIKEY': self.api_key,
        }
    
    def get_asset_balance(self, assets=['BTC', 'USDT']):
        account_info = self.get_account_info()
        balances = account_info.get('balances', [])
        filtered_balances = {asset: next((bal for bal in balances if bal['asset'] == asset), None) for asset in assets}
        return filtered_balances
    
    def get_open_orders_summary(self, symbol='BTCUSDT'):
        open_orders = self.get_open_orders(symbol)
        summary = []
        for order in open_orders:
            # 基本訂單信息
            order_info = {
                'orderId': order['orderId'],
                'price': order['price'],
                'quantity': order['origQty']
                # 止盈止損和浮盈浮虧需要額外的邏輯或計算
                # 'stop_loss': '需要實現',  # 這裡需要額外的實現
                # 'take_profit': '需要實現',  # 這裡需要額外的實現
                # 'unrealized_profit_loss': '需要實現'  # 這裡需要額外的實現
            }
            summary.append(order_info)
        return summary
    def cancel_order(self, symbol, order_id):
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self._sign_request(params)
        response = requests.delete(f'{self.base_url}/v3/order', headers=self._get_headers(), params=params)
        return response.json()
    
    def cancel_all_orders(self, symbol):
        params = {
            'symbol': symbol,
            'timestamp': int(time.time() * 1000),
            'recvWindow': 5000
        }
        self._sign_request(params)
        response = requests.delete(f'{self.base_url}/v3/openOrders', headers=self._get_headers(), params=params)
        return response.json()



# 使用範例
# api = BinanceAPI()
# response = api.create_order('BTCUSDT', 'BUY', 'LIMIT', '0.0100000', '19000')
# # print(response)
# account_info = api.get_asset_balance()
# print("Account Info:", account_info)

# open_orders = api.get_open_orders_summary()
# print("Open Orders:", open_orders)

# cancel_all_response = api.cancel_all_orders('BTCUSDT')
# print("Cancel All orders")