import requests
import time
import hmac
import hashlib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up authentication
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

# Set up the request parameters
params = {
    'symbol':       'BTCUSDT',
    'side':         'SELL',
    'type':         'LIMIT',
    'timeInForce':  'GTC',
    'quantity':     '1.0000000',
    'price':        '19000',
    'timestamp':    int(time.time() * 1000),  # UNIX timestamp in milliseconds
    'recvWindow':   5000  # Adding recvWindow parameter
}

# Create a query string
query_string = '&'.join([f"{k}={v}" for k, v in params.items()])

# Sign the request
signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
params['signature'] = signature

# Set the headers
headers = {
    'X-MBX-APIKEY': API_KEY,
}

# Send the request
url = 'https://testnet.binance.vision/api/v3/order'
response = requests.post(url, headers=headers, params=params)

print(response.json())
