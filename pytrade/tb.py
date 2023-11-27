
import requests
import time
import hmac
import hashlib
from dotenv import load_dotenv
import os
from goldcorse import golden_cross_strategy, get_historical_prices

# Function to create a signature
def create_signature(query_string, api_secret):
    return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

# Function to send a request
def send_request(url, headers, params):
    response = requests.post(url, headers=headers, params=params)
    print(response.status_code)
    print(response.text)
    return response.json()


def trading_strategy():
    symbol = 'BTCUSDT'
    interval = '1d'
    limit = 500

    prices = get_historical_prices(symbol, interval, limit)
    action = golden_cross_strategy(prices, short_window=50, long_window=200)
    quantity = 1.0000000  # Define your own logic for quantity

    return action, quantity
# Main function
def main():
    # Load environment variables
    load_dotenv()

    # Set up authentication
    API_KEY = os.getenv('API_KEY')
    API_SECRET = os.getenv('API_SECRET')

    # Call the trading strategy
    action, quantity = trading_strategy()

    # Set up the request parameters
    params = {
        'symbol':       'BTCUSDT',
        'side':         action,
        'type':         'LIMIT',
        'timeInForce':  'GTC',
        'quantity':     str(quantity),
        'price':        '0.20',  # You may want to calculate this dynamically
        'timestamp':    int(time.time() * 1000) # UNIX timestamp in milliseconds
    }

    # Create a query string
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])

    # Sign the request
    signature = create_signature(query_string, API_SECRET)
    params['signature'] = signature

    # Set the headers
    headers = {
        'X-MBX-APIKEY': API_KEY,
    }

    # Send the request
    url = 'https://testnet.binance.vision/api/v3/order'
    response = send_request(url, headers, params)

    print(response)

if __name__ == "__main__":
    main()
