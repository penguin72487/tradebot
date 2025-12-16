import os
import pandas as pd
fp = r"c:\gitproject\tradebot\ML\TAE\BYBIT_BTCUSDT15.csv"
print('exists:', os.path.exists(fp))
try:
    df = pd.read_csv(fp)
    print('shape:', df.shape)
    print('first 10 columns:', df.columns.tolist()[:10])
    print('\nhead:\n', df.head(3).to_string())
except Exception as e:
    print('read_csv exception:', repr(e))
