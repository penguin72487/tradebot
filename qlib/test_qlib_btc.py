import qlib
from qlib.data import D
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
provider_uri = os.path.join(base_dir, "qlib_data_btc")

qlib.init(provider_uri=provider_uri, region="cn")

print("Calendar sample:", D.calendar(freq="15min")[:5])

print("Instruments:", D.instruments(market="all")[:5])

df = D.features(
    instruments=["BTCUSDT"],
    fields=["$close", "$volume"],  # 注意前面有 $
    start_time="2024-01-01",
    end_time="2100-01-01",
    freq="15min",
)

print(df.head())
print(df.tail())