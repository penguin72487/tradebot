import csv
import os
from typing import List, Dict
from datetime import datetime

class Kbar:
    def __init__(self, time: int, data: List[float]):
        self.time = time  # UNIX time
        self.data = data

    def __repr__(self):
        dt = datetime.fromtimestamp(self.time)
        return f"Kbar(time {dt.strftime('%Y-%m-%d')}, price {self.data})"
    

class Security:
    def __init__(self, symbol: str, start: int, end: int, kbar: Dict[int, Kbar]):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.kbar = kbar  # key is UNIX time

class SimuExchange:
    def __init__(self, min_time: int = 995850000):
        base_path = os.path.abspath("FinTechML/Dataset")
        files = {
            "TSMC": "TWSE_DLY_2330, 1D.csv",
            "MediaTek": "TWSE_DLY_2454, 1D.csv",
        }

        securities = {}
        for symbol, filename in files.items():
            path = os.path.join(base_path, filename)
            sec = self._load_security_from_csv(symbol, path, min_time=min_time)
            securities[symbol] = sec

        self.securities = securities

        # 這裡可以做後續預處理（像時間對齊）喵～

    def query(self, symbol: str, start: int, end: int):
        sec = self.securities.get(symbol)
        if not sec:
            return []
        return [bar for t, bar in sec.kbar.items() if start <= t <= end]

    def query_by_time(self, symbol: str, time: int):
        sec = self.securities.get(symbol)
        return sec.kbar.get(time) if sec else None

    @staticmethod
    def _load_security_from_csv(symbol: str, path: str, min_time: int) -> Security:
        kbars = {}
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time = int(row['time'])
                if time < min_time:
                    continue

                data = [
                    float(row['close']),  # 這裡你目前只用 close
                ]
                kbars[time] = Kbar(time, data)

        if not kbars:
            raise ValueError(f"No data after {min_time} for {symbol}")

        start = min(kbars.keys())
        end = max(kbars.keys())
        return Security(symbol=symbol, start=start, end=end, kbar=kbars)


        

# 初始化交易所
exchange = SimuExchange()

for symbol, sec in exchange.securities.items():
    print(f"Symbol: {symbol}, Start: {datetime.fromtimestamp(sec.start)}, End: {datetime.fromtimestamp(sec.end)}")
    for time, kbar in sec.kbar.items():
        print(f"  Time: {datetime.fromtimestamp(time)}, Data: {kbar.data}")

print(exchange.query("TSMC", 995850000, 1000000000))
print(exchange.query_by_time("TSMC", 995850000))
