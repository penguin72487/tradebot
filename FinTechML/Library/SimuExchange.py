from typing import List, Dict
from datetime import datetime

class Kbar:
    def __init__(self, time: int, data: List[float]):
        self.time = time  # UNIX time
        self.data = data

class Security:
    def __init__(self, symbol: str, start: datetime, end: datetime, kbar: Dict[int, Kbar]):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.kbar = kbar  # key is UNIX time

class SimuExchange:
    def __init__(self, securities: Dict[str, Security]):
        self.securities = securities

    def query(self, symbol: str, start: datetime, end: datetime):
        # 可以根據 symbol 與時間區間查詢
        pass

    def query_by_time(self, symbol: str, time: datetime):
        # 可以根據 symbol 與某個時間點查詢
        pass