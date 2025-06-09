import SimuExchange
from SimuExchange import SimuExchange, Kbar, Security
from typing import List, Dict
from datetime import datetime
from Records import Transaction, Record, Records



class Position:
    def __init__(self, symbol: str, position: float):
        self.symbol = symbol
        self.size = position

    def __repr__(self):
        return f"Position(symbol={self.symbol}, position={self.position})"

class TestPlatform:
    def __init__(self, exchange: SimuExchange, initial_captial : float, records: Records):
        self.exchange = exchange # 交易所
        self.initial_captial  = initial_captial # 初始資金
        self.records = records # 交易紀錄
        self.positions = {}    # 持有的部位，key 是 symbol, value 是 Position 物件
        self.cash = initial_captial
        self.commission = 0 # 交易手續費
        self.TimeSet = set()  # 先用 set 去重喵～

        for symbol, sec in self.exchange.securities.items():
            self.TimeSet.update(sec.kbar.keys())  # 加進所有時間點

        self.TimeSet = sorted(self.TimeSet)  # 排序後變成 list
        self.TimeSetIndex = 0
        self.Now = self.TimeSet[self.TimeSetIndex]
        # self.Now = 995850000 # 開測時間



        

    def buy(self, symbol: str, cash: float):
        # 購買股票，symbol 是股票代碼，cash 是購買金額
        # 取得當前價格
        kbar = self.exchange.query_by_time(symbol, self.Now)
        if not kbar:
            raise ValueError(f"No data for {symbol} at time {self.Now}")
        
        price = kbar.data[0]
        # 計算購買數量
        # 手續費
        commission = cash * self.commission
        cash -= commission
        qty = cash / price
        # 更新持倉
        if symbol in self.positions:
            self.positions[symbol].size += qty
        else:
            self.positions[symbol] = Position(symbol, qty)
        # 更新現金
        self.cash -= cash
        # 更新交易紀錄
        transaction = Transaction(self.Now, qty, self.positions[symbol].size, price, cash)
        self.records.transactions.append(transaction)
    
    def sell(self, symbol: str, cash: float):
        # 賣出股票，symbol 是股票代碼，cash 是賣出金額
        
        # 取得當前價格
        kbar = self.exchange.query_by_time(symbol, self.Now)
        if not kbar:
            raise ValueError(f"No data for {symbol} at time {self.Now}")
        
        price = kbar.data[0]
        # 計算賣出數量
        commission = cash * self.commission
        cash -= commission
        qty = cash / price
        
        # 更新持倉
        self.positions[symbol].size -= qty
        if self.positions[symbol].size <= 0:
            del self.positions[symbol]
        
        # 更新現金
        self.cash += cash
        
        # 更新交易紀錄
        transaction = Transaction(self.Now, -qty, self.positions[symbol].size, price, cash)
        self.records.transactions.append(transaction)

    def flatten(self):
        # 平倉所有部位
        for symbol, pos in self.positions.items():
            kbar = self.exchange.query_by_time(symbol, self.Now)
            if not kbar:
                continue
            price = kbar.data[0]
            cash = pos.size * price
            commission = cash * self.commission
            cash -= commission
            self.cash += cash
            transaction = Transaction(self.Now, -pos.size, 0, price, cash)
            self.records.transactions.append(transaction)
        
        self.positions.clear()


    def next(self):
        self.TimeSetIndex += 1
        if self.TimeSetIndex >= len(self.TimeSet):
            raise IndexError("已經沒有下一根K棒了喵~")
        self.Now = self.TimeSet[self.TimeSetIndex]

    def qty(self):
        # 取得目前持有的部位價值總和
        total_value = 0
        for symbol, pos in self.positions.items():
            kbar = self.exchange.query_by_time(symbol, self.Now)
            if not kbar:
                continue
            price = kbar.data[0]
            total_value += pos.size * price
        return total_value + self.cash
    

    def report(self):
        # 報告目前的資金狀況
        print(f"目前時間: {datetime.fromtimestamp(self.Now).strftime('%Y-%m-%d')}")
        print(f"現金: {self.cash}")
        print(f"持有部位: {self.positions}")
        print(f"資產總值: {self.qty()}")
    def __repr__(self):
        return f"TestPlatform(exchange={self.exchange}, initial_capital={self.initial_captial}, records={self.records})"
# 測試 TestPlatform
    
    

        

exchange = SimuExchange()
record = Record(995850000, 1000000, 0.001, []) 
TestPlatform = TestPlatform(exchange, 1000000, Records([]))
