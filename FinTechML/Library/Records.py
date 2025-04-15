from typing import List, Dict
from datetime import datetime
import csv
import os

class Transaction:
    def __init__(self, time: int, buy: float, position: float, price: float, cash: float):
        self.time = time  # UNIX time
        self.buy = buy
        self.position = position
        self.price = price
        self.cash = cash

class Record:
    def __init__(self, start: int, end: int, initial_capital: float, commission: float, transactions: List[Transaction]):
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.commission = commission
        self.transactions = transactions

    def report(self):
        # 計算 qty_over_time, sharp, Max_draw_down, TV 等等
        pass

    @staticmethod
    def from_csv(path: str) -> 'Record':
        transactions = []
        initial_capital = 0.0
        commission = 0.0
        start = end = 0

        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                time = int(row['time'])
                buy = float(row['buy'])
                position = float(row['position'])
                price = float(row['price'])
                cash = float(row['cash'])

                # 第一筆資料中抓 initial_capital, commission
                if i == 0:
                    initial_capital = float(row['initial_capital'])
                    commission = float(row['commission'])
                    start = time
                end = time

                transactions.append(Transaction(time, buy, position, price, cash))

        return Record(start, end, initial_capital, commission, transactions)


def to_csv(self, path: str):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'buy', 'position', 'price', 'cash', 'initial_capital', 'commission'])
        for tx in self.transactions:
            writer.writerow([
                tx.time,
                tx.buy,
                tx.position,
                tx.price,
                tx.cash,
                self.initial_capital,
                self.commission
            ])


class Records:
    def __init__(self):
        self.records = []

    def report(self, index: int):
        # 可自訂你要看的 Record 報告
        if 0 <= index < len(self.records):
            return self.records[index].report()

    def save_all(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        for i, record in enumerate(self.records):
            path = os.path.join(folder_path, f"record_{i:03d}.csv")
            record.to_csv(path)

    @staticmethod
    def load_all(folder_path: str, capital: float = 1000000, commission: float = 0.0) -> 'Records':
        records = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".csv"):
                path = os.path.join(folder_path, filename)
                record = Record.from_csv(path, capital=capital, commission=commission)
                records.append(record)
        return Records(records)

record = Records()
record.load_all("FinTechML/Dataset/Records")
record.save_all("FinTechML/Dataset/Records")