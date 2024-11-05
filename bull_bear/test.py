import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 定義下載市場數據的函數
def get_market_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    # 選擇需要的欄位，如收盤價和交易量
    data = data[['Close', 'Volume']]
    return data

# 2. 標記市場情境
def label_market_conditions(data):
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['Condition'] = 'Neutral'

    # 設定牛市與熊市的條件，從2%調整為1.5%
    data.loc[data['Close'] > data['SMA_30'] * 1.015, 'Condition'] = 'Bull'  # 當價格高於30日均線1.5%為牛市
    data.loc[data['Close'] < data['SMA_30'] * 0.985, 'Condition'] = 'Bear'  # 低於30日均線1.5%為熊市

    return data[['Close', 'Volume', 'Condition']]

# 3. 準備數據，用於LSTM模型訓練
def prepare_data(data):
    X, y = [], []
    for i in range(30, len(data)):  # 以30天為一組數據
        X.append(data[i-30:i])  # 包含 Close 和 Volume
        y.append(data[i, 0])  # 預測的目標為 Close 價格
    X, y = np.array(X), np.array(y)
    return X, y

# 4. 建立LSTM模型
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model

# 5. 模擬市場情境
def simulate_market_scenario(model, data, scenario):
    simulated_data = []
    for i in range(30, len(data)):  # 確保有足夠的數據進行預測
        if data.iloc[i]['Condition'] == scenario:  # 使用 iloc 來避免 FutureWarning
            input_data = data[['Close', 'Volume']].values[i-30:i]  # 包含 Close 和 Volume
            input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))  # 調整形狀以符合模型輸入
            prediction = model.predict(input_data)
            simulated_data.append(prediction[0][0])  # 假設預測結果是單一數值
        else:
            simulated_data.append(np.nan)  # 使用 NaN 作為非此情境的數據點佔位
    # 使用插值填補NaN
    return pd.Series(simulated_data).interpolate().fillna(method='bfill').fillna(method='ffill')

# 6. 執行流程：下載數據、標記情境、訓練模型並模擬市場情境
def main():
    # 下載市場數據並標記情境
    market_data = get_market_data('^GSPC', '2020-01-01', '2023-01-01')
    labeled_data = label_market_conditions(market_data)

    # 將數據標準化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(labeled_data[['Close', 'Volume']])
    labeled_data[['Close', 'Volume']] = scaled_data

    # 準備數據並訓練LSTM模型
    X_train, y_train = prepare_data(scaled_data)
    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # 模擬不同市場情境
    bull_market = simulate_market_scenario(model, labeled_data, scenario='Bull')
    bear_market = simulate_market_scenario(model, labeled_data, scenario='Bear')
    neutral_market = simulate_market_scenario(model, labeled_data, scenario='Neutral')

    # 繪製結果
    plt.figure(figsize=(12, 8))
    plt.plot(bull_market, label='Bull Market Simulation', color='green')
    plt.plot(bear_market, label='Bear Market Simulation', color='red')
    plt.plot(neutral_market, label='Neutral Market Simulation', color='blue')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Simulated Close Price')
    plt.title('Market Scenario Simulation with LSTM')
    plt.show()

# 7. 執行主函數
if __name__ == "__main__":
    main()
