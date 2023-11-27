//@version=5
strategy('fibtun 1 15', overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=5, commission_type=strategy.commission.percent, commission_value=0.05, margin_long=0.04, margin_short=0.04, initial_capital=50)
startYear = input(2022, 'Start Year')
startMonth = input(1, 'Start Month')
startDay = input(1, 'Start Day')
endYear = year(timenow)
endMonth = month(timenow)
endDay = dayofmonth(timenow)
startTime = timestamp(startYear, startMonth, startDay, 00, 00)
endTime = timestamp(endYear, endMonth, endDay, 23, 59)

// 判斷當前條是否在指定的時間範圍內
inDateRange = time >= startTime and time <= endTime
fibStartLength = input.int(1125, minval=1, title='Fibonacci Lookback Period Start')  // Reduced for faster response
fibEndLength = input.int(375, minval=1, title='Fibonacci Lookback Period End')  // Reduced for faster response
volMultiplier = input(1.5, title='Volume Multiplier for Entry')  // Lowered to trigger trades on smaller volume changes
riskPct = input(0.5, title='Risk Percentage for Stop Loss')
atrLength = input(210, title='ATR Length for Stop Loss')
leverage = input(0.001, title='Leverage')
trailStopAtrMultiplier = input(0.5, title='Trailing Stop ATR Multiplier')
forcestoppercent = input(0.10, title='Force stop percent')
atrFilter = input(0, title='atr filter')

// Adjusted trend filter for higher sensitivity
emaFilterLength = input(375, title='EMA Trend Filter Length')  // Shortened for quicker trend detection
longTrendFilter = ta.ema(close, emaFilterLength) < close
shortTrendFilter = ta.ema(close, emaFilterLength) > close

// Average volume calculation remains the same
avgVol = ta.sma(volume, fibStartLength)



// ATR and risk management remains the same
atr = ta.atr(atrLength)
riskPctTrade = strategy.equity * (riskPct / 100) * leverage 
riskPctTrade := math.abs(riskPctTrade)
// Fibonacci channel high and low
highLevel = ta.highest(high, fibStartLength)
lowLevel = ta.lowest(low, fibEndLength)
// Adjusted Fibonacci retracement levels for more frequent trades
fibRange = highLevel - lowLevel
fib786 = highLevel - fibRange * 0.786
fib618 = highLevel - fibRange * 0.618
fib500 = highLevel - fibRange * 0.500
fib382 = highLevel - fibRange * 0.382
fib236 = highLevel - fibRange * 0.236
fib1 = highLevel - fibRange * 1.0

var fibLevels = array.new_float(9, na)
if (barstate.isfirst)
    array.set(fibLevels, 0, lowLevel)                       // 最低水平
    array.set(fibLevels, 1, highLevel - fibRange * 0.236)  // Fib 0.236
    array.set(fibLevels, 2, highLevel - fibRange * 0.382)  // Fib 0.382
    array.set(fibLevels, 3, highLevel - fibRange * 0.500)  // Fib 0.500
    array.set(fibLevels, 4, highLevel - fibRange * 0.618)  // Fib 0.618
    array.set(fibLevels, 5, highLevel - fibRange * 0.786)  // Fib 0.786
    array.set(fibLevels, 6, highLevel)                 // 小于最高水平的值
    array.set(fibLevels, 7, highLevel)         
    array.set(fibLevels, 8, highLevel)             // 最高水平
    
// Draw Fibonacci channel with additional level

plot(highLevel, color=color.new(color.red, 0), linewidth=2, title='High Level')
plot(fib1, color=color.new(color.orange, 0), linewidth=2, title='Fib 1.000 Level')
plot(fib786, color=color.new(color.green, 0), linewidth=2, title='Fib 0.786 Level')
plot(fib618, color=color.new(color.blue, 0), linewidth=2, title='Fib 0.618 Level')
plot(fib500, color=color.new(color.purple, 0), linewidth=2, title='Fib 0.500 Level')  // New mid-level
plot(fib382, color=color.new(color.orange, 0), linewidth=2, title='Fib 0.382 Level')
plot(fib236, color=color.new(color.green, 0), linewidth=2, title='Fib 0.236 Level')
plot(lowLevel, color=color.new(color.red, 0), linewidth=2, title='Low Level')

// Adjusted trade signal logic for higher frequency

initial_state(d_close) =>
    var v_state = 0
    
    if d_close >= highLevel
        v_state := 7        
    else if d_close > fib786
        v_state := 6
    else if d_close > fib618
        v_state := 5
    else if d_close > fib500
        v_state := 4
    else if d_close > fib382
        v_state := 3
    else if d_close > fib236
        v_state := 2
    else if d_close > lowLevel
        v_state := 1
    else
        v_state := 0
    
    v_state

var pre_fibState = initial_state(close) // 0 0~0.236 1 0.236~0.382 2 0.382~0.5 3 0.5~0.618 4 0.618~0.786 5 0.786~1
var pre_Volume = 0.0
<<<<<<< HEAD

=======
>>>>>>> acddb17b916d5c84564103a07da772da3b5f913d

next_fibStateV (d_close, pre_fibState) =>
    v_state = initial_state(d_close)
    res = v_state-pre_fibState

    res

pre_fibStateV = next_fibStateV(close, pre_fibState)
//pre_fibState+=pre_fibStateV

next_volumeVPercent (d_volume, pre_Volume) =>
    v_state = d_volume/pre_Volume
    v_state

pre_volumeVPercent = next_volumeVPercent(volume, pre_Volume)
<<<<<<< HEAD
//pre_Volume:=pre_Volume*pre_volumeVPercent

// 確保在使用之前聲明並初始化 tradeqty
// tradeqty = strategy.equity * 0.05*leverage > 0 ? strategy.equity * 0.05*leverage : 0
//tradeqty = strategy.equity * 0.05
var trade_Status = 0 // 0 沒開倉, 1 多倉, -1 空倉
var tradeId = 0
var float highestPrice = na
var float lowestPrice = na
//我想要達到的效果是，trade_Status 來表示現在的開倉狀態，如果沒開倉，pre_fibStateV是正的，交易量大過某程度就開倉，並且紀錄現在的交易量，開倉的時候都會默認帶強制止損，如果現在是多倉狀態pre_fibStateV是正的，交易量大於之前開倉的時候就加倉，如果沒有加上追蹤止盈，追蹤止盈是只會一直跟蹤價格，直到價格突然回撤到某個程度就止盈，

if strategy.position_size == 0
    trade_Status := 0
    pre_Volume := 0
    highestPrice:=0
    lowestPrice := 99999999999999999999.9999999999


tradeLogic(trade_Status, pre_fibState, pre_Volume) =>
    fpre_fibStateV = next_fibStateV(close, pre_fibState)
    new_pre_fibState = pre_fibState + pre_fibStateV
    fpre_volumeVPercent = next_volumeVPercent(volume, pre_Volume)
    new_pre_Volume = pre_Volume * pre_volumeVPercent

    new_trade_Status = trade_Status
    new_tradeId = tradeId

    if trade_Status == 0
        if pre_fibStateV > 0 and volume > avgVol * volMultiplier
            new_tradeId := tradeId + 1
            new_trade_Status := 1
            fibStopLossPrice = (array.get(fibLevels, fibState) + array.get(fibLevels, math.max(0,fibState - 1))) * 0.5
            absoluteStopLossPrice = close * (1 - forcestoppercent/100)
            stopLossPrice = math.max(fibStopLossPrice, absoluteStopLossPrice)
            strategy.entry('Long', strategy.long, qty=riskPctTrade,stop=stopLossPrice, comment='Long')
        else if pre_fibStateV < 0 and volume > avgVol * volMultiplier
            new_tradeId := tradeId + 1
            new_trade_Status := -1
            fibStopLossPrice = (array.get(fibLevels, fibState) + array.get(fibLevels, math.min(7,fibState + 1))) * 0.5
            absoluteStopLossPrice = close * (1 + forcestoppercent/100)
            stopLossPrice = math.min(fibStopLossPrice, absoluteStopLossPrice)
            strategy.entry('Short', strategy.short, qty=riskPctTrade,stop=stopLossPrice ,comment='Short')

    [new_trade_Status, new_pre_fibState, new_pre_Volume, new_tradeId]

// 使用函數
[ftrade_Status, fpre_fibState, fpre_Volume, ftradeId] = tradeLogic(trade_Status, pre_fibState, pre_Volume)
trade_Status := ftrade_Status
pre_fibState := fpre_fibState
pre_Volume := fpre_Volume
tradeId := ftradeId

plot(pre_Volume, title="Pre Volume", color=color.blue)

// setStopLossAndTakeProfit(trade_Status, fibState) =>
//     if trade_Status == 1
//     // 为多头设置止损和止盈
//         fibStopLossPrice = (array.get(fibLevels, fibState) + array.get(fibLevels, math.max(0,fibState - 1))) * 0.5
//         absoluteStopLossPrice = close * (1 - forcestoppercent/100)
//         stopLossPrice = math.max(fibStopLossPrice, absoluteStopLossPrice)
//         strategy.exit('Long Exit', 'Long', stop=stopLossPrice)
//     else if trade_Status == -1
//     // 为空头设置止损和止盈
//         fibStopLossPrice = (array.get(fibLevels, fibState) + array.get(fibLevels, math.min(7,fibState + 1))) * 0.5
//         absoluteStopLossPrice = close * (1 + forcestoppercent/100)
//         stopLossPrice = math.min(fibStopLossPrice, absoluteStopLossPrice)
//         strategy.exit('Short Exit', 'Short', stop=stopLossPrice)
        

// // 使用止盈止損函數
// setStopLossAndTakeProfit(trade_Status, pre_fibState)



updateTrailingStop(trade_Status, fibState) =>
    stopPrice = 0.0

    if trade_Status == 1
        // 更新多头交易的最高价格
        fhighestPrice = na(highestPrice) ? high : math.max(highestPrice, high)
        // 设置止盈价格为最高价格减去半个通道的价格
        stopPrice := highestPrice - (array.get(fibLevels, fibState) - array.get(fibLevels, math.max(0, fibState - 1))) * 0.5
        strategy.exit('Trailing Stop', 'Long', stop=stopPrice)
    else if trade_Status == -1
        // 更新空头交易的最低价格
        flowestPrice = na(lowestPrice) ? low : math.min(lowestPrice, low)
        // 设置止盈价格为最低价格加上半个通道的价格
        stopPrice := lowestPrice + (array.get(fibLevels, math.min(7, fibState + 1)) - array.get(fibLevels, fibState)) * 0.5
        strategy.exit('Trailing Stop', 'Short', stop=stopPrice)
    [highestPrice, lowestPrice, stopPrice ]
    
=======
pre_Volume:=pre_Volume*pre_volumeVPercent

// 確保在使用之前聲明並初始化 tradeqty
tradeqty = strategy.equity * 0.05 > 0 ? strategy.equity * 0.05 : 0

var trade_Status = 0 // 0 沒開倉, 1 多倉, -1 空倉
var tradeId = 0

//我想要達到的效果是，trade_Status 來表示現在的開倉狀態，如果沒開倉，pre_fibStateV是正的，交易量大過某程度就開倉，並且紀錄現在的交易量，開倉的時候都會默認帶強制止損，如果現在是多倉狀態pre_fibStateV是正的，交易量大於之前開倉的時候就加倉，如果沒有加上追蹤止盈，追蹤止盈是只會一直跟蹤價格，直到價格突然回撤到某個程度就止盈，
>>>>>>> acddb17b916d5c84564103a07da772da3b5f913d

[fhighestPrice, flowestPrice, fstopPrice]= updateTrailingStop(trade_Status, pre_fibState)
highestPrice := fhighestPrice
lowestPrice := flowestPrice
fstopPrice := fstopPrice

var line stopPriceLine = na
if na(stopPriceLine) == false
    line.delete(stopPriceLine)

<<<<<<< HEAD
// 绘制新的止盈价格线
stopPriceLine := line.new(bar_index, fstopPrice, bar_index - 1, fstopPrice, width=2, color=color.red)
=======
tradeLogic(trade_Status, pre_fibState, pre_Volume) =>
    fpre_fibStateV = next_fibStateV(close, pre_fibState)
    new_pre_fibState = pre_fibState + pre_fibStateV
    fpre_volumeVPercent = next_volumeVPercent(volume, pre_Volume)
    new_pre_Volume = pre_Volume * pre_volumeVPercent

    new_trade_Status = trade_Status
    new_tradeId = tradeId

    if trade_Status == 0
        if pre_fibStateV > 0 and volume > avgVol * volMultiplier
            new_tradeId := tradeId + 1
            new_trade_Status := 1
            strategy.entry('Long', strategy.long, qty=tradeqty, comment='Long', stop=close*(1-forcestoppercent/100))
        else if pre_fibStateV < 0 and volume > avgVol * volMultiplier
            new_tradeId := tradeId + 1
            new_trade_Status := -1
            strategy.entry('Short', strategy.short, qty=tradeqty, comment='Short', stop=close*(1+forcestoppercent/100))

    [new_trade_Status, new_pre_fibState, new_pre_Volume, new_tradeId]

// 使用函數
[ftrade_Status, fpre_fibState, fpre_Volume, ftradeId] = tradeLogic(trade_Status, pre_fibState, pre_Volume)
trade_Status := ftrade_Status
pre_fibState := fpre_fibState
pre_Volume := fpre_Volume
tradeId := ftradeId


setStopLossAndTakeProfit(trade_Status) =>
    if trade_Status == 1
        // 為多頭設置止損和止盈
        strategy.exit('Long Exit', 'Long', stop=close*(1-forcestoppercent/100), limit=close*(1+trailStopAtrMultiplier*atr))
    else if trade_Status == -1
        // 為空頭設置止損和止盈
        strategy.exit('Short Exit', 'Short', stop=close*(1+forcestoppercent/100), limit=close*(1-trailStopAtrMultiplier*atr))
    [fpre_Volume, ftrade_Status]
// 使用止盈止損函數
setStopLossAndTakeProfit(trade_Status)

if strategy.position_size == 0
    trade_Status := 0
    pre_Volume := 0
>>>>>>> acddb17b916d5c84564103a07da772da3b5f913d
