//@version=5
strategy('fibtun', overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=5, commission_type=strategy.commission.percent, commission_value=0.05, margin_long=0.04, margin_short=0.04, initial_capital=50)
// startYear = input(2022, 'Start Year')
// startMonth = input(1, 'Start Month')
// startDay = input(1, 'Start Day')
// endYear = year(timenow)
// endMonth = month(timenow)
// endDay = dayofmonth(timenow)
// startTime = timestamp(startYear, startMonth, startDay, 00, 00)
// endTime = timestamp(endYear, endMonth, endDay, 23, 59)

// // 判斷當前條是否在指定的時間範圍內
// inDateRange = time >= startTime and time <= endTime
// Adjusted strategy parameters for higher frequency
fibStartLength = input.int(75, minval=1, title='Fibonacci Lookback Period Start')  // Reduced for faster response
fibEndLength = input.int(25, minval=1, title='Fibonacci Lookback Period End')  // Reduced for faster response
volMultiplier = input(1.5, title='Volume Multiplier for Entry')  // Lowered to trigger trades on smaller volume changes
riskPct = input(0.5, title='Risk Percentage for Stop Loss')
atrLength = input(14, title='ATR Length for Stop Loss')
leverage = input(1, title='Leverage')
trailStopAtrMultiplier = input(0.5, title='Trailing Stop ATR Multiplier')
forcestoppercent = input(0.5, title='Force stop percent')
atrFilter = input(80, title='atr filter')

// Adjusted trend filter for higher sensitivity
emaFilterLength = input(25, title='EMA Trend Filter Length')  // Shortened for quicker trend detection
longTrendFilter = ta.ema(close, emaFilterLength) < close
shortTrendFilter = ta.ema(close, emaFilterLength) > close

// Average volume calculation remains the same
avgVol = ta.sma(volume, fibStartLength)

// ATR and risk management remains the same
atr = ta.atr(atrLength)
riskPerTrade = strategy.equity * (riskPct / 100) * leverage

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
    if d_close > fib786
        v_state := 5
    else if d_close > fib618
        v_state := 4
    else if d_close > fib500
        v_state := 3
    else if d_close > fib382
        v_state := 2
    else if d_close > fib236
        v_state := 1
    else
        v_state := 0
    
    v_state

var pre_fibState = initial_state(close) // 0 0~0.236 1 0.236~0.382 2 0.382~0.5 3 0.5~0.618 4 0.618~0.786 5 0.786~1
var pre_Volume = volume

next_fibStateV (d_close, pre_fibState) =>
    v_state = initial_state(d_close)
    res = v_state-pre_fibState

    res

pre_fibStateV = next_fibStateV(close, pre_fibState)
pre_fibState+=pre_fibStateV

next_volumeV (d_volume, pre_Volume) =>
    res = d_volume-pre_Volume

    res

pre_volumeV = next_volumeV(volume, pre_Volume)
pre_Volume+=pre_volumeV

longCondition (d_pre_fibStateV, d_pre_VolumeV) =>
    v_longCondition = d_pre_fibStateV > 0 and d_pre_VolumeV >0 
    
    v_longCondition

shortCondition (d_pre_fibStateV, d_pre_VolumeV) =>
    v_shortCondition = d_pre_fibStateV < 0 and d_pre_VolumeV <0 
    
    v_shortCondition



