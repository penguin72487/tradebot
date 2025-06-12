import yfinance as yf

try:
    ticker = yf.Ticker("2330.TW")
    print("✅ 成功連線！")
    print("公司名稱：", ticker.info.get("longName", "查不到公司名稱"))
except Exception as e:
    print("❌ 無法連線：", e)
