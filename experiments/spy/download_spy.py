import os
import yfinance as yf

os.makedirs("data", exist_ok=True)

prices = yf.download("SPY", start="1999-12-31", end="2025-01-01",
                     auto_adjust=True, progress=False)
close = prices["Close"]
close.to_csv("data/spy_close.csv")

print("Rows:", len(close))
print(close.head(3))
print(close.tail(3))