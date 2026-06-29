from openbb import obb

quote = obb.equity.price.quote("ry", provider="tmx")
hist = obb.equity.price.historical("ry", provider="tmx")
# print (f"The hist is : {hist} ")

import yfinance as yf
ticker = yf.Ticker("RY.TO")
# info = ticker.income_stmt  # dict of pre-computed metric
info = ticker.recommendations_summary
print (f"The info is : {info} ")