# ============================================================================
# Testing yfinance.py
# ============================================================================
from src.data.providers.boc import BOCMacroDataProvider
# from openbb import obb

# quote = obb.equity.price.quote("ry", provider="tmx")
# hist = obb.equity.price.historical("ry", provider="tmx")
# # print (f"The hist is : {hist} ")

# import yfinance as yf
# ticker = yf.Ticker("RY.TO")
# # info = ticker.income_stmt  # dict of pre-computed metric
# info = ticker.recommendations_summary
# print (f"The info is : {info} ")


# ============================================================================
# Testing boc.py
# ============================================================================
import asyncio

async def test_boc():
    async with BOCMacroDataProvider() as provider:
        # ✅ All async calls must use await
        rates = await provider.get_interest_rates("2026-07-23", "2026-07-24")
        exchange = await provider.get_exchange_rates("USDCAD")
        macro = await provider.get_macro_data(["FXUSDCAD"], "2026-07-23", "2026-07-24")
        
        print(f"Rates: {rates}")
        print(f"Exchange: {exchange}")
        print(f"Macro: {macro}")

# Run it
asyncio.run(test_boc())
# ============================================================================
# Testing fmp.py
# ============================================================================