import sys
sys.path.insert(0, 'src')
# ============================================================================
# Testing yfinance.py
# ============================================================================

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
from src.data.providers.boc import BOCMacroDataProvider
async def test_boc():
    async with BOCMacroDataProvider() as provider:
        # ✅ All async calls must use await
        rates = await provider.get_interest_rates()
        exchange = await provider.get_exchange_rates("USDCAD")
        macro = await provider.get_macro_data(["FXUSDCAD"])
        
        # print(f"Rates: {rates}")
        # print(f"Exchange: {exchange}")
        # print(f"Macro: {macro}")

# Run it
asyncio.run(test_boc())
# ============================================================================
# Testing fmp.py
# ============================================================================
from src.data.providers.fmp import FMPDataProvider
async def test_fmp():
    async with FMPDataProvider() as provider:
        # ✅ All async calls must use await
        hist = await provider.get_price_history("AAPL", "1mo", "1d")
        # print(f"hist: {hist}")
        info = await provider.get_company_info("AAPL")
        # print(f"info: {info}")
        analysis_est = await provider.get_analyst_estimates("AAPL")
        # print(f"analysis_est: {analysis_est}")

# Run it
asyncio.run(test_fmp())
# ============================================================================
# Testing openbb_tmx.py
# ============================================================================
from src.data.providers.openbb_tmx import OpenBBTMXProvider
async def test_get_price_history():
    provider = OpenBBTMXProvider()
    df = await provider.get_price_history(ticker="SHOP", period="1y", interval="1d")
    
    print(f"Rows returned: {len(df)}")
    print(df.head())
    
    assert not df.empty, "Expected non-empty DataFrame"
    assert "close" in df.columns.str.lower(), "Expected a 'close' column"
    
    print("✅ get_price_history test passed")

# Run it
asyncio.run(test_get_price_history())