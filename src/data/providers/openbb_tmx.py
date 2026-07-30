"""OpenBB TMX (Toronto Stock Exchange) provider for Canadian stock data.

OpenBB is a free, open-source market data library. This provider leverages
openbb.equity.fundamental and openbb.equity.price for Canadian stocks.

Coverage: Canadian stocks (tickers ending in .TO)
Limitations: Free tier has quotas and some endpoints require paid accounts
"""

import asyncio
import pandas as pd
from openbb import obb
from data.providers.base import StockDataProvider, NewsProvider

class OpenBBTMXProvider(StockDataProvider, NewsProvider):
    """OpenBB (TMX extension) provider for Canadian equities.

    Covers price history, fundamentals, company info, and dividends via
    the `openbb-tmx` data provider. Analyst estimates/ratings, insider
    trading, peers, and earnings calendar are NOT supported by TMX and
    are intentionally left unimplemented — use FMP/Finnhub for those.
    """

    PROVIDER = "tmx"

    # ---------- StockDataProvider ----------

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        result = await asyncio.to_thread(
            obb.equity.price.historical,
            symbol=ticker,
            interval=interval,
            provider=self.PROVIDER,
        )
        return result.to_df()

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        # statement: "income" | "balance" | "cash"
        fetcher = {
            "income": obb.equity.fundamental.income,
            "balance": obb.equity.fundamental.balance,
            "cash": obb.equity.fundamental.cash,
        }.get(statement)

        if fetcher is None:
            raise ValueError(f"Unknown statement type: {statement}")

        result = await asyncio.to_thread(
            fetcher, symbol=ticker, period=period, provider=self.PROVIDER
        )
        return result.to_df()

    async def get_company_info(self, ticker: str) -> dict:
        result = await asyncio.to_thread(
            obb.equity.profile, symbol=ticker, provider=self.PROVIDER
        )
        return result.to_df().iloc[0].to_dict()

    async def get_analyst_estimates(self, ticker: str) -> dict:
        # Not supported by TMX provider.
        raise NotImplementedError("Analyst estimates are not available via openbb-tmx.")

    async def get_analyst_ratings(self, ticker: str) -> dict:
        # Not supported by TMX provider.
        raise NotImplementedError("Analyst ratings are not available via openbb-tmx.")

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        # Not supported by TMX provider.
        raise NotImplementedError("Insider trading data is not available via openbb-tmx.")

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        # Not supported by TMX provider.
        raise NotImplementedError("Peer comparison is not available via openbb-tmx.")

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        # Not supported by TMX provider.
        raise NotImplementedError("Earnings calendar is not available via openbb-tmx.")

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        result = await asyncio.to_thread(
            obb.equity.fundamental.dividends,
            symbol=ticker,
            provider=self.PROVIDER,
        )
        df = result.to_df()
        mask = (df.index >= from_date) & (df.index <= to_date)
        return df.loc[mask].reset_index().to_dict("records")

    # ---------- NewsProvider ----------

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        # TMX doesn't have its own news fetcher registered in OpenBB;
        # fall back to the general equity news endpoint with tmx as symbol source.
        raise NotImplementedError("News is not available via openbb-tmx; use FMP or Finnhub instead.")

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        # Not supported by TMX provider.
        raise NotImplementedError("Recommendation trends are not available via openbb-tmx.")