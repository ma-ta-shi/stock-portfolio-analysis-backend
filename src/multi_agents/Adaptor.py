from abc import ABC, abstractmethod
import pandas as pd

class StockDataProvider(ABC):
    """Abstract base for stock-centric financial data (FMP, yfinance)."""
    
    @abstractmethod
    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame: ...
    @abstractmethod
    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame: ...
    @abstractmethod
    async def get_company_info(self, ticker: str) -> dict: ...
    @abstractmethod
    async def get_analyst_estimates(self, ticker: str) -> dict: ...
    @abstractmethod
    async def get_analyst_ratings(self, ticker: str) -> dict: ...
    @abstractmethod
    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]: ...
    @abstractmethod
    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]: ...
    @abstractmethod
    async def get_earnings_calendar(self, ticker: str) -> list[dict]: ...
    @abstractmethod
    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Fetch historical dividends for total return calculations.
        Used by the checkpoint evaluator to compute actual total return
        (price change + dividends) rather than price-only return.
        Source:
          - Canadian stocks: openbb-tmx .equity.fundamental.dividends
          - US stocks: FMP /api/v3/historical-price-full/stock_dividend/{ticker}"""
        ...


class NewsProvider(ABC):
    """Abstract base for news data (FMP news, Finnhub news with sentiment)."""
    
    @abstractmethod
    async def get_news(self, ticker: str, days: int) -> list[dict]: ...
    @abstractmethod
    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]: ...


class MacroDataProvider(ABC):
    """Abstract base for macroeconomic data (FRED, Bank of Canada Valet)."""
    
    @abstractmethod
    async def get_macro_data(self, series_ids: list[str]) -> dict[str, pd.Series]: ...
    @abstractmethod
    async def get_interest_rates(self) -> dict: ...
    @abstractmethod
    async def get_exchange_rates(self, pair: str = "CADUSD") -> dict: ...