"""OpenBB TMX (Toronto Stock Exchange) provider for Canadian stock data.

OpenBB is a free, open-source market data library. This provider leverages
openbb.equity.fundamental and openbb.equity.price for Canadian stocks.

Coverage: Canadian stocks (tickers ending in .TO)
Limitations: Free tier has quotas and some endpoints require paid accounts
"""

import pandas as pd
import structlog
from datetime import datetime, timedelta

from data.providers.base import StockDataProvider, NewsProvider

logger = structlog.get_logger(__name__)


class OpenBBTMXProvider(StockDataProvider, NewsProvider):
    """Canadian stock data via OpenBB TMX module.

    Implements:
    - StockDataProvider: Canadian prices, company info, dividends
    - NewsProvider: Not implemented (Finnhub is the sole news source)

    Features confirmed working:
    - get_price_history(): Daily OHLCV data via openbb.equity.price.historical
    - get_dividend_history(): Dividend records via openbb.equity.fundamental.dividends
    - get_company_info(): Company profile via openbb.equity.info

    Limitations (free tier / not implemented):
    - No analyst data (estimates, ratings, recommendations) — paid-only
    - No insider trading or peer data
    - No earnings calendar
    - No financial statements (edgartools.py owns US financials; openbb-tmx
      does not provide comprehensive Canadian GAAP statements)
    - No news (Finnhub is the sole news source per routing rule)
    """

    def __init__(self) -> None:
        """Initialize OpenBB TMX provider.

        Note: OpenBB loads credentials from environment or config files automatically.
        No explicit API key management needed for free tier.
        """
        try:
            import openbb

            self.openbb = openbb
        except ImportError:
            raise RuntimeError(
                "OpenBB is not installed. Install it with: pip install openbb"
            )

    async def __aenter__(self) -> "OpenBBTMXProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit. OpenBB doesn't need cleanup."""
        pass

    # ============================================================================
    # StockDataProvider Implementation
    # ============================================================================

    async def get_price_history(
        self, ticker: str, period: str = "1y", interval: str = "daily"
    ) -> pd.DataFrame:
        """Fetch historical price data for a Canadian stock.

        Args:
            ticker: Stock symbol, typically ending in .TO (e.g., 'RY.TO', 'TD.TO').
            period: Time period ('1mo', '3mo', '1y', '5y', 'max'). Converted to from/to dates.
            interval: Frequency ('daily' supported; intraday not available in OpenBB free).

        Returns:
            DataFrame with columns: date, open, high, low, close, volume (lowercase).
            Empty DataFrame if data unavailable.
        """
        try:
            # Convert period to from_date
            from_date = self._period_to_from_date(period)
            to_date = datetime.now().strftime("%Y-%m-%d")

            # OpenBB obb.equity.price.historical returns OBBject with results attribute
            response = self.openbb.obb.equity.price.historical(
                symbol=ticker, start_date=from_date, end_date=to_date
            )

            if response is None or not hasattr(response, 'results'):
                logger.warning("openbb_no_price_history", ticker=ticker)
                return pd.DataFrame()

            # Convert results list to DataFrame
            results = response.results
            if not results:
                logger.warning("openbb_no_price_history", ticker=ticker)
                return pd.DataFrame()

            # Convert list of data objects to DataFrame
            df = pd.DataFrame([vars(item) for item in results])
            df.columns = [col.lower() for col in df.columns]

            # Ensure date is datetime and sorted
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            return df.sort_values("date")

        except Exception as e:
            logger.error("openbb_price_history_error", ticker=ticker, error=str(e))
            return pd.DataFrame()

    async def get_company_info(self, ticker: str) -> dict:
        """Fetch company profile information.

        Args:
            ticker: Stock symbol (e.g., 'RY.TO').

        Returns:
            Dictionary with company profile data (name, sector, industry, etc.).
            Empty dict if unavailable.
        """
        try:
            response = self.openbb.obb.equity.profile(symbol=ticker)
            print(response)
            if response is None or not hasattr(response, 'results'):
                logger.warning("openbb_no_company_info", ticker=ticker)
                return {}

            results = response.results
            if not results:
                logger.warning("openbb_no_company_info", ticker=ticker)
                return {}

            # Convert first result to dict (returns single company profile)
            if isinstance(results, list) and len(results) > 0:
                return vars(results[0]) if hasattr(results[0], '__dict__') else results[0]
            elif isinstance(results, dict):
                return results
            else:
                return vars(results) if hasattr(results, '__dict__') else {}

        except Exception as e:
            logger.error("openbb_company_info_error", ticker=ticker, error=str(e))
            return {}

    async def get_dividend_history(
        self, ticker: str, from_date: str = None, to_date: str = None
    ) -> list[dict]:
        """Fetch dividend history for a Canadian stock.

        Args:
            ticker: Stock symbol (e.g., 'RY.TO', 'TD.TO').
            from_date: Start date (YYYY-MM-DD). If None, defaults to 5 years ago.
            to_date: End date (YYYY-MM-DD). If None, defaults to today.

        Returns:
            List of dividend records with date and amount.
            Used by checkpoint evaluator for total return calculations.
        """
        try:
            # Set defaults for date range
            if to_date is None:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if from_date is None:
                from_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

            # OpenBB obb.equity.fundamental.dividends returns OBBject with results attribute
            response = self.openbb.obb.equity.fundamental.dividends(
                symbol=ticker, start_date=from_date, end_date=to_date
            )

            if response is None or not hasattr(response, 'results'):
                logger.warning("openbb_no_dividends", ticker=ticker)
                return []

            results = response.results
            if not results:
                logger.warning("openbb_no_dividends", ticker=ticker)
                return []

            # Convert list of data objects to list of dicts
            return [vars(item) for item in results]

        except Exception as e:
            logger.error("openbb_dividend_error", ticker=ticker, error=str(e))
            return []

    # ============================================================================
    # Unsupported Methods (Not Available in OpenBB Free Tier)
    # ============================================================================

    async def get_financials(
        self, ticker: str, statement: str = "income", period: str = "annual"
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "Financial statements are edgartools.py's job for US stocks. "
            "For Canadian stocks, use direct SEC/SEDAR queries or a paid service."
        )

    async def get_analyst_estimates(self, ticker: str) -> dict:
        raise NotImplementedError(
            "Analyst estimates are paid-only in OpenBB. Not available on free tier."
        )

    async def get_analyst_ratings(self, ticker: str) -> dict:
        raise NotImplementedError(
            "Analyst ratings are paid-only in OpenBB. Not available on free tier."
        )

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        raise NotImplementedError(
            "Insider trading (Form 4) is edgartools.py's job for US stocks. "
            "For Canadian stocks, use SEDAR+ filings or a paid service."
        )

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        raise NotImplementedError(
            "Peer data is not available in OpenBB free tier. Use sector research manually."
        )

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        raise NotImplementedError(
            "Earnings calendar is paid-only in OpenBB. Not available on free tier."
        )

    async def get_news(self, ticker: str, days: int = 30) -> list[dict]:
        raise NotImplementedError(
            "News is Finnhub's job per routing rule. "
            "OpenBB news endpoint may not have sufficient Canadian stock coverage."
        )

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        raise NotImplementedError(
            "Analyst recommendation trends are paid-only in OpenBB. Not available on free tier."
        )

    # ============================================================================
    # Helper Methods
    # ============================================================================

    @staticmethod
    def _period_to_from_date(period: str) -> str:
        """Convert yfinance-style period to from_date for OpenBB.

        Args:
            period: One of '1mo', '3mo', '6mo', '1y', '5y', 'max'.

        Returns:
            YYYY-MM-DD formatted date string.
        """
        period = period.lower().strip()

        if period == "max":
            return "1970-01-01"

        # Parse period string (e.g., '1y', '3mo', '5y')
        if period.endswith("mo"):
            months = int(period.replace("mo", ""))
            delta_days = months * 30
        elif period.endswith("y"):
            years = int(period.replace("y", ""))
            delta_days = years * 365
        else:
            # Default to 1 year if unrecognized
            delta_days = 365

        from_date = (datetime.now() - timedelta(days=delta_days)).strftime("%Y-%m-%d")
        return from_date
