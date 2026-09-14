import yfinance as yf
import pandas as pd
import structlog
from datetime import datetime, timedelta
import pandas_datareader.data as web
from data.providers.base import (
    StockDataProvider,
    NewsProvider,
    MacroDataProvider,
    NormalizedAnalystEstimates,
    NormalizedCompanyInfo,
    NormalizedDividendRecord,
    NormalizedFinancials,
    NormalizedQuote,
)
import time

logger = structlog.get_logger(__name__)

# Map the combination of period and statement to the correct yfinance property
MAPPING = {
    ("annual", "income"): "financials",
    ("annual", "balance"): "balance_sheet",
    ("annual", "cashflow"): "cashflow",
    ("quarterly", "income"): "quarterly_financials",
    ("quarterly", "balance"): "quarterly_balance_sheet",
    ("quarterly", "cashflow"): "quarterly_cash_flow",
}


# Row-label maps for normalize_financials() (86bbb001k) — confirmed live against
# AAPL's quarterly_financials/quarterly_balance_sheet/quarterly_cashflow (2026-08-07),
# not guessed. "Interest Expense" is a real row for debt-heavy companies (confirmed
# on T) but genuinely absent for cash-rich ones like AAPL — a normal None, not a bug.
_INCOME_ROWS = {
    "revenue": "Total Revenue",
    "net_income": "Net Income",
    # Net income attributable to common shareholders (after preferred dividends).
    # For preferred-heavy names (banks, insurers) this runs 2-8% below "Net Income";
    # it's the correct denominator for a per-common-share P/E. Genuinely absent for
    # some filers -> a normal None, and fundamentals.py falls back to net_income.
    "net_income_common": "Net Income Common Stockholders",
    "eps": "Diluted EPS",
    "operating_income": "Operating Income",
    "interest_expense": "Interest Expense",
    "tax_expense": "Tax Provision",
    "cost_of_revenue": "Cost Of Revenue",
    "shares_outstanding": "Diluted Average Shares",
}
_CASHFLOW_ROWS = {
    "depreciation_amortization": "Depreciation And Amortization",
    "dividends_paid": "Cash Dividends Paid",
    "operating_cash_flow": "Operating Cash Flow",
    "capital_expenditures": "Capital Expenditure",
}
_BALANCE_ROWS = {
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest",
    "total_equity": "Stockholders Equity",
    "total_debt": "Total Debt",
    "cash_and_equivalents": "Cash And Cash Equivalents",
    "current_assets": "Current Assets",
    "current_liabilities": "Current Liabilities",
}


def _extract_row(df: pd.DataFrame, row_label: str, column) -> float | None:
    """Defensive single-cell lookup — a missing row (e.g. no Interest Expense
    for a cash-rich company) or missing period column returns None rather
    than raising, matching NormalizedFinancials' graceful-degradation design.
    Also guards against a duplicate-labeled index (unverified but plausible
    for messier statements than AAPL/RY.TO) — df.loc[row_label, column]
    returns a Series, not a scalar, when row_label repeats, which would
    otherwise crash pd.isna()/float() with an ambiguous-truth-value error."""
    if row_label not in df.index or column not in df.columns:
        return None
    value = df.loc[row_label, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return None if pd.isna(value) else float(value)


def _safe_float(value) -> float | None:
    """Same numpy-leak guard as _extract_row above, for values read
    directly off a DataFrame row (e.g. get_earnings_surprises) rather
    than via .loc — pandas cells are numpy.float64, not plain float."""
    return None if value is None or pd.isna(value) else float(value)


def correct_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and correct the yfinance one-year column misalignment bug
    (financial-data-api-research.md §2). The most recent column should be
    within the last 18 months; if it's older, shift all column labels
    forward by one year."""
    if df.empty:
        return df
    most_recent = df.columns[0]  # yfinance returns newest first
    months_old = (pd.Timestamp.now() - most_recent).days / 30
    if months_old > 18:
        df.columns = [c + pd.DateOffset(years=1) for c in df.columns]
    return df


def _correct_alignment_like(reference: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Applies correct_alignment()'s shift decision from `reference` (the
    always-fetched income statement) to `df`, rather than letting each
    statement decide independently. Real desync risk otherwise: income and
    cashflow are separate get_financials() calls, so if their staleness
    happened to straddle the 18-month threshold differently, correct_alignment()
    could shift one but not the other, silently misaligning their columns
    against each other even though the merge in _periods() assumes they
    share the same column labels."""
    if df.empty or reference.empty:
        return df
    most_recent = reference.columns[0]
    months_old = (pd.Timestamp.now() - most_recent).days / 30
    if months_old > 18:
        df.columns = [c + pd.DateOffset(years=1) for c in df.columns]
    return df


class YFinanceDataProvider(StockDataProvider):
    """Abstract base for stock-centric financial data yfinance."""

    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        # auto_adjust=False: keep `Close` the raw (split-adjusted, NOT
        # dividend-back-adjusted) price, matching openbb-tmx and FMP so a CA
        # name and a US name are on the same basis (86bbq7dkv). Drop the
        # `Adj Close` column it adds — we've settled on unadjusted, and this
        # keeps the column set identical to before.
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval, auto_adjust=False)
        return history.drop(columns=["Adj Close"], errors="ignore")

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        key = (period.lower(), statement.lower())
        if key not in MAPPING:
            raise ValueError(
                "Invalid statement or period. Use 'income'/'balance'/'cashflow' and 'annual'/'quarterly'"
            )
        # Dynamically extract the attribute from the Ticker object
        attribute_name = MAPPING[key]
        df = getattr(stock, attribute_name)
        return df

    async def _periods(self, ticker: str, period: str) -> list[dict]:
        """One entry per period (newest first), merging income + cashflow
        rows via _INCOME_ROWS/_CASHFLOW_ROWS. Cashflow rows are genuinely
        absent from the balance sheet's own period columns — a period with
        income data but no matching cashflow column just gets None for
        those fields, not dropped entirely."""
        raw_income = await self.get_financials(ticker, "income", period)
        cashflow = _correct_alignment_like(raw_income, await self.get_financials(ticker, "cashflow", period))
        income = correct_alignment(raw_income)
        if income.empty:
            return []
        periods = []
        for column in income.columns:
            row = {"period_end": column.date().isoformat()}
            for field, label in _INCOME_ROWS.items():
                row[field] = _extract_row(income, label, column)
            for field, label in _CASHFLOW_ROWS.items():
                row[field] = _extract_row(cashflow, label, column)
            # yfinance materialises a column for the newest period as soon as the
            # filing is docketed, before the actual statement data lands — an
            # all-None placeholder that _ttm()/_cagr() would otherwise poison the
            # whole trailing window with. Drop any period carrying neither revenue
            # nor net income (eps alone being None is normal for CA filers, so it
            # isn't part of the sentinel).
            if row["revenue"] is None and row["net_income"] is None:
                continue
            periods.append(row)
        return periods

    async def normalize_financials(self, ticker: str) -> NormalizedFinancials:
        """Builds NormalizedFinancials (86bbb001k) from this adapter's own
        get_financials() calls — 4 calls (income/cashflow x quarterly/annual)
        for quarters/annual, plus one quarterly balance-sheet call for the
        single latest balance_sheet dict (no per-period balance history
        needed, per NormalizedFinancials' own "latest only" design)."""
        quarters = await self._periods(ticker, "quarterly")
        annual = await self._periods(ticker, "annual")

        balance = correct_alignment(await self.get_financials(ticker, "balance", "quarterly"))
        balance_sheet: dict = {}
        if not balance.empty:
            # Same recent-filer lag as _periods(): the newest balance-sheet column
            # can be an all-None placeholder while the data is still landing (seen
            # on recent CA filers). Take the newest column that actually has
            # equity or assets, not blindly columns[0].
            latest_column = next(
                (
                    c
                    for c in balance.columns
                    if _extract_row(balance, _BALANCE_ROWS["total_equity"], c) is not None
                    or _extract_row(balance, _BALANCE_ROWS["total_assets"], c) is not None
                ),
                balance.columns[0],
            )
            for field, label in _BALANCE_ROWS.items():
                balance_sheet[field] = _extract_row(balance, label, latest_column)

        info = yf.Ticker(ticker).info
        return NormalizedFinancials(
            quarters=quarters,
            annual=annual,
            balance_sheet=balance_sheet,
            # financialCurrency, NOT currency: for a Canadian-listed company that
            # reports in USD (ATD.TO, NTR.TO, QSR.TO, AEM.TO, BN.TO, CSU.TO, ...)
            # yfinance gives currency="CAD" (the trading currency) while the
            # statement line items are actually in USD. The accurate label lets
            # compute_all() detect the CAD-quote / USD-statement mismatch and flag
            # the FX-distorted valuation multiples (P/E, P/S, EV/EBITDA — mcap in
            # CAD over earnings in USD, off by the CAD/USD rate ~1.37). FX-aware
            # reconciliation is ClickUp 86bbxucf0; for now the numbers still flow,
            # marked. Fall back to currency when financialCurrency is absent.
            # .get(key) or "" (not .get(key, "")) — yfinance can hold an explicit
            # None here, not just omit the key.
            currency=info.get("financialCurrency") or info.get("currency") or "",
        )

    async def get_company_info(self, ticker: str) -> NormalizedCompanyInfo:
        """Maps yfinance's .info onto NormalizedCompanyInfo (86bbb001k).
        currency/exchange/industry/country/market_cap were previously never
        extracted at all despite .info having them — real bug, not just a
        rename.

        Real gap caught in a final integration-level review: for a genuinely
        invalid ticker, yfinance's .info doesn't raise or come back empty —
        confirmed live it returns a near-empty dict with one unrelated key
        ({'trailingPegRatio': None}), no real data at all. Without a guard,
        every field below would default to "" and this method would return
        a full 7-key NormalizedCompanyInfo that LOOKS successful. That
        breaks router.py's fallback chain: _is_empty()'s len(dict) == 0
        check can never see a fully-blank-but-7-key dict as empty, so a
        real FMP failure correctly falling back to yfinance would silently
        "succeed" with a useless, all-blank result instead of the chain
        correctly reporting total failure. fmp.py/openbb_tmx.py don't have
        this problem — they already guard with `if not data: return {}`
        before building anything, since their APIs cleanly signal "no
        results" up front. name is the one field a real ticker should
        always have, so it's the guard."""
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info.get("longName"):
            return {}
        # .get(key) or "" (not .get(key, "")) — yfinance's .info can hold an
        # explicit None for these keys, not just omit them; .get(key, "")
        # only covers the omitted case and would silently store None in a
        # str field.
        return NormalizedCompanyInfo(
            name=info.get("longName") or "",
            sector=info.get("sector") or "",
            industry=info.get("industry") or "",
            market_cap=info.get("marketCap"),
            currency=info.get("currency") or "",
            country=info.get("country") or "",
            primary_exchange=info.get("exchange") or "",
        )

    async def get_analyst_estimates(self, ticker: str) -> NormalizedAnalystEstimates:
        """Maps onto NormalizedAnalystEstimates (86bbdu04a). yfinance
        doesn't expose forward EPS via analyst_price_targets or
        recommendations (checked) — it's a plain .info field instead,
        confirmed live for both AAPL and RY.TO. Bare {} on a missing
        value — see NormalizedAnalystEstimates's docstring in base.py
        for why."""
        stock = yf.Ticker(ticker)
        forward_eps = stock.info.get("forwardEps")
        return {"forward_eps": forward_eps} if forward_eps is not None else {}

    async def get_earnings_surprises(self, ticker: str) -> list[dict]:
        """Ticker.earnings_history — confirmed live to work for both US and
        CA tickers (4 rows each, yfinance's own hard cap), unlike most
        analyst-data fields on this provider. No revenue-surprise fields
        (unlike FMP) — revenue_actual/revenue_estimated stay None. Period
        comes from the DataFrame's "quarter" index, not a column.

        Real gap caught via live integration check: earnings_history's
        cells are numpy.float64, not plain float (DataFrame-sourced,
        unlike .info's plain-dict values elsewhere in this file) — same
        numpy-leak class _extract_row() above already guards against for
        financials. Cast explicitly; numpy.float64 isn't always
        JSON-serializable downstream and this project has no other
        established tolerance for it leaking past the provider layer."""
        stock = yf.Ticker(ticker)
        eh = stock.earnings_history
        if eh is None or eh.empty:
            return []
        return [
            {
                "period_end": period.date().isoformat(),
                "eps_actual": _safe_float(row.get("epsActual")),
                "eps_estimated": _safe_float(row.get("epsEstimate")),
                "eps_surprise_pct": (
                    _safe_float(row.get("surprisePercent")) * 100
                    if row.get("surprisePercent") is not None
                    else None
                ),
                "revenue_actual": None,
                "revenue_estimated": None,
            }
            for period, row in eh.iterrows()
        ]

    async def get_analyst_ratings(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        # Fetch the overall consensus text from .info safely
        try:
            info = stock.info
            consensus_text = info.get("recommendationKey", "N/A").lower()
            current_price = info.get("currentPrice", "N/A")
        except Exception:
            consensus_text = "N/A"
            current_price = "N/A"
        # Fetch the recommendation trend matrix
        recs_df = stock.recommendations
        breakdown = {}
        if recs_df is not None and not recs_df.empty:
            # Sort to ensure we get the row for the most recent period ('0m' is current)
            # Typically yfinance returns rows representing 0m, -1m, -2m, -3m
            latest_row = recs_df.iloc[0]
            breakdown = {
                "strong_buy": int(latest_row.get("strongBuy", 0)),
                "buy": int(latest_row.get("buy", 0)),
                "hold": int(latest_row.get("hold", 0)),
                "sell": int(latest_row.get("sell", 0)),
                "strong_sell": int(latest_row.get("strongSell", 0)),
            }
        # Construct the clean dictionary output
        ratings_data = {
            "symbol": ticker.upper(),
            "current_price": current_price,
            "consensus": consensus_text,  # e.g., 'buy', 'hold', 'strong_buy'
            "rating_breakdown": breakdown or "No breakdown data available",
        }
        return ratings_data

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        """Real, confirmed bug fixed here (found in a later sweep, same
        class as get_dividend_history()'s pre-fix bug): declared
        -> list[dict] (matching StockDataProvider's ABC signature) but
        every code path actually returned a pd.DataFrame — the "no data"
        branch, the "no recognizable date column" branch, and the main
        filtered-results path. Also used print() instead of structlog
        (CLAUDE.md violation). Currently unreachable via router.py (US
        routes get_insider_trading to edgartools only, per CLAUDE.md's
        hard "never FMP" rule; CA routes to openbb_tmx), but a real bug in
        the adapter regardless — directly callable on its own."""
        stock = yf.Ticker(ticker)
        df_insider = stock.insider_transactions
        if df_insider is None or df_insider.empty:
            logger.info("yfinance_no_insider_trading", ticker=ticker)
            return []
        if "Start Date" in df_insider.columns:
            date_col = "Start Date"
        elif "Date" in df_insider.columns:
            date_col = "Date"
        else:
            logger.warning("yfinance_insider_trading_no_date_column", ticker=ticker)
            return df_insider.to_dict("records")
        df_insider[date_col] = pd.to_datetime(df_insider[date_col])
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_df = df_insider[df_insider[date_col] >= cutoff_date]
        return filtered_df.to_dict("records")

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        """
        Finds industry peers for a given stock using yfinance screeners.
        :param ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        :param limit: Maximum number of peers to return (default 5)
        :return: A list of ticker strings representing peer companies
        """
        try:
            # Initialize the target stock to find its classification
            target_stock = yf.Ticker(ticker)
            info = target_stock.info
            # Extract industry key (safely fallback to sector if missing)
            industry_name = info.get("industry") or info.get("sector")
            if not industry_name:
                print(f"Could not identify industry classification for {ticker}.")
                return []
            # Use the yfinance Screener to fetch companies matching this industry
            # We create an EquityQuery filtering by the parsed industry name
            query = yf.EquityQuery("eq", ["industry", industry_name])
            screener = yf.Screener()
            screener.set_body(
                {
                    "size": limit + 5,  # Pull slightly more to filter out the target ticker itself
                    "offset": 0,
                    "sortField": "intradaymarketcap",  # Sort by market cap to get closest major competitors
                    "sortType": "DESC",
                    "quoteType": "EQUITY",
                    "query": query.to_dict(),
                }
            )
            # Execute search
            results = screener.response
            quotes = results.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            # Filter the response to clean the final array
            peer_list = []
            target_upper = ticker.upper()
            for item in quotes:
                peer_symbol = item.get("symbol", "").upper()
                # Ensure we don't include the input company in its own peer list
                if peer_symbol and peer_symbol != target_upper:
                    peer_list.append(peer_symbol)
                if len(peer_list) >= limit:
                    break
            return peer_list
        except Exception as e:
            print(f"Error retrieving peer data: {e}")
            return []

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        """Retrieves the upcoming earnings dates and EPS consensus estimates.

        Real, confirmed bug fixed here (found in a later sweep, same class
        as get_dividend_history()'s pre-fix bug): declared -> list[dict] but
        every code path actually returned a single dict, never a list —
        including the "no data" branch, which returned a dict with a
        human-readable "Status" message instead of the empty list the type
        contract promises. yfinance.calendar only ever exposes one upcoming
        earnings event per ticker (unlike FMP's get_earnings_calendar,
        which returns many companies' events in a date window), so the fix
        here is to wrap that single record in a list, not to build a
        multi-event lookup the underlying data doesn't support. Currently
        unreachable via router.py (neither US_CHAINS nor CA_CHAINS route
        get_earnings_calendar to yfinance), but a real bug in the adapter
        regardless — directly callable on its own.

        No shared record shape exists yet between this and FMP's raw,
        unmapped earnings-calendar rows — deliberately not inventing one
        here, same reasoning as get_price_history() staying unnormalized
        in 86bbb001k: no real consumer exists yet to ground it against.
        """
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty) or not cal:
            return []
        earnings_date = cal.get("Earnings Date", ["N/A"])
        if isinstance(earnings_date, list) and len(earnings_date) > 0:
            formatted_dates = [
                d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in earnings_date
            ]
        else:
            formatted_dates = [str(earnings_date)]
        return [
            {
                "Symbol": ticker.upper(),
                "Upcoming Earnings Dates": formatted_dates,
                "EPS Estimate": cal.get("Earnings Average", "N/A"),
                "EPS High Estimate": cal.get("Earnings High", "N/A"),
                "EPS Low Estimate": cal.get("Earnings Low", "N/A"),
                "Revenue Estimate": cal.get("Revenue Average", "N/A"),
            }
        ]

    async def get_dividend_history(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[NormalizedDividendRecord]:
        """Retrieves historical dividend payments for a stock within a
        specific date range, mapped onto NormalizedDividendRecord (86bbb001k).

        Real, confirmed bug fixed here: this method declared -> list[dict]
        but every code path actually returned a pd.DataFrame — a type-
        contract violation (not just an unnormalized shape) that would have
        silently iterated column-name strings instead of dividend records
        for any caller treating the result as list[dict], per the ABC's own
        contract. payment_date is a real, disclosed gap for yfinance
        specifically: its raw .dividends Series has only the ex-dividend
        date, no separate payment date (unlike FMP/openbb_tmx, both
        confirmed live to have a real paymentDate/payment_date field).

        :param ticker: Stock ticker symbol (e.g., 'KO', 'MSFT')
        :param from_date: Start date string in 'YYYY-MM-DD' format
        :param to_date: End date string in 'YYYY-MM-DD' format
        """
        stock = yf.Ticker(ticker)
        dividends_series = stock.dividends
        if dividends_series is None or dividends_series.empty:
            logger.info("yfinance_no_dividend_history", ticker=ticker)
            return []
        # Normalize the index to plain date strings first so from_date/to_date
        # comparison is uniform regardless of yfinance's tz-aware index —
        # avoids the previous code's separate KeyError-catch branch entirely.
        df = dividends_series.to_frame(name="amount_per_share").reset_index()
        df.columns = ["ex_date", "amount_per_share"]
        df["ex_date"] = df["ex_date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        filtered = df[(df["ex_date"] >= from_date) & (df["ex_date"] <= to_date)]
        return [
            NormalizedDividendRecord(
                ex_date=row["ex_date"],
                payment_date=None,
                amount_per_share=float(row["amount_per_share"]),
            )
            for row in filtered.to_dict("records")
        ]

    async def get_quote(self, ticker: str) -> NormalizedQuote:
        """Not on StockDataProvider. Maps onto NormalizedQuote (86bbb17pw),
        matching FMPDataProvider.get_quote's now-normalized shape so the
        US-equity composite router can fall back here transparently — both
        return the same fields now, so no provider-specific shape-mirroring
        is needed the way the old raw-FMP-shaped version required. Yahoo's
        fast_info raises KeyError rather than returning empty for an
        invalid/delisted ticker, so that's the one failure mode this needs
        to catch explicitly. Confirmed live 2026-08-10: fast_info already
        has year_high/year_low directly, same as FMP's yearHigh/yearLow —
        no derivation from get_price_history() needed.

        The camelCase keys below (lastPrice/marketCap/yearHigh/yearLow) are
        NOT what dir(fast_info) shows — its real attributes are snake_case
        (last_price/market_cap/year_high/year_low), and fast_info.get()
        with the snake_case name actually returns None. This looks like a
        bug on sight. It isn't: FastInfo.get() supports a separate set of
        legacy camelCase aliases as a backward-compat layer, confirmed live
        to resolve to the exact same values as the real attributes for all
        four keys used here. Don't switch these to snake_case without
        re-verifying live first."""
        stock = yf.Ticker(ticker)
        try:
            fi = stock.fast_info
            price = fi.get("lastPrice")
        except Exception:
            return {}
        if price is None:
            return {}
        return NormalizedQuote(
            current_price=price,
            market_cap=fi.get("marketCap"),
            # .get("currency") or "" (not .get("currency", "")) — confirmed
            # live that FastInfo.get(key, default) only falls back to
            # `default` for a genuinely unrecognized key, not when a
            # recognized key's value is None (e.g. fi.get("marketCap", "X")
            # on ^GSPC returns real None, not "X") — same bug pattern as
            # get_company_info()'s earlier .get(key, "") fix, caught here
            # by testing a ticker (^GSPC) where market_cap is genuinely None.
            currency=fi.get("currency") or "",
            high_52w=fi.get("yearHigh"),
            low_52w=fi.get("yearLow"),
        )


class YFinanceNewsProvider(NewsProvider):
    """news data (FMP news, Finnhub news with sentiment)."""

    async def get_news(self, ticker: str, days: int) -> list[dict]:
        """
        Retrieves recent news articles for a given stock ticker, filtered by a day-based lookback window.
        :param ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA')
        :param days: Maximum age of articles in days to include
        :return: A list of dictionaries containing filtered article metadata
        """
        # Instantiate the Ticker object and fetch the raw news feed
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        # Guard clause: Handle instances where no news or feed items are available
        if not raw_news:
            print(f"No recent news articles found for {ticker.upper()}.")
            return []
        # Calculate the historical cutoff boundary in UNIX epoch time
        # providerPublishTime is formatted in seconds since Jan 1, 1970
        cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
        filtered_news = []
        # 3. Iterate through the articles and apply the chronological filter
        for article in raw_news:
            publish_time = article.get("providerPublishTime")
            if publish_time and publish_time >= cutoff_timestamp:
                # Convert the raw epoch integer into a human-readable string format
                readable_date = datetime.fromtimestamp(publish_time).strftime("%Y-%m-%d %H:%M:%S")
                # Construct a cleaned response item mapping the most actionable fields
                cleaned_article = {
                    "headline": article.get("title", "N/A"),
                    "publisher": article.get("publisher", "N/A"),
                    "url": article.get("link", "N/A"),
                    "published_at": readable_date,
                    "type": article.get("type", "N/A"),  # e.g., 'STORY' or 'VIDEO'
                }
                filtered_news.append(cleaned_article)
        return filtered_news

    async def get_analyst_recommendation_trends(self, ticker: str) -> list[dict]:
        """
        Retrieves historical month-by-month analyst consensus rating trends.
        :param ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        :return: A list of dictionaries containing historical rating breakdowns.
        """
        stock = yf.Ticker(ticker)
        # Fetch the raw recommendation trend DataFrame
        recs_df = stock.recommendations
        # Guard clause: Handle instances where Yahoo Finance has no historical trend record
        if recs_df is None or recs_df.empty:
            print(f"No recommendation trend data found for {ticker.upper()}.")
            return []
        # Reset index to ensure the period identifier (e.g., '0m', '-1m') becomes a regular column
        # yfinance often leaves 'period' as a column or index depending on the version
        df_clean = recs_df.copy()
        if df_clean.index.name is not None:
            df_clean = df_clean.reset_index()
        # Standardize column names to lowercase/snake_case for clean API dictionaries (optional but recommended)
        rename_map = {
            "period": "period",
            "strongBuy": "strong_buy",
            "buy": "buy",
            "hold": "hold",
            "sell": "sell",
            "strongSell": "strong_sell",
        }
        df_clean = df_clean.rename(columns=rename_map)
        # Convert the DataFrame rows directly into a list of Python dictionaries
        trend_list = df_clean.to_dict(orient="records")
        return trend_list


class MacroDataProvider(MacroDataProvider):
    """macroeconomic data (FRED, Bank of Canada Valet)."""

    async def get_macro_data(self, series_ids: list[str]) -> dict[str, pd.Series]:
        """
        Retrieves macroeconomic time-series data from FRED.
        :param series_ids: List of FRED series identification strings
                           (e.g., ['FEDFUNDS', 'CPIAUCSNS', 'UNRATE'])
        :return: A dictionary mapping each series ID to its corresponding Pandas Series
        """
        macro_dict = {}
        # Set a reasonable historical start date for macro trends
        start_date = datetime(2010, 1, 1)
        end_date = datetime.now()
        for series_id in series_ids:
            clean_id = series_id.upper().strip()
            try:
                # Fetch data directly from the 'fred' data source
                df = web.DataReader(clean_id, "fred", start_date, end_date)
                if df is not None and not df.empty:
                    # DataReader returns a DataFrame; extract the column as a pure Series
                    macro_dict[clean_id] = df[clean_id]
                else:
                    macro_dict[clean_id] = pd.Series(dtype="float64")
            except Exception as e:
                print(f"Error fetching macroeconomic series '{clean_id}': {e}")
                # Provide an empty series as a fallback to prevent pipeline crashes
                macro_dict[clean_id] = pd.Series(dtype="float64")
        return macro_dict

    async def get_interest_rates(self) -> dict:
        """
        Retrieves the most recent values for major US benchmark interest rates from FRED.
        :return: A dictionary mapping rate names to their latest percentage values.
        """
        # Define the core macro rate ticker mappings from FRED
        rate_series = {
            "fed_funds_effective": "FEDFUNDS",  # Federal Funds Effective Rate
            "treasury_3mo": "TB3MS",  # 3-Month Treasury Bill Secondary Market Rate
            "treasury_2yr": "DGS2",  # 2-Year Treasury Constant Maturity Yield
            "treasury_10yr": "DGS10",  # 10-Year Treasury Constant Maturity Yield
        }
        # Set a short 30-day window to minimize payload while ensuring we capture daily/monthly prints
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        rates_output = {}
        try:
            # Batch fetch all data points from FRED in one call to reduce API overhead
            df = web.DataReader(list(rate_series.values()), "fred", start_date, end_date)
            # Map the response series IDs back to human-readable dictionary keys
            for human_name, fred_id in rate_series.items():
                if fred_id in df.columns:
                    # Drop NaN rows for this column and isolate the most recent valid numeric index value
                    valid_series = df[fred_id].dropna()
                    if not valid_series.empty:
                        # Extract the final number and round to 2 decimal places
                        rates_output[human_name] = round(float(valid_series.iloc[-1]), 2)
                    else:
                        rates_output[human_name] = None
                else:
                    rates_output[human_name] = None
        except Exception as e:
            print(f"Error connecting to FRED endpoint: {e}")
            # Fallback values if the user's local network/API context crashes
            return {name: None for name in rate_series.keys()}
        return rates_output

    async def get_exchange_rates(self, pair: str = "CADUSD") -> dict:
        """
        Retrieves live and historical benchmark pricing for a specific fiat currency pair.

        :param pair: Currency combo string, 6 characters long (e.g., 'CADUSD', 'EURUSD')
        :return: A dictionary containing structural exchange rate data metrics.
        """
        # Clean the input and format it with the required Yahoo Finance '=X' suffix
        clean_pair = pair.strip().upper()
        if len(clean_pair) == 6:
            ticker_symbol = f"{clean_pair}=X"
        elif len(clean_pair) == 8 and clean_pair.endswith("=X"):
            ticker_symbol = clean_pair
            clean_pair = clean_pair.replace("=X", "")
        else:
            raise ValueError("Invalid format. Use a 6-character string like 'CADUSD'.")
        try:
            # Instantiate the Ticker object
            fx_ticker = yf.Ticker(ticker_symbol)
            info = fx_ticker.info

            # Guard clause: Verify that Yahoo Finance recognized the FX ticker symbol
            if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
                return {"pair": clean_pair, "status": "No exchange rate data found."}

            # Build a structured dictionary output mapping core pricing metrics
            # yfinance uses 'regularMarketPrice' or 'currentPrice' for FX pairs depending on version
            exchange_data = {
                "pair": clean_pair,
                "base_currency": clean_pair[:3],
                "quote_currency": clean_pair[3:],
                "exchange_rate": info.get("regularMarketPrice") or info.get("currentPrice", "N/A"),
                "previous_close": info.get("previousClose", "N/A"),
                "day_open": info.get("open", "N/A"),
                "day_low": info.get("dayLow", "N/A"),
                "day_high": info.get("dayHigh", "N/A"),
            }
            return exchange_data

        except Exception as e:
            print(f"Error querying Yahoo Finance for currency pair {clean_pair}: {e}")
            return {"pair": clean_pair, "status": "Error executing API extraction loop."}
