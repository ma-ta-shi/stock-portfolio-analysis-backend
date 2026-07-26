import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pandas_datareader.data as web
from base import StockDataProvider, NewsProvider, MacroDataProvider
import time

# Map the combination of period and statement to the correct yfinance property
MAPPING = {
    ('annual', 'income'): 'financials',
    ('annual', 'balance'): 'balance_sheet',
    ('annual', 'cashflow'): 'cashflow',
    ('quarterly', 'income'): 'quarterly_financials',
    ('quarterly', 'balance'): 'quarterly_balance_sheet',
    ('quarterly', 'cashflow'): 'quarterly_cash_flow'
}

class YFinanceDataProvider(StockDataProvider):
    """Abstract base for stock-centric financial data yfinance."""
    
    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        history = stock.history(period, interval)
        return history # histroy hould be a panda data frame already

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        key = (period.lower(), statement.lower())
        if key not in MAPPING:
            raise ValueError("Invalid statement or period. Use 'income'/'balance'/'cashflow' and 'annual'/'quarterly'")
        # Dynamically extract the attribute from the Ticker object
        attribute_name = MAPPING[key]
        df = getattr(stock, attribute_name)
        return df
    
    async def get_company_info(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Safely extract key metadata points using .get() to prevent KeyError if data is missing
        company_profile = {
            "Symbol": info.get("symbol", ticker.upper()),
            "Company Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Country": info.get("country", "N/A"),
            "Full-Time Employees": info.get("fullTimeEmployees", "N/A"),
            "Website": info.get("website", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Business Summary": info.get("longBusinessSummary", "N/A")
        }
        return company_profile

    async def get_analyst_estimates(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        # Fetch Price Targets safely
        targets = stock.analyst_price_targets
        # Fetch Recommendation Trends (DataFrame)
        recs_df = stock.recommendations
        # Extract the most recent month's rating breakdown if available
        latest_consensus = {}
        if recs_df is not None and not recs_df.empty:
            # The first row (index 0) typically represents the current month ('0m')
            latest_row = recs_df.iloc[0]
            latest_consensus = {
                "Period": latest_row.get("period", "Current"),
                "Strong Buy": int(latest_row.get("strongBuy", 0)),
                "Buy": int(latest_row.get("buy", 0)),
                "Hold": int(latest_row.get("hold", 0)),
                "Sell": int(latest_row.get("sell", 0)),
                "Strong Sell": int(latest_row.get("strongSell", 0))
            }
        # Consolidate the data structure
        estimates = {
            "Symbol": ticker.upper(),
            "Price Targets": {
                "Low": targets.get("low", "N/A"),
                "High": targets.get("high", "N/A"),
                "Mean": targets.get("mean", "N/A"),
                "Median": targets.get("median", "N/A")
            },
            "Latest Consensus Counts": latest_consensus or "No recommendation data available"
        }
        return estimates
    
    async def get_analyst_ratings(self, ticker: str) -> dict:
        stock = yf.Ticker(ticker)
        #Fetch the overall consensus text from .info safely
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
                "strong_sell": int(latest_row.get("strongSell", 0))
            }
        # Construct the clean dictionary output
        ratings_data = {
            "symbol": ticker.upper(),
            "current_price": current_price,
            "consensus": consensus_text,  # e.g., 'buy', 'hold', 'strong_buy'
            "rating_breakdown": breakdown or "No breakdown data available"
        }
        return ratings_data

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        stock = yf.Ticker(ticker)
        # Fetch raw insider transaction data (returns a Pandas DataFrame)
        df_insider = stock.insider_transactions
        # Guard clause: Handle instances where Yahoo Finance has no transaction record
        if df_insider is None or df_insider.empty:
            print(f"No insider trading data found for {ticker.upper()}.")
            return pd.DataFrame() # Return empty DataFrame to prevent breaking downstream pipelines
        # Ensure the date column is parsed as datetime objects for mathematical comparison
        # Note: yfinance typically puts the transaction date in the 'Start Date' column
        if 'Start Date' in df_insider.columns:
            date_col = 'Start Date'
        elif 'Date' in df_insider.columns:
            date_col = 'Date'
        else:
            # If no obvious date column exists, return the raw data safely
            return df_insider
        df_insider[date_col] = pd.to_datetime(df_insider[date_col])
        # Calculate the boundary cutoff date based on the 'days' parameter
        cutoff_date = datetime.now() - timedelta(days=days)
        # Filter for rows where the transaction occurred after or on the cutoff date
        filtered_df = df_insider[df_insider[date_col] >= cutoff_date]
        return filtered_df

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
            query = yf.EquityQuery('eq', ['industry', industry_name])
            screener = yf.Screener()
            screener.set_body({
                "size": limit + 5, # Pull slightly more to filter out the target ticker itself
                "offset": 0,
                "sortField": "intradaymarketcap", # Sort by market cap to get closest major competitors
                "sortType": "DESC",
                "quoteType": "EQUITY",
                "query": query.to_dict()
            })
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
        """
        Retrieves the upcoming earnings dates and EPS consensus estimates.
        :param ticker: Stock ticker symbol (e.g., 'AAPL', 'AMD')
        :return: A dictionary containing scheduled reporting metrics.
        """
        stock = yf.Ticker(ticker)
        # Fetch calendar data (typically returns a dictionary or empty DataFrame)
        cal = stock.calendar
        # Guard clause: Handle stocks that do not have an upcoming date scheduled
        if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty) or not cal:
            return {
                "Symbol": ticker.upper(),
                "Status": f"No upcoming earnings calendar data found for {ticker.upper()}."
            }
        # Extract components safely (handling list-based or scalar values)
        earnings_date = cal.get("Earnings Date", ["N/A"])
        # yfinance often packages dates as a list of timestamps if the time is unconfirmed
        if isinstance(earnings_date, list) and len(earnings_date) > 0:
            # Format dates to clean string format (YYYY-MM-DD)
            formatted_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in earnings_date]
        else:
            formatted_dates = [str(earnings_date)]
        # Build a structured output dictionary
        calendar_data = {
            "Symbol": ticker.upper(),
            "Upcoming Earnings Dates": formatted_dates,
            "EPS Estimate": cal.get("Earnings Average", "N/A"),
            "EPS High Estimate": cal.get("Earnings High", "N/A"),
            "EPS Low Estimate": cal.get("Earnings Low", "N/A"),
            "Revenue Estimate": cal.get("Revenue Average", "N/A")
        }
        return calendar_data

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """
        Retrieves historical dividend payments for a stock within a specific date range.
        :param ticker: Stock ticker symbol (e.g., 'KO', 'MSFT')
        :param from_date: Start date string in 'YYYY-MM-DD' format
        :param to_date: End date string in 'YYYY-MM-DD' format
        :return: A Pandas DataFrame with dates and dividend amounts
        """
        stock = yf.Ticker(ticker)
        # Fetch the complete raw dividend history Series
        dividends_series = stock.dividends
        # Guard clause: Check if the stock pays a dividend at all
        if dividends_series is None or dividends_series.empty:
            print(f"No dividend history found for {ticker.upper()}.")
            return pd.DataFrame(columns=["Date", "Dividend"])
        # Filter the Series using the date parameters
        # Slicing works directly with string dates in Pandas Series
        try:
            filtered_series = dividends_series.loc[from_date:to_date]
        except KeyError:
            # Handle edge case where exact boundary dates cause lookup errors
            # Convert series index to string dates for uniform comparison
            df_temp = dividends_series.to_frame().reset_index()
            df_temp['Date'] = df_temp['Date'].dt.strftime('%Y-%m-%d')
            filtered_df = df_temp[(df_temp['Date'] >= from_date) & (df_temp['Date'] <= to_date)]
            filtered_df.columns = ["Date", "Dividend"]
            return filtered_df
        # Restructure the sliced Series into a clean DataFrame
        df_dividends = filtered_series.to_frame().reset_index()
        df_dividends.columns = ["Date", "Dividend"]
        # Clean up the Date column format (remove timezone offset if present)
        df_dividends['Date'] = df_dividends['Date'].dt.tz_localize(None)
        return df_dividends

class NewsProvider(NewsProvider):
    """ news data (FMP news, Finnhub news with sentiment)."""
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
                readable_date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M:%S')
                # Construct a cleaned response item mapping the most actionable fields
                cleaned_article = {
                    "headline": article.get("title", "N/A"),
                    "publisher": article.get("publisher", "N/A"),
                    "url": article.get("link", "N/A"),
                    "published_at": readable_date,
                    "type": article.get("type", "N/A") # e.g., 'STORY' or 'VIDEO'
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
            "strongSell": "strong_sell"
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
                df = web.DataReader(clean_id, 'fred', start_date, end_date)
                if df is not None and not df.empty:
                    # DataReader returns a DataFrame; extract the column as a pure Series
                    macro_dict[clean_id] = df[clean_id]
                else:
                    macro_dict[clean_id] = pd.Series(dtype='float64')
            except Exception as e:
                print(f"Error fetching macroeconomic series '{clean_id}': {e}")
                # Provide an empty series as a fallback to prevent pipeline crashes
                macro_dict[clean_id] = pd.Series(dtype='float64')  
        return macro_dict
    
    async def get_interest_rates(self) -> dict:
        """
        Retrieves the most recent values for major US benchmark interest rates from FRED.
        :return: A dictionary mapping rate names to their latest percentage values.
        """
        # Define the core macro rate ticker mappings from FRED
        rate_series = {
            "fed_funds_effective": "FEDFUNDS",   # Federal Funds Effective Rate
            "treasury_3mo": "TB3MS",             # 3-Month Treasury Bill Secondary Market Rate
            "treasury_2yr": "DGS2",              # 2-Year Treasury Constant Maturity Yield
            "treasury_10yr": "DGS10",            # 10-Year Treasury Constant Maturity Yield
        }
        # Set a short 30-day window to minimize payload while ensuring we capture daily/monthly prints
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        rates_output = {}
        try:
            # Batch fetch all data points from FRED in one call to reduce API overhead
            df = web.DataReader(list(rate_series.values()), 'fred', start_date, end_date)
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
                "day_high": info.get("dayHigh", "N/A")
            }
            return exchange_data

        except Exception as e:
            print(f"Error querying Yahoo Finance for currency pair {clean_pair}: {e}")
            return {"pair": clean_pair, "status": "Error executing API extraction loop."}
