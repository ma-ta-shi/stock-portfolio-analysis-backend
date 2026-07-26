import pandas as pd
import aiohttp
from .base import MacroDataProvider
import logging

logger = logging.getLogger(__name__)

class BOCMacroDataProvider(MacroDataProvider):
    """
    Bank of Canada Valet API provider.
    
    - MacroDataProvider: Fully supported (interest rates, exchange rates, macro data)
    - BOCStockDataProvider: Not supported (returns empty/passes)
    - NewsProvider: Not supported (returns empty/passes)
    
    API Docs: https://www.bankofcanada.ca/valet/
    """
    
    BASE_URL = "https://www.bankofcanada.ca/valet"
    
    def __init__(self, session: aiohttp.ClientSession = None):
        """Initialize the Bank of Canada Valet provider.
        
        Args:
            session: Optional aiohttp ClientSession for making requests.
        """
        self.session = session
        self._owns_session = session is None
    
    async def __aenter__(self):
        if self._owns_session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session and self.session:
            await self.session.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._owns_session = True
        return self.session
    
    async def _fetch_json(self, url: str, params: dict = None) -> dict:
        """Fetch JSON data from Bank of Canada Valet API.
        
        Args:
            url: API endpoint URL.
            params: Query parameters.
        
        Returns:
            Parsed JSON response.
        """
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from {url}: {e}")
            raise
    
    # ============================================================================
    # MacroDataProvider Implementation (Supported)
    # ============================================================================
    
    async def get_macro_data(self, series_ids: list[str], start_date:str, end_date:str) -> dict[str, pd.Series]:
        """Fetch macroeconomic data from Bank of Canada Valet.
        
        Args:
            series_ids: List of Bank of Canada series IDs (e.g., ['FXUSDCAD', 'V39079']).
        
        Returns:
            Dictionary mapping series_id to pd.Series with dates and values.
        
        Example series IDs:
            - FXUSDCAD: USD/CAD exchange rate
            - FXCADAUD: CAD/AUD exchange rate
            - V80691311: Prime rate
            - INTMP02: Alternative rate
            - V39079: Overnight rate
        """
        result = {}
        for series_id in series_ids:
            try:
                url = f"{self.BASE_URL}/observations/{series_id}/json?start_date={start_date}&end_date={end_date}"
                data = await self._fetch_json(url)
                
                if "observations" in data:
                    observations = data["observations"]
                    dates = []
                    values = []
                    
                    for obs in observations:
                        date_str = obs.get("d")
                        value_str = obs.get(series_id).get('v')
                        if date_str and value_str:
                            try:
                                date = pd.to_datetime(date_str)
                                value = float(value_str)
                                dates.append(date)
                                values.append(value)
                            except (ValueError, TypeError):
                                continue
                    if dates and values:
                        result[series_id] = pd.Series(values, index=dates, name=series_id)
                    else:
                        logger.warning(f"No valid data found for series {series_id}")
                
            except Exception as e:
                logger.error(f"Error fetching macro data for {series_id}: {e}")
                continue
        
        return result
    
    async def get_interest_rates(self, start_date:str, end_date:str) -> dict:
        """Fetch current interest rates from Bank of Canada.
        
        Returns:
            Dictionary with interest rate data.
            Example: {
                'overnight_rate': 5.0,
                'prime_rate': 7.2,
                'date': '2024-01-15'
            }
        """
        try:
            # Fetch overnight rate (V39079)
            overnight_url = f"{self.BASE_URL}/observations/V39079/json?start_date={start_date}&end_date={end_date}"
            overnight_data = await self._fetch_json(overnight_url)
            print(f"overnight_data:{overnight_data}")
            
            # Fetch prime rate (V80691311)
            prime_url = f"{self.BASE_URL}/observations/V80691311/json"
            prime_data = await self._fetch_json(prime_url)
            
            result = {}
            
            if "observations" in overnight_data and overnight_data["observations"]:
                latest = overnight_data["observations"][-1]
                result["overnight_rate"] = float(latest.get("V39079", 0).get('v', None))
                result["date"] = latest.get("d")
            
            if "observations" in prime_data and prime_data["observations"]:
                latest = prime_data["observations"][-1]
                result["prime_rate"] = float(latest.get("V80691311", 0).get('v', None))
                if "date" not in result:
                    result["date"] = latest.get("d")
            
            return result
        except Exception as e:
            logger.error(f"Error fetching interest rates: {e}")
            return {}
    
    async def get_exchange_rates(self, pair: str = "CADUSD") -> dict:
        """Fetch exchange rate data from Bank of Canada.
        
        Args:
            pair: Currency pair (e.g., 'CADUSD', 'USDCAD').
                  Default is 'CADUSD' (Canadian dollars per USD).
        
        Returns:
            Dictionary with exchange rate info.
            Example: {
                'pair': 'USDCAD',
                'rate': 1.25,
                'date': '2024-01-15'
            }
        """
        try:
            # Common exchange rate series IDs
            series_map = {
                "USDCAD": "FXUSDCAD",  # USD to CAD
                "CADUSD": "FXUSDCAD",  # CAD to USD (inverse)
                "EURCAD": "FXEURCAD",  # EUR to CAD
                "JPYCAD": "FXJPYCAD",  # JPY to CAD
                "GBPCAD": "FXGBPCAD",  # GBP to CAD
            }
            
            series_id = series_map.get(pair.upper(), f"FX{pair.upper()}")
            url = f"{self.BASE_URL}/observations/{series_id}/json"
            data = await self._fetch_json(url)
            
            result = {"pair": pair}
            
            if "observations" in data and data["observations"]:
                latest = data["observations"][-1]
                rate = latest.get(series_id)
                
                if rate:
                    rate_value = float(rate['v'])
                    
                    # If asking for CADUSD, invert the USDCAD rate
                    if pair.upper() == "CADUSD" and series_id == "FXUSDCAD":
                        rate_value = 1.0 / rate_value
                    
                    result["rate"] = rate_value
                    result["date"] = latest.get("d")
            
            return result
        except Exception as e:
            logger.error(f"Error fetching exchange rate for {pair}: {e}")
            return {"pair": pair, "rate": None, "date": None}
