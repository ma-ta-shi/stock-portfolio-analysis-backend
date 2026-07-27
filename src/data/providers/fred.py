import asyncio
import os

import pandas as pd
import structlog
from fredapi import Fred

from data.providers.base import MacroDataProvider

logger = structlog.get_logger(__name__)

# Documented series set (financial-data-api-research.md §6) — covers US macro plus
# Canadian CPI/unemployment mirrored through OECD series. Default for get_macro_data()
# when the caller doesn't need a custom subset.
DEFAULT_SERIES_IDS = [
    "FEDFUNDS",
    "CPIAUCSL",
    "CPILFESL",
    "GDP",
    "UNRATE",
    "VIXCLS",
    "DGS10",
    "CPALTT01CAM657N",
    "LRUNTTTTCAM156S",
]

# Same benchmark-rate block yfinance.py's stopgap implementation already returns —
# keep the shape stable so the Macro Economist consumer contract doesn't change.
RATE_SERIES = {
    "fed_funds_effective": "FEDFUNDS",
    "treasury_3mo": "TB3MS",
    "treasury_2yr": "DGS2",
    "treasury_10yr": "DGS10",
}


class FredMacroDataProvider(MacroDataProvider):
    """US macro data via the official FRED REST API (not pandas_datareader scraping)."""

    def __init__(self) -> None:
        api_key = os.environ.get("SP_FRED_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SP_FRED_API_KEY is not set. Register for a free key at "
                "https://fred.stlouisfed.org and set it in the environment."
            )
        # Loaded once here, not per request — same pattern as edgartools' set_identity().
        self._client = Fred(api_key=api_key)

    async def get_macro_data(self, series_ids: list[str] | None = None) -> dict[str, pd.Series]:
        """Fetch FRED series (defaults to DEFAULT_SERIES_IDS). One bad series logs and
        falls back to an empty Series rather than failing the whole batch — 120 calls/min
        is generous for the ~10-series/batch volume, so these run sequentially, not
        concurrently. fredapi's HTTP call is blocking, so it's offloaded to a thread
        rather than run directly on the event loop."""
        result: dict[str, pd.Series] = {}
        for series_id in DEFAULT_SERIES_IDS if series_ids is None else series_ids:
            clean_id = series_id.strip().upper()
            try:
                result[clean_id] = await asyncio.to_thread(self._client.get_series, clean_id)
            except Exception:
                logger.warning("fred_series_fetch_failed", series_id=clean_id, exc_info=True)
                result[clean_id] = pd.Series(dtype="float64")
        return result

    async def get_interest_rates(self) -> dict:
        """Latest values for the standard US benchmark-rate block used every batch run."""
        rates: dict[str, float | None] = {}
        for name, series_id in RATE_SERIES.items():
            try:
                series = await asyncio.to_thread(self._client.get_series, series_id)
                series = series.dropna()
                rates[name] = round(float(series.iloc[-1]), 2) if not series.empty else None
            except Exception:
                logger.warning("fred_rate_fetch_failed", series_id=series_id, exc_info=True)
                rates[name] = None
        return rates

    async def get_exchange_rates(self, pair: str = "CADUSD") -> dict:
        """FRED has no FX data — CAD/USD is Bank of Canada Valet's job. Returns a
        status dict rather than raising, consistent with this class's "one bad thing
        shouldn't crash the caller" behavior elsewhere."""
        logger.debug("fred_exchange_rates_not_supported", pair=pair)
        return {
            "pair": pair.strip().upper(),
            "status": "not supported by FredMacroDataProvider — see BOCMacroDataProvider in boc.py",
        }
