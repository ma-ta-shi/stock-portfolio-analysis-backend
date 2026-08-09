"""MacroDataProvider live checks (ClickUp 86bb7j0kh).

MacroDataProvider isn't routed through router.py (confirmed in router.py's
own docstring: "Out of scope: ... MacroDataProvider routing") — FRED and
BoC are called directly in production, so there's no chain-fallback
dimension here, just per-provider completeness.
"""

import pandas as pd
import pytest

from data.providers.boc import BOCMacroDataProvider
from data.providers.fred import FredMacroDataProvider
from data.providers.yfinance import MacroDataProvider as YFinanceMacroDataProvider

pytestmark = pytest.mark.live


# ---------- FRED ----------


async def test_fred_valid_series_returns_real_data():
    provider = FredMacroDataProvider()
    result = await provider.get_macro_data(["FEDFUNDS"])
    assert isinstance(result["FEDFUNDS"], pd.Series)
    assert not result["FEDFUNDS"].empty


async def test_fred_dead_series_returns_empty_series_not_a_crash():
    """fred.py's own get_macro_data logs via logger.warning(...,
    exc_info=True) on a bad series — the same exc_info=True pattern that
    crashed router.py's fallback logging with UnicodeEncodeError on this
    Windows console (see test_router_chains.py). This test exercises that
    exact code path for FRED; if it errors instead of returning a clean
    empty Series, that's the same systemic logging bug, not a FRED-specific
    one."""
    provider = FredMacroDataProvider()
    result = await provider.get_macro_data(["NOTAREALSERIESID123"])
    assert isinstance(result["NOTAREALSERIESID123"], pd.Series)
    assert result["NOTAREALSERIESID123"].empty


async def test_fred_get_interest_rates_returns_real_values():
    provider = FredMacroDataProvider()
    result = await provider.get_interest_rates()
    assert isinstance(result, dict)
    assert result.get("fed_funds_effective") is not None


# ---------- Bank of Canada Valet ----------


async def test_boc_get_interest_rates_returns_real_values():
    """Also live-exercises the `latest.get(series_id, 0).get("v", None)`
    line flagged as a latent AttributeError risk during planning (if the
    key is genuinely absent from an observation, .get returns the int 0
    default, and .get("v") on an int raises) — this test only confirms
    normal-path behavior; forcing the missing-key case would need a
    mocked response, which isn't in scope for a live-only suite. Flagged
    as a code-review finding, not something this test can force-trigger."""
    async with BOCMacroDataProvider() as provider:
        result = await provider.get_interest_rates()
    assert isinstance(result, dict)
    assert result.get("overnight_rate") is not None
    assert result.get("prime_rate") is not None


async def test_boc_get_exchange_rates_returns_real_cadusd_rate():
    async with BOCMacroDataProvider() as provider:
        result = await provider.get_exchange_rates("CADUSD")
    assert isinstance(result, dict)
    assert result.get("rate") is not None
    assert 0 < result["rate"] < 2  # sanity bound, CAD/USD has never been outside this


# ---------- yfinance MacroDataProvider.get_exchange_rates ----------
# The real exchange-rate implementation this ticket means to test — FRED's
# own get_exchange_rates is a permanent "not supported" stub (see fred.py).


async def test_yfinance_exchange_rates_valid_pair_returns_real_data():
    provider = YFinanceMacroDataProvider()
    result = await provider.get_exchange_rates("CADUSD")
    assert isinstance(result, dict)
    assert "exchange_rate" in result


async def test_yfinance_exchange_rates_malformed_pair_raises_value_error():
    """Ticket's explicit ask: confirm yfinance's own malformed-pair guard
    still raises ValueError."""
    provider = YFinanceMacroDataProvider()
    with pytest.raises(ValueError):
        await provider.get_exchange_rates("NOTAPAIR")
