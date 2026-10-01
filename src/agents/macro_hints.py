"""The two per-stock hints the Macro Economist prompt is written around.

Rule 5 of the prompt tells the model to prioritise "the SECTOR MACRO SENSITIVITIES list",
and rule 6 to apply "the currency exposure hint". Both placeholders were being filled with
an empty string, so neither rule had anything to act on. The wording is the design doc's
(`docs/agents/macro_economist.md`, "Sector macro sensitivity hint" and "Currency exposure
hint").

Only the currency rows that can be told apart today are reachable: the two Canadian rows
that depend on `usd_revenue_exposure_pct` (30-60% and over 60% USD revenue) need a revenue
source the provider layer does not have, so every Canadian listing gets the "unknown"
row, exactly as the doc specifies for unknown revenue.
"""

import structlog

from data.sector_names import normalize_sector

logger = structlog.get_logger(__name__)

_SECTOR_HINTS = {
    "Financials": "net_interest_margin, yield_curve_slope, credit_loss_provisioning, unemployment_trend",
    "Real Estate": "10yr_yield (cap_rate_driver), mortgage_rates, inflation_pass_through_to_rents",
    "Utilities": "10yr_yield (bond_proxy), natural_gas_price, regulatory_rate_base",
    "Energy": "oil_price (WTI), usd_strength, inventory_levels",
    "Materials": "industrial_commodity_prices, usd_strength, china_demand",
    "Industrials": "PMI, capex_cycle, interest_rates (financing)",
    "Consumer Staples": "inflation_pass_through_to_price, input_costs (agriculture), unemployment_trend",
    "Consumer Discretionary": "unemployment_trend, real_wage_growth, interest_rates (big-ticket financing)",
    "Healthcare": "demographic_trends, regulatory_environment, usd_strength (if multinational)",
    "Technology": "interest_rates (duration risk), usd_strength (if multinational), capex_cycle",
    "Communication Services": "interest_rates (capex financing), advertising_cycle, regulatory_environment",
}
_DEFAULT_SECTOR_HINT = "interest_rates, inflation, economic_growth, usd_strength"

_US_CURRENCY_HINT = (
    "Currency exposure: primarily_usd. Reports in USD, primarily US market. CAD/USD moves "
    "affect a Canadian investor's after-tax return but are not a direct operational factor."
)
_CANADIAN_CURRENCY_HINT = (
    "Currency exposure: primarily_cad. Reports in CAD, primarily Canadian revenue. CAD/USD "
    "moves mainly affect input-cost lines (USD-priced commodities or imports) and price "
    "relative to US peers."
)


def sector_macro_hint(sector: str | None) -> str:
    """The transmission mechanisms that matter for this sector; the doc's Default row
    when the sector is missing or not one of the 11 (logged, so a new provider spelling
    is noticed rather than silently getting generic macro advice)."""
    canonical = normalize_sector(sector)
    if canonical is None:
        logger.warning("macro_sector_hint_unmapped_sector", sector=sector)
        return _DEFAULT_SECTOR_HINT
    return _SECTOR_HINTS[canonical]


def currency_exposure_hint(is_canadian: bool) -> str:
    return _CANADIAN_CURRENCY_HINT if is_canadian else _US_CURRENCY_HINT
