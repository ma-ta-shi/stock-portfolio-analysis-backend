"""macro_sources.py — full input contract for the Macro Economist agent
(ClickUp 86ban0wgz). Fetches FRED + BoC series, computes deltas/trends,
resolves sector-commodity relevance, pulls Fed/BoC-filtered central-bank
commentary, and assembles a MacroSourcesBundle. Pure orchestration +
arithmetic — no LLM calls, per CLAUDE.md's "pipeline does the math" rule.

Doc/contract reconciliation (disclosed, not silently resolved — several
real conflicts between the ticket's series list, data-pipeline.md's
MacroSourcesBundle dataclass, and financial-data-api-research.md):

- data-pipeline.md's precompute/macro_sources.py section (a) lists
  CANRGDPR and LRUNTTTTCAM156S among "core series to fetch," but neither
  has a destination field in MacroSourcesBundle (no ca_gdp/ca_unemployment
  field exists) — Canada unemployment is instead covered by the
  StatCan-sourced statcan_unemployment_ca field via a different provider,
  and Canada GDP has no modeled destination at all. Not fetched here.
- CPALTT01CAM657N (Canada CPI via OECD) IS fetched, despite
  MacroSourcesBundle's own inline comment grouping canada_cpi under
  "# BoC Valet series" — financial-data-api-research.md's agent-data-
  mapping table explicitly states "Canadian CPI, unemployment | FRED (via
  OECD)", and fred.py's own DEFAULT_SERIES_IDS already includes this
  series, confirming FRED (not BoC) is the real source. canada_cpi is
  FRED-sourced here.
- The ticket's (a) list groups FXCADUSD under "FRED:", but
  financial-data-api-research.md §7 lists FXCADUSD as a Bank of Canada
  Valet series, and boc.py's get_exchange_rates() already uses it (via
  FXUSDCAD inverted) for the contract's cad_usd field. cad_usd_fred
  instead uses FRED's DEXCAUS (live-verified 2026-08-07), which the
  ticket doesn't name at all — an omission in the other direction.
- usd_revenue_exposure_pct has no wired-up data source anywhere in the
  provider layer (no FMP revenue-by-geography method exists) — always
  None for this build, consistent with the field's Optional design.
  Building that provider method is out of this ticket's scope.
- Single delta/trend fields (policy_rate_90d_delta_bp/rate_trend,
  cpi_3m_delta_pp/cpi_trend) are always US-sourced (FEDFUNDS/CPIAUCSL) —
  matching gdp/unemployment/vix, which are unconditionally US-only in
  this contract (no CA-variant field exists for any of them). Canada gets
  its own CPI reading (canada_cpi) and its own bond-yield/BoC-rate block,
  but not a parallel delta/trend apparatus — this mirrors the contract's
  own asymmetric field design, not an invented convention.
- Materials sector maps to copper only, not "copper/gold" as the doc
  lists — GOLDAMGBD228NLBM and GOLDPMGBD228NLBM (the standard FRED gold
  fixing series) are both discontinued (confirmed live 2026-08-07, empty
  response from both). A working gold series is a known follow-up gap,
  not silently dropped — see _SECTOR_COMMODITY below.
- The doc's "12-row lookup table" claim is aspirational: only 4 sector
  mappings (Energy, Materials, Industrials, Utilities) are documented
  anywhere found; all other sectors fall through to
  sector_commodity_relevant=False.

Trend-classification thresholds (rate_trend/cpi_trend/cad_trend/
vix_regime) are first-pass, not sourced from any doc — same disclosure
pattern as fundamentals.py's health_rating thresholds.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Literal

import pandas as pd
import structlog

from data.providers.boc import BOCMacroDataProvider
from data.providers.finnhub import FinnhubDataProvider
from data.providers.fred import FredMacroDataProvider
from data.providers.stats_canada import StatsCanadaProvider
from data.schemas.common import CBCommentaryItem
from data.schemas.macro_sources_bundle import MacroSourcesBundle

logger = structlog.get_logger(__name__)

_FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "treasury_2y": "DGS2",
    "treasury_5y": "DGS5",
    "treasury_10y": "DGS10",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "gdp": "GDP",
    "unemployment": "UNRATE",
    "vix": "VIXCLS",
    "cad_usd_fred": "DEXCAUS",
    "wti_crude": "DCOILWTICO",
    "canada_cpi": "CPALTT01CAM657N",
}

_BOC_BOND_SERIES = {
    "canada_bond_2y": "BD.CDN.2YR.DQ.YLD",
    "canada_bond_5y": "BD.CDN.5YR.DQ.YLD",
    "canada_bond_10y": "BD.CDN.10YR.DQ.YLD",
}

# sector (lowercased GICS name) -> (display name, FRED series id)
_SECTOR_COMMODITY = {
    "energy": ("WTI Crude Oil", "DCOILWTICO"),
    "materials": ("Copper", "PCOPPUSDM"),
    "industrials": ("Copper", "PCOPPUSDM"),
    "utilities": ("Natural Gas", "DHHNGSP"),
}

# Real bug caught live (2026-08-07): the original list included generic
# monetary-policy terms ("interest rate", "rate hike", "rate cut",
# "monetary policy", "rate decision") that match ANY central bank's
# coverage, not just Fed/BoC — confirmed live when Finnhub's general-news
# feed returned an RBI (Reserve Bank of India) rate-outlier article that
# matched purely on generic wording in its summary. The ticket explicitly
# scopes this to "Finnhub general news filtered to Fed/BoC topics," not
# "any central bank" — keywords below are Fed/BoC-specific proper nouns
# only, deliberately narrower than a first pass would suggest.
_CB_KEYWORDS = (
    "federal reserve",
    "fed ",
    "fomc",
    "jerome powell",
    "bank of canada",
    " boc ",
    "tiff macklem",
)

_TIMELINE_CB_WINDOW_DAYS = {
    "short_term": 14,
    "medium_term": 45,
    "long_term": 120,
}


# ---------- series arithmetic helpers ----------


def _sorted_dropna(series: pd.Series) -> pd.Series:
    """Drops NaNs and sorts ascending by date before any "take the last
    value = latest" logic runs. Neither fred.py nor boc.py documents a
    guaranteed ordering on the Series they return (both happen to come
    back ascending today, from fredapi and BoC Valet's own observation
    order respectively) — relying on that silently would mean a future
    provider quirk picks the wrong "latest" value without raising,
    exactly the class of bug this module already had to fix once for
    news_id_assignment.py's timezone handling."""
    return series.dropna().sort_index()


def _latest_value(series: pd.Series) -> float | None:
    dropped = _sorted_dropna(series)
    if dropped.empty:
        return None
    return round(float(dropped.iloc[-1]), 4)


def _latest_age_days(series: pd.Series, as_of: date) -> int | None:
    dropped = _sorted_dropna(series)
    if dropped.empty:
        return None
    latest_date = dropped.index[-1]
    if hasattr(latest_date, "date"):
        latest_date = latest_date.date()
    return (as_of - latest_date).days


def _value_n_days_ago(series: pd.Series, as_of: date, days: int) -> float | None:
    """Closest observation at or before (as_of - days) — economic series
    don't have data every calendar day (weekends, monthly/quarterly
    releases), so an exact-date lookup would almost always miss."""
    dropped = _sorted_dropna(series)
    if dropped.empty:
        return None
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=days)
    eligible = dropped[dropped.index <= cutoff]
    if eligible.empty:
        return None
    return round(float(eligible.iloc[-1]), 4)


def _point_delta(series: pd.Series, as_of: date, lookback_days: int) -> float | None:
    """Raw point delta (latest - past), for fields already expressed as a
    rate/percentage (e.g. unemployment: 4.2% -> 4.5% = +0.3)."""
    latest = _latest_value(series)
    past = _value_n_days_ago(series, as_of, lookback_days)
    if latest is None or past is None:
        return None
    return round(latest - past, 2)


def _pct_change(series: pd.Series, as_of: date, lookback_days: int) -> float | None:
    """Percent change (latest vs past), for fields expressed as an index
    level or price (CPI index, CAD/USD, commodity levels)."""
    latest = _latest_value(series)
    past = _value_n_days_ago(series, as_of, lookback_days)
    if latest is None or past is None or past == 0:
        return None
    return round((latest / past - 1) * 100, 2)


def _policy_rate_delta_bp(fed_funds: pd.Series, as_of: date) -> float | None:
    delta_pp = _point_delta(fed_funds, as_of, 90)
    if delta_pp is None:
        return None
    return round(delta_pp * 100, 1)


# ---------- trend classification (first-pass thresholds, not sourced from
# any doc) ----------


def _rate_trend(delta_bp: float | None) -> Literal["tightening", "pausing", "easing"] | None:
    if delta_bp is None:
        return None
    if delta_bp > 10:
        return "tightening"
    if delta_bp < -10:
        return "easing"
    return "pausing"


def _cpi_trend(delta_pct: float | None) -> Literal["rising", "stable", "falling"] | None:
    if delta_pct is None:
        return None
    if delta_pct > 0.3:
        return "rising"
    if delta_pct < -0.3:
        return "falling"
    return "stable"


def _cad_trend(
    change_pct: float | None,
) -> Literal["cad_strengthening", "stable", "cad_weakening"] | None:
    if change_pct is None:
        return None
    if change_pct > 2:
        return "cad_strengthening"
    if change_pct < -2:
        return "cad_weakening"
    return "stable"


def _vix_regime(vix: float | None) -> Literal["low", "elevated", "high"] | None:
    if vix is None:
        return None
    if vix < 15:
        return "low"
    if vix > 25:
        return "high"
    return "elevated"


def _curve_shape(
    short_yield: float | None, long_yield: float | None
) -> Literal["normal", "flat", "inverted"] | None:
    """Classified from the 10yr-minus-2yr spread in basis points. Same
    ±10bp deadband as _rate_trend, for consistency within this module —
    no doc gives an authoritative threshold, first-pass design. None
    unless both endpoints resolved (enforced by MacroSourcesBundle's own
    bidirectional validator — a shape can't be classified from one
    missing yield, and must be classified when both are present)."""
    if short_yield is None or long_yield is None:
        return None
    spread_bp = (long_yield - short_yield) * 100
    if spread_bp > 10:
        return "normal"
    if spread_bp < -10:
        return "inverted"
    return "flat"


# ---------- sector-commodity relevance ----------


async def _resolve_sector_commodity(
    sector: str | None, fred: FredMacroDataProvider, as_of: date
) -> dict:
    """Returns the 5 sector_commodity_* fields + commodity_90d_change_pct
    as a dict. All-None when the sector isn't in the (documented, only
    4-row) lookup table — matches MacroSourcesBundle's
    _check_sector_commodity_nulls validator, which requires exactly that
    combination for sector_commodity_relevant=False."""
    normalized = (sector or "").strip().lower()
    commodity = _SECTOR_COMMODITY.get(normalized)
    if commodity is None:
        return {
            "sector_commodity_relevant": False,
            "sector_commodity_name": None,
            "sector_commodity_level": None,
            "sector_commodity_direction": None,
            "sector_commodity_age_days": None,
            "commodity_90d_change_pct": None,
        }
    name, series_id = commodity
    fetched = await fred.get_macro_data([series_id])
    series = fetched.get(series_id, pd.Series(dtype="float64"))
    change_pct = _pct_change(series, as_of, 90)
    direction = None
    if change_pct is not None:
        direction = "rising" if change_pct > 0 else "falling" if change_pct < 0 else "flat"
    return {
        "sector_commodity_relevant": True,
        "sector_commodity_name": name,
        "sector_commodity_level": _latest_value(series),
        "sector_commodity_direction": direction,
        "sector_commodity_age_days": _latest_age_days(series, as_of),
        "commodity_90d_change_pct": change_pct,
    }


# ---------- central bank commentary ----------


def _is_cb_related(headline: str, summary: str) -> bool:
    """Padded with a leading AND trailing space: real bug caught on
    review — " boc " (leading+trailing space, to avoid matching "boc" as
    a substring of an unrelated word) could never match a headline that
    *starts* with "BoC", since `f"{headline} {summary}"` has no space
    before position 0. A headline like "BoC raises overnight rate" would
    have been silently excluded from central-bank commentary — exactly
    the article this feature exists to catch."""
    text = f" {headline} {summary} ".lower()
    return any(keyword in text for keyword in _CB_KEYWORDS)


async def _fetch_cb_commentary(
    finnhub: FinnhubDataProvider,
    timeline: Literal["short_term", "medium_term", "long_term"],
    as_of: datetime,
) -> list[CBCommentaryItem]:
    window_days = _TIMELINE_CB_WINDOW_DAYS[timeline]
    cutoff = as_of - timedelta(days=window_days)
    try:
        raw_articles = await finnhub.get_general_news(category="general")
    except Exception:
        logger.warning("macro_sources_cb_commentary_fetch_failed", exc_info=True)
        return []
    items = []
    for article in raw_articles:
        headline = article.get("headline", "")
        if not headline or not _is_cb_related(headline, article.get("summary", "")):
            continue
        published_ts = article.get("datetime")
        if published_ts is None:
            continue
        published_at = datetime.fromtimestamp(published_ts, tz=UTC)
        if published_at < cutoff:
            continue
        items.append(CBCommentaryItem(date=published_at.date(), headline=headline))
    return items


def _cb_stance_note(items: list[CBCommentaryItem]) -> str | None:
    """The most recent CB-related headline, if any — already filtered for
    Fed/BoC materiality by _is_cb_related(). Deliberately does not attempt
    to classify a dovish/hawkish direction: that's an interpretive
    judgment for the Macro Economist LLM to make from the headline text
    (per CLAUDE.md's "pipeline does the math, the LLM does the thinking"
    rule), not something this precompute module should pre-decide. None
    when the list is empty — enforced by MacroSourcesBundle's own
    one-directional validator.

    Tie-breaking on same-day items is arbitrary (whichever came first in
    Finnhub's response order) — CBCommentaryItem.date is day-only by its
    own disclosed contract design (no intraday timestamp survives that
    far), so there's no finer-grained signal available here to break the
    tie meaningfully. Low-stakes in practice: multiple genuinely Fed/BoC-
    specific articles landing the same day is rare given the narrow
    keyword filter."""
    if not items:
        return None
    return max(items, key=lambda item: item.date).headline


# ---------- Statistics Canada supplementary fields ----------


async def _fetch_statcan_fields(stats_canada: StatsCanadaProvider, as_of: date) -> dict:
    unemployment = await stats_canada.get_unemployment_rate()
    housing = await stats_canada.get_housing_starts()
    retail = await stats_canada.get_retail_sales_yoy()
    cpi_by_province = await stats_canada.get_cpi_by_province()

    # statcan_age_days is one field, not one per statcan_* value — no doc
    # says which underlying fetch it should track, so it's taken from
    # whichever of the four resolved first (unemployment preferred, since
    # it's the statcan_* field explicitly named in the reliability-scorer
    # ticket text). get_cpi_by_province() never returns a "released" date
    # (confirmed in stats_canada.py — always None), so it can't anchor
    # this even as a last resort.
    age_days = None
    for result in (unemployment, housing, retail):
        if result and result.get("released"):
            try:
                released_date = datetime.fromisoformat(result["released"]).date()
                age_days = (as_of - released_date).days
            except ValueError:
                continue
            break

    return {
        "statcan_unemployment_ca": unemployment["value"] if unemployment else None,
        "statcan_housing_starts": housing["value"] if housing else None,
        "statcan_retail_sales_yoy": retail["value"] if retail else None,
        "statcan_cpi_by_province": cpi_by_province["value"] if cpi_by_province else None,
        "statcan_age_days": age_days,
    }


_EMPTY_STATCAN_FIELDS = {
    "statcan_unemployment_ca": None,
    "statcan_housing_starts": None,
    "statcan_retail_sales_yoy": None,
    "statcan_cpi_by_province": None,
    "statcan_age_days": None,
}


# ---------- entry point ----------


async def compute_macro_sources(
    sector: str | None,
    is_canadian_stock: bool,
    timeline: Literal["short_term", "medium_term", "long_term"],
    fred: FredMacroDataProvider,
    boc: BOCMacroDataProvider,
    finnhub: FinnhubDataProvider,
    stats_canada: StatsCanadaProvider | None = None,
    as_of: datetime | None = None,
) -> MacroSourcesBundle:
    """Assembles one MacroSourcesBundle. Callers own provider lifecycle
    (e.g. `async with BOCMacroDataProvider() as boc: ...`) — this function
    only calls already-constructed provider instances, it doesn't create
    or close sessions itself.

    stats_canada is only called when is_canadian_stock is True — per the
    contract's own docstring, statcan_* fields are None both on fetch
    failure and when the stock isn't Canadian, so there's no reason to
    spend an API round trip on a US stock.

    as_of defaults to the real current time; overridable so age/delta
    computations are deterministic and testable rather than coupled to
    wall-clock time at every call site.
    """
    # Real gap caught on review: a caller-supplied naive (tzinfo-less)
    # as_of would make _fetch_cb_commentary's `published_at < cutoff`
    # comparison mix tz-aware and naive datetimes and raise TypeError —
    # the same class of bug already fixed once in news_id_assignment.py
    # this session. Normalize rather than let an ambiguous caller input
    # crash unpredictably deep inside a helper function.
    as_of_dt = as_of if as_of is not None else datetime.now(UTC)
    if as_of_dt.tzinfo is None:
        as_of_dt = as_of_dt.replace(tzinfo=UTC)
    as_of_date = as_of_dt.date()

    fred_series = await fred.get_macro_data(list(_FRED_SERIES.values()))
    boc_bonds = await boc.get_macro_data(list(_BOC_BOND_SERIES.values()))
    boc_rates = await boc.get_interest_rates()
    boc_fx = await boc.get_exchange_rates("CADUSD")

    def fred_series_for(field: str) -> pd.Series:
        return fred_series.get(_FRED_SERIES[field], pd.Series(dtype="float64"))

    def boc_series_for(field: str) -> pd.Series:
        return boc_bonds.get(_BOC_BOND_SERIES[field], pd.Series(dtype="float64"))

    fed_funds_series = fred_series_for("fed_funds_rate")
    cpi_series = fred_series_for("cpi")
    cad_usd_fred_series = fred_series_for("cad_usd_fred")
    vix_series = fred_series_for("vix")
    unemployment_series = fred_series_for("unemployment")
    gdp_series = fred_series_for("gdp")

    policy_rate_delta_bp = _policy_rate_delta_bp(fed_funds_series, as_of_date)
    cpi_delta_pct = _pct_change(cpi_series, as_of_date, 90)
    cad_change_pct = _pct_change(cad_usd_fred_series, as_of_date, 90)
    unemployment_delta = _point_delta(unemployment_series, as_of_date, 180)  # ~6 months
    latest_vix = _latest_value(vix_series)

    sector_commodity = await _resolve_sector_commodity(sector, fred, as_of_date)
    cb_items = await _fetch_cb_commentary(finnhub, timeline, as_of_dt)
    cb_stance_note = _cb_stance_note(cb_items)

    treasury_2y = _latest_value(fred_series_for("treasury_2y"))
    treasury_10y = _latest_value(fred_series_for("treasury_10y"))
    canada_bond_2y = _latest_value(boc_series_for("canada_bond_2y"))
    canada_bond_10y = _latest_value(boc_series_for("canada_bond_10y"))
    us_curve_shape = _curve_shape(treasury_2y, treasury_10y)
    ca_curve_shape = _curve_shape(canada_bond_2y, canada_bond_10y)

    statcan_fields = _EMPTY_STATCAN_FIELDS
    if is_canadian_stock and stats_canada is not None:
        statcan_fields = await _fetch_statcan_fields(stats_canada, as_of_date)

    return MacroSourcesBundle(
        fed_funds_rate=_latest_value(fed_funds_series),
        treasury_2y=treasury_2y,
        treasury_5y=_latest_value(fred_series_for("treasury_5y")),
        treasury_10y=treasury_10y,
        cpi=_latest_value(cpi_series),
        core_cpi=_latest_value(fred_series_for("core_cpi")),
        gdp=_latest_value(gdp_series),
        unemployment=_latest_value(unemployment_series),
        vix=latest_vix,
        cad_usd_fred=_latest_value(cad_usd_fred_series),
        wti_crude=_latest_value(fred_series_for("wti_crude")),
        boc_rate=boc_rates.get("overnight_rate"),
        cad_usd=boc_fx.get("rate"),
        canada_bond_2y=canada_bond_2y,
        canada_bond_5y=_latest_value(boc_series_for("canada_bond_5y")),
        canada_bond_10y=canada_bond_10y,
        canada_cpi=_latest_value(fred_series_for("canada_cpi")),
        policy_rate_90d_delta_bp=policy_rate_delta_bp,
        cpi_3m_delta_pp=cpi_delta_pct,
        cad_usd_90d_change_pct=cad_change_pct,
        commodity_90d_change_pct=sector_commodity["commodity_90d_change_pct"],
        unemployment_6m_delta=unemployment_delta,
        rate_trend=_rate_trend(policy_rate_delta_bp),
        cpi_trend=_cpi_trend(cpi_delta_pct),
        cad_trend=_cad_trend(cad_change_pct),
        vix_regime=_vix_regime(latest_vix),
        us_curve_shape=us_curve_shape,
        ca_curve_shape=ca_curve_shape,
        sector_commodity_relevant=sector_commodity["sector_commodity_relevant"],
        sector_commodity_name=sector_commodity["sector_commodity_name"],
        sector_commodity_level=sector_commodity["sector_commodity_level"],
        sector_commodity_direction=sector_commodity["sector_commodity_direction"],
        sector_commodity_age_days=sector_commodity["sector_commodity_age_days"],
        bond_yields_available=canada_bond_10y is not None,
        usd_revenue_exposure_pct=None,
        cb_commentary_items=cb_items,
        cb_commentary_count=len(cb_items),
        cb_stance_note=cb_stance_note,
        policy_rate_age_days=_latest_age_days(fed_funds_series, as_of_date),
        cpi_age_days=_latest_age_days(cpi_series, as_of_date),
        gdp_age_days=_latest_age_days(gdp_series, as_of_date),
        unemployment_age_days=_latest_age_days(unemployment_series, as_of_date),
        cad_usd_age_days=_latest_age_days(cad_usd_fred_series, as_of_date),
        vix_age_days=_latest_age_days(vix_series, as_of_date),
        sector_commodity_age_days_reliability=sector_commodity["sector_commodity_age_days"],
        **statcan_fields,
    )
