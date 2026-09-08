import aiohttp
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest"

# Verified live against the real WDS API (2026-07-27 — see ClickUp 86bb4757n).
#
# Vector IDs below were discovered via a one-time getCubeMetadata (cube
# dimension/member structure) + getSeriesInfoFromCubePidCoord
# (productId + coordinate -> vectorId) pass, not re-derived per request —
# those two endpoints are for discovery/maintenance, not per-call use.
#
# Table 20100008 ("Retail trade sales") is officially archived (confirmed
# live via getCubeMetadata's archiveStatusEn = "ARCHIVED... will no longer
# be updated") — do not use it. 20100056 ("Monthly retail trade sales by
# province and territory") is its confirmed-current replacement.

_UNEMPLOYMENT_VECTOR = 2062815  # table 14100287, Canada, overall unemployment rate
_HOUSING_STARTS_VECTOR = 52300157  # table 34100158, Canada, all areas, SAAR
_RETAIL_SALES_VECTOR = (
    1446859481  # table 20100056, Canada, Retail trade, Total retail sales, Unadjusted
)

# table 18100004, Canada;All-items (index level, not the dead FRED
# CPALTT01CAM657N MoM-%-change series — see ClickUp 86bbahum6). Verified
# live 2026-08-15 via getSeriesInfoFromCubePidCoord, coordinate
# "2.2.0.0.0.0.0.0.0.0" (Geography=Canada, Products=All-items).
_CPI_NATIONAL_VECTOR = 41690973

# table 36100105, Canada;Real gross domestic product, volume index
# 2017=100 (quarterly). Verified live 2026-08-15 via
# getSeriesInfoFromCubePidCoord, coordinate "1.1.0.0.0.0.0.0.0.0"
# (Geography=Canada, Estimates=Real GDP volume index).
_GDP_VOLUME_INDEX_VECTOR = 61992650

# get_cpi_by_province / _CPI_VECTORS_BY_PROVINCE removed 2026-09-08 (86bbq8rj1):
# a 10-province API fan-out feeding statcan_cpi_by_province, which no payload
# placeholder or reliability field consumed, and which couldn't anchor
# statcan_age_days either (getDataFromVectors gives no releaseTime for these).
# The audit's "surface or drop" call, resolved as drop.


def _shift_year(ref_per: str, delta_years: int) -> str:
    """'2026-04-01' shifted by -1 -> '2025-04-01'."""
    year, month, day = ref_per.split("-")
    return f"{int(year) + delta_years}-{month}-{day}"


def _shift_months(ref_per: str, delta_months: int) -> str:
    """'2026-04-01' shifted by -3 -> '2026-01-01'. Uses pd.Period for the
    month-rollover arithmetic rather than hand-rolled modulo math — pandas
    is already a hard dependency of this module."""
    period = pd.Period(ref_per, freq="M") + delta_months
    return f"{period.year:04d}-{period.month:02d}-01"


class StatsCanadaProvider:
    """Statistics Canada Web Data Service (WDS) — supplementary Canadian
    macro data (unemployment level + 6m delta, housing starts, retail
    sales YoY, national CPI, real GDP index) for MacroSourcesBundle's
    statcan_*/ca_*/ca_cpi_*/ca_gdp_* fields. Runs alongside boc.py for
    Canadian stocks; supplementary, not primary — Bank of Canada Valet is.

    Deliberately doesn't inherit MacroDataProvider — that ABC's shape
    (get_macro_data(series_ids), get_interest_rates(), get_exchange_rates())
    is FRED/BoC Valet's "one ID = one series" model. WDS has none of that: no
    interest rates, no exchange rates, and "one ID" (a table/product ID) is
    actually a whole multi-dimensional cube — table 14100287 alone holds
    8,991 series (every province x age group x sex x labour-force-
    characteristic combination). Getting one specific series needs a
    separate vector ID (see module docstring above on how the constants below
    were found). Exposes curated named methods matching MacroSourcesBundle's
    fields directly, same pattern as fred.py's get_interest_rates() — not a
    generic fetch-by-table-ID method.

    No API key needed. Rate limits aren't documented anywhere confirmable —
    don't assume unlimited at batch scale.

    Return shape: each method returns {"value": ..., "reference_period":
    "YYYY-MM-DD", "released": ISO datetime or None} rather than a bare value.
    `value` is what maps directly onto the matching MacroSourcesBundle field;
    `reference_period`/`released` exist because the four series update on
    different cadences (unemployment/CPI monthly, housing starts monthly,
    retail sales monthly with a lag) — `statcan_age_days` needs to be tracked
    per-field, not assumed uniform, and this is what that computation needs.
    """

    BASE_URL = BASE_URL

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self.session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "StatsCanadaProvider":
        if self._owns_session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._owns_session and self.session:
            await self.session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._owns_session = True
        return self.session

    async def _fetch_vectors(self, vector_ids: list[int], latest_n: int = 1) -> list[dict]:
        """POST getDataFromVectorsAndLatestNPeriods for one or more vectors.
        An unknown/invalid vector ID comes back as HTTP 200 with
        {"status": "FAILED", ...} — confirmed live — not an HTTP error, so
        callers must check each row's "status", not just the response code."""
        session = await self._get_session()
        payload = [{"vectorId": vid, "latestN": latest_n} for vid in vector_ids]
        async with session.post(
            f"{self.BASE_URL}/getDataFromVectorsAndLatestNPeriods", json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()

    @staticmethod
    def _scaled_value(point: dict) -> float:
        """StatCan's `value` is scaled by 10**scalarFactorCode — e.g. a
        housing-starts point of 238.971 with scalarFactorCode=3 means
        238,971 units (SAAR), not 238.971. Confirmed live against a
        realistic Canada housing-starts figure."""
        return point["value"] * (10 ** point["scalarFactorCode"])

    @staticmethod
    def _latest_point(row: dict) -> dict | None:
        if row.get("status") != "SUCCESS":
            return None
        points = row.get("object", {}).get("vectorDataPoint", [])
        valid = [p for p in points if p.get("value") is not None]
        if not valid:
            return None
        return max(valid, key=lambda p: p["refPer"])

    async def get_unemployment_rate(self) -> dict | None:
        """LFS overall unemployment rate for Canada. Maps to
        MacroSourcesBundle.statcan_unemployment_ca (value) and
        ca_unemployment_6m_delta (delta_6m_pp, 86bbq8rj1).

        delta_6m_pp is a raw percentage-point change (rate 6 months ago
        subtracted from now) — unemployment is already a rate, so unlike
        get_cpi_national's delta_3m_pp this is not a change-in-the-YoY.
        Needs the point 6 months back, so latest_n=8 (7 minimum + 1 month
        buffer, matching get_cpi_national's convention). Lookback by exact
        refPer match, not list position."""
        data = await self._fetch_vectors([_UNEMPLOYMENT_VECTOR], latest_n=8)
        row = data[0]
        if row.get("status") != "SUCCESS":
            logger.warning("statcan_unemployment_missing")
            return None
        points = {
            p["refPer"]: p for p in row["object"]["vectorDataPoint"] if p.get("value") is not None
        }
        if not points:
            logger.warning("statcan_unemployment_missing")
            return None
        latest_period = max(points)
        latest_value = self._scaled_value(points[latest_period])

        delta_6m_pp = None
        prior = points.get(_shift_months(latest_period, -6))
        if prior is not None:
            delta_6m_pp = latest_value - self._scaled_value(prior)
        else:
            logger.warning("statcan_unemployment_no_prior_6m", latest_period=latest_period)

        return {
            "value": latest_value,
            "delta_6m_pp": delta_6m_pp,
            "reference_period": latest_period,
            "released": points[latest_period]["releaseTime"],
        }

    async def get_housing_starts(self) -> dict | None:
        """Annualized housing starts (SAAR), Canada, all areas. Maps to
        MacroSourcesBundle.statcan_housing_starts."""
        data = await self._fetch_vectors([_HOUSING_STARTS_VECTOR])
        point = self._latest_point(data[0])
        if point is None:
            logger.warning("statcan_housing_starts_missing")
            return None
        return {
            "value": self._scaled_value(point),
            "reference_period": point["refPer"],
            "released": point["releaseTime"],
        }

    async def get_retail_sales_yoy(self) -> dict | None:
        """Retail sales YoY % change, Canada, Total retail sales, unadjusted
        (unadjusted so the same-month-prior-year comparison itself cancels
        seasonality, rather than double-correcting on top of the API's own
        adjustment). WDS returns raw monthly sales, not a pre-computed YoY
        rate — computed locally here from the latest month vs. the same
        month one year prior. Maps to
        MacroSourcesBundle.statcan_retail_sales_yoy.

        Sourced from table 20100056 — see module docstring on why not
        20100008 (archived, confirmed dead)."""
        data = await self._fetch_vectors([_RETAIL_SALES_VECTOR], latest_n=14)
        row = data[0]
        if row.get("status") != "SUCCESS":
            logger.warning("statcan_retail_sales_missing")
            return None
        points = {
            p["refPer"]: p for p in row["object"]["vectorDataPoint"] if p.get("value") is not None
        }
        if not points:
            logger.warning("statcan_retail_sales_missing")
            return None
        latest_period = max(points)
        latest = points[latest_period]
        prior_period = _shift_year(latest_period, -1)
        prior = points.get(prior_period)
        if prior is None:
            logger.warning("statcan_retail_sales_no_prior_year", latest_period=latest_period)
            return None
        current_value = self._scaled_value(latest)
        prior_value = self._scaled_value(prior)
        yoy_pct = ((current_value - prior_value) / prior_value) * 100
        return {
            "value": yoy_pct,
            "reference_period": latest_period,
            "released": latest["releaseTime"],
        }

    @staticmethod
    def _yoy_pct_for_period(points: dict[str, dict], period: str) -> float | None:
        """YoY % change for a specific period, using already-fetched
        points. Pure computation, no logging — callers log with their own
        context-specific event names, since 'which point was missing'
        means something different depending on what's being computed
        from the result.

        Returns unrounded, matching every other method's convention on
        this provider. Note (86bbeu0jy): get_cpi_national()'s delta_3m_pp
        is the difference of two unrounded calls to this method, rounded
        once at the end — the US-side equivalent (macro_sources.py's
        _yoy_pct_at) instead rounds each YoY component before subtracting,
        then rounds the difference again. Both are defensible and the
        numeric gap between them is negligible in practice, but it's an
        unexplained asymmetry between two code paths doing the same
        conceptual thing. Not worth unifying on its own; worth knowing if
        the two ever need to agree to the last digit for some future
        comparison (e.g. cross-jurisdiction calibration)."""
        current = points.get(period)
        prior = points.get(_shift_year(period, -1))
        if current is None or prior is None:
            return None
        current_value = StatsCanadaProvider._scaled_value(current)
        prior_value = StatsCanadaProvider._scaled_value(prior)
        if prior_value == 0:
            return None
        return ((current_value - prior_value) / prior_value) * 100

    async def get_cpi_national(self) -> dict | None:
        """All-items CPI index level for Canada (replaces the dead FRED
        CPALTT01CAM657N series — see ClickUp 86bbahum6). Maps to
        MacroSourcesBundle.canada_cpi/ca_cpi_yoy/ca_cpi_3m_delta, and
        indirectly ca_cpi_trend (precompute classifies delta_3m_pp into
        the trend enum, not this method).

        delta_3m_pp is a percentage-point change in the YoY inflation
        rate (YoY now minus YoY as of 3 months ago), not a percent change
        of the index — see ClickUp 86bbeu0jy. That needs the index value
        15 months back (3 + 12) as the oldest point, so latest_n=17
        (16 minimum + 1 month buffer, matching get_real_gdp_index()'s
        buffer-over-minimum convention). Lookback points are found by
        exact refPer match, not list position — same rationale as
        get_retail_sales_yoy(): a positional lookup would silently return
        the wrong calendar period if the series has a gap."""
        data = await self._fetch_vectors([_CPI_NATIONAL_VECTOR], latest_n=17)
        row = data[0]
        if row.get("status") != "SUCCESS":
            logger.warning("statcan_cpi_national_missing")
            return None
        points = {
            p["refPer"]: p for p in row["object"]["vectorDataPoint"] if p.get("value") is not None
        }
        if not points:
            logger.warning("statcan_cpi_national_missing")
            return None
        latest_period = max(points)
        latest = points[latest_period]
        latest_value = self._scaled_value(latest)

        yoy_pct = self._yoy_pct_for_period(points, latest_period)
        if yoy_pct is None:
            logger.warning("statcan_cpi_national_no_prior_year", latest_period=latest_period)

        delta_3m_pp = None
        three_m_ago_period = _shift_months(latest_period, -3)
        yoy_3mo_ago = self._yoy_pct_for_period(points, three_m_ago_period)
        if yoy_pct is not None and yoy_3mo_ago is not None:
            delta_3m_pp = yoy_pct - yoy_3mo_ago
        elif yoy_pct is not None:
            # Only warn about the 3-month-specific lookback here when the
            # 12-month one (yoy_pct) actually succeeded — otherwise this
            # would double-log a single root cause (e.g. a large series
            # gap) as two seemingly-unrelated problems; the no_prior_year
            # warning above already covers the yoy_pct failure case.
            logger.warning("statcan_cpi_national_no_prior_3m_yoy", latest_period=latest_period)

        return {
            "value": latest_value,
            "yoy_pct": yoy_pct,
            "delta_3m_pp": delta_3m_pp,
            "reference_period": latest_period,
            "released": latest["releaseTime"],
        }

    async def get_real_gdp_index(self) -> dict | None:
        """Real GDP volume index (2017=100) for Canada (replaces the dead
        FRED CANRGDPR series — see ClickUp 86bbahum6). Maps to
        MacroSourcesBundle.ca_gdp_qoq/ca_gdp_4q_trend (via the
        precompute layer's yoy-based trend classifier — yoy_pct itself
        isn't a MacroSourcesBundle field, the prompt has no placeholder
        for it).

        latest_n=6 covers the latest point, the prior quarter (QoQ), and
        4-quarters-ago (YoY) — the oldest point either comparison needs is
        4 quarters back (5 points), plus one quarter of buffer so a
        single delayed/missing quarterly release doesn't blow out the
        YoY lookback, matching get_cpi_national()'s one-period margin.
        Lookback points are found by exact refPer match, same rationale
        as get_cpi_national()/get_retail_sales_yoy()."""
        data = await self._fetch_vectors([_GDP_VOLUME_INDEX_VECTOR], latest_n=6)
        row = data[0]
        if row.get("status") != "SUCCESS":
            logger.warning("statcan_gdp_index_missing")
            return None
        points = {
            p["refPer"]: p for p in row["object"]["vectorDataPoint"] if p.get("value") is not None
        }
        if not points:
            logger.warning("statcan_gdp_index_missing")
            return None
        latest_period = max(points)
        latest = points[latest_period]
        latest_value = self._scaled_value(latest)

        qoq_annualized_pct = None
        prior_quarter = points.get(_shift_months(latest_period, -3))
        if prior_quarter is not None:
            prior_quarter_value = self._scaled_value(prior_quarter)
            qoq_annualized_pct = ((latest_value / prior_quarter_value) ** 4 - 1) * 100
        else:
            logger.warning("statcan_gdp_index_no_prior_quarter", latest_period=latest_period)

        yoy_pct = None
        prior_year = points.get(_shift_year(latest_period, -1))
        if prior_year is not None:
            prior_year_value = self._scaled_value(prior_year)
            yoy_pct = ((latest_value - prior_year_value) / prior_year_value) * 100
        else:
            logger.warning("statcan_gdp_index_no_prior_year", latest_period=latest_period)

        return {
            "value": latest_value,
            "qoq_annualized_pct": qoq_annualized_pct,
            "yoy_pct": yoy_pct,
            "reference_period": latest_period,
            "released": latest["releaseTime"],
        }
