import aiohttp
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

# table 18100004, All-items CPI vector per province. Territories aren't
# included — WDS only carries them at the city level (Whitehorse/Yellowknife/
# Iqaluit), not province/territory-wide, so there's no matching vector.
_CPI_VECTORS_BY_PROVINCE = {
    "NL": 41691244,
    "PE": 41691379,
    "NS": 41691513,
    "NB": 41691648,
    "QC": 41691783,
    "ON": 41691919,
    "MB": 41692055,
    "SK": 41692191,
    "AB": 41692327,
    "BC": 41692462,
}


def _shift_year(ref_per: str, delta_years: int) -> str:
    """'2026-04-01' shifted by -1 -> '2025-04-01'."""
    year, month, day = ref_per.split("-")
    return f"{int(year) + delta_years}-{month}-{day}"


class StatsCanadaProvider:
    """Statistics Canada Web Data Service (WDS) — supplementary Canadian
    macro data (unemployment, housing starts, retail sales YoY, CPI by
    province) for MacroSourcesBundle's statcan_* fields. Runs alongside
    boc.py for Canadian stocks; supplementary, not primary — Bank of Canada
    Valet is.

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
        MacroSourcesBundle.statcan_unemployment_ca."""
        data = await self._fetch_vectors([_UNEMPLOYMENT_VECTOR])
        point = self._latest_point(data[0])
        if point is None:
            logger.warning("statcan_unemployment_missing")
            return None
        return {
            "value": self._scaled_value(point),
            "reference_period": point["refPer"],
            "released": point["releaseTime"],
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

    async def get_cpi_by_province(self) -> dict | None:
        """All-items CPI index by province (10 provinces; territories aren't
        included — see module docstring). Maps to
        MacroSourcesBundle.statcan_cpi_by_province.

        Matches each response row back to its province via the `vectorId`
        the row itself reports, NOT by response position. Confirmed live
        that getDataFromVectorsAndLatestNPeriods does not preserve request
        order when multiple vectors share a productId (which all 10 of
        these do, on 18100004) — it comes back sorted ascending by vectorId
        instead. A positional zip() would silently misassign CPI values to
        the wrong provinces the moment _CPI_VECTORS_BY_PROVINCE's insertion
        order didn't happen to already be ascending."""
        vector_to_province = {vid: code for code, vid in _CPI_VECTORS_BY_PROVINCE.items()}
        data = await self._fetch_vectors(list(_CPI_VECTORS_BY_PROVINCE.values()))
        by_province: dict[str, float] = {}
        periods: set[str] = set()
        for row in data:
            vector_id = row.get("object", {}).get("vectorId")
            province_code = vector_to_province.get(vector_id)
            if province_code is None:
                logger.warning("statcan_cpi_unrecognized_vector_in_response", vector_id=vector_id)
                continue
            point = self._latest_point(row)
            if point is None:
                logger.warning("statcan_cpi_missing_province", province=province_code)
                continue
            by_province[province_code] = self._scaled_value(point)
            periods.add(point["refPer"])
        if not by_province:
            return None
        if len(periods) > 1:
            logger.warning("statcan_cpi_period_mismatch_across_provinces", periods=sorted(periods))
        return {
            "value": by_province,
            "reference_period": max(periods),
            "released": None,
        }
