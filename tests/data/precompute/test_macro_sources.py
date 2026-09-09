from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from data.precompute.macro_sources import (
    _yoy_pct_at,
    _yoy_pct_latest,
    compute_macro_sources,
)

_AS_OF = datetime(2026, 8, 7, 12, 0, 0, tzinfo=UTC)


def _series(values_by_days_ago: dict[int, float]) -> pd.Series:
    dates = [pd.Timestamp(_AS_OF.date()) - pd.Timedelta(days=d) for d in values_by_days_ago]
    return pd.Series(list(values_by_days_ago.values()), index=pd.DatetimeIndex(dates)).sort_index()


class _FakeFred:
    def __init__(self, series: dict[str, pd.Series] | None = None):
        self._series = series or {}

    async def get_macro_data(self, series_ids):
        return {sid: self._series.get(sid, pd.Series(dtype="float64")) for sid in series_ids}


class _FakeBoc:
    def __init__(self, bond_series=None, rates=None, fx=None):
        self._bond_series = bond_series or {}
        self._rates = rates if rates is not None else {"overnight_rate": 4.5}
        self._fx = fx if fx is not None else {"pair": "CADUSD", "rate": 0.73}

    async def get_macro_data(self, series_ids):
        return {sid: self._bond_series.get(sid, pd.Series(dtype="float64")) for sid in series_ids}

    async def get_interest_rates(self):
        return self._rates

    async def get_exchange_rates(self, pair="CADUSD"):
        return self._fx


class _FakeFinnhub:
    def __init__(self, articles=None):
        self._articles = articles or []

    async def get_general_news(self, category="general"):
        return self._articles


class _FakeStatsCanada:
    def __init__(
        self,
        unemployment=None,
        housing=None,
        retail=None,
        cpi_national=None,
        gdp_index=None,
    ):
        self._unemployment = unemployment
        self._housing = housing
        self._retail = retail
        self._cpi_national = cpi_national
        self._gdp_index = gdp_index
        self.call_count = 0

    async def get_unemployment_rate(self):
        self.call_count += 1
        return self._unemployment

    async def get_housing_starts(self):
        return self._housing

    async def get_retail_sales_yoy(self):
        return self._retail

    async def get_cpi_national(self):
        return self._cpi_national

    async def get_real_gdp_index(self):
        return self._gdp_index


def _full_fred_series() -> dict[str, pd.Series]:
    return {
        "FEDFUNDS": _series({90: 5.00, 0: 5.25}),
        "DGS2": _series({0: 4.25}),
        "DGS5": _series({0: 4.40}),
        "DGS10": _series({0: 4.70}),
        "CPIAUCSL": _series({455: 310.0, 365: 313.0, 90: 320.0, 0: 322.0}),
        "CPILFESL": _series({365: 320.0, 0: 330.0}),
        "GDP": _series({0: 28000.0}),
        # GDPC1: 5 quarterly points so _real_gdp_growth can do QoQ + YoY
        "GDPC1": _series({365: 22800.0, 273: 22900.0, 182: 23000.0, 91: 23100.0, 0: 23300.0}),
        "UNRATE": _series({180: 3.9, 0: 4.2}),
        "VIXCLS": _series({20: 15.0, 10: 16.0, 0: 16.5}),
        "DEXCAUS": _series({90: 1.35, 0: 1.40}),
        "DCOILWTICO": _series({0: 78.5}),
    }


def _full_boc_series() -> dict[str, pd.Series]:
    return {
        "BD.CDN.2YR.DQ.YLD": _series({0: 3.5}),
        "BD.CDN.5YR.DQ.YLD": _series({0: 3.6}),
        "BD.CDN.10YR.DQ.YLD": _series({0: 3.8}),
        "V39079": _series({90: 4.75, 0: 4.25}),  # BoC overnight rate, -50bp over 90d
    }


async def _compute(
    sector="Technology",
    is_canadian_stock=False,
    timeline="medium_term",
    fred=None,
    boc=None,
    finnhub=None,
    stats_canada=None,
):
    return await compute_macro_sources(
        sector=sector,
        is_canadian_stock=is_canadian_stock,
        timeline=timeline,
        fred=fred or _FakeFred(_full_fred_series()),
        boc=boc or _FakeBoc(_full_boc_series()),
        finnhub=finnhub or _FakeFinnhub(),
        stats_canada=stats_canada,
        as_of=_AS_OF,
    )


# ---------- happy path ----------


@pytest.mark.asyncio
async def test_full_construction_with_all_series_present():
    bundle = await _compute()
    assert bundle.fed_funds_rate == 5.25
    assert bundle.boc_rate == 4.5
    assert bundle.cad_usd == 0.73
    assert bundle.canada_bond_10y == 3.8
    assert bundle.canada_cpi is None  # StatsCanada-sourced now; no provider passed here


@pytest.mark.asyncio
async def test_policy_rate_delta_computed_in_basis_points():
    """FEDFUNDS moved 5.00 -> 5.25 over 90 days = +25bp."""
    bundle = await _compute()
    assert bundle.policy_rate_90d_delta_bp == 25.0
    assert bundle.rate_trend == "tightening"


@pytest.mark.asyncio
async def test_cad_usd_change_computed_as_pct_not_point_delta():
    """1.35 -> 1.40 over 90 days = +3.70% (not +0.05)."""
    bundle = await _compute()
    assert bundle.cad_usd_90d_change_pct == pytest.approx(3.70, abs=0.01)
    assert bundle.cad_trend == "cad_strengthening"


@pytest.mark.asyncio
async def test_unemployment_delta_is_point_delta_not_pct():
    """3.9% -> 4.2% over ~6mo = +0.3 (point delta, not percent change)."""
    bundle = await _compute()
    assert bundle.unemployment_6m_delta == pytest.approx(0.3, abs=0.001)


@pytest.mark.asyncio
async def test_cpi_delta_is_pp_change_in_yoy_rate_not_pct_of_index():
    """CPIAUCSL: 310 (455d ago), 313 (365d ago), 320 (90d ago), 322 (now).
    YoY now = 322 vs 313 = +2.88%. YoY 3mo ago = 320 vs 310 = +3.23%.
    cpi_3m_delta_pp = 2.88 - 3.23 = -0.35pp (not a percent change of the
    index itself, and not the same number a naive _pct_change(90d) would
    give)."""
    bundle = await _compute()
    assert bundle.cpi_3m_delta_pp == pytest.approx(-0.35, abs=0.01)
    assert bundle.cpi_trend == "falling"  # -0.35 < -0.3 deadband


@pytest.mark.asyncio
async def test_bond_yields_available_true_when_10y_resolves():
    bundle = await _compute()
    assert bundle.bond_yields_available is True


@pytest.mark.asyncio
async def test_us_and_boc_derived_fields_populated():
    """86bbq8rj1: the 5 US CPI/GDP/VIX fields + the BoC rate delta/trend.
    us_cpi_yoy via _yoy_pct_latest (322 now vs 313 a year back = +2.88%);
    core CPI 330 vs 320 = +3.12%; GDPC1 23300 vs 23100 QoQ-annualized and
    23300 vs 22800 YoY; V39079 4.25 vs 4.75 = -50bp; VIX mean 15/16/16.5."""
    bundle = await _compute()
    assert bundle.us_cpi_yoy == pytest.approx(2.88, abs=0.02)
    assert bundle.us_core_cpi_yoy == pytest.approx(3.12, abs=0.02)
    assert bundle.us_gdp_qoq == pytest.approx(3.51, abs=0.05)
    assert bundle.us_gdp_4q_trend == "rising"  # yoy ~2.2% > 1.0 deadband
    assert bundle.vix_30d_avg == pytest.approx(15.83, abs=0.02)
    assert bundle.boc_rate_90d_delta_bp == pytest.approx(-50.0, abs=0.1)
    assert bundle.boc_rate_trend == "easing"


# ---------- graceful degradation ----------


@pytest.mark.asyncio
async def test_missing_series_degrades_the_whole_dependent_chain_to_none():
    """No FEDFUNDS data -> fed_funds_rate, its delta, its trend, and its
    age all fall back to None instead of raising."""
    fred = _FakeFred({k: v for k, v in _full_fred_series().items() if k != "FEDFUNDS"})
    bundle = await _compute(fred=fred)
    assert bundle.fed_funds_rate is None
    assert bundle.policy_rate_90d_delta_bp is None
    assert bundle.rate_trend is None
    assert bundle.policy_rate_age_days is None


@pytest.mark.asyncio
async def test_missing_10y_bond_makes_bond_yields_available_false():
    boc = _FakeBoc({"BD.CDN.2YR.DQ.YLD": _series({0: 3.5}), "BD.CDN.5YR.DQ.YLD": _series({0: 3.6})})
    bundle = await _compute(boc=boc)
    assert bundle.canada_bond_10y is None
    assert bundle.bond_yields_available is False


@pytest.mark.asyncio
async def test_totally_empty_fred_and_boc_still_constructs():
    bundle = await _compute(fred=_FakeFred({}), boc=_FakeBoc({}, rates={}, fx={"pair": "CADUSD"}))
    assert bundle.fed_funds_rate is None
    assert bundle.boc_rate is None
    assert bundle.cad_usd is None
    assert bundle.bond_yields_available is False
    # 86bbq8rj1 fields all degrade to None when their source series is
    # absent — no GDPC1 (us_gdp_*), no VIX window (vix_30d_avg), no
    # V39079 lookback (boc_rate_*).
    for field in (
        "us_cpi_yoy",
        "us_core_cpi_yoy",
        "us_gdp_qoq",
        "us_gdp_4q_trend",
        "vix_30d_avg",
        "boc_rate_90d_delta_bp",
        "boc_rate_trend",
    ):
        assert getattr(bundle, field) is None, field


@pytest.mark.asyncio
async def test_us_gdp_yoy_none_when_under_five_quarters_but_qoq_still_computes():
    """_real_gdp_growth needs 5 points for YoY, 2 for QoQ — a short GDPC1
    series gives us_gdp_qoq but leaves us_gdp_4q_trend None."""
    fred = _FakeFred({**_full_fred_series(), "GDPC1": _series({91: 23100.0, 0: 23300.0})})
    bundle = await _compute(fred=fred)
    assert bundle.us_gdp_qoq is not None
    assert bundle.us_gdp_4q_trend is None


@pytest.mark.asyncio
async def test_delta_is_none_when_only_the_latest_point_exists_no_lookback():
    """A series with just one recent point has nothing 90 days back —
    latest value still resolves, but the delta/trend don't."""
    fred = _FakeFred({**_full_fred_series(), "FEDFUNDS": _series({0: 5.25})})
    bundle = await _compute(fred=fred)
    assert bundle.fed_funds_rate == 5.25
    assert bundle.policy_rate_90d_delta_bp is None
    assert bundle.rate_trend is None


# ---------- YoY helpers: publication-lag handling ----------


def _dated_series(values_by_date: dict[str, float]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in values_by_date])
    return pd.Series(list(values_by_date.values()), index=idx).sort_index()


def test_yoy_pct_latest_anchors_lookback_to_latest_observation_not_as_of():
    """A monthly series published ~2 months late: latest point is
    2026-06-01, as_of is 2026-08-07. _yoy_pct_at pins the prior anchor to
    as_of - 365d (lands on 2025-08-01) and compares across only ~10
    months; _yoy_pct_latest pins it to latest - 365d (2025-06-01) for a
    true 12-month reading."""
    as_of = date(2026, 8, 7)
    series = _dated_series(
        {
            "2025-06-01": 96.0,  # 12 months before the latest point
            "2025-08-01": 97.0,  # decoy: what _yoy_pct_at picks as "prior"
            "2026-06-01": 100.0,  # latest available (publication lag)
        }
    )
    assert _yoy_pct_latest(series, as_of) == pytest.approx(4.17, abs=0.01)  # 100/96
    assert _yoy_pct_at(series, as_of, 0) == pytest.approx(3.09, abs=0.01)  # 100/97, compressed


def test_yoy_pct_latest_none_when_no_point_a_full_year_before_latest():
    as_of = date(2026, 8, 7)
    series = _dated_series({"2026-01-01": 98.0, "2026-06-01": 100.0})
    assert _yoy_pct_latest(series, as_of) is None


def test_yoy_pct_latest_none_on_empty_series():
    assert _yoy_pct_latest(pd.Series(dtype="float64"), date(2026, 8, 7)) is None


# ---------- trend classification thresholds ----------


@pytest.mark.parametrize(
    "delta_bp,expected",
    [
        (25.0, "tightening"),
        (-25.0, "easing"),
        (0.0, "pausing"),
        (10.0, "pausing"),
        (15.0, "tightening"),
    ],
)
@pytest.mark.asyncio
async def test_rate_trend_thresholds(delta_bp, expected):
    fred = _FakeFred(
        {**_full_fred_series(), "FEDFUNDS": _series({90: 5.00, 0: 5.00 + delta_bp / 100})}
    )
    bundle = await _compute(fred=fred)
    assert bundle.rate_trend == expected


@pytest.mark.parametrize(
    "vix_level,expected", [(10.0, "low"), (14.9, "low"), (20.0, "elevated"), (30.0, "high")]
)
@pytest.mark.asyncio
async def test_vix_regime_thresholds(vix_level, expected):
    fred = _FakeFred({**_full_fred_series(), "VIXCLS": _series({0: vix_level})})
    bundle = await _compute(fred=fred)
    assert bundle.vix_regime == expected


# ---------- sector-commodity relevance ----------


@pytest.mark.asyncio
async def test_energy_sector_maps_to_wti():
    bundle = await _compute(sector="Energy")
    assert bundle.sector_commodity_relevant is True
    assert bundle.sector_commodity_name == "WTI Crude Oil"
    assert bundle.sector_commodity_level == 78.5


@pytest.mark.asyncio
async def test_sector_matching_is_case_insensitive():
    bundle = await _compute(sector="ENERGY")
    assert bundle.sector_commodity_relevant is True


@pytest.mark.asyncio
async def test_materials_and_industrials_both_map_to_copper():
    fred = _FakeFred({**_full_fred_series(), "PCOPPUSDM": _series({0: 4.2})})
    materials = await _compute(sector="Materials", fred=fred)
    industrials = await _compute(sector="Industrials", fred=fred)
    assert materials.sector_commodity_name == "Copper"
    assert industrials.sector_commodity_name == "Copper"


@pytest.mark.asyncio
async def test_unmapped_sector_is_fully_none():
    bundle = await _compute(sector="Consumer Discretionary")
    assert bundle.sector_commodity_relevant is False
    assert bundle.sector_commodity_name is None
    assert bundle.commodity_90d_change_pct is None


@pytest.mark.asyncio
async def test_none_sector_is_fully_none():
    bundle = await _compute(sector=None)
    assert bundle.sector_commodity_relevant is False


# ---------- central bank commentary ----------


def _article(headline, days_ago=1, summary=""):
    published = _AS_OF - timedelta(days=days_ago)
    return {"headline": headline, "summary": summary, "datetime": int(published.timestamp())}


@pytest.mark.asyncio
async def test_fed_related_headline_is_kept():
    finnhub = _FakeFinnhub([_article("Federal Reserve holds rates steady")])
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_commentary_count == 1
    assert bundle.cb_commentary_items[0].headline == "Federal Reserve holds rates steady"


@pytest.mark.asyncio
async def test_unrelated_headline_is_dropped():
    finnhub = _FakeFinnhub([_article("Local bakery wins award")])
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_commentary_count == 0


@pytest.mark.asyncio
async def test_other_central_banks_generic_rate_coverage_is_dropped():
    """Real bug caught live (2026-08-07): a real Finnhub general-news pull
    returned an RBI (Reserve Bank of India) article that matched the
    original generic keyword list ("interest rate"/"rate decision"/etc.)
    purely because any central bank's coverage uses that wording. The
    ticket scopes this to Fed/BoC only — an RBI/ECB/BOJ article, even one
    that explicitly discusses "interest rate" and "monetary policy," must
    not be included."""
    finnhub = _FakeFinnhub(
        [
            _article(
                "Why the RBI is emerging as an Asian rate outlier",
                summary="The central bank's interest rate decision and monetary policy stance diverge from peers.",
            )
        ]
    )
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_commentary_count == 0


@pytest.mark.asyncio
async def test_headline_starting_with_boc_is_kept():
    """Real bug caught on review: " boc " (leading+trailing space, to
    avoid matching "boc" as a substring of an unrelated word) could never
    match a headline starting with "BoC" — there's no space before
    position 0 in the joined text. Confirmed fixed by padding the whole
    joined string with edge spaces before matching."""
    finnhub = _FakeFinnhub([_article("BoC raises overnight rate")])
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_commentary_count == 1


@pytest.mark.asyncio
async def test_article_outside_timeline_window_is_dropped():
    """medium_term window is 45 days — an article 60 days old must be excluded."""
    finnhub = _FakeFinnhub([_article("Bank of Canada raises rates", days_ago=60)])
    bundle = await _compute(finnhub=finnhub, timeline="medium_term")
    assert bundle.cb_commentary_count == 0


@pytest.mark.asyncio
async def test_article_inside_longer_timeline_window_is_kept():
    """The same 60-day-old article IS inside the long_term (120-day) window."""
    finnhub = _FakeFinnhub([_article("Bank of Canada raises rates", days_ago=60)])
    bundle = await _compute(finnhub=finnhub, timeline="long_term")
    assert bundle.cb_commentary_count == 1


@pytest.mark.asyncio
async def test_cb_commentary_count_matches_items_length():
    finnhub = _FakeFinnhub(
        [_article("FOMC meeting minutes released"), _article("Jerome Powell speaks on inflation")]
    )
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_commentary_count == len(bundle.cb_commentary_items) == 2


@pytest.mark.asyncio
async def test_cb_commentary_fetch_failure_degrades_to_empty_not_raise():
    class _BrokenFinnhub:
        async def get_general_news(self, category="general"):
            raise RuntimeError("finnhub is down")

    bundle = await _compute(finnhub=_BrokenFinnhub())
    assert bundle.cb_commentary_count == 0
    assert bundle.cb_commentary_items == []


# ---------- StatCan gating ----------


@pytest.mark.asyncio
async def test_statcan_not_called_for_a_us_stock():
    stats_canada = _FakeStatsCanada(unemployment={"value": 6.8, "released": "2026-07-04"})
    bundle = await _compute(is_canadian_stock=False, stats_canada=stats_canada)
    assert stats_canada.call_count == 0
    assert bundle.statcan_unemployment_ca is None
    assert bundle.canada_cpi is None
    assert bundle.ca_gdp_qoq is None


@pytest.mark.asyncio
async def test_statcan_called_and_populated_for_a_canadian_stock():
    stats_canada = _FakeStatsCanada(
        unemployment={"value": 6.8, "delta_6m_pp": 0.3, "released": "2026-07-04"},
        cpi_national={
            "value": 169.0,
            "yoy_pct": 10.0,
            "delta_3m_pp": 0.5,
            "reference_period": "2026-06-01",
            "released": "2026-07-20T08:30",
        },
        gdp_index={
            "value": 116.8,
            "qoq_annualized_pct": 2.1,
            "yoy_pct": -1.5,
            "reference_period": "2026-01-01",
            "released": "2026-05-29T08:30",
        },
    )
    bundle = await _compute(is_canadian_stock=True, stats_canada=stats_canada)
    assert stats_canada.call_count == 1
    assert bundle.statcan_unemployment_ca == 6.8
    assert bundle.ca_unemployment_6m_delta == 0.3
    assert bundle.statcan_age_days is not None
    assert bundle.canada_cpi == 169.0
    assert bundle.ca_cpi_yoy == 10.0
    assert bundle.ca_cpi_3m_delta == 0.5
    assert bundle.ca_cpi_trend == "rising"  # 0.5 > 0.3 deadband
    assert bundle.ca_gdp_qoq == 2.1
    assert bundle.ca_gdp_4q_trend == "falling"  # yoy_pct -1.5 < -1.0 deadband


@pytest.mark.asyncio
async def test_statcan_age_days_falls_back_to_cpi_or_gdp_when_others_fail():
    """Regression test: unemployment/housing/retail all failing must not
    report statcan_age_days=None if cpi_national/gdp_index actually
    succeeded — real Canadian macro data was fetched, the age tracking
    must reflect that."""
    stats_canada = _FakeStatsCanada(
        unemployment=None,
        housing=None,
        retail=None,
        cpi_national={
            "value": 169.0,
            "yoy_pct": 2.8,
            "delta_3m_pp": 0.5,
            "reference_period": "2026-06-01",
            "released": "2026-07-20T08:30",
        },
    )
    bundle = await _compute(is_canadian_stock=True, stats_canada=stats_canada)
    assert bundle.statcan_age_days is not None
    assert bundle.canada_cpi == 169.0


@pytest.mark.asyncio
async def test_ca_cpi_yoy_and_delta_rounded_to_2_decimals():
    """StatsCanadaProvider always returns raw, unrounded floats — rounding
    is macro_sources.py's job, matching cpi_3m_delta_pp's precision on
    the US side. Regression test for 86bbeu0jy."""
    stats_canada = _FakeStatsCanada(
        cpi_national={
            "value": 169.0,
            "yoy_pct": 2.7980535279805316,
            "delta_3m_pp": 0.9557945041815976,
            "reference_period": "2026-06-01",
            "released": "2026-07-20T08:30",
        },
    )
    bundle = await _compute(is_canadian_stock=True, stats_canada=stats_canada)
    assert bundle.ca_cpi_yoy == pytest.approx(2.80)
    assert bundle.ca_cpi_3m_delta == pytest.approx(0.96)


@pytest.mark.asyncio
async def test_canadian_stock_with_no_stats_canada_provider_stays_none():
    """is_canadian_stock=True but no provider instance passed — must not
    crash, just skip (matches the None-default parameter)."""
    bundle = await _compute(is_canadian_stock=True, stats_canada=None)
    assert bundle.statcan_unemployment_ca is None
    assert bundle.canada_cpi is None
    assert bundle.ca_cpi_trend is None
    assert bundle.ca_gdp_qoq is None
    assert bundle.ca_gdp_4q_trend is None


# ---------- reliability age fields ----------


@pytest.mark.asyncio
async def test_age_days_reflects_the_latest_observation_date():
    fred = _FakeFred({**_full_fred_series(), "VIXCLS": _series({5: 16.5})})
    bundle = await _compute(fred=fred)
    assert bundle.vix_age_days == 5


@pytest.mark.asyncio
async def test_latest_value_is_correct_even_if_the_provider_returns_unsorted_data():
    """Real gap caught on review: neither fred.py nor boc.py documents a
    guaranteed ascending order on the Series they return — taking
    .iloc[-1] on an unsorted series would silently pick the wrong "latest"
    value instead of raising. This constructs a deliberately out-of-order
    series (newest point first) to confirm the module sorts before
    reading, not just when the test happens to build one in order."""
    out_of_order = pd.Series(
        [5.25, 5.00],
        index=pd.DatetimeIndex(
            [pd.Timestamp(_AS_OF.date()), pd.Timestamp(_AS_OF.date()) - pd.Timedelta(days=90)]
        ),
    )
    fred = _FakeFred({**_full_fred_series(), "FEDFUNDS": out_of_order})
    bundle = await _compute(fred=fred)
    assert bundle.fed_funds_rate == 5.25
    assert bundle.policy_rate_90d_delta_bp == 25.0


# ---------- yield curve shape (2026-08-07 contract-vs-consumer audit) ----------


@pytest.mark.asyncio
async def test_us_curve_shape_normal_when_10y_well_above_2y():
    """Default fixtures: DGS2=4.25, DGS10=4.70 -> +45bp spread -> normal."""
    bundle = await _compute()
    assert bundle.us_curve_shape == "normal"


@pytest.mark.asyncio
async def test_ca_curve_shape_normal_when_10y_well_above_2y():
    """Default fixtures: 2yr=3.5, 10yr=3.8 -> +30bp spread -> normal."""
    bundle = await _compute()
    assert bundle.ca_curve_shape == "normal"


@pytest.mark.asyncio
async def test_us_curve_shape_inverted_when_2y_above_10y():
    fred = _FakeFred(
        {**_full_fred_series(), "DGS2": _series({0: 4.70}), "DGS10": _series({0: 4.25})}
    )
    bundle = await _compute(fred=fred)
    assert bundle.us_curve_shape == "inverted"


@pytest.mark.asyncio
async def test_us_curve_shape_flat_when_spread_within_deadband():
    fred = _FakeFred(
        {**_full_fred_series(), "DGS2": _series({0: 4.25}), "DGS10": _series({0: 4.30})}
    )
    bundle = await _compute(fred=fred)
    assert bundle.us_curve_shape == "flat"


@pytest.mark.asyncio
async def test_us_curve_shape_none_when_2y_missing():
    fred = _FakeFred({k: v for k, v in _full_fred_series().items() if k != "DGS2"})
    bundle = await _compute(fred=fred)
    assert bundle.treasury_2y is None
    assert bundle.us_curve_shape is None


@pytest.mark.asyncio
async def test_ca_curve_shape_none_when_10y_missing():
    boc = _FakeBoc({"BD.CDN.2YR.DQ.YLD": _series({0: 3.5})})
    bundle = await _compute(boc=boc)
    assert bundle.canada_bond_10y is None
    assert bundle.ca_curve_shape is None


# ---------- cb_stance_note (2026-08-07 contract-vs-consumer audit) ----------


@pytest.mark.asyncio
async def test_cb_stance_note_none_when_no_commentary():
    bundle = await _compute(finnhub=_FakeFinnhub([]))
    assert bundle.cb_stance_note is None


@pytest.mark.asyncio
async def test_cb_stance_note_uses_most_recent_headline():
    finnhub = _FakeFinnhub(
        [
            _article("Fed holds rates steady", days_ago=10),
            _article("Fed signals openness to a cut", days_ago=2),
        ]
    )
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_stance_note == "Fed signals openness to a cut"


@pytest.mark.asyncio
async def test_cb_stance_note_matches_the_single_item_when_only_one_present():
    finnhub = _FakeFinnhub([_article("Bank of Canada holds overnight rate")])
    bundle = await _compute(finnhub=finnhub)
    assert bundle.cb_stance_note == "Bank of Canada holds overnight rate"


# ---------- naive as_of normalization (2026-08-07 review) ----------


@pytest.mark.asyncio
async def test_naive_as_of_does_not_crash_the_cb_commentary_fetch():
    """Real gap caught on review: a caller-supplied naive (tzinfo-less)
    as_of would previously make _fetch_cb_commentary's tz-aware/naive
    datetime comparison raise TypeError. This bypasses the _compute()
    helper (which always passes a tz-aware as_of) to actually exercise
    the naive-input path."""
    naive_as_of = datetime(2026, 8, 7, 12, 0, 0)  # no tzinfo
    finnhub = _FakeFinnhub([_article("Fed holds rates steady", days_ago=2)])
    bundle = await compute_macro_sources(
        sector="Technology",
        is_canadian_stock=False,
        timeline="medium_term",
        fred=_FakeFred(_full_fred_series()),
        boc=_FakeBoc(_full_boc_series()),
        finnhub=finnhub,
        as_of=naive_as_of,
    )
    assert bundle.cb_stance_note == "Fed holds rates steady"
