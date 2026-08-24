import json

import pytest
from pydantic import ValidationError

from data.schemas.common import CBCommentaryItem
from data.schemas.macro_sources_bundle import MacroSourcesBundle


def _bundle(**overrides) -> MacroSourcesBundle:
    defaults = dict(
        # FRED series
        fed_funds_rate=5.25,
        treasury_2y=4.8,
        treasury_5y=4.5,
        treasury_10y=4.3,
        cpi=3.1,
        core_cpi=2.9,
        gdp=2.1,
        unemployment=4.0,
        vix=15.2,
        cad_usd_fred=0.73,
        wti_crude=78.5,
        # BoC Valet series
        boc_rate=5.0,
        cad_usd=0.73,
        canada_bond_2y=4.2,
        canada_bond_5y=3.9,
        canada_bond_10y=3.7,
        canada_cpi=2.8,
        # Orchestrator-computed deltas
        policy_rate_90d_delta_bp=25.0,
        cpi_3m_delta_pp=-0.2,
        cad_usd_90d_change_pct=1.5,
        commodity_90d_change_pct=None,
        unemployment_6m_delta=0.1,
        # Trend enums
        rate_trend="pausing",
        cpi_trend="stable",
        cad_trend="stable",
        vix_regime="low",
        # Yield curve shape
        us_curve_shape="normal",
        ca_curve_shape="normal",
        # Sector-commodity relevance
        sector_commodity_relevant=False,
        sector_commodity_name=None,
        sector_commodity_level=None,
        sector_commodity_direction=None,
        sector_commodity_age_days=None,
        # Availability flags
        bond_yields_available=True,
        usd_revenue_exposure_pct=None,
        # CB commentary
        cb_commentary_items=[],
        cb_commentary_count=0,
        cb_stance_note=None,
        # Series age fields
        policy_rate_age_days=1,
        cpi_age_days=15,
        gdp_age_days=45,
        unemployment_age_days=15,
        cad_usd_age_days=0,
        vix_age_days=0,
        sector_commodity_age_days_reliability=None,
        # StatCan supplementary
        statcan_unemployment_ca=None,
        statcan_housing_starts=None,
        statcan_retail_sales_yoy=None,
        statcan_cpi_by_province=None,
        statcan_age_days=None,
        # StatCan CPI/GDP trend fields (86bbahum6)
        ca_cpi_yoy=None,
        ca_cpi_3m_delta=None,
        ca_cpi_trend=None,
        ca_gdp_qoq=None,
        ca_gdp_4q_trend=None,
    )
    return MacroSourcesBundle(**{**defaults, **overrides})


def test_valid_construction():
    bundle = _bundle()
    assert bundle.rate_trend == "pausing"


def test_rejects_invalid_rate_trend():
    with pytest.raises(ValidationError):
        _bundle(rate_trend="hiking")


def test_rejects_invalid_cpi_trend():
    with pytest.raises(ValidationError):
        _bundle(cpi_trend="surging")


def test_rejects_invalid_cad_trend():
    with pytest.raises(ValidationError):
        _bundle(cad_trend="crashing")


def test_rejects_invalid_vix_regime():
    with pytest.raises(ValidationError):
        _bundle(vix_regime="extreme")


# ---------- yield curve shape (2026-08-07 contract-vs-consumer audit) ----------


@pytest.mark.parametrize("shape", ["normal", "flat", "inverted"])
def test_accepts_all_real_curve_shapes(shape):
    bundle = _bundle(us_curve_shape=shape, ca_curve_shape=shape)
    assert bundle.us_curve_shape == shape


def test_rejects_invalid_us_curve_shape():
    with pytest.raises(ValidationError):
        _bundle(us_curve_shape="steep")


def test_rejects_invalid_ca_curve_shape():
    with pytest.raises(ValidationError):
        _bundle(ca_curve_shape="steep")


def test_curve_shape_accepts_none_gracefully():
    """A curve shape can only be None when at least one of its two
    underlying yield points is also None — see the bidirectional
    _check_curve_shape_requires_both_yield_points validator."""
    bundle = _bundle(
        treasury_2y=None,
        treasury_10y=None,
        canada_bond_2y=None,
        canada_bond_10y=None,
        us_curve_shape=None,
        ca_curve_shape=None,
    )
    assert bundle.us_curve_shape is None
    assert bundle.ca_curve_shape is None


def test_rejects_curve_shape_set_when_a_yield_point_is_missing():
    with pytest.raises(ValidationError):
        _bundle(treasury_10y=None, us_curve_shape="normal")


def test_rejects_curve_shape_none_when_both_yield_points_present():
    with pytest.raises(ValidationError):
        _bundle(us_curve_shape=None)


# ---------- cb_stance_note <-> cb_commentary_items (2026-08-07 audit) ----------


def test_cb_stance_note_none_with_empty_items_is_valid():
    bundle = _bundle(cb_commentary_items=[], cb_commentary_count=0, cb_stance_note=None)
    assert bundle.cb_stance_note is None


def test_cb_stance_note_rejects_real_value_with_empty_items():
    with pytest.raises(ValidationError):
        _bundle(cb_commentary_items=[], cb_commentary_count=0, cb_stance_note="BoC: dovish pivot")


def test_cb_stance_note_can_be_none_even_with_real_items():
    """One-directional: having commentary items doesn't guarantee one was
    material enough to produce a stance note."""
    items = [CBCommentaryItem(date="2026-08-01", headline="Fed holds rates")]
    bundle = _bundle(cb_commentary_items=items, cb_commentary_count=1, cb_stance_note=None)
    assert bundle.cb_stance_note is None


def test_cb_stance_note_real_value_with_real_items_is_valid():
    items = [CBCommentaryItem(date="2026-08-01", headline="Fed holds rates")]
    bundle = _bundle(
        cb_commentary_items=items,
        cb_commentary_count=1,
        cb_stance_note="Fed: holding steady, no near-term move signaled",
    )
    assert bundle.cb_stance_note == "Fed: holding steady, no near-term move signaled"


# ---------- sector-commodity conditional-null cluster ----------


def test_sector_commodity_irrelevant_allows_all_none():
    bundle = _bundle(sector_commodity_relevant=False)
    assert bundle.sector_commodity_name is None


def test_sector_commodity_relevant_with_real_values_is_valid():
    bundle = _bundle(
        sector_commodity_relevant=True,
        sector_commodity_name="Crude Oil",
        sector_commodity_level=78.5,
        sector_commodity_direction="rising",
        sector_commodity_age_days=1,
        commodity_90d_change_pct=5.2,
    )
    assert bundle.sector_commodity_name == "Crude Oil"


@pytest.mark.parametrize(
    "field,value",
    [
        ("sector_commodity_name", "Crude Oil"),
        ("sector_commodity_level", 78.5),
        ("sector_commodity_direction", "rising"),
        ("sector_commodity_age_days", 1),
        ("commodity_90d_change_pct", 5.2),
    ],
)
def test_sector_commodity_irrelevant_rejects_any_non_null_field(field, value):
    """Ticket's explicit validator requirement for name/level/direction/
    age_days, extended to commodity_90d_change_pct per the doc's own inline
    comment tying it to the same sector_commodity_relevant flag."""
    with pytest.raises(ValidationError):
        _bundle(sector_commodity_relevant=False, **{field: value})


def test_sector_commodity_relevant_missing_all_fields_is_still_none_and_valid():
    """relevant=True doesn't itself force non-null fields — a genuinely
    unresolved sector-commodity lookup can still be all-None even when
    relevant. Only the False -> must-be-None direction is enforced."""
    bundle = _bundle(sector_commodity_relevant=True)
    assert bundle.sector_commodity_name is None


# ---------- cb_commentary_count consistency ----------


def test_cb_commentary_count_must_match_list_length():
    with pytest.raises(ValidationError):
        _bundle(cb_commentary_items=[], cb_commentary_count=3)


def test_cb_commentary_count_matching_is_valid():
    items = [CBCommentaryItem(date="2026-08-01", headline="Fed holds rates")]
    bundle = _bundle(cb_commentary_items=items, cb_commentary_count=1)
    assert bundle.cb_commentary_count == 1


# ---------- non-negative age/count fields ----------


def test_rejects_negative_age_field():
    with pytest.raises(ValidationError):
        _bundle(policy_rate_age_days=-1)


def test_rejects_negative_optional_age_field():
    with pytest.raises(ValidationError):
        _bundle(statcan_age_days=-1)


def test_accepts_zero_age():
    bundle = _bundle(vix_age_days=0)
    assert bundle.vix_age_days == 0


# ---------- StatCan fields all-None (non-Canadian stock) ----------


def test_statcan_fields_all_none_is_valid():
    bundle = _bundle()
    assert bundle.statcan_unemployment_ca is None
    assert bundle.statcan_cpi_by_province is None


# ---------- StatCan CPI/GDP trend fields (86bbahum6) ----------


def test_ca_cpi_gdp_fields_accept_real_values():
    bundle = _bundle(
        canada_cpi=169.0,
        ca_cpi_yoy=10.0,
        ca_cpi_3m_delta=0.5,
        ca_cpi_trend="rising",
        ca_gdp_qoq=2.1,
        ca_gdp_4q_trend="falling",
    )
    assert bundle.ca_cpi_yoy == 10.0
    assert bundle.ca_cpi_trend == "rising"
    assert bundle.ca_gdp_qoq == 2.1
    assert bundle.ca_gdp_4q_trend == "falling"


def test_ca_cpi_gdp_fields_all_none_is_valid():
    bundle = _bundle()
    assert bundle.ca_cpi_yoy is None
    assert bundle.ca_cpi_3m_delta is None
    assert bundle.ca_cpi_trend is None
    assert bundle.ca_gdp_qoq is None
    assert bundle.ca_gdp_4q_trend is None


def test_statcan_cpi_by_province_accepts_real_dict():
    bundle = _bundle(statcan_cpi_by_province={"ON": 3.1, "BC": 2.9})
    assert bundle.statcan_cpi_by_province["ON"] == 3.1


# ---------- base behavior ----------


def test_is_frozen():
    bundle = _bundle()
    with pytest.raises(ValidationError):
        bundle.vix = 25.0


def test_forbids_extra_fields():
    with pytest.raises(ValidationError):
        _bundle(macro_us={"legacy": True})


# ---------- graceful degradation: a single failed series must not block
# constructing the bundle (2026-08-05 design decision) ----------


def test_accepts_none_for_a_single_failed_fred_series():
    """One series failing (e.g. FRED down for treasury_10y) shouldn't
    block constructing the rest of the bundle."""
    bundle = _bundle(treasury_10y=None, us_curve_shape=None)
    assert bundle.treasury_10y is None
    assert bundle.fed_funds_rate == 5.25  # everything else still resolved


def test_accepts_none_for_a_single_failed_boc_series():
    bundle = _bundle(boc_rate=None)
    assert bundle.boc_rate is None


def test_accepts_none_for_delta_when_underlying_series_unresolved():
    bundle = _bundle(policy_rate_90d_delta_bp=None)
    assert bundle.policy_rate_90d_delta_bp is None


def test_accepts_none_for_trend_when_underlying_series_unresolved():
    bundle = _bundle(vix_regime=None, vix=None, vix_age_days=None)
    assert bundle.vix_regime is None


def test_accepts_none_for_age_field_paired_with_missing_series():
    bundle = _bundle(cpi=None, cpi_age_days=None)
    assert bundle.cpi_age_days is None


def test_accepts_totally_empty_macro_data_without_raising():
    """Extreme case: every fetchable series failed. Still constructs —
    the whole point of graceful degradation is the run doesn't fail."""
    bundle = _bundle(
        fed_funds_rate=None,
        treasury_2y=None,
        treasury_5y=None,
        treasury_10y=None,
        cpi=None,
        core_cpi=None,
        gdp=None,
        unemployment=None,
        vix=None,
        cad_usd_fred=None,
        wti_crude=None,
        boc_rate=None,
        cad_usd=None,
        canada_bond_2y=None,
        canada_bond_5y=None,
        canada_bond_10y=None,
        canada_cpi=None,
        policy_rate_90d_delta_bp=None,
        cpi_3m_delta_pp=None,
        cad_usd_90d_change_pct=None,
        unemployment_6m_delta=None,
        rate_trend=None,
        cpi_trend=None,
        cad_trend=None,
        vix_regime=None,
        us_curve_shape=None,
        ca_curve_shape=None,
        policy_rate_age_days=None,
        cpi_age_days=None,
        gdp_age_days=None,
        unemployment_age_days=None,
        cad_usd_age_days=None,
        vix_age_days=None,
    )
    assert bundle.fed_funds_rate is None
    assert bundle.rate_trend is None


# ---------- model_dump() round-trip (ClickUp 86bawp88h) ----------


def test_round_trips_through_model_dump():
    bundle = _bundle()
    assert MacroSourcesBundle.model_validate(bundle.model_dump()) == bundle


def test_round_trips_with_nested_cb_commentary_items():
    """The nested CBCommentaryItem list is the part of this model most
    worth checking — confirms it survives dump as real structured data,
    not collapsed into an opaque blob."""
    items = [CBCommentaryItem(date="2026-08-01", headline="Fed holds rates")]
    bundle = _bundle(cb_commentary_items=items, cb_commentary_count=1)
    dumped = bundle.model_dump()
    assert dumped["cb_commentary_items"] == [{"date": items[0].date, "headline": "Fed holds rates"}]
    assert MacroSourcesBundle.model_validate(dumped) == bundle


def test_round_trips_totally_empty_macro_data():
    """The graceful-degradation all-None case must also survive a dump/
    reload cycle unchanged, not just construct."""
    bundle = _bundle(
        fed_funds_rate=None,
        treasury_2y=None,
        treasury_10y=None,
        boc_rate=None,
        rate_trend=None,
        vix_regime=None,
        us_curve_shape=None,
        policy_rate_age_days=None,
    )
    assert MacroSourcesBundle.model_validate(bundle.model_dump()) == bundle


def test_json_mode_dump_is_json_serializable_and_round_trips():
    items = [CBCommentaryItem(date="2026-08-01", headline="Fed holds rates")]
    bundle = _bundle(cb_commentary_items=items, cb_commentary_count=1)
    dumped = bundle.model_dump(mode="json")
    json.dumps(dumped)  # raises if the nested CBCommentaryItem.date isn't JSON-primitive
    assert dumped["cb_commentary_items"][0]["date"] == "2026-08-01"
    assert MacroSourcesBundle.model_validate(dumped) == bundle
