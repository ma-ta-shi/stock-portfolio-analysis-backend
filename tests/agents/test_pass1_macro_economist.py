"""Tests for agents/pass1_macro_economist.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_macro_economist import (
    _COVERAGE_GAP_SENTENCES,
    _anomalies,
    _coverage_presence,
    _data_coverage_line,
    _stale_data,
    _validate_with_caveats,
    build_user_message,
)
from agents.utils import render_data_coverage_line, to_data_coverage


def _macro(**overrides) -> SimpleNamespace:
    defaults = dict(
        fed_funds_rate=5.25, rate_trend="pausing", policy_rate_90d_delta_bp=0.0,
        policy_rate_age_days=1,  # 86bbwachy Phase 4 -- real fetch age, drives field_presence["rate"]
        boc_rate=5.0, boc_rate_trend="pausing", boc_rate_90d_delta_bp=0.0,
        cb_stance_note=None,
        treasury_2y=4.8, treasury_5y=4.5, treasury_10y=4.3, us_curve_shape="inverted",
        canada_bond_2y=4.1, canada_bond_5y=3.9, canada_bond_10y=3.7, ca_curve_shape="inverted",
        bond_yields_available=True,
        us_cpi_yoy=3.1, us_core_cpi_yoy=2.9, cpi_trend="falling", cpi_3m_delta_pp=-0.2,
        cpi_age_days=5,
        ca_cpi_yoy=2.8, ca_cpi_trend="stable", ca_cpi_3m_delta=0.0,
        us_gdp_qoq=2.1, us_gdp_4q_trend="rising", gdp_age_days=30,
        ca_gdp_qoq=1.4, ca_gdp_4q_trend="stable",
        unemployment=4.1, unemployment_6m_delta=0.2, unemployment_age_days=7,
        statcan_unemployment_ca=6.5, ca_unemployment_6m_delta=0.1,
        cad_usd=0.73, cad_trend="stable", cad_usd_90d_change_pct=1.2, cad_usd_age_days=0,
        vix=15.2, vix_regime="low", vix_30d_avg=16.0, vix_age_days=0,
        sector_commodity_relevant=False, sector_commodity_name=None,
        sector_commodity_level=None, sector_commodity_direction=None,
        sector_commodity_age_days=None, commodity_90d_change_pct=None,
        wti_crude=78.5,
        statcan_housing_starts=245000.0, statcan_retail_sales_yoy=2.1, statcan_age_days=15,
    )
    return SimpleNamespace(**{**defaults, **overrides})


def _bundle(is_ca=True, **macro_overrides) -> SimpleNamespace:
    return SimpleNamespace(
        stock=SimpleNamespace(ticker="RY.TO" if is_ca else "AAPL",
                                currency="CAD" if is_ca else "USD",
                                exchange="TSX" if is_ca else "NASDAQ"),
        company_info={"name": "Royal Bank of Canada" if is_ca else "Apple Inc.",
                       "sector": "Financials" if is_ca else "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        canadian_data_flags=SimpleNamespace() if is_ca else None,
        macro_sources=_macro(**macro_overrides),
    )


def test_renders_real_header_fields():
    msg, _ = build_user_message(_bundle())
    assert "RY.TO (Royal Bank of Canada) | Financials | TSX | CAD" in msg


def test_renders_pre_computed_trend_fields_not_raw_only():
    """The real prompt's own text says trends arrive pre-computed ('Direction
    is already computed (cite RATE)') -- must render the real trend
    classification, not just raw numbers."""
    msg, _ = build_user_message(_bundle())
    assert "Trend: pausing" in msg
    assert "Trend: falling" in msg  # cpi_trend


def test_ca_specific_fields_render_for_ca_stock():
    msg, _ = build_user_message(_bundle(is_ca=True))
    assert "BoC Rate: 5.0%" in msg
    assert "Canada Bond 2y/5y/10y" in msg
    assert "Canada CPI YoY: 2.8%" in msg
    assert "Canada GDP QoQ" in msg
    assert "Canada Unemployment Rate (LFS): 6.5%" in msg
    assert "STATISTICS CANADA" in msg
    assert "Housing starts (SAAR): 245000.0" in msg


def test_ca_specific_fields_absent_for_us_stock():
    msg, _ = build_user_message(_bundle(is_ca=False))
    assert "BoC Rate" not in msg
    assert "Canada Bond" not in msg
    assert "Canada CPI" not in msg
    assert "STATISTICS CANADA" not in msg


def test_us_fields_always_render():
    msg, _ = build_user_message(_bundle(is_ca=False))
    assert "AAPL (Apple Inc.) | Technology | NASDAQ | USD" in msg
    assert "US CPI YoY: 3.1%" in msg
    assert "US GDP QoQ" in msg


def test_commodity_block_gated_on_sector_commodity_relevant():
    msg, _ = build_user_message(_bundle(sector_commodity_relevant=False))
    assert "Not commodity-sensitive for this sector." in msg
    assert "WTI Crude" not in msg  # only rendered inside the relevant branch


def test_commodity_block_renders_when_relevant():
    bundle = _bundle(
        sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_level=78.5, sector_commodity_direction="rising",
        sector_commodity_age_days=1, commodity_90d_change_pct=5.2,
    )
    msg, _ = build_user_message(bundle)
    assert "WTI Crude: 78.5" in msg
    assert "rising" in msg


def test_cb_stance_note_renders_when_present():
    bundle = _bundle(cb_stance_note="Fed signaled a pause through year-end.")
    msg, _ = build_user_message(bundle)
    assert "Central bank stance: Fed signaled a pause through year-end." in msg


def test_cb_stance_note_absent_when_none():
    msg, _ = build_user_message(_bundle())  # default None
    assert "Central bank stance:" not in msg


def test_no_sector_tailwinds_headwinds_fabricated_as_input():
    """sector_tailwinds/sector_headwinds are Macro's own OUTPUT fields, not
    input data -- confirmed via agents/pass2_view.py's MACRO branch, which
    reads them from structured_data, not bundle input."""
    msg, _ = build_user_message(_bundle())
    assert "sector_tailwinds" not in msg
    assert "sector_headwinds" not in msg


def test_missing_bond_yields_flagged():
    bundle = _bundle(bond_yields_available=False)
    msg, _ = build_user_message(bundle)
    assert "(bond yield data unavailable this run)" in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_all_true_when_default_bundle_is_fully_populated_ca():
    _, presence = build_user_message(_bundle(is_ca=True))
    assert presence == {
        "rate": True, "yield_curve": True, "cpi": True, "gdp": True,
        "employment": True, "fx": True, "vix": True, "commodities": False,
        "statcan": True,
    }


def test_field_presence_no_statcan_key_for_us_stock():
    """statcan simply isn't in the map for a US stock -- that block never
    renders at all, so "attempted and absent" would misrepresent it."""
    _, presence = build_user_message(_bundle(is_ca=False))
    assert "statcan" not in presence


def test_field_presence_rate_false_on_fetch_failure():
    bundle = _bundle(policy_rate_age_days=None)
    _, presence = build_user_message(bundle)
    assert presence["rate"] is False


def test_field_presence_yield_curve_tracks_bond_yields_available():
    bundle = _bundle(bond_yields_available=False)
    _, presence = build_user_message(bundle)
    assert presence["yield_curve"] is False


def test_field_presence_commodities_true_when_sector_relevant_and_resolved():
    bundle = _bundle(
        sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_level=78.5, sector_commodity_direction="rising",
        sector_commodity_age_days=2, commodity_90d_change_pct=3.1,
    )
    _, presence = build_user_message(bundle)
    assert presence["commodities"] is True


def test_field_presence_statcan_false_on_fetch_failure():
    bundle = _bundle(is_ca=True, statcan_age_days=None)
    _, presence = build_user_message(bundle)
    assert presence["statcan"] is False


# ---------- _data_coverage_line (86bbummwp Tier 1a) ----------


def test_data_coverage_line_standard_for_default_non_commodity_sector():
    """Default fixture is not sector-commodity-relevant -- that's a real,
    intended N/A (the schema's own validator enforces
    sector_commodity_age_days=None for a non-commodity sector), NOT a data
    gap. Found during critical review: field_presence["commodities"] is
    unconditionally False whenever sector_commodity_relevant is False, which
    would otherwise make every non-commodity-sensitive stock (the large
    majority) report a false "no sector commodity data available" gap --
    live-verified against 5 real tickers, none commodity-sensitive, all
    showing this false positive before the fix."""
    _, presence = build_user_message(_bundle(is_ca=True))
    line = _data_coverage_line(presence, sector_commodity_relevant=False)
    assert line == "standard."


def test_data_coverage_line_standard_when_commodity_relevant_and_resolved():
    bundle = _bundle(
        is_ca=True, sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_level=78.5, sector_commodity_direction="rising",
        sector_commodity_age_days=2, commodity_90d_change_pct=3.1,
    )
    _, presence = build_user_message(bundle)
    assert _data_coverage_line(presence, sector_commodity_relevant=True) == "standard."


def test_data_coverage_line_flags_real_commodity_fetch_failure_when_relevant():
    """Distinct from "not relevant": a sector that IS commodity-sensitive but
    whose fetch genuinely failed (age_days is None) is a real gap, still
    correctly flagged."""
    bundle = _bundle(
        is_ca=True, sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_age_days=None,
    )
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence, sector_commodity_relevant=True)
    assert "no sector commodity data available" in line


def test_data_coverage_line_us_stock_never_mentions_statcan():
    """statcan is absent from field_presence entirely for a US stock -- must
    be treated as not applicable, never rendered as a gap."""
    bundle = _bundle(is_ca=False)
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence, sector_commodity_relevant=False)
    assert "Statistics Canada" not in line


def test_data_coverage_line_flags_real_statcan_fetch_failure_for_ca_stock():
    bundle = _bundle(is_ca=True, statcan_age_days=None)
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence, sector_commodity_relevant=False)
    assert "no Statistics Canada demand indicator data available" in line


def test_data_coverage_line_flags_multiple_real_fetch_failures():
    bundle = _bundle(is_ca=True, policy_rate_age_days=None, cpi_age_days=None)
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence, sector_commodity_relevant=False)
    assert "no policy rate data available" in line
    assert "no inflation data available" in line


# ---------- _coverage_presence (86bbummwp Tier 2) ----------


def test_coverage_presence_drops_commodities_when_not_relevant():
    _, presence = build_user_message(_bundle(is_ca=True))  # default: not commodity-relevant
    result = _coverage_presence(presence, sector_commodity_relevant=False)
    assert "commodities" not in result
    assert result == {k: v for k, v in presence.items() if k != "commodities"}


def test_coverage_presence_keeps_commodities_when_relevant():
    bundle = _bundle(
        is_ca=True, sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_age_days=2,
    )
    _, presence = build_user_message(bundle)
    result = _coverage_presence(presence, sector_commodity_relevant=True)
    assert result == presence  # unchanged -- commodities is a real, applicable field here


def test_coverage_presence_shared_by_prose_and_structured_flag():
    """The one real conditional (drop commodities when not relevant) must
    produce identical results whether consumed via _data_coverage_line() or
    _coverage_presence() directly -- confirms there's no drift between the
    two representations of data_coverage."""
    _, presence = build_user_message(_bundle(is_ca=True))
    coverage_presence = _coverage_presence(presence, sector_commodity_relevant=False)
    prose = render_data_coverage_line(coverage_presence, _COVERAGE_GAP_SENTENCES)
    structured = to_data_coverage(coverage_presence, _COVERAGE_GAP_SENTENCES)
    assert prose == _data_coverage_line(presence, sector_commodity_relevant=False)
    assert structured["absent"] == []  # nothing missing in the default fixture once commodities is dropped


# ---------- _stale_data (86bbummwp Tier 2) ----------


def test_stale_data_empty_for_default_fixture():
    """Default fixture: every real age field is well within its own
    cadence-aware threshold."""
    assert _stale_data(_bundle(is_ca=True)) == []


def test_stale_data_flags_stale_daily_series():
    bundle = _bundle(is_ca=True, policy_rate_age_days=10, cad_usd_age_days=8, vix_age_days=8)
    result = _stale_data(bundle)
    assert set(result) == {"rate", "fx", "vix"}


def test_stale_data_daily_threshold_not_applied_to_monthly_series():
    """A CPI age of 10 days is completely normal (monthly release cadence) --
    must NOT be flagged using the daily threshold."""
    bundle = _bundle(is_ca=True, cpi_age_days=10)
    assert "cpi" not in _stale_data(bundle)


def test_stale_data_flags_stale_monthly_series_past_normal_lag():
    bundle = _bundle(is_ca=True, cpi_age_days=50, unemployment_age_days=50)
    result = _stale_data(bundle)
    assert "cpi" in result
    assert "employment" in result


def test_stale_data_monthly_series_within_normal_lag_not_flagged():
    """CPI age of 40 days is within a normal release cycle -- not stale."""
    bundle = _bundle(is_ca=True, cpi_age_days=40)
    assert "cpi" not in _stale_data(bundle)


def test_stale_data_flags_stale_gdp_past_quarterly_lag():
    bundle = _bundle(is_ca=True, gdp_age_days=110)
    assert "gdp" in _stale_data(bundle)


def test_stale_data_gdp_within_normal_quarterly_lag_not_flagged():
    bundle = _bundle(is_ca=True, gdp_age_days=85)
    assert "gdp" not in _stale_data(bundle)


def test_stale_data_fetch_failure_not_flagged_as_stale():
    """age=None means the fetch failed -- that's a data_coverage gap, not a
    staleness signal (nothing to measure an age from)."""
    bundle = _bundle(is_ca=True, policy_rate_age_days=None)
    assert "rate" not in _stale_data(bundle)


def test_stale_data_commodities_only_flagged_when_relevant():
    bundle = _bundle(
        is_ca=True, sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_age_days=10,
    )
    assert "commodities" in _stale_data(bundle)


def test_stale_data_commodities_not_flagged_when_not_relevant():
    """Default fixture has sector_commodity_age_days=None (not relevant) --
    must never be flagged, matching data_coverage's own "not applicable"
    treatment of the same field."""
    assert "commodities" not in _stale_data(_bundle(is_ca=True))


def test_stale_data_statcan_only_flagged_for_ca_stock():
    bundle = _bundle(is_ca=True, statcan_age_days=60)
    assert "statcan" in _stale_data(bundle)


def test_stale_data_statcan_not_checked_for_us_stock():
    bundle = _bundle(is_ca=False, statcan_age_days=60)
    assert "statcan" not in _stale_data(bundle)


# ---------- _anomalies (86bbummwp Tier 2) ----------


def test_anomalies_flags_inverted_us_curve_by_default():
    """Default fixture is genuinely inverted -- a real signal, not a fixture
    artifact to work around."""
    result = _anomalies(_bundle(is_ca=False))
    assert "US yield curve is inverted" in result
    assert not any("Canada" in f for f in result)  # not a CA stock


def test_anomalies_flags_inverted_ca_curve_only_for_ca_stock():
    result = _anomalies(_bundle(is_ca=True))
    assert "Canada yield curve is inverted" in result


def test_anomalies_normal_curve_not_flagged():
    bundle = _bundle(is_ca=False, us_curve_shape="normal")
    assert "US yield curve is inverted" not in _anomalies(bundle)


def test_anomalies_flags_high_vix_regime():
    bundle = _bundle(is_ca=False, us_curve_shape="normal", vix_regime="high", vix=35.0)
    result = _anomalies(bundle)
    assert any("high-volatility" in f for f in result)


def test_anomalies_low_vix_regime_not_flagged():
    bundle = _bundle(is_ca=False, us_curve_shape="normal")  # default vix_regime="low"
    assert not any("volatility" in f for f in _anomalies(bundle))


def test_anomalies_flags_large_policy_rate_move():
    bundle = _bundle(is_ca=False, us_curve_shape="normal", policy_rate_90d_delta_bp=150.0)
    result = _anomalies(bundle)
    assert any("policy_rate_90d_delta_bp" in f for f in result)


def test_anomalies_small_policy_rate_move_not_flagged():
    bundle = _bundle(is_ca=False, us_curve_shape="normal", policy_rate_90d_delta_bp=25.0)
    assert not any("policy_rate_90d_delta_bp" in f for f in _anomalies(bundle))


def test_anomalies_flags_negative_large_delta_too():
    """abs() must be applied -- a large NEGATIVE move is just as anomalous as
    a large positive one."""
    bundle = _bundle(is_ca=False, us_curve_shape="normal", cad_usd_90d_change_pct=-12.0)
    result = _anomalies(bundle)
    assert any("cad_usd_90d_change_pct" in f for f in result)


def test_anomalies_clean_fixture_has_no_delta_anomalies():
    bundle = _bundle(is_ca=False, us_curve_shape="normal")  # default deltas are all small
    result = _anomalies(bundle)
    assert not any("delta" in f or "change_pct" in f for f in result)


# ---------- _validate_with_caveats (86bbummwp follow-on -- new here, this agent
# had no caveat-specific validator to compose with before now) ----------


def _valid_macro_output(**overrides) -> dict:
    base = {
        "assessment_summary": "A mixed macro picture with rate headwinds offset by AI tailwinds.",
        "analysis_confidence": "high",
        "caveats": ["Rate and inflation data reflect the most recent available release."],
        "key_factors": [
            {"factor": "Rate environment", "importance": "high", "sentiment": "negative", "evidence": "RATE: 5.25%"},
            {"factor": "Sector tailwind", "importance": "high", "sentiment": "positive", "evidence": "AI infrastructure spend"},
        ],
        "risks": [
            {"risk": "Macro slowdown", "severity": "medium", "evidence": "GDP: 2.4% growth cooling"},
        ],
        "narrative": (
            "The macro environment presents a mixed picture for this stock. Elevated Fed Funds "
            "Rate at 5.25% (RATE) compresses growth multiples, though the yield curve remains "
            "stable. CPI creates mild input cost pressure but margins are largely insulated. "
            "GDP growth supports enterprise IT budgets while USD/CAD is neutral for this "
            "USD-denominated name. VIX signals normal volatility. AI infrastructure spending "
            "remains a powerful sector tailwind offsetting rate headwinds over the medium term."
        ),
        "structured_data": {
            "overall_macro_environment": "neutral",
            "sector_cycle_position": "mid_cycle",
        },
    }
    base.update(overrides)
    return base


def test_validate_with_caveats_passes_when_schema_valid_and_no_gap():
    passed, errors = _validate_with_caveats(
        _valid_macro_output(), material_absent=[], anomalies=[], stale_data=[]
    )
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_gap():
    out = _valid_macro_output(structured_data={
        **_valid_macro_output()["structured_data"], "overall_macro_environment": "positive",
    })
    passed, errors = _validate_with_caveats(out, material_absent=[], anomalies=[], stale_data=[])
    assert not passed
    assert any("overall_macro_environment" in e for e in errors)


def test_validate_with_caveats_flags_high_confidence_stale_data_with_no_caveat():
    out = _valid_macro_output(caveats=[])
    passed, errors = _validate_with_caveats(
        out, material_absent=[], anomalies=[], stale_data=["rate", "cpi", "gdp", "employment"]
    )
    assert not passed
    assert any("caveats is empty" in e for e in errors)


def test_validate_with_caveats_passes_high_confidence_stale_data_with_caveat_present():
    out = _valid_macro_output(caveats=["Rate, CPI, GDP and employment data are all beyond their normal release lag."])
    passed, errors = _validate_with_caveats(
        out, material_absent=[], anomalies=[], stale_data=["rate", "cpi", "gdp", "employment"]
    )
    assert passed, errors
