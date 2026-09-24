"""Tests for agents/pass1_macro_economist.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_macro_economist import build_user_message


def _macro(**overrides) -> SimpleNamespace:
    defaults = dict(
        fed_funds_rate=5.25, rate_trend="pausing", policy_rate_90d_delta_bp=0.0,
        boc_rate=5.0, boc_rate_trend="pausing", boc_rate_90d_delta_bp=0.0,
        cb_stance_note=None,
        treasury_2y=4.8, treasury_5y=4.5, treasury_10y=4.3, us_curve_shape="inverted",
        canada_bond_2y=4.1, canada_bond_5y=3.9, canada_bond_10y=3.7, ca_curve_shape="inverted",
        bond_yields_available=True,
        us_cpi_yoy=3.1, us_core_cpi_yoy=2.9, cpi_trend="falling", cpi_3m_delta_pp=-0.2,
        ca_cpi_yoy=2.8, ca_cpi_trend="stable", ca_cpi_3m_delta=0.0,
        us_gdp_qoq=2.1, us_gdp_4q_trend="rising",
        ca_gdp_qoq=1.4, ca_gdp_4q_trend="stable",
        unemployment=4.1, unemployment_6m_delta=0.2,
        statcan_unemployment_ca=6.5, ca_unemployment_6m_delta=0.1,
        cad_usd=0.73, cad_trend="stable", cad_usd_90d_change_pct=1.2,
        vix=15.2, vix_regime="low", vix_30d_avg=16.0,
        sector_commodity_relevant=False, sector_commodity_name=None,
        sector_commodity_level=None, sector_commodity_direction=None,
        sector_commodity_age_days=None, commodity_90d_change_pct=None,
        wti_crude=78.5,
        statcan_housing_starts=245000.0, statcan_retail_sales_yoy=2.1,
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
    msg = build_user_message(_bundle())
    assert "RY.TO (Royal Bank of Canada) | Financials | TSX | CAD" in msg


def test_renders_pre_computed_trend_fields_not_raw_only():
    """The real prompt's own text says trends arrive pre-computed ('Direction
    is already computed (cite RATE)') -- must render the real trend
    classification, not just raw numbers."""
    msg = build_user_message(_bundle())
    assert "Trend: pausing" in msg
    assert "Trend: falling" in msg  # cpi_trend


def test_ca_specific_fields_render_for_ca_stock():
    msg = build_user_message(_bundle(is_ca=True))
    assert "BoC Rate: 5.0%" in msg
    assert "Canada Bond 2y/5y/10y" in msg
    assert "Canada CPI YoY: 2.8%" in msg
    assert "Canada GDP QoQ" in msg
    assert "Canada Unemployment Rate (LFS): 6.5%" in msg
    assert "STATISTICS CANADA" in msg
    assert "Housing starts (SAAR): 245000.0" in msg


def test_ca_specific_fields_absent_for_us_stock():
    msg = build_user_message(_bundle(is_ca=False))
    assert "BoC Rate" not in msg
    assert "Canada Bond" not in msg
    assert "Canada CPI" not in msg
    assert "STATISTICS CANADA" not in msg


def test_us_fields_always_render():
    msg = build_user_message(_bundle(is_ca=False))
    assert "AAPL (Apple Inc.) | Technology | NASDAQ | USD" in msg
    assert "US CPI YoY: 3.1%" in msg
    assert "US GDP QoQ" in msg


def test_commodity_block_gated_on_sector_commodity_relevant():
    msg = build_user_message(_bundle(sector_commodity_relevant=False))
    assert "Not commodity-sensitive for this sector." in msg
    assert "WTI Crude" not in msg  # only rendered inside the relevant branch


def test_commodity_block_renders_when_relevant():
    bundle = _bundle(
        sector_commodity_relevant=True, sector_commodity_name="WTI Crude",
        sector_commodity_level=78.5, sector_commodity_direction="rising",
        sector_commodity_age_days=1, commodity_90d_change_pct=5.2,
    )
    msg = build_user_message(bundle)
    assert "WTI Crude: 78.5" in msg
    assert "rising" in msg


def test_cb_stance_note_renders_when_present():
    bundle = _bundle(cb_stance_note="Fed signaled a pause through year-end.")
    msg = build_user_message(bundle)
    assert "Central bank stance: Fed signaled a pause through year-end." in msg


def test_cb_stance_note_absent_when_none():
    msg = build_user_message(_bundle())  # default None
    assert "Central bank stance:" not in msg


def test_no_sector_tailwinds_headwinds_fabricated_as_input():
    """sector_tailwinds/sector_headwinds are Macro's own OUTPUT fields, not
    input data -- confirmed via agents/pass2_view.py's MACRO branch, which
    reads them from structured_data, not bundle input."""
    msg = build_user_message(_bundle())
    assert "sector_tailwinds" not in msg
    assert "sector_headwinds" not in msg


def test_missing_bond_yields_flagged():
    bundle = _bundle(bond_yields_available=False)
    msg = build_user_message(bundle)
    assert "(bond yield data unavailable this run)" in msg
