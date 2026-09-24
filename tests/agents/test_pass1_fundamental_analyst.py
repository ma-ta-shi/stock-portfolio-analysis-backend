"""Tests for agents/pass1_fundamental_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_fundamental_analyst import build_user_message


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        price_info={"current_price": 105.0, "market_cap": 1.3e11, "currency": "CAD",
                     "high_52w": 120.0, "low_52w": 60.0},
        valuation_metrics={"pe_ratio": 25.4, "forward_pe": 22.1, "peg_ratio": 1.8},
        growth_metrics={"revenue_growth_yoy": 0.142, "revenue_growth_3yr_cagr": 0.11,
                         "eps_growth_yoy": 0.08, "earnings_surprises": []},
        profitability_metrics={"gross_margin": 0.49, "operating_margin": 0.12,
                                "net_margin": 0.09, "roe": 0.15, "fcf_to_net_income": 1.1},
        balance_sheet_metrics={"debt_to_equity": 0.4, "current_ratio": 1.8,
                                "interest_coverage": 12.0, "cash_position": 5.2e9},
        dividend_info={"dividend_yield": None, "payout_ratio": None,
                       "dividend_growth_5yr": None, "dividend_regularity": "none"},
        analyst_consensus={"consensus_rating": "buy", "num_analysts": 32, "target_mean": 118.5,
                            "buy_count": 22, "hold_count": 8, "sell_count": 2},
        peer_metrics={"sector_medians": {"sector_median_pe": 20.1}, "peer_records": []},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg = build_user_message(_bundle())
    assert "SHOP.TO (Shopify Inc) | Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg


def test_renders_percentage_fields_as_percent_not_raw_fraction():
    """DataBundle fields are raw fractions (0.142), not the harness's
    already-scaled _pct values (14.2) -- must be *100'd on render."""
    msg = build_user_message(_bundle())
    assert "14.20%" in msg  # revenue_growth_yoy
    assert "0.142" not in msg


def test_renders_sector_median_from_peer_metrics_not_a_stale_key_name():
    msg = build_user_message(_bundle())
    assert "vs Sector median: 20.1" in msg


def test_missing_sector_median_renders_honestly():
    bundle = _bundle(peer_metrics={"sector_medians": {}, "peer_records": []})
    msg = build_user_message(bundle)
    assert "vs Sector median: N/A" in msg


def test_renders_analyst_consensus_from_real_field_names():
    msg = build_user_message(_bundle())
    assert "Coverage: 32 analysts" in msg
    assert "Buy: 22" in msg
    assert "Hold: 8" in msg
    assert "Sell: 2" in msg
    assert "Consensus: buy" in msg
    assert "Avg target: 118.5" in msg


def test_earnings_surprises_rendered_as_raw_data_not_a_verdict():
    """guidance_vs_consensus is an LLM OUTPUT field, not input data -- the
    raw earnings-surprise history is rendered instead so the LLM can form
    its own judgment."""
    bundle = _bundle(growth_metrics={
        "revenue_growth_yoy": 0.1, "revenue_growth_3yr_cagr": 0.1, "eps_growth_yoy": 0.1,
        "earnings_surprises": [
            {"period_end": "2026-06-30", "eps_actual": 1.2, "eps_estimated": 1.1,
             "eps_surprise_pct": 9.09, "revenue_actual": None, "revenue_estimated": None},
        ],
    })
    msg = build_user_message(bundle)
    assert "2026-06-30: EPS actual 1.2 vs estimate 1.1 (9.09% surprise)" in msg
    assert "guidance_vs_consensus" not in msg


def test_no_earnings_surprises_renders_honestly():
    msg = build_user_message(_bundle())  # defaults: empty list
    assert "N/A — no earnings surprise history available." in msg


def test_no_dividend_data_renders_honestly():
    msg = build_user_message(_bundle())
    assert "Yield: N/A | Payout: N/A" in msg


def test_dividend_data_renders_as_percentage():
    bundle = _bundle(dividend_info={"dividend_yield": 0.032, "payout_ratio": 0.45,
                                     "dividend_growth_5yr": 0.05, "dividend_regularity": "regular"})
    msg = build_user_message(bundle)
    assert "Yield: 3.20% | Payout: 45.00%" in msg
    assert "Regularity: regular" in msg


def test_peer_data_block_renders_when_present():
    bundle = _bundle(peer_metrics={
        "sector_medians": {"sector_median_pe": 20.1},
        "peer_records": [{"ticker": "ETSY", "pe_ratio": 18.2, "roe": 0.22}],
    })
    msg = build_user_message(bundle)
    assert "PEER_1 (ETSY):" in msg
    assert "pe_ratio=18.2" in msg


def test_no_peer_data_omits_peer_block_not_fabricated():
    msg = build_user_message(_bundle())  # defaults: peer_records=[]
    assert "PEER_1" not in msg
    assert "PEER DATA" not in msg
