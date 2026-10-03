"""Tests for agents/pass1_fundamental_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_fundamental_analyst import (
    _anomalies,
    _data_coverage_line,
    _stale_data,
    _validate_with_caveats,
    build_user_message,
)


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
        peer_metrics={"industry_benchmark": None},
        missing_fields=[],
        latest_financials_period_end="2026-06-30",  # 86bbummwp Tier 2 -- well within threshold
        currency_mismatch=None,
        not_applicable=None,
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg, _ = build_user_message(_bundle())
    assert "SHOP.TO (Shopify Inc) | Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg


def test_renders_percentage_fields_as_percent_not_raw_fraction():
    """DataBundle fields are raw fractions (0.142), not the harness's
    already-scaled _pct values (14.2) -- must be *100'd on render."""
    msg, _ = build_user_message(_bundle())
    assert "14.20%" in msg  # revenue_growth_yoy
    assert "0.142" not in msg


_BENCH = {"industry_benchmark": {"industry": "Software—Infrastructure", "market": "US", "companies": 82, "median_pe": 23.5, "p25_pe": 12.9, "p75_pe": 50.7, "closest": ["ORCL", "PLTR"]}, "closest": ["ORCL", "PLTR"]}


def test_the_industry_pe_benchmark_is_rendered_with_its_basis_and_the_stocks_position():
    """MSFT 2026-10-03: P/E 28.6 against a median of 23.5 is +22% but inside the 12.9 to 50.7 middle half: in line."""
    bundle = _bundle(valuation_metrics={"pe_ratio": 28.6, "forward_pe": 22.1, "peg_ratio": 1.8}, peer_metrics=_BENCH)
    bundle.stock.ticker = "MSFT"
    msg, _ = build_user_message(bundle)
    assert "Industry P/E: median 23.5, middle half 12.9 to 50.7 (82 companies, Software—Infrastructure, US, Yahoo trailing P/E)" in msg
    assert "This P/E is within_range (+22% against the median)" in msg


def test_a_pe_outside_the_middle_half_is_labelled_above_or_below():
    above = build_user_message(_bundle(valuation_metrics={"pe_ratio": 80.0, "forward_pe": 1.0, "peg_ratio": 1.0}, peer_metrics=_BENCH))[0]
    below = build_user_message(_bundle(valuation_metrics={"pe_ratio": 10.0, "forward_pe": 1.0, "peg_ratio": 1.0}, peer_metrics=_BENCH))[0]
    assert "above_range" in above and "below_range" in below


def test_a_canadian_stock_on_the_us_industry_says_why():
    bundle = _bundle(peer_metrics=_BENCH)
    bundle.stock.ticker = "SHOP.TO"
    msg, _ = build_user_message(bundle)
    assert "the TSX industry has too few companies" in msg


def test_a_missing_benchmark_says_so_instead_of_inventing_a_median():
    msg, _ = build_user_message(_bundle())
    assert "Industry P/E: N/A — no industry benchmark available" in msg
    assert "median" not in msg.split("Industry P/E:")[1].split("\n")[0].replace("no industry benchmark", "")


def test_renders_analyst_consensus_from_real_field_names():
    msg, _ = build_user_message(_bundle())
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
    msg, _ = build_user_message(bundle)
    assert "2026-06-30: EPS actual 1.2 vs estimate 1.1 (9.09% surprise)" in msg
    assert "guidance_vs_consensus" not in msg


def test_no_earnings_surprises_renders_honestly():
    msg, _ = build_user_message(_bundle())  # defaults: empty list
    assert "N/A — no earnings surprise history available." in msg


def test_no_dividend_data_renders_honestly():
    msg, _ = build_user_message(_bundle())
    assert "Yield: N/A | Payout: N/A" in msg


def test_dividend_data_renders_as_percentage():
    bundle = _bundle(dividend_info={"dividend_yield": 0.032, "payout_ratio": 0.45,
                                     "dividend_growth_5yr": 0.05, "dividend_regularity": "regular"})
    msg, _ = build_user_message(bundle)
    assert "Yield: 3.20% | Payout: 45.00%" in msg
    assert "Regularity: regular" in msg


def test_there_is_no_per_peer_table_any_more():
    msg, _ = build_user_message(_bundle(peer_metrics=_BENCH))
    assert "PEER_1" not in msg and "PEER DATA" not in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_all_true_with_default_bundle_except_no_earnings_or_benchmark():
    """Default fixture has every VAL/GROWTH/PROF/BAL field populated and an
    empty missing_fields list -- only earnings_surprises/industry_benchmark (both
    genuinely empty in the default fixture) should read False."""
    _, presence = build_user_message(_bundle())
    assert presence == {
        "pe_ratio": True, "forward_pe": True, "peg_ratio": True,
        "revenue_growth_yoy": True, "revenue_growth_3yr_cagr": True, "eps_growth_yoy": True,
        "gross_margin": True, "operating_margin": True, "net_margin": True,
        "roe": True, "fcf_to_net_income": True,
        "debt_to_equity": True, "current_ratio": True,
        "interest_coverage": True, "cash_position": True,
        "earnings_surprises": False, "industry_benchmark": False,
    }


def test_field_presence_reads_missing_fields_list_not_the_raw_dict():
    """A field can still be a real key in valuation_metrics (e.g. present
    but None) -- presence must come from bundle.missing_fields, the same
    signal compute_all() itself already flags it with, not a second,
    independently-maintained None-check against the raw dict."""
    bundle = _bundle(missing_fields=["valuation_metrics.peg_ratio", "balance_sheet_metrics.current_ratio"])
    _, presence = build_user_message(bundle)
    assert presence["peg_ratio"] is False
    assert presence["current_ratio"] is False
    assert presence["pe_ratio"] is True  # untouched entries stay real


def test_field_presence_earnings_surprises_true_with_real_history():
    bundle = _bundle(growth_metrics={
        "revenue_growth_yoy": 0.1, "revenue_growth_3yr_cagr": 0.1, "eps_growth_yoy": 0.1,
        "earnings_surprises": [
            {"period_end": "2026-06-30", "eps_actual": 1.2, "eps_estimated": 1.1,
             "eps_surprise_pct": 9.09, "revenue_actual": None, "revenue_estimated": None},
        ],
    })
    _, presence = build_user_message(bundle)
    assert presence["earnings_surprises"] is True


def test_field_presence_industry_benchmark_true_when_there_is_one():
    _, presence = build_user_message(_bundle(peer_metrics=_BENCH))
    assert presence["industry_benchmark"] is True
    assert build_user_message(_bundle())[1]["industry_benchmark"] is False


def test_field_presence_has_no_entry_for_dividend_or_analyst_consensus():
    """dividend_regularity='none' is itself the honest signal for "no
    dividend" (not an N/A gap), and analyst_consensus is already tracked by
    Sentiment Analyst -- neither is duplicated here."""
    _, presence = build_user_message(_bundle())
    assert "dividend_yield" not in presence
    assert "analyst_consensus" not in presence


# ---------- _data_coverage_line (86bbummwp Tier 1a) ----------


def test_data_coverage_line_standard_when_the_benchmark_and_earnings_surprises_are_present():
    _, presence = build_user_message(_bundle(peer_metrics=_BENCH, growth_metrics={
        "revenue_growth_yoy": 0.1, "revenue_growth_3yr_cagr": 0.1, "eps_growth_yoy": 0.1,
        "earnings_surprises": [{"period_end": "2026-06-30", "eps_actual": 1.2,
                                 "eps_estimated": 1.1, "eps_surprise_pct": 9.09}],
    }))
    assert _data_coverage_line(presence) == "standard."


def test_data_coverage_line_flags_a_missing_benchmark_and_earnings_surprises():
    """Default fixture has neither -- the coarse coverage line mentions both,
    unlike the 15 per-ratio keys in the same field_presence dict, which stay
    granular-only in input_field_coverage."""
    _, presence = build_user_message(_bundle())
    line = _data_coverage_line(presence)
    assert "no industry P/E benchmark available" in line
    assert "no earnings surprise history available" in line


def test_data_coverage_line_ignores_the_15_per_ratio_keys():
    """A missing per-ratio field (e.g. peg_ratio) must NOT surface in this
    prose line -- too granular, stays in input_field_coverage only."""
    bundle = _bundle(missing_fields=["valuation_metrics.peg_ratio"])
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence)
    assert "peg_ratio" not in line
    assert "no industry P/E benchmark available" in line  # still real (default fixture has none)


# ---------- _anomalies (86bbummwp Tier 2) ----------


def test_anomalies_empty_when_all_within_plausible_ranges():
    assert _anomalies(_bundle()) == []


def test_anomalies_flags_absurd_pe_ratio():
    bundle = _bundle(valuation_metrics={"pe_ratio": 5000.0, "forward_pe": 22.1, "peg_ratio": 1.8})
    result = _anomalies(bundle)
    assert len(result) == 1
    assert "pe_ratio=5000.0" in result[0]


def test_anomalies_flags_negative_gross_margin_above_range():
    """Range is (0.0, 0.99) -- confirms the lower bound is checked too, not
    just an absurdly high value."""
    bundle = _bundle(profitability_metrics={
        "gross_margin": -0.1, "operating_margin": 0.12, "net_margin": 0.09,
        "roe": 0.15, "fcf_to_net_income": 1.1,
    })
    result = _anomalies(bundle)
    assert any("gross_margin" in f for f in result)


def test_anomalies_none_value_is_not_flagged():
    """A missing value (None) is a coverage gap, not an anomaly -- must not
    be flagged here (that's data_coverage's job)."""
    bundle = _bundle(valuation_metrics={"pe_ratio": None, "forward_pe": 22.1, "peg_ratio": 1.8})
    assert _anomalies(bundle) == []


def test_anomalies_can_flag_multiple_metrics_at_once():
    bundle = _bundle(
        valuation_metrics={"pe_ratio": 5000.0, "forward_pe": 22.1, "peg_ratio": 1.8},
        balance_sheet_metrics={"debt_to_equity": 999.0, "current_ratio": 1.8,
                                "interest_coverage": 12.0, "cash_position": 5.2e9},
    )
    result = _anomalies(bundle)
    assert len(result) == 2
    assert any("pe_ratio" in f for f in result)
    assert any("debt_to_equity" in f for f in result)


def test_anomalies_ignores_peer_only_metrics_not_on_bundle():
    """pb_ratio/ev_ebitda have no primary-stock field anywhere -- must never
    be referenced, not even as a crash."""
    result = _anomalies(_bundle())
    assert not any("pb_ratio" in f or "ev_ebitda" in f for f in result)


# ---------- _stale_data (86bbummwp Tier 2) ----------


def test_stale_data_empty_for_default_fixture():
    assert _stale_data(_bundle()) == []


def test_stale_data_flags_stale_financials():
    bundle = _bundle(latest_financials_period_end="2025-01-01")  # well over 180 days before 2026-09-23
    assert _stale_data(bundle) == ["financials"]


def test_stale_data_within_threshold_not_flagged():
    bundle = _bundle(latest_financials_period_end="2026-04-01")  # ~175 days before 2026-09-23
    assert _stale_data(bundle) == []


def test_stale_data_no_financials_at_all_not_flagged_as_stale():
    """None means no quarters exist at all -- a data_coverage-shaped gap
    (nothing to measure an age from), not a staleness signal."""
    assert _stale_data(_bundle(latest_financials_period_end=None)) == []


# ---------- _validate_with_caveats (86bbummwp follow-on -- new here, this agent
# had no caveat-specific validator to compose with before now) ----------


def _valid_fundamental_output(**overrides) -> dict:
    base = {
        "assessment_summary": "A solid technology company with strong fundamentals.",
        "analysis_confidence": "high",
        "caveats": ["Coverage limited to public filings and earnings calls."],
        "key_factors": [
            {"factor": "Revenue growth", "importance": "high", "sentiment": "positive", "evidence": "FILING:10-K: 14.2% YoY"},
            {"factor": "FCF generation", "importance": "high", "sentiment": "positive", "evidence": "FILING:MD&A: FCF yield 4.8%"},
        ],
        "risks": [
            {"risk": "Competitive pressure", "severity": "medium", "evidence": "N1: new entrant"},
        ],
        "narrative": (
            "The company demonstrates strong compounding fundamentals with revenue growth of "
            "14.2% driven by cloud migration and steady margin expansion. Free cash flow "
            "generation remains healthy relative to peers, supporting continued reinvestment "
            "in growth initiatives without excessive leverage. Management has a consistent "
            "track record of meeting guidance, which supports confidence in forward estimates. "
            "The balance sheet remains conservatively positioned with ample interest coverage "
            "and a comfortable current ratio, leaving room to weather a cyclical downturn "
            "without needing to raise capital on unfavorable terms. Overall the thesis rests "
            "on durable competitive advantages and disciplined capital allocation over the "
            "medium term horizon, with valuation broadly in line with sector peers on a "
            "forward earnings basis, leaving upside contingent on continued execution."
        ),
        "interpretive_fields": {
            "valuation_vs_sector": "fair",
            "health_rating": "healthy",
            "guidance_vs_consensus": "inline",
            "dividend_sustainability": "strong",
            "peer_comparison_summary": "Leads sector on FCF yield and margins.",
        },
    }
    base.update(overrides)
    return base


def test_validate_with_caveats_passes_when_schema_valid_and_no_gap():
    passed, errors = _validate_with_caveats(
        _valid_fundamental_output(), material_absent=[], anomalies=[], stale_data=[]
    )
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_gap():
    out = _valid_fundamental_output(interpretive_fields={
        **_valid_fundamental_output()["interpretive_fields"], "health_rating": "not_a_real_rating",
    })
    passed, errors = _validate_with_caveats(out, material_absent=[], anomalies=[], stale_data=[])
    assert not passed
    assert any("health_rating" in e for e in errors)


def test_validate_with_caveats_flags_high_confidence_stale_data_with_no_caveat():
    out = _valid_fundamental_output(caveats=[])
    passed, errors = _validate_with_caveats(
        out, material_absent=[], anomalies=[], stale_data=["financials"]
    )
    assert not passed
    assert any("caveats is empty" in e for e in errors)


def test_validate_with_caveats_passes_high_confidence_stale_data_with_caveat_present():
    out = _valid_fundamental_output(caveats=["Financials are 200 days old, beyond the usual reporting cycle."])
    passed, errors = _validate_with_caveats(
        out, material_absent=[], anomalies=[], stale_data=["financials"]
    )
    assert passed, errors


def test_a_banks_nonexistent_metrics_read_not_applicable_not_n_a():
    bundle = _bundle(
        profitability_metrics={"gross_margin": None, "operating_margin": None, "net_margin": 0.32, "roe": 0.15,
                               "fcf_to_net_income": 1.1},
        balance_sheet_metrics={"debt_to_equity": 4.1, "current_ratio": None, "interest_coverage": None, "cash_position": 5.2e9},
        not_applicable={"reason": "bank", "fields": ["gross_margin", "operating_margin", "current_ratio", "interest_coverage"]},
    )
    msg, _ = build_user_message(bundle)
    assert "Gross margin: not applicable (bank) | Operating margin: not applicable (bank)" in msg
    assert "Current ratio: not applicable (bank)" in msg and "Interest coverage: not applicable (bank)" in msg
    assert "N/A" not in msg.split("PROFITABILITY")[1].split("BALANCE SHEET")[0]


def test_a_gap_that_is_not_on_the_not_applicable_list_still_reads_n_a():
    bundle = _bundle(balance_sheet_metrics={"debt_to_equity": 0.4, "current_ratio": 1.8, "interest_coverage": None,
                                            "cash_position": 5.2e9})
    msg, _ = build_user_message(bundle)
    assert "Interest coverage: N/A" in msg


def test_a_converted_currency_is_stated_in_the_payload_and_the_de_convention_is_labelled():
    bundle = _bundle(currency_mismatch={"financials_currency": "USD", "quote_currency": "CAD", "converted": True, "usd_cad": 1.4243})
    msg, _ = build_user_message(bundle)
    assert "NOTE: statements are reported in USD and shown here converted to CAD at 1.4243 CAD per USD (Bank of Canada)." in msg
    assert "D/E (financial debt, excluding leases):" in msg


def test_an_unconvertible_currency_mismatch_says_the_multiples_are_omitted():
    bundle = _bundle(currency_mismatch={"financials_currency": "EUR", "quote_currency": "CAD", "converted": False, "usd_cad": None})
    msg, _ = build_user_message(bundle)
    assert "no exchange rate was available, so price-based multiples are omitted" in msg


def test_revenue_growth_shows_its_basis_and_the_fiscal_year_figure():
    bundle = _bundle(growth_metrics={"revenue_growth_yoy": 0.0674, "revenue_growth_yoy_basis": "quarter ended 2026-07-03 vs 2025-06-27",
                                     "revenue_growth_annual": 0.0187, "revenue_growth_3yr_cagr": 0.11, "eps_growth_yoy": 0.08,
                                     "earnings_surprises": []})
    msg, _ = build_user_message(bundle)
    assert "Revenue growth YoY: 6.74% (quarter ended 2026-07-03 vs 2025-06-27) | Last fiscal year: 1.87%" in msg


def test_a_reit_is_told_its_industry_pe_is_not_applicable():
    bundle = _bundle(peer_metrics=_BENCH, not_applicable={"reason": "REIT", "fields": ["pe_ratio"]})
    msg, presence = build_user_message(bundle)
    assert "Industry P/E: not applicable (REIT" in msg and "median 23.5" not in msg


def test_a_pe_from_near_zero_earnings_is_called_not_meaningful():
    msg, _ = build_user_message(_bundle(valuation_metrics={"pe_ratio": 9001.0, "forward_pe": 1.0, "peg_ratio": 1.0}, peer_metrics=_BENCH))
    assert "not_meaningful against the industry (earnings are near zero)" in msg
    assert "against the median" not in msg
