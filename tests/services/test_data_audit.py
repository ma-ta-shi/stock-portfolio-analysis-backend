"""The pure parts of the Pass 1 data audit (services/data_audit.py): the calculations and the classification of what
differs. The fetching functions need the network and are exercised by running scripts/audit_pass1_data.py."""
import pytest

from services.data_audit import (
    Finding,
    compare,
    empty_pass2_fields,
    format_report,
    hand_ratios,
    rel_diff,
    summarize,
    trailing_sum,
)

RAW = {"net_income": 400.0, "revenue": 4000.0, "operating_income": 800.0, "gross_profit": 2000.0, "equity": 2000.0,
       "total_debt": 500.0, "current_assets": 900.0, "current_liabilities": 600.0, "operating_cash_flow": 700.0, "capex": -200.0}


def test_rel_diff_and_missing_inputs():
    assert rel_diff(110.0, 100.0) == pytest.approx(0.10)
    assert rel_diff(None, 100.0) is None and rel_diff(5.0, None) is None and rel_diff(5.0, 0) is None


def test_trailing_sum_needs_four_quarters_newest_first():
    assert trailing_sum([1.0, 2.0, 3.0, 4.0, 100.0]) == 10.0
    assert trailing_sum([1.0, 2.0, 3.0]) is None and trailing_sum(None) is None


def test_hand_ratios_from_raw_statements():
    r = hand_ratios(RAW, market_cap=40000.0)
    assert r["pe_ratio"] == pytest.approx(100.0) and r["pb_ratio"] == pytest.approx(20.0) and r["ps_ratio"] == pytest.approx(10.0)
    assert r["net_margin"] == pytest.approx(0.1) and r["operating_margin"] == pytest.approx(0.2) and r["gross_margin"] == pytest.approx(0.5)
    assert r["roe"] == pytest.approx(0.2) and r["debt_to_equity"] == pytest.approx(0.25) and r["current_ratio"] == pytest.approx(1.5)
    assert r["free_cash_flow"] == pytest.approx(500.0) and r["fcf_to_net_income"] == pytest.approx(1.25)


def test_the_fx_factor_converts_statement_amounts_but_not_ratios():
    """A USD reporter on a CAD quote: market cap in CAD over statement amounts converted to CAD."""
    r = hand_ratios(RAW, market_cap=40000.0, factor=1.4)
    assert r["pe_ratio"] == pytest.approx(100.0 / 1.4) and r["ps_ratio"] == pytest.approx(10.0 / 1.4)
    assert r["free_cash_flow"] == pytest.approx(500.0 * 1.4)
    assert r["net_margin"] == pytest.approx(0.1) and r["debt_to_equity"] == pytest.approx(0.25)


def test_hand_ratios_leave_out_what_the_inputs_cannot_give():
    r = hand_ratios({"revenue": 100.0}, market_cap=10.0)
    assert set(r) == {"ps_ratio"}
    assert "pe_ratio" not in hand_ratios({"net_income": -5.0, "equity": 10.0}, market_cap=10.0)


def test_compare_classifies_ok_known_mismatch_and_gap():
    ours = {"pe_ratio": 100.0, "debt_to_equity": 0.1, "current_ratio": 20.0, "roe": None}
    theirs = {"pe_ratio": 101.0, "debt_to_equity": 0.3, "current_ratio": 15.0, "roe": 0.2}
    by_field = {f.field: f for f in compare("X", "t", ours, theirs, tol=0.02)}
    assert by_field["pe_ratio"].status == "ok"
    assert by_field["debt_to_equity"].status == "known" and "lease" in by_field["debt_to_equity"].note
    assert by_field["current_ratio"].status == "mismatch" and by_field["current_ratio"].note == "+33.3%"
    assert by_field["roe"].status == "gap"


def test_a_known_difference_inside_tolerance_is_just_ok():
    assert compare("X", "t", {"debt_to_equity": 0.100}, {"debt_to_equity": 0.101}, tol=0.02)[0].status == "ok"


def test_the_known_table_can_be_overridden_per_call():
    out = compare("X", "t", {"debt_to_equity": 0.1}, {"debt_to_equity": 0.3}, known={})
    assert out[0].status == "mismatch"


def test_empty_pass2_fields_skip_not_applicable_and_expected_empties():
    views = {"FUND": {"pe_ratio": "not_applicable", "operating_margin": None, "industry_pe_median": None},
             "MACRO": {"commodity_context": None, "inflation_trend": None}}
    assert empty_pass2_fields(views) == [("FUND", "operating_margin"), ("MACRO", "inflation_trend")]


def test_summarize_and_report_hide_ok_lines_unless_asked():
    findings = [Finding("A", "c", "f1", "ok", 1.0, 1.0), Finding("A", "c", "f2", "mismatch", 2.0, 1.0, "+100.0%"),
                Finding("B", "c", "f1", "ok", 1.0, 1.0), Finding("B", "sec", "revenue", "skipped", note="no XBRL")]
    assert summarize(findings) == {"ok": 2, "known": 0, "mismatch": 1, "gap": 0, "skipped": 1}
    report = format_report(findings)
    assert "MISMATCH c/f2" in report and "c/f1" not in report and "SKIPPED" in report
    assert "MISMATCH 1" in report and "all checks agree" not in report.split("=== A ===")[1].split("===")[0]
    assert "c/f1" in format_report(findings, show_ok=True)


def test_a_ticker_with_only_ok_findings_says_all_agree():
    assert "all checks agree" in format_report([Finding("A", "c", "f", "ok", 1.0, 1.0)])


def test_an_absolute_tolerance_applies_to_scale_bound_fields_like_rsi():
    """RSI is on a 0 to 100 scale: 37.4 against 39.6 is 5.5% apart but only 2.2 points."""
    from services.data_audit import compare

    near = compare("X", "t", {"rsi_14": 37.4}, {"rsi_14": 39.6}, tol=0.02, abs_tol={"rsi_14": 3.0})
    far = compare("X", "t", {"rsi_14": 30.0}, {"rsi_14": 39.6}, tol=0.02, abs_tol={"rsi_14": 3.0})
    assert near[0].status == "ok" and far[0].status == "mismatch"


def test_fields_marked_not_applicable_are_not_compared():
    from services.data_audit import without_not_applicable

    theirs = {"operating_margin": 0.46, "net_margin": 0.31}
    assert without_not_applicable(theirs, {"reason": "bank", "fields": ["operating_margin"]}) == {"net_margin": 0.31}
    assert without_not_applicable(theirs, None) == theirs


def test_a_mismatch_in_an_old_bundle_becomes_known_but_a_fresh_one_stays():
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from services.data_audit import downgrade_if_stale

    mismatch = [Finding("X", "macro", "treasury_2y", "mismatch", 4.88, 4.78), Finding("X", "macro", "cpi", "ok", 1.0, 1.0)]
    old = SimpleNamespace(data_vintage=datetime.now(UTC) - timedelta(hours=10))
    fresh = SimpleNamespace(data_vintage=datetime.now(UTC) - timedelta(hours=1))
    aged = downgrade_if_stale(mismatch, old)
    assert [f.status for f in aged] == ["known", "ok"] and "10h old" in aged[0].note
    assert [f.status for f in downgrade_if_stale(mismatch, fresh)] == ["mismatch", "ok"]


def _research_bundle(ticker, canadian, sections, profile=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        stock=SimpleNamespace(ticker=ticker), canadian_data_flags=object() if canadian else None,
        research_sources=SimpleNamespace(filing_digests=[SimpleNamespace(section=s) for s in sections],
                                         business_profile=profile))


def test_a_us_name_without_its_filing_digest_is_a_gap_not_a_quiet_omission():
    """2026-10-03 to 2026-10-06 every US name lost its digest and nothing said so."""
    from services.data_audit import check_research

    by_field = {f.field: f for f in check_research(_research_bundle("KO", False, []))}

    assert by_field["filing_digest.Business"].status == "gap" and by_field["filing_digest.MDA"].status == "gap"
    assert by_field["business_profile"].status == "gap"  # no digest and no fallback


def test_a_name_with_both_sections_is_clean():
    from services.data_audit import check_research

    findings = check_research(_research_bundle("KO", False, ["Business", "MDA"]))

    assert {f.status for f in findings} == {"ok"}


def test_a_pure_canadian_name_is_known_to_have_no_filing_and_needs_the_profile_fallback():
    from services.data_audit import check_research

    findings = {f.field: f for f in check_research(_research_bundle("ZZZ.TO", True, [], profile="A profile."))}

    assert findings["filing_digest.Business"].status == "known"
    assert findings["business_profile"].status == "ok"



def _fundamental_bundle(**overrides):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    defaults = dict(
        stock=SimpleNamespace(ticker="TD.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Toronto-Dominion Bank", "sector": "Finance", "industry": "Banking"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"), data_vintage=datetime(2026, 10, 6, tzinfo=UTC),
        price_info={"current_price": 168.3, "market_cap": 2.7e11, "currency": "CAD", "high_52w": 175.3, "low_52w": 109.5},
        valuation_metrics={"pe_ratio": 17.6, "forward_pe": 15.0, "peg_ratio": None, "pb_ratio": 2.16, "ps_ratio": 4.2, "ev_ebitda": None},
        growth_metrics={"revenue_growth_yoy": 0.08, "revenue_growth_yoy_basis": "q", "revenue_growth_annual": 0.09,
                        "revenue_growth_3yr_cagr": 0.1, "eps_growth_yoy": 0.2,
                        "earnings_surprises": [{"period_end": "2026-07-31", "eps_surprise_pct": 5.0}, {"period_end": "2026-04-30", "eps_surprise_pct": 4.0}]},
        profitability_metrics={"gross_margin": None, "operating_margin": None, "net_margin": 0.25, "roe": 0.13, "roa": 0.008,
                               "fcf_to_net_income": 0.31, "margin_trend": "expanding"},
        balance_sheet_metrics={"debt_to_equity": 3.58, "current_ratio": None, "interest_coverage": None, "cash_position": 1e11,
                               "equity_to_assets": 0.06},
        dividend_info={"dividend_yield": 0.026, "payout_ratio": 0.48, "dividend_growth_5yr": 0.065, "dividend_regularity": "regular"},
        analyst_consensus={"consensus_rating": "buy", "num_analysts": 10, "target_mean": 183.4, "buy_count": 6, "hold_count": 3, "sell_count": 1},
        peer_metrics={"industry_benchmark": {"industry": "Banks", "market": "CA", "companies": 5, "median_pe": 17.0, "p25_pe": 16.8, "p75_pe": 17.6}},
        missing_fields=[], currency_mismatch=None, not_applicable=None, latest_financials_period_end="2026-07-31",
        metric_profile={"group": "financials", "lens": "Value on P/B against ROE, with P/E.", "limits": "",
                        "hidden": ["debt_to_equity", "fcf_to_net_income", "operating_margin", "gross_margin", "current_ratio",
                                   "interest_coverage", "ev_ebitda", "ps_ratio"]},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_the_fundamental_input_check_passes_a_clean_bank():
    from services.data_audit import check_fundamental_inputs

    findings = check_fundamental_inputs(_fundamental_bundle())

    assert {f.status for f in findings} == {"ok"}


def test_it_flags_oldest_first_surprises_a_negative_interest_cover_and_a_leaked_hidden_metric():
    from services.data_audit import check_fundamental_inputs

    bundle = _fundamental_bundle(
        balance_sheet_metrics={"debt_to_equity": 3.58, "current_ratio": None, "interest_coverage": -88.0, "cash_position": 1e11, "equity_to_assets": 0.06},
        metric_profile={"group": "standard", "lens": "", "limits": "", "hidden": []},
    )
    bundle.growth_metrics["earnings_surprises"].reverse()  # oldest first, as yfinance returns them

    by_field = {f.field: f for f in check_fundamental_inputs(bundle)}

    assert by_field["surprises_newest_first"].status == "mismatch"
    assert by_field["interest_coverage_sign"].status == "mismatch"
    assert by_field["hidden_metrics_absent"].status == "ok"  # standard hides nothing

    leaky = _fundamental_bundle()
    leaky.metric_profile = {**leaky.metric_profile, "hidden": [*leaky.metric_profile["hidden"], "roe"]}
    assert {f.field: f for f in check_fundamental_inputs(leaky)}["hidden_metrics_absent"].status == "mismatch"


def test_empty_pass2_fields_skip_what_the_metric_profile_hides_and_other_groups_extras():
    views = {"FUND": {"debt_to_equity": None, "operating_margin": None, "roa": None, "cash_runway_quarters": None,
                      "valuation_lens": None, "pe_ratio": None}}
    bank = {"group": "financials", "hidden": ["debt_to_equity", "operating_margin"]}

    # D/E and operating margin are hidden, the runway and lens belong elsewhere; the bank's own empty ROA and an empty
    # P/E are still real gaps
    assert empty_pass2_fields(views, bank) == [("FUND", "roa"), ("FUND", "pe_ratio")]


def test_without_not_applicable_also_drops_the_profile_hidden_fields():
    from services.data_audit import without_not_applicable

    theirs = {"net_margin": 0.31, "debt_to_equity": 3.5}
    assert without_not_applicable(theirs, None, ["debt_to_equity"]) == {"net_margin": 0.31}
