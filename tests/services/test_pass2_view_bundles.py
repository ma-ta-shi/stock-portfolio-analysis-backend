"""The Pass 2 FUND slice carries the industry P/E benchmark, and where the stock sits in it, computed in code."""
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.pass2_view import build_pass2_view
from services.orchestrator import _build_pass2_view_bundles


def _bench(median=23.5, p25=12.9, p75=50.7, companies=82):
    return {"industry": "Software—Infrastructure", "market": "US", "companies": companies, "median_pe": median,
            "p25_pe": p25, "p75_pe": p75, "closest": ["ORCL", "PLTR"]}


def _bundle(benchmark=None, pe=28.5, canadian=False):
    return SimpleNamespace(canadian_data_flags=object() if canadian else None,
        valuation_metrics={"pe_ratio": pe}, growth_metrics={}, profitability_metrics={}, balance_sheet_metrics={}, dividend_info={},
        peer_metrics={"industry_benchmark": benchmark},
        technical_indicators={}, support_resistance={}, macro_sources=MagicMock(),
        insider_activity={"transactions": []}, price_info={"market_cap": 1e12}, analyst_consensus={}, not_applicable=None, metric_profile=None,
        data_vintage=datetime(2026, 10, 3, tzinfo=UTC), short_interest=None, analyst_rating_changes=None,
    )


def test_the_fund_slice_carries_the_industry_benchmark_its_range_and_its_count():
    fund = _build_pass2_view_bundles(_bundle(_bench()))["FUND"]
    assert fund["industry_pe_median"] == 23.5 and fund["industry_pe_range"] == "12.9 to 50.7" and fund["industry_pe_count"] == 82
    assert "peer_pe_median" not in fund and "pe_vs_peer_median_pct" not in fund


def test_a_pe_inside_the_middle_half_is_in_line_even_when_it_is_above_the_median():
    """MSFT 2026-10-03: 28.5 is +21% against the median 23.5 but inside 12.9 to 50.7: in line, not a premium."""
    fund = _build_pass2_view_bundles(_bundle(_bench()))["FUND"]
    assert fund["pe_vs_industry_median_pct"] == 21.3 and fund["pe_vs_industry"] == "within_range"


def test_a_pe_outside_the_middle_half_is_above_or_below():
    assert _build_pass2_view_bundles(_bundle(_bench(), pe=80.0))["FUND"]["pe_vs_industry"] == "above_range"
    assert _build_pass2_view_bundles(_bundle(_bench(), pe=10.0))["FUND"]["pe_vs_industry"] == "below_range"


def test_no_benchmark_leaves_every_industry_field_empty_not_a_crash():
    fund = _build_pass2_view_bundles(_bundle(None))["FUND"]
    for key in ("industry_pe_median", "industry_pe_range", "industry_pe_count", "pe_vs_industry_median_pct", "pe_vs_industry"):
        assert fund[key] is None


def test_a_loss_or_missing_pe_has_no_position():
    for pe in (None, -5.0):
        fund = _build_pass2_view_bundles(_bundle(_bench(), pe=pe))["FUND"]
        assert fund["pe_vs_industry_median_pct"] is None and fund["pe_vs_industry"] is None


def test_the_pass2_view_passes_the_industry_fields_through():
    fund = _build_pass2_view_bundles(_bundle(_bench()))["FUND"]
    view = build_pass2_view("FUND", {}, fund)
    assert view["industry_pe_median"] == 23.5 and view["industry_pe_range"] == "12.9 to 50.7"
    assert view["industry_pe_count"] == 82 and view["pe_vs_industry"] == "within_range"


def test_the_sent_slice_carries_insider_dollars_and_materiality_into_the_view():
    b = _bundle()
    recent = datetime.now(UTC).date().isoformat()
    b.insider_activity = {"transactions": [{"date": recent, "is_issuer": False, "transaction_type": "sale", "shares": 1, "value": 5e6}]}
    b.price_info = {"market_cap": 1e10}
    sent = _build_pass2_view_bundles(b)["SENT"]
    assert sent["insider_materiality"] == "notable" and "net selling $5.0M" in sent["insider_activity_90d"]
    view = build_pass2_view("SENT", {}, sent)
    assert view["insider_materiality"] == "notable"


def test_analyst_consensus_comes_from_the_bundle_for_both_markets_not_from_the_model_output():
    """Empty in 44 of 46 real runs: the view read these from an object the Sentiment agent never writes."""
    for consensus in ({"consensus_rating": "strong_buy", "target_mean": 244.23},   # SHOP.TO, tmx, CAD
                      {"consensus_rating": "buy", "target_mean": 518.4}):           # MSFT, yfinance, USD
        b = _bundle()
        b.analyst_consensus = consensus
        sent = _build_pass2_view_bundles(b)["SENT"]
        # the model's own analyst_sentiment object carries no consensus, as in the real outputs
        view = build_pass2_view("SENT", {"structured_data": {"analyst_sentiment": {"consensus_trend": "stable"}}}, sent)
        assert view["consensus_rating"] == consensus["consensus_rating"]
        assert view["average_price_target"] == consensus["target_mean"]


def test_missing_analyst_consensus_stays_none_not_a_crash():
    b = _bundle()
    b.analyst_consensus = {}
    view = build_pass2_view("SENT", {}, _build_pass2_view_bundles(b)["SENT"])
    assert view["consensus_rating"] is None and view["average_price_target"] is None


def test_not_applicable_metrics_reach_pass_2_as_not_applicable_not_none():
    b = _bundle()
    b.profitability_metrics = {"operating_margin": None}
    b.not_applicable = {"reason": "bank", "fields": ["operating_margin"]}
    fund = _build_pass2_view_bundles(b)["FUND"]
    assert fund["operating_margin"] == "not_applicable"
    assert build_pass2_view("FUND", {}, fund)["operating_margin"] == "not_applicable"


def test_a_reit_without_a_pe_also_gets_the_industry_pe_fields_marked_not_applicable():
    b = _bundle(_bench())
    b.valuation_metrics = {"pe_ratio": None}
    b.not_applicable = {"reason": "REIT", "fields": ["pe_ratio"]}
    fund = _build_pass2_view_bundles(b)["FUND"]
    assert fund["pe_ratio"] == fund["pe_vs_industry_median_pct"] == fund["pe_vs_industry"] == "not_applicable"
    assert fund["industry_pe_median"] == "not_applicable"  # a REIT's industry multiple is not meaningful either


def test_the_technical_levels_reach_pass_2_as_plain_floats_not_numpy_scalars():
    import numpy as np

    b = _bundle()
    b.support_resistance = {"nearest_support": np.float64(509.03), "nearest_resistance": np.float64(515.94)}
    tech = _build_pass2_view_bundles(b)["TECH"]
    assert type(tech["nearest_support"]) is float and tech["nearest_resistance"] == 515.94


def test_a_reits_industry_pe_is_not_applicable_even_though_the_screener_has_one():
    """CAR-UN.TO 2026-10-03: the pipeline treats P/E as not applicable for a REIT, so a REIT industry median (11.4) must not
    reach Pass 2 as if it meant something."""
    b = _bundle(_bench(median=11.4))
    b.valuation_metrics = {"pe_ratio": None}
    b.not_applicable = {"reason": "REIT", "fields": ["pe_ratio"]}
    fund = _build_pass2_view_bundles(b)["FUND"]
    for key in ("industry_pe_median", "industry_pe_range", "industry_pe_count", "pe_vs_industry_median_pct", "pe_vs_industry"):
        assert fund[key] == "not_applicable"


def test_a_pe_from_near_zero_earnings_is_not_meaningful_and_has_no_percentage():
    """CRWD 2026-10-03: a trailing P/E of 9,001 printed as '+38,619% above range'."""
    fund = _build_pass2_view_bundles(_bundle(_bench(), pe=9001.0))["FUND"]
    assert fund["pe_vs_industry"] == "not_meaningful" and fund["pe_vs_industry_median_pct"] is None


def test_the_sent_slice_decides_short_interest_and_analyst_changes_in_code():
    """The model used to write short_interest_interpretation (27 of 48 first attempts failed the validator on it)."""
    b = _bundle()
    b.short_interest = {"short_interest_pct": 1.79, "days_to_cover": 8.79, "shares_short": 28_890_182,
                        "shares_short_prior_month": 27_719_808, "as_of_date": "2026-09-15"}
    b.price_info = {"market_cap": 1e12, "current_price": 168.07}
    b.analyst_consensus = {"target_mean": 183.4}
    b.analyst_rating_changes = [{"date": "2026-09-20", "action": "up", "price_target_action": "Raises"}]

    sent = _build_pass2_view_bundles(b)["SENT"]

    assert sent["short_interest_interpretation"] == {"trend": "stable", "interpretation": "normal"}
    assert sent["analyst_changes_90d"].startswith("1 upgrade, 0 downgrades, 0 new initiations in the last 90d")
    assert "average target +9.1% against the price" in sent["analyst_changes_90d"]


def test_the_sent_slice_without_short_interest_or_changes_is_honest_not_a_crash():
    sent = _build_pass2_view_bundles(_bundle())["SENT"]
    assert sent["short_interest_interpretation"] == {"trend": "unknown", "interpretation": "insufficient_data"}
    assert sent["analyst_changes_90d"] == "no data"


def _macro_bundle(canadian):
    bundle = _bundle(canadian=canadian)
    bundle.macro_sources = SimpleNamespace(
        rate_trend="tightening", cpi_trend="rising", cad_trend="stable", sector_commodity_direction=None,
        vix_regime="low", boc_rate_trend="pausing", ca_cpi_trend="stable")
    return bundle


def test_a_canadian_stocks_macro_view_carries_the_bank_of_canada_and_canadian_cpi_trends():
    """TD.TO 2026-10-06: the BoC was pausing and Canadian CPI stable, but Bull, Bear and the CIO were given the Fed's
    "tightening" and US CPI's trend."""
    macro = _build_pass2_view_bundles(_macro_bundle(canadian=True))["MACRO"]
    assert macro["interest_rate_direction"] == "pausing" and macro["inflation_trend"] == "stable"


def test_a_us_stocks_macro_view_still_carries_the_fed_and_us_cpi_trends():
    macro = _build_pass2_view_bundles(_macro_bundle(canadian=False))["MACRO"]
    assert macro["interest_rate_direction"] == "tightening" and macro["inflation_trend"] == "rising"


def test_a_missing_canadian_trend_is_none_not_the_us_one():
    bundle = _macro_bundle(canadian=True)
    bundle.macro_sources.boc_rate_trend = None
    assert _build_pass2_view_bundles(bundle)["MACRO"]["interest_rate_direction"] is None


def test_the_fund_slice_leaves_out_what_the_stocks_profile_hides_and_carries_the_lens():
    """TD.TO: D/E 3.58 reached Pass 2 (cited in 14 of 16 Bull outputs) beside a not_applicable token for operating margin."""
    bundle = _bundle()
    bundle.valuation_metrics = {"pe_ratio": 17.6, "pb_ratio": 2.16}
    bundle.profitability_metrics = {"roe": 0.1275, "roa": 0.008, "fcf_to_net_income": 0.31, "operating_margin": None, "net_margin": 0.25}
    bundle.balance_sheet_metrics = {"debt_to_equity": 3.58, "equity_to_assets": 0.06}
    bundle.metric_profile = {"group": "financials", "lens": "Value on P/B against ROE, with P/E.", "limits": "",
                             "hidden": ["debt_to_equity", "fcf_to_net_income", "operating_margin"]}

    fund = _build_pass2_view_bundles(bundle)["FUND"]

    assert fund["debt_to_equity"] is None and fund["fcf_to_net_income"] is None and fund["operating_margin"] is None
    assert fund["pb_ratio"] == 2.16 and fund["roa"] == 0.008 and fund["equity_to_assets"] == 0.06
    assert fund["valuation_lens"] == "Value on P/B against ROE, with P/E."
    assert "not_applicable" not in fund.values()


def test_a_hidden_pe_takes_the_industry_pe_position_with_it():
    bundle = _bundle(_bench())
    bundle.metric_profile = {"group": "pre_profit", "lens": "No earnings yet.", "limits": "", "hidden": ["pe_ratio"]}

    fund = _build_pass2_view_bundles(bundle)["FUND"]

    assert fund["pe_ratio"] is None and fund["industry_pe_median"] is None and fund["pe_vs_industry"] is None


def test_the_fund_and_tech_slices_carry_the_figures_pass2_was_quoting_from_the_narratives():
    """About a quarter of Bull's and Bear's figures came from a Pass 1 narrative alone (PEG, forward P/E, interest cover,
    current ratio, payout; RSI, distance to the averages, ATR distance, volume ratio)."""
    bundle = _bundle()
    bundle.valuation_metrics.update({"forward_pe": 15.0, "peg_ratio": 0.6, "ps_ratio": 4.2, "ev_ebitda": 12.0})
    bundle.profitability_metrics["gross_margin"] = 0.45
    bundle.balance_sheet_metrics.update({"interest_coverage": 2.64, "current_ratio": 0.9})
    bundle.dividend_info = {"dividend_yield": 0.058, "payout_ratio": 1.26, "dividend_growth_5yr": 0.03}
    bundle.technical_indicators = {"rsi_14": 48.6, "price_vs_sma20_pct": -1.09, "price_vs_sma50_pct": -5.23,
                                   "price_vs_sma200_pct": -8.42, "volume_ratio_today": 2.13}
    bundle.support_resistance = {"nearest_support": 167.3, "atr_to_support": 0.4, "atr_to_resistance": 1.9}

    views = _build_pass2_view_bundles(bundle)

    assert views["FUND"]["forward_pe"] == 15.0 and views["FUND"]["payout_ratio"] == 1.26 and views["FUND"]["interest_coverage"] == 2.64
    assert views["TECH"]["rsi_14"] == 48.6 and views["TECH"]["price_vs_sma200_pct"] == -8.42
    assert views["TECH"]["support_atr_distance"] == 0.4 and views["TECH"]["resistance_atr_distance"] == 1.9


def test_a_bank_does_not_get_the_figures_its_profile_hides_in_the_new_fund_fields():
    bundle = _bundle()
    bundle.balance_sheet_metrics.update({"interest_coverage": 2.64, "current_ratio": 0.9})
    bundle.profitability_metrics["gross_margin"] = 0.45
    bundle.metric_profile = {"group": "financials", "lens": "", "limits": "",
                             "hidden": ["interest_coverage", "current_ratio", "gross_margin", "ev_ebitda", "ps_ratio"]}

    fund = _build_pass2_view_bundles(bundle)["FUND"]

    assert fund["interest_coverage"] is None and fund["current_ratio"] is None and fund["gross_margin"] is None
