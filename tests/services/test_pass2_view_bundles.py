"""The Pass 2 FUND slice must call the P/E comparison what it is: the median of a few named peers."""
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.pass2_view import build_pass2_view
from services.orchestrator import _build_pass2_view_bundles


def _bundle(peer_records, median):
    return SimpleNamespace(
        valuation_metrics={"pe_ratio": 28.5}, growth_metrics={}, profitability_metrics={}, balance_sheet_metrics={},
        peer_metrics={"sector_medians": {"sector_median_pe": median}, "peer_records": peer_records},
        technical_indicators={}, support_resistance={}, macro_sources=MagicMock(),
        insider_activity={"transactions": []}, price_info={"market_cap": 1e12}, analyst_consensus={}, not_applicable=None,
    )


def test_the_fund_slice_names_a_peer_median_and_counts_the_peers_behind_it():
    # MSFT 2026-10-01: five peers, three with a P/E (ORCL 24.5, NOW 83.0, FTNT 61.8): the "sector median" was FTNT.
    records = [{"ticker": "ORCL", "pe_ratio": 24.5}, {"ticker": "PANW", "pe_ratio": None},
               {"ticker": "CRWD", "pe_ratio": None}, {"ticker": "NOW", "pe_ratio": 83.0}, {"ticker": "FTNT", "pe_ratio": 61.8}]
    fund = _build_pass2_view_bundles(_bundle(records, 61.8))["FUND"]
    assert fund["peer_pe_median"] == 61.8 and fund["peer_pe_count"] == 3
    assert "sector_pe_median" not in fund


def test_no_peers_gives_a_zero_count_not_a_crash():
    fund = _build_pass2_view_bundles(_bundle([], None))["FUND"]
    assert fund["peer_pe_median"] is None and fund["peer_pe_count"] == 0


def test_the_pass2_view_passes_both_fields_through():
    fund = _build_pass2_view_bundles(_bundle([{"ticker": "A", "pe_ratio": 20.0}], 20.0))["FUND"]
    view = build_pass2_view("FUND", {}, fund)
    assert view["peer_pe_median"] == 20.0 and view["peer_pe_count"] == 1


def test_the_pe_premium_or_discount_is_computed_in_code_with_its_sign():
    # MSFT 2026-10-02: P/E 28.59 against a peer median of 62.81 is about a 54.6% discount.
    fund = _build_pass2_view_bundles(_bundle([{"ticker": "A", "pe_ratio": 62.81}, {"ticker": "B", "pe_ratio": 62.81}], 62.81))["FUND"]
    assert fund["pe_vs_peer_median_pct"] == -54.6
    assert build_pass2_view("FUND", {}, fund)["pe_vs_peer_median_pct"] == -54.6


def test_the_premium_is_none_unless_both_numbers_are_positive():
    from services.orchestrator import _pe_vs_peer_median_pct

    assert _pe_vs_peer_median_pct(30.0, 20.0) == 50.0
    for pe, med in ((None, 20.0), (30.0, None), (-5.0, 20.0), (30.0, 0.0)):
        assert _pe_vs_peer_median_pct(pe, med) is None


def test_the_sent_slice_carries_insider_dollars_and_materiality_into_the_view():
    b = _bundle([], None)
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
        b = _bundle([], None)
        b.analyst_consensus = consensus
        sent = _build_pass2_view_bundles(b)["SENT"]
        # the model's own analyst_sentiment object carries no consensus, as in the real outputs
        view = build_pass2_view("SENT", {"structured_data": {"analyst_sentiment": {"consensus_trend": "stable"}}}, sent)
        assert view["consensus_rating"] == consensus["consensus_rating"]
        assert view["average_price_target"] == consensus["target_mean"]


def test_missing_analyst_consensus_stays_none_not_a_crash():
    b = _bundle([], None)
    b.analyst_consensus = {}
    view = build_pass2_view("SENT", {}, _build_pass2_view_bundles(b)["SENT"])
    assert view["consensus_rating"] is None and view["average_price_target"] is None


def test_not_applicable_metrics_reach_pass_2_as_not_applicable_not_none():
    b = _bundle([], None)
    b.profitability_metrics = {"operating_margin": None}
    b.not_applicable = {"reason": "bank", "fields": ["operating_margin"]}
    fund = _build_pass2_view_bundles(b)["FUND"]
    assert fund["operating_margin"] == "not_applicable"
    assert build_pass2_view("FUND", {}, fund)["operating_margin"] == "not_applicable"


def test_a_reit_without_a_pe_also_gets_the_peer_pe_fields_marked_not_applicable():
    b = _bundle([], None)
    b.valuation_metrics = {"pe_ratio": None}
    b.not_applicable = {"reason": "REIT", "fields": ["pe_ratio"]}
    fund = _build_pass2_view_bundles(b)["FUND"]
    assert fund["pe_ratio"] == fund["peer_pe_median"] == fund["pe_vs_peer_median_pct"] == "not_applicable"


def test_the_technical_levels_reach_pass_2_as_plain_floats_not_numpy_scalars():
    import numpy as np

    b = _bundle([], None)
    b.support_resistance = {"nearest_support": np.float64(509.03), "nearest_resistance": np.float64(515.94)}
    tech = _build_pass2_view_bundles(b)["TECH"]
    assert type(tech["nearest_support"]) is float and tech["nearest_resistance"] == 515.94
