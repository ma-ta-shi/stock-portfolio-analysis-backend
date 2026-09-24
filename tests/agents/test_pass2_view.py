"""Tests for build_pass2_view — the Pass1→Pass2 contract builder.

These are unit tests of a pure mapping function, so hand-built inputs are the
right tool: the thing under test is whether each agent's declared field lands in
the right place, not how a provider behaves. Real-data verification of the same
path is in backend/_sent_rig.py's live run.
"""
import pytest

from agents.compression import compress_pass1_outputs
from agents.pass2_view import build_pass2_view
from agents.validators.compression import validate_pass2_view_shape


# ---------- the container trap: RSRCH/SENT/MACRO nest under structured_data,
# ---------- FUND/TECH under interpretive_fields. A generic reader breaks 2 of 5.

def test_fund_reads_interpretive_fields_not_structured_data():
    out = {"interpretive_fields": {"valuation_vs_sector": "fair", "health_rating": "healthy"}}
    view = build_pass2_view("FUND", out)
    assert view["valuation_vs_sector"] == "fair"
    assert view["health_rating"] == "healthy"


def test_fund_ignores_structured_data_container():
    """FUND output placed under the wrong key must not silently populate."""
    out = {"structured_data": {"valuation_vs_sector": "overvalued"}}
    assert build_pass2_view("FUND", out)["valuation_vs_sector"] is None


def test_tech_reads_interpretive_fields():
    out = {"interpretive_fields": {"primary_trend": "bullish", "confluence_score": 3}}
    view = build_pass2_view("TECH", out)
    assert view["primary_trend"] == "bullish"
    assert view["confluence_score"] == 3


def test_sent_reads_structured_data():
    out = {"structured_data": {"peer_sentiment_comparison": "in line with PEER_1"}}
    assert build_pass2_view("SENT", out)["peer_sentiment_comparison"] == "in line with PEER_1"


# ---------- nested paths flatten correctly

def test_sent_flattens_nested_news_sentiment():
    """The prompt declares a FLAT view; news_sentiment is nested in the schema."""
    out = {"structured_data": {"news_sentiment": {"overall": "positive", "sentiment_trend": "improving"}}}
    view = build_pass2_view("SENT", out)
    assert view["news_sentiment_overall"] == "positive"
    assert view["sentiment_trend"] == "improving"
    assert "news_sentiment" not in view


def test_rsrch_flattens_moat_assessment():
    out = {"structured_data": {"moat_assessment": {"overall_moat_durability": "strong", "moat_trend": "stable"}}}
    view = build_pass2_view("RSRCH", out)
    assert view["overall_moat_durability"] == "strong"
    assert view["moat_trend"] == "stable"


# ---------- top_* are slices, not separate fields

def test_rsrch_top_fields_are_slices_of_full_arrays():
    out = {"structured_data": {
        "growth_drivers": ["d1", "d2", "d3", "d4"],
        "competitive_threats": ["t1", "t2", "t3"],
        "recent_developments": [{"news_id": "N1"}, {"news_id": "N2"}, {"news_id": "N3"}],
    }}
    view = build_pass2_view("RSRCH", out)
    assert view["top_growth_drivers"] == ["d1", "d2"]
    assert view["top_competitive_threats"] == ["t1", "t2"]
    assert len(view["top_recent_developments"]) == 2


def test_top_fields_tolerate_non_list():
    out = {"structured_data": {"growth_drivers": None}}
    assert build_pass2_view("RSRCH", out)["top_growth_drivers"] == []


# ---------- orchestrator-owned fields come from the bundle, never the LLM

def test_fund_numeric_passthrough_comes_from_bundle():
    """pe_ratio and margin_trend are DataBundle passthrough (margin_trend moved out
    of the LLM's interpretive fields in v5.1)."""
    out = {"interpretive_fields": {"health_rating": "healthy"}}
    view = build_pass2_view("FUND", out, {"pe_ratio": 17.9, "margin_trend": "improving"})
    assert view["pe_ratio"] == 17.9
    assert view["margin_trend"] == "improving"


def test_tech_orchestrator_fields_come_from_bundle():
    out = {"interpretive_fields": {"primary_trend": "bearish"}}
    bundle = {"nearest_support": 120.5, "nearest_resistance": 141.0,
              "volatility_regime_derived": "contracting"}
    view = build_pass2_view("TECH", out, bundle)
    assert view["nearest_support"] == 120.5
    assert view["volatility_regime_derived"] == "contracting"


def test_missing_bundle_yields_none_not_error():
    """Graceful degradation, matching the precompute/ pattern."""
    view = build_pass2_view("TECH", {"interpretive_fields": {}})
    assert view["nearest_support"] is None
    assert view["volatility_regime_derived"] is None


# ---------- shape agreement with the validator

@pytest.mark.parametrize("agent_id,output,bundle", [
    ("RSRCH", {"structured_data": {
        "thesis_archetype": "dividend_compounder", "competitive_position": "strong",
        "moat_assessment": {"overall_moat_durability": "moderate", "moat_trend": "stable"},
        "management_assessment": "competent", "growth_drivers": ["g"],
        "competitive_threats": ["t"], "recent_developments": [{"news_id": "N1"}],
        "peer_comparison_summary": "vs PEER_1"}}, None),
    ("FUND", {"interpretive_fields": {
        "valuation_vs_sector": "fair", "health_rating": "healthy",
        "guidance_vs_consensus": "inline", "dividend_sustainability": "strong"}},
     {"pe_ratio": 17.9, "margin_trend": "stable"}),
    ("TECH", {"interpretive_fields": {
        "primary_trend": "bullish", "trend_strength": "moderate", "momentum_zone": "neutral",
        "momentum_direction": "improving", "volume_confirmation": "confirming",
        "nearest_level_bias": "midrange", "confluence_score": 2}},
     {"nearest_support": 1.0, "nearest_resistance": 2.0, "volatility_regime_derived": "normal"}),
    ("SENT", {"structured_data": {
        "news_sentiment": {"overall": "positive", "sentiment_trend": "stable"},
        "insider_activity_interpretation": "neutral",
        "short_interest_interpretation": {"trend": "unknown", "interpretation": "normal"},
        "peer_sentiment_comparison": "vs PEER_1"}}, None),
    ("MACRO", {"structured_data": {
        "overall_macro_environment": "neutral", "sector_cycle_position": "mid_cycle",
        "sector_tailwinds": ["tw"], "sector_headwinds": ["hw"]}},
     {"volatility_regime": "elevated"}),
])
def test_built_view_satisfies_shape_validator(agent_id, output, bundle):
    """Every agent's built view must pass the shape validator it is written against."""
    ok, errors = validate_pass2_view_shape(agent_id, build_pass2_view(agent_id, output, bundle))
    assert ok, f"{agent_id}: {errors}"


def test_unknown_agent_returns_empty():
    assert build_pass2_view("NOPE", {"structured_data": {"x": 1}}) == {}


# ---------- the 51.1 regression: the compressor must BUILD, not read

def test_compressor_builds_pass2_view_when_absent():
    """Pass 1 agents never emit pass2_view. Before the fix the compressor read it,
    got {}, and every agent rendered as NOT AVAILABLE with a zeroed score."""
    pass1 = {"SENT": {
        "assessment_summary": "s", "analysis_confidence": "medium", "narrative": "n", "caveats": [],
        "structured_data": {"news_sentiment": {"overall": "positive", "sentiment_trend": "stable"},
                            "insider_activity_interpretation": "neutral",
                            "short_interest_interpretation": {"trend": "unknown", "interpretation": "normal"},
                            "peer_sentiment_comparison": "vs PEER_1"}}}
    view = compress_pass1_outputs(pass1)["SENT"]["pass2_view"]
    assert view, "pass2_view must be built, not read as empty"
    assert view["news_sentiment_overall"] == "positive"


def test_explicit_pass2_view_on_output_wins():
    """A caller-supplied view overrides the builder."""
    pass1 = {"SENT": {"assessment_summary": "s", "analysis_confidence": "medium", "narrative": "n",
                      "pass2_view": {"news_sentiment_overall": "override"},
                      "structured_data": {"news_sentiment": {"overall": "positive"}}}}
    view = compress_pass1_outputs(pass1)["SENT"]["pass2_view"]
    assert view["news_sentiment_overall"] == "override"


def test_none_output_still_compresses_to_none():
    assert compress_pass1_outputs({"FUND": None})["FUND"] is None


def test_string_pass2_view_from_model_is_ignored_not_trusted():
    """A model can emit `pass2_view` as a STRING. Observed live on RY.TO: it reached
    build_pass2_user_message and raised AttributeError: 'str' object has no
    attribute 'items'. The explicit-view override must check isinstance, not
    truthiness — third instance of this shape-trust bug class in this pass."""
    pass1 = {"SENT": {"assessment_summary": "s", "analysis_confidence": "medium", "narrative": "n",
                      "pass2_view": "positive overall sentiment",
                      "structured_data": {"news_sentiment": {"overall": "positive"}}}}
    view = compress_pass1_outputs(pass1)["SENT"]["pass2_view"]
    assert isinstance(view, dict), "a string pass2_view must be rebuilt, not passed through"
    assert view["news_sentiment_overall"] == "positive"


def test_renderer_reads_real_fields_off_a_databundle_shaped_object():
    """86bbuhjup: build_pass2_user_message's header line is real DataBundle-field
    translation, not a pure port -- confirm it actually reads the bundle's fields,
    not just that a call doesn't crash. Regression coverage for the exact bug class
    86bbt1k1p/86bbt1kct found in the CIO/Shadow payload builders (fields silently
    rendering NOT AVAILABLE/N/A instead of the real value)."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from agents.compression import build_pass2_user_message
    compressed = {
        "RSRCH": {
            "assessment_summary": "Durable moat, steady growth.",
            "analysis_confidence": "high",
            "caveats": [],
            "pass2_view": {"thesis_archetype": "secular_grower"},
            "narrative_truncated": "n",
        }
    }
    bundle = SimpleNamespace(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
    )
    msg = build_pass2_user_message(bundle, compressed)
    assert "SHOP.TO (Shopify Inc) | Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg
    assert "2026-09-23" in msg
    assert "thesis_archetype" in msg


def test_renderer_survives_a_non_dict_pass2_view():
    """Even if a bad shape gets that far, rendering must degrade, not crash."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from agents.compression import build_pass2_user_message
    compressed = {"SENT": {"assessment_summary": "s", "analysis_confidence": "medium",
                           "caveats": [],
                           "pass2_view": "not a dict", "narrative_truncated": "n"}}
    # SimpleNamespace, not a real DataBundle -- build_pass2_user_message only reads
    # a handful of attributes off it (see the module docstring's field mapping), and
    # this test is about rendering degradation, not DataBundle validity. Same
    # duck-typing convention data/pipeline.py already uses for a narrower-than-full
    # input to a function that only needs a few fields off it.
    bundle = SimpleNamespace(
        stock=SimpleNamespace(ticker="X", currency="CAD", exchange="E"),
        company_info={"name": "X", "sector": "S"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 8, 31, tzinfo=UTC),
    )
    msg = build_pass2_user_message(bundle, compressed)
    assert "NOT AVAILABLE" in msg
