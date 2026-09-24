"""pytest tests for Pass 1 validators.

Tests pass/fail for each validator using representative outputs.
"""
import pytest
from agents.validators.pass1 import (
    validate_stock_researcher,
    validate_fundamental_analyst,
    validate_technical_analyst,
    validate_sentiment_analyst,
    validate_macro_economist,
    validate_earnings_proximity_caveat,
    validate_thin_volume_caveat,
    validate_canadian_caveat,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _base_pass1_output() -> dict:
    """Minimal valid Pass 1 output skeleton."""
    return {
        "assessment_summary": "A solid US large-cap technology company with strong fundamentals and growing AI exposure.",
        "analysis_confidence": "high",
        "caveats": ["Coverage limited to public filings and earnings calls."],
        "key_factors": [
            {"factor": "Revenue growth", "importance": "high", "sentiment": "positive", "evidence": "FILING:10-K: 14.2% YoY"},
            {"factor": "FCF generation", "importance": "high", "sentiment": "positive", "evidence": "FILING:MD&A: FCF yield 4.8%"},
        ],
        "risks": [
            {"risk": "Competitive pressure from hyperscalers", "severity": "medium", "evidence": "N1: AWS native products"}
        ],
        "narrative": "GlobalTech demonstrates strong compounding fundamentals. Revenue grew 14.2% (FILING:10-K) driven by cloud migration (N1) and AI copilot adoption. Operating leverage is materializing with margins expanding 120bps (FILING:MD&A). Management credibility is high based on consistent guidance track record (TRANSCRIPT:Q1). Peer comparison shows GlobalTech trailing MSFT on growth but leading on FCF yield (PEER_1). Key risk is SAP's AI roadmap which could erode switching costs (N2). Overall thesis is a quality compounder with mid-term catalysts. Confidence is high given breadth of data sources. The DataStream acquisition adds incremental cloud revenue but introduces integration risk warranting close monitoring over the next two quarters.",
    }


def _stock_researcher_output() -> dict:
    """No `pass2_view` here on purpose: that's built by the orchestrator downstream of this
    agent (compression.py), never produced by the LLM directly -- validate_stock_researcher no
    longer checks it (86bbuhk82; see the removal note in simulation/validators/pass1.py)."""
    out = _base_pass1_output()
    out["structured_data"] = {
        "thesis_archetype": "quality_compounder",
        # Real schema declares both as enums (dominant|strong|average|weak|deteriorating and
        # excellent|competent|concerning|poor|insufficient_data respectively) -- this fixture
        # used to put descriptive prose here instead, which is what the real schema's separate
        # free-text companion fields (revenue_mix_notes, management_notes) are for.
        "competitive_position": "strong",
        "moat_assessment": {"overall_moat_durability": "strong", "moat_trend": "stable"},
        "management_assessment": "competent",
        "growth_drivers": ["AI copilot attach rates (N1)", "Cloud migration cycle (N2)"],
        "competitive_threats": ["SAP AI roadmap (N2)", "Oracle cloud expansion (PEER_1)"],
        "recent_developments": ["DataStream acquisition (N1)", "AI copilot launch (N2)"],
        "peer_comparison_summary": "GlobalTech compares favorably to peers on FCF but trails on revenue growth vs MSFT.",
    }
    return out


def _fundamental_analyst_output() -> dict:
    """`interpretive_fields` is the sole real container as of 86bbdutn6 -- the old
    `structured_data` + `pass2_view` shape this fixture used to build is gone from the
    validator entirely (86bbuhk82 item 3), not just deprioritized."""
    out = _base_pass1_output()
    out["interpretive_fields"] = {
        "valuation_vs_sector": "fair",
        "health_rating": "healthy",
        "guidance_vs_consensus": "inline",
        "dividend_sustainability": "strong",
        "peer_comparison_summary": "Leads sector on FCF yield and margins.",
    }
    return out


def _technical_analyst_output() -> dict:
    """`interpretive_fields` is the sole real container as of 86bbdutn6 (see the
    equivalent note on `_fundamental_analyst_output`)."""
    out = _base_pass1_output()
    out["interpretive_fields"] = {
        "primary_trend": "bullish",
        "trend_strength": "moderate",
        "momentum_zone": "neutral",
        "momentum_direction": "improving",
        "momentum_divergence": "none",
        "volume_confirmation": "confirming",
        "nearest_level_bias": "near_support",
        "pattern_signal": "none",
        "pattern_confirmed": False,
        "confluence_score": 1,
        "suggested_invalidation_level": 175.0,
        "nearest_support": 178.50,
        "nearest_resistance": 195.20,
        "rsi_zone_adjusted": "neutral",
        "volatility_regime_derived": "normal",
        "weekly_trend": "sideways_to_up",
    }
    return out


def _sentiment_analyst_output() -> dict:
    out = _base_pass1_output()
    out["structured_data"] = {
        "news_sentiment": {
            "overall": "positive",
            "dominant_themes": ["AI adoption", "margin expansion"],
            "sentiment_trend": "stable",
        },
        "analyst_sentiment": {
            "consensus_direction": "bullish",
            "recent_changes": "3 upgrades, 1 downgrade in past 30 days.",
            "avg_price_target": 205.0,
        },
        "insider_activity_interpretation": "CEO bought shares at $183 in March — positive signal of management conviction.",
        "short_interest_interpretation": {"trend": "stable", "interpretation": "normal"},
        "social_sentiment": "unknown",
        "narrative_momentum": "stable",
        "positioning_assessment": "neutral",
        "analyst_consensus": "buy",
        "peer_sentiment_comparison": "Sector broadly positive; PEER_1 (MSFT) and PEER_2 (ORCL) showing similar momentum.",
    }
    out["contrarian_signals"] = []
    out["pass2_view"] = {
        "news_sentiment": {
            "overall": "positive",
            "dominant_themes": ["AI adoption", "margin expansion"],
            "sentiment_trend": "stable",
        },
        "analyst_sentiment": {
            "consensus_direction": "bullish",
            "recent_changes": "3 upgrades, 1 downgrade.",
            "avg_price_target": 205.0,
        },
        "insider_activity_interpretation": "CEO bought shares — positive signal.",
        "short_interest_interpretation": {"trend": "stable", "interpretation": "normal"},
        "peer_sentiment_comparison": "PEER_1 and PEER_2 showing similar positive sentiment.",
        "contrarian_signals": [],
    }
    return out


def _macro_economist_output() -> dict:
    """Valid Macro Economist output with narrative ~80-120 words."""
    out = _base_pass1_output()
    out["narrative"] = (
        "The macro environment presents a mixed picture for GlobalTech. "
        "Elevated Fed Funds Rate at 4.50% (RATE) compresses tech multiples, "
        "though the yield curve at 4.28% (YIELD) remains stable. "
        "CPI at 3.1% (CPI) creates mild input cost pressure but software margins are largely insulated. "
        "GDP growth of 2.4% (GDP) supports enterprise IT budgets. "
        "USD/CAD at 1.362 (FX) is neutral for this USD-denominated stock. "
        "VIX at 16.8 (VIX) signals normal volatility. "
        "AI infrastructure spending remains a powerful sector tailwind offsetting rate headwinds."
    )
    out["structured_data"] = {
        "overall_macro_environment": "neutral",
        "interest_rate_environment": {"impact_on_stock": "negative", "current_direction": "stable", "rationale": "Elevated rates compress growth multiples."},
        "inflation_environment": {"impact_on_stock": "neutral", "trend": "falling", "rationale": "Software margins largely insulated from inflation."},
        "economic_growth": {"outlook": "stable", "impact_on_stock": "positive", "rationale": "2.4% GDP growth supports IT budgets."},
        "currency_impact": {"impact_on_stock": "neutral", "cad_usd_trend": "stable", "rationale": "USD-denominated stock; CAD/USD stable."},
        "volatility_regime": "normal",
        "sector_cycle_position": "mid_cycle",
        "sector_tailwinds": ["AI infrastructure spending", "Enterprise digital transformation"],
        "sector_headwinds": ["Elevated rates compress multiples", "Macro slowdown risk"],
        "commodity_context": "not applicable",
    }
    out["pass2_view"] = {
        "overall_macro_environment": "neutral",
        "interest_rate_environment": {"impact_on_stock": "negative", "current_direction": "stable", "rationale": "Elevated rates compress growth multiples."},
        "inflation_environment": {"impact_on_stock": "neutral", "trend": "falling", "rationale": "Software insulated."},
        "economic_growth": {"outlook": "stable", "impact_on_stock": "positive", "rationale": "2.4% GDP supports IT budgets."},
        "currency_impact": {"impact_on_stock": "neutral", "cad_usd_trend": "stable", "rationale": "USD stock."},
        "volatility_regime": "normal",
        "sector_cycle_position": "mid_cycle",
        "sector_tailwinds": ["AI infrastructure spending", "Enterprise digital transformation"],
        "sector_headwinds": ["Elevated rates", "Macro slowdown"],
        "commodity_context": "not applicable",
    }
    return out


# ─── Stock Researcher Tests ───────────────────────────────────────────────────

class TestStockResearcher:
    def test_valid_output_passes(self):
        out = _stock_researcher_output()
        passed, errors = validate_stock_researcher(out)
        assert passed, f"Valid output should pass: {errors}"

    def test_missing_thesis_archetype_fails(self):
        out = _stock_researcher_output()
        del out["structured_data"]["thesis_archetype"]
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("thesis_archetype" in e for e in errors)

    def test_invalid_thesis_archetype_fails(self):
        out = _stock_researcher_output()
        out["structured_data"]["thesis_archetype"] = "growth_stock"  # invalid
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("thesis_archetype" in e for e in errors)

    def test_assessment_summary_too_long_fails(self):
        out = _stock_researcher_output()
        out["assessment_summary"] = " ".join(["word"] * 85)  # 85 words
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("assessment_summary" in e for e in errors)

    def test_narrative_too_short_fails(self):
        out = _stock_researcher_output()
        out["narrative"] = "Short narrative."  # way under 720 chars
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("narrative" in e and "short" in e for e in errors)

    def test_narrative_too_long_fails(self):
        out = _stock_researcher_output()
        out["narrative"] = "A" * 1100  # over 1080 chars
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("narrative" in e and "long" in e for e in errors)

    def test_too_few_key_factors_fails(self):
        out = _stock_researcher_output()
        out["key_factors"] = out["key_factors"][:1]  # only 1 item
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("key_factors" in e for e in errors)

    def test_invalid_moat_durability_fails(self):
        out = _stock_researcher_output()
        out["structured_data"]["moat_assessment"]["overall_moat_durability"] = "excellent"
        passed, errors = validate_stock_researcher(out)
        assert not passed

    def test_invalid_moat_type_fails(self):
        """Declared in the real schema (9-value), unenforced until 86bbuhk82 -- surfaced by
        the permanent enum-inventory test, confirmed live (moats[0].type='cost_advantage' in
        a real call)."""
        out = _stock_researcher_output()
        out["structured_data"]["moat_assessment"]["moats"] = [
            {"type": "first_mover_advantage", "strength": "strong", "evidence": "N1: early market entry"}
        ]
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("moats[0].type" in e for e in errors)

    def test_valid_moat_type_passes(self):
        out = _stock_researcher_output()
        out["structured_data"]["moat_assessment"]["moats"] = [
            {"type": "cost_advantage", "strength": "strong", "evidence": "N1: lowest cost producer"}
        ]
        passed, errors = validate_stock_researcher(out)
        assert passed, errors

    def test_invalid_competitive_position_fails(self):
        out = _stock_researcher_output()
        out["structured_data"]["competitive_position"] = "leader"
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("competitive_position" in e for e in errors)

    def test_valid_competitive_position_passes(self):
        out = _stock_researcher_output()
        out["structured_data"]["competitive_position"] = "deteriorating"
        passed, errors = validate_stock_researcher(out)
        assert passed, errors

    def test_invalid_management_assessment_fails(self):
        out = _stock_researcher_output()
        out["structured_data"]["management_assessment"] = "great"
        passed, errors = validate_stock_researcher(out)
        assert not passed
        assert any("management_assessment" in e for e in errors)

    def test_valid_management_assessment_passes(self):
        out = _stock_researcher_output()
        out["structured_data"]["management_assessment"] = "insufficient_data"
        passed, errors = validate_stock_researcher(out)
        assert passed, errors


# ─── Fundamental Analyst Tests ────────────────────────────────────────────────

class TestFundamentalAnalyst:
    def test_valid_output_passes(self):
        out = _fundamental_analyst_output()
        passed, errors = validate_fundamental_analyst(out)
        assert passed, f"Valid output should pass: {errors}"

    def test_invalid_health_rating_fails(self):
        out = _fundamental_analyst_output()
        out["interpretive_fields"]["health_rating"] = "strong"  # invalid 5-tier value
        passed, errors = validate_fundamental_analyst(out)
        assert not passed
        assert any("health_rating" in e for e in errors)

    def test_insufficient_confidence_requires_empty_pass2_view(self):
        out = _fundamental_analyst_output()
        out["analysis_confidence"] = "insufficient"
        out["pass2_view"] = {}
        passed, errors = validate_fundamental_analyst(out)
        assert passed, f"Insufficient with empty pass2_view should pass: {errors}"

    def test_insufficient_confidence_with_nonempty_pass2_view_fails(self):
        out = _fundamental_analyst_output()
        out["analysis_confidence"] = "insufficient"
        out["pass2_view"] = {"health_rating": "healthy"}  # non-empty -- this should fail
        passed, errors = validate_fundamental_analyst(out)
        assert not passed
        assert any("pass2_view" in e and "empty" in e.lower() for e in errors)

    def test_low_confidence_is_not_exempted_from_content_checks(self):
        """Exact match on "insufficient" only, not loosened to include "low" (86bbummwp):
        live-verified a real "low"-confidence Fundamental call still produces fully compliant
        content on its own merits, so "low" should go through normal checks, not skip them."""
        out = _fundamental_analyst_output()
        out["analysis_confidence"] = "low"
        passed, errors = validate_fundamental_analyst(out)
        assert passed, f"A compliant 'low'-confidence output should pass normal checks: {errors}"

    def test_invalid_dividend_sustainability_fails(self):
        out = _fundamental_analyst_output()
        out["interpretive_fields"]["dividend_sustainability"] = "paying"  # invalid
        passed, errors = validate_fundamental_analyst(out)
        assert not passed


# ─── Technical Analyst Tests ─────────────────────────────────────────────────

class TestTechnicalAnalyst:
    def test_valid_output_passes(self):
        out = _technical_analyst_output()
        passed, errors = validate_technical_analyst(out)
        assert passed, f"Valid output should pass: {errors}"

    def test_invalid_primary_trend_fails(self):
        out = _technical_analyst_output()
        out["interpretive_fields"]["primary_trend"] = "ranging"  # invalid
        passed, errors = validate_technical_analyst(out)
        assert not passed
        assert any("primary_trend" in e for e in errors)

    def test_invalid_momentum_zone_fails(self):
        out = _technical_analyst_output()
        out["interpretive_fields"]["momentum_zone"] = "very_bearish"  # invalid
        passed, errors = validate_technical_analyst(out)
        assert not passed

    def test_earnings_proximity_caveat_missing(self):
        out = _technical_analyst_output()
        out["caveats"] = ["Some unrelated caveat."]
        out["narrative"] = out["narrative"].replace("earnings", "")
        passed, errors = validate_earnings_proximity_caveat(out, earnings_days=3)
        assert not passed

    def test_earnings_proximity_caveat_present(self):
        out = _technical_analyst_output()
        out["caveats"] = ["Earnings in ≤5 days: technical signals may be overridden by binary event outcome; analysis_confidence capped at medium."]
        out["analysis_confidence"] = "medium"
        passed, errors = validate_earnings_proximity_caveat(out, earnings_days=3)
        assert passed, f"Should pass with correct caveat: {errors}"

    def test_earnings_proximity_no_check_when_far(self):
        out = _technical_analyst_output()
        passed, errors = validate_earnings_proximity_caveat(out, earnings_days=45)
        assert passed  # no check needed when >5 days

    def test_thin_volume_caveat_missing(self):
        out = _technical_analyst_output()
        out["caveats"] = ["Some unrelated caveat."]
        out["narrative"] = "Technical analysis of price action and momentum indicators only."
        passed, errors = validate_thin_volume_caveat(out, avg_dollar_volume=773640)
        assert not passed

    def test_thin_volume_no_check_when_normal(self):
        out = _technical_analyst_output()
        passed, errors = validate_thin_volume_caveat(out, avg_dollar_volume=5_000_000)
        assert passed  # no check needed when >=1M


# ─── Sentiment Analyst Tests ─────────────────────────────────────────────────

class TestSentimentAnalyst:
    def test_valid_output_passes(self):
        out = _sentiment_analyst_output()
        passed, errors = validate_sentiment_analyst(out)
        assert passed, f"Valid output should pass: {errors}"

    def test_social_sentiment_not_unknown_fails(self):
        out = _sentiment_analyst_output()
        out["structured_data"]["social_sentiment"] = "positive"  # must be "unknown"
        passed, errors = validate_sentiment_analyst(out)
        assert not passed
        assert any("social_sentiment" in e for e in errors)

    def test_pass2_view_has_social_sentiment_fails(self):
        out = _sentiment_analyst_output()
        out["pass2_view"]["social_sentiment"] = "unknown"  # must NOT be in pass2_view
        passed, errors = validate_sentiment_analyst(out)
        assert not passed
        assert any("social_sentiment" in e for e in errors)

    def test_invalid_short_interest_interpretation_fails(self):
        out = _sentiment_analyst_output()
        out["structured_data"]["short_interest_interpretation"] = {"trend": "stable", "interpretation": "high"}  # invalid 3-value enum
        passed, errors = validate_sentiment_analyst(out)
        assert not passed

    def test_invalid_news_sentiment_overall_fails(self):
        """Declared in the real schema (5-value, distinct from key_factors[].sentiment's
        3-value vocabulary), unenforced until 86bbuhk82."""
        out = _sentiment_analyst_output()
        out["structured_data"]["news_sentiment"]["overall"] = "mixed"  # not one of the 5 values
        passed, errors = validate_sentiment_analyst(out)
        assert not passed
        assert any("news_sentiment.overall" in e for e in errors)

    def test_valid_news_sentiment_overall_passes(self):
        out = _sentiment_analyst_output()
        out["structured_data"]["news_sentiment"]["overall"] = "very_positive"
        passed, errors = validate_sentiment_analyst(out)
        assert passed, errors

    def test_too_many_contrarian_signals_fails(self):
        out = _sentiment_analyst_output()
        out["contrarian_signals"] = ["signal1", "signal2", "signal3"]  # max 2
        passed, errors = validate_sentiment_analyst(out)
        assert not passed

    def test_canadian_caveat_required(self):
        out = _sentiment_analyst_output()
        out["caveats"] = ["Some other caveat."]  # missing Finnhub caveat
        out["reliability_score"] = 65
        passed, errors = validate_canadian_caveat(out, "SENT")
        assert not passed
        assert any("Finnhub" in e for e in errors)


# ─── Macro Economist Tests ────────────────────────────────────────────────────

class TestMacroEconomist:
    def test_valid_output_passes(self):
        out = _macro_economist_output()
        passed, errors = validate_macro_economist(out)
        assert passed, f"Valid output should pass: {errors}"

    def test_narrative_too_long_fails(self):
        out = _macro_economist_output()
        out["narrative"] = "A " * 400  # 800 chars, over 720 max
        passed, errors = validate_macro_economist(out)
        assert not passed
        assert any("narrative" in e and "long" in e for e in errors)

    def test_invalid_macro_environment_fails(self):
        out = _macro_economist_output()
        out["structured_data"]["overall_macro_environment"] = "positive"  # invalid (should be favorable)
        passed, errors = validate_macro_economist(out)
        assert not passed
        assert any("overall_macro_environment" in e for e in errors)

    def test_invalid_sector_cycle_fails(self):
        out = _macro_economist_output()
        out["structured_data"]["sector_cycle_position"] = "peak"  # invalid
        passed, errors = validate_macro_economist(out)
        assert not passed

    def test_pass2_view_invalid_macro_env_fails(self):
        out = _macro_economist_output()
        out["pass2_view"]["overall_macro_environment"] = "good"  # invalid
        passed, errors = validate_macro_economist(out)
        assert not passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
