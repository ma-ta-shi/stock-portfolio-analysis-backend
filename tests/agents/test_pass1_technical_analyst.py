"""Tests for agents/pass1_technical_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_technical_analyst import (
    _data_coverage_line,
    _data_warnings_line,
    _validate_with_caveats,
    build_user_message,
)


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        price_info={"current_price": 105.0, "currency": "CAD"},
        risk_metrics={"beta": 1.8},
        technical_indicators={
            "stack_order": "bullish", "sma_50_slope": "rising", "sma_200_slope": "rising",
            "price_vs_sma20_pct": 2.1, "price_vs_sma50_pct": 5.4, "price_vs_sma200_pct": 12.0,
            "rsi_14": 62.5, "rsi_zone_adjusted": "neutral",
            "macd_line": 1.2, "macd_signal": 0.9, "macd_histogram": 0.3,
            "macd_recent_cross": "bullish_cross", "divergence": "none",
            "volume_avg_20": 1_200_000, "volume_ratio_today": 1.1, "volume_today": 1_320_000,
            "bb_position": "mean_reverting", "bb_width_pct": 4.2,
            "atr_14": 2.3, "atr_60_avg": 2.1, "volatility_regime_derived": "normal",
        },
        support_resistance={
            "nearest_support": 98.0, "pct_to_support": -6.7, "atr_to_support": 3.0,
            "support_touch_count": 4,
            "nearest_resistance": 112.0, "pct_to_resistance": 6.7, "atr_to_resistance": 3.0,
            "resistance_touch_count": 2,
        },
        trend_structure={
            "swing_structure_20d": "HH", "trend_structure_weekly": "HH",
            "rs_vs_sector_3mo": 8.5, "rs_leadership": "leading",
        },
        multi_timeframe={"weekly_trend": "bullish", "weekly_rsi_zone": "neutral"},
        pattern_metrics={"squeeze_release": False, "breakout_direction": None},
        market_context={"market_regime": "bull"},
        liquidity_flags={"avg_dollar_volume_20": 5_000_000},
        earnings_proximity={"next_earnings_date": "2026-11-01", "earnings_proximity_days": 30},
        price_position={"high_52w": 120.0, "low_52w": 60.0,
                         "pct_from_52w_high": -12.5, "pct_from_52w_low": 75.0},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_renders_real_header_fields():
    msg, _ = build_user_message(_bundle())
    assert "SHOP.TO (Shopify Inc) | Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg


def test_renders_previously_stale_fields_that_harness_hardcoded_to_na():
    """rs_leadership/rsi_zone_adjusted/weekly_rsi_zone are real fields today
    -- the harness's own runner marked these 'not present in this fixture's
    schema' and hardcoded them to 'N/A'; this port must use the real value."""
    msg, _ = build_user_message(_bundle())
    assert "Leadership: leading" in msg
    assert "RSI zone (adjusted): neutral" in msg
    assert "Weekly RSI zone: neutral" in msg


def test_renders_beta_from_risk_metrics():
    msg, _ = build_user_message(_bundle())
    assert "Beta: 1.8" in msg


def test_earnings_proximity_days_from_earnings_proximity_field():
    """Real field lives on bundle.earnings_proximity, not fundamental_data
    (where the harness fixture put it) -- cross-agent field-location
    difference, not a data gap."""
    msg, _ = build_user_message(_bundle())
    assert "EARNINGS PROXIMITY: 30 days" in msg


def test_earnings_flag_fires_within_5_days():
    bundle = _bundle(earnings_proximity={"next_earnings_date": "2026-09-26", "earnings_proximity_days": 3})
    msg, _ = build_user_message(bundle)
    assert "⚠️ EARNINGS IN 3 DAY(S)" in msg


def test_earnings_flag_absent_when_no_upcoming_earnings_known():
    bundle = _bundle(earnings_proximity={"next_earnings_date": None, "earnings_proximity_days": None})
    msg, _ = build_user_message(bundle)
    assert "⚠️ EARNINGS IN" not in msg
    assert "EARNINGS PROXIMITY: N/A" in msg


def test_thin_volume_flag_fires_below_1m():
    bundle = _bundle(liquidity_flags={"avg_dollar_volume_20": 450_000})
    msg, _ = build_user_message(bundle)
    assert "⚠️ THIN VOLUME: avg_dollar_volume_20=$450,000" in msg


def test_thin_volume_flag_absent_above_1m():
    msg, _ = build_user_message(_bundle())  # default 5,000,000
    assert "⚠️ THIN VOLUME" not in msg


def test_missing_avg_dollar_volume_renders_honestly_not_as_zero():
    bundle = _bundle(liquidity_flags={"avg_dollar_volume_20": None})
    msg, _ = build_user_message(bundle)
    assert "⚠️ THIN VOLUME" not in msg
    assert "Avg dollar volume (20d): N/A" in msg


def test_renders_52w_position_fields():
    msg, _ = build_user_message(_bundle())
    assert "52w High: 120.0 | 52w Low: 60.0" in msg
    assert "% from 52w high: -12.5%" in msg


def test_no_confluence_score_or_pattern_confirmed_fabricated():
    """Neither field has a real precompute source -- must never appear."""
    msg, _ = build_user_message(_bundle())
    assert "confluence_score" not in msg
    assert "pattern_confirmed" not in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_all_true_when_default_bundle_is_fully_populated():
    _, presence = build_user_message(_bundle())
    assert presence == {
        "earnings_proximity": True,
        "weekly_timeframe": True,
        "sector_relative_strength": True,
        "support_resistance": True,
    }


def test_field_presence_earnings_proximity_false_without_a_calendar():
    bundle = _bundle(earnings_proximity={"next_earnings_date": None, "earnings_proximity_days": None})
    _, presence = build_user_message(bundle)
    assert presence["earnings_proximity"] is False


def test_field_presence_weekly_timeframe_false_with_insufficient_weekly_history():
    bundle = _bundle(multi_timeframe={"weekly_trend": None, "weekly_rsi_zone": None})
    _, presence = build_user_message(bundle)
    assert presence["weekly_timeframe"] is False


def test_field_presence_sector_relative_strength_false_without_sector_etf():
    bundle = _bundle(trend_structure={
        "swing_structure_20d": "HH", "trend_structure_weekly": "HH",
        "rs_vs_sector_3mo": None, "rs_leadership": None,
    })
    _, presence = build_user_message(bundle)
    assert presence["sector_relative_strength"] is False


def test_field_presence_support_resistance_false_when_pivots_unresolved():
    bundle = _bundle(support_resistance={})
    _, presence = build_user_message(bundle)
    assert presence["support_resistance"] is False


# ---------- _data_coverage_line / _data_warnings_line (86bbummwp Tier 1a/1b) ----------


def test_data_coverage_line_standard_with_full_field_presence():
    _, presence = build_user_message(_bundle())
    assert _data_coverage_line(presence) == "standard."


def test_data_coverage_line_flags_multiple_real_gaps():
    bundle = _bundle(
        earnings_proximity={"next_earnings_date": None, "earnings_proximity_days": None},
        support_resistance={},
    )
    _, presence = build_user_message(bundle)
    line = _data_coverage_line(presence)
    assert "no earnings calendar data available" in line
    assert "no support/resistance levels could be resolved" in line
    assert "weekly timeframe" not in line  # that one's still present in this fixture


def test_data_warnings_line_empty_when_no_preflight_warnings():
    assert _data_warnings_line([]) == ""


def test_data_warnings_line_joins_real_anomalies():
    warnings = ["zero-volume session on 2026-09-10", "unexplained 28% gap on 2026-09-15"]
    assert _data_warnings_line(warnings) == (
        "zero-volume session on 2026-09-10; unexplained 28% gap on 2026-09-15"
    )


# ---------- _validate_with_caveats (86bbummwp Tier 1c) ----------


def _valid_technical_output(**overrides) -> dict:
    base = {
        "assessment_summary": "A solid technical setup with bullish momentum.",
        "analysis_confidence": "high",
        "caveats": ["Coverage limited to daily price action."],
        "key_factors": [
            {"factor": "Trend", "importance": "high", "sentiment": "positive", "evidence": "SMA stack bullish"},
            {"factor": "Momentum", "importance": "medium", "sentiment": "positive", "evidence": "RSI 62"},
        ],
        "risks": [
            {"risk": "Approaching resistance", "severity": "medium", "evidence": "ATR 3.0 to resistance"},
        ],
        "narrative": (
            "SHOP.TO shows a clean bullish setup with price trading above all major moving "
            "averages and a rising trend structure across both daily and weekly timeframes. "
            "The RSI at 62 suggests room to run before overbought conditions are reached, while "
            "the MACD line remains above its signal line following a recent bullish crossover, "
            "confirming the underlying momentum picture. Support sits at 98.00 with four prior "
            "touches, while resistance at 112.00 has been tested twice without a clean break to "
            "the upside so far this quarter. Bollinger Band width suggests a normal volatility "
            "regime rather than a compression squeeze setting up. Relative strength versus the "
            "sector remains a modest positive tailwind for the name. Overall the weight of "
            "evidence favors continuation of the current uptrend over the medium term horizon, "
            "with resistance as the key level to watch for a potential stall in price action."
        ),
        "interpretive_fields": {
            "primary_trend": "bullish",
            "trend_strength": "moderate",
            "momentum_zone": "neutral",
        },
    }
    base.update(overrides)
    return base


def test_validate_with_caveats_passes_when_no_condition_and_schema_valid():
    passed, errors = _validate_with_caveats(_valid_technical_output(), earnings_days=45, avg_dollar_vol=5_000_000)
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_caveats():
    out = _valid_technical_output(interpretive_fields={"primary_trend": "ranging", "trend_strength": "moderate", "momentum_zone": "neutral"})
    passed, errors = _validate_with_caveats(out, earnings_days=45, avg_dollar_vol=5_000_000)
    assert not passed
    assert any("primary_trend" in e for e in errors)


def test_validate_with_caveats_flags_missing_earnings_caveat():
    out = _valid_technical_output()  # no earnings mention
    passed, errors = _validate_with_caveats(out, earnings_days=3, avg_dollar_vol=5_000_000)
    assert not passed
    assert any("earnings proximity caveat" in e for e in errors)


def test_validate_with_caveats_flags_missing_thin_volume_caveat():
    out = _valid_technical_output()  # no volume/liquidity mention
    passed, errors = _validate_with_caveats(out, earnings_days=45, avg_dollar_vol=500_000)
    assert not passed
    assert any("thin volume" in e for e in errors)


def test_validate_with_caveats_merges_errors_from_both_checks():
    out = _valid_technical_output()
    passed, errors = _validate_with_caveats(out, earnings_days=3, avg_dollar_vol=500_000)
    assert not passed
    assert any("earnings proximity caveat" in e for e in errors)
    assert any("thin volume" in e for e in errors)


def test_validate_with_caveats_earnings_none_skips_that_check():
    """No earnings calendar data at all -- nothing to flag, not a crash."""
    out = _valid_technical_output()
    passed, errors = _validate_with_caveats(out, earnings_days=None, avg_dollar_vol=5_000_000)
    assert passed, errors


def test_validate_with_caveats_avg_dollar_vol_none_skips_that_check():
    """No liquidity data at all -- nothing to flag, not a crash."""
    out = _valid_technical_output()
    passed, errors = _validate_with_caveats(out, earnings_days=45, avg_dollar_vol=None)
    assert passed, errors
