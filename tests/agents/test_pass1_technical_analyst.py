"""Tests for agents/pass1_technical_analyst.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass1_technical_analyst import build_user_message


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
    msg = build_user_message(_bundle())
    assert "SHOP.TO (Shopify Inc) | Technology | TSX | CAD" in msg
    assert "Timeline: medium_term | Account: tfsa" in msg


def test_renders_previously_stale_fields_that_harness_hardcoded_to_na():
    """rs_leadership/rsi_zone_adjusted/weekly_rsi_zone are real fields today
    -- the harness's own runner marked these 'not present in this fixture's
    schema' and hardcoded them to 'N/A'; this port must use the real value."""
    msg = build_user_message(_bundle())
    assert "Leadership: leading" in msg
    assert "RSI zone (adjusted): neutral" in msg
    assert "Weekly RSI zone: neutral" in msg


def test_renders_beta_from_risk_metrics():
    msg = build_user_message(_bundle())
    assert "Beta: 1.8" in msg


def test_earnings_proximity_days_from_earnings_proximity_field():
    """Real field lives on bundle.earnings_proximity, not fundamental_data
    (where the harness fixture put it) -- cross-agent field-location
    difference, not a data gap."""
    msg = build_user_message(_bundle())
    assert "EARNINGS PROXIMITY: 30 days" in msg


def test_earnings_flag_fires_within_5_days():
    bundle = _bundle(earnings_proximity={"next_earnings_date": "2026-09-26", "earnings_proximity_days": 3})
    msg = build_user_message(bundle)
    assert "⚠️ EARNINGS IN 3 DAY(S)" in msg


def test_earnings_flag_absent_when_no_upcoming_earnings_known():
    bundle = _bundle(earnings_proximity={"next_earnings_date": None, "earnings_proximity_days": None})
    msg = build_user_message(bundle)
    assert "⚠️ EARNINGS IN" not in msg
    assert "EARNINGS PROXIMITY: N/A" in msg


def test_thin_volume_flag_fires_below_1m():
    bundle = _bundle(liquidity_flags={"avg_dollar_volume_20": 450_000})
    msg = build_user_message(bundle)
    assert "⚠️ THIN VOLUME: avg_dollar_volume_20=$450,000" in msg


def test_thin_volume_flag_absent_above_1m():
    msg = build_user_message(_bundle())  # default 5,000,000
    assert "⚠️ THIN VOLUME" not in msg


def test_missing_avg_dollar_volume_renders_honestly_not_as_zero():
    bundle = _bundle(liquidity_flags={"avg_dollar_volume_20": None})
    msg = build_user_message(bundle)
    assert "⚠️ THIN VOLUME" not in msg
    assert "Avg dollar volume (20d): N/A" in msg


def test_renders_52w_position_fields():
    msg = build_user_message(_bundle())
    assert "52w High: 120.0 | 52w Low: 60.0" in msg
    assert "% from 52w high: -12.5%" in msg


def test_no_confluence_score_or_pattern_confirmed_fabricated():
    """Neither field has a real precompute source -- must never appear."""
    msg = build_user_message(_bundle())
    assert "confluence_score" not in msg
    assert "pattern_confirmed" not in msg
