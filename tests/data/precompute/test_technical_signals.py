"""Code-decided Technical fields (data/precompute/technical_signals.py)."""

from data.precompute.technical_signals import (
    decided_fields,
    invalidation,
    momentum_direction,
    nearest_level_bias,
    primary_trend,
)


def _ti(hist=0.36, before=0.10, atr=11.6, **extra):
    return {"macd_histogram": hist, "macd_histogram_3d_ago": before, "atr_14": atr, **extra}


def test_momentum_direction_compares_the_histogram_with_three_sessions_ago_in_atr():
    assert momentum_direction(_ti(hist=1.0, before=0.1)) == "improving"  # +0.9 / 11.6 = 0.078 ATR
    assert momentum_direction(_ti(hist=0.1, before=1.0)) == "deteriorating"
    assert momentum_direction(_ti(hist=0.5, before=0.4)) == "flat"  # 0.1 / 11.6 = 0.009 ATR, inside the deadband


def test_momentum_direction_is_none_without_the_inputs():
    for ti in ({}, _ti(before=None), _ti(hist=None), _ti(atr=None), _ti(atr=0)):
        assert momentum_direction(ti) is None


def test_invalidation_is_the_stop_side_level_for_the_trend():
    sr = {"nearest_support": 512.75, "atr_to_support": 0.41, "support_basis": "swing",
          "nearest_resistance": 553.72, "atr_to_resistance": 3.12, "resistance_basis": "52w high + swing"}
    assert invalidation("bullish", sr) == {"level": 512.75, "atr_distance": 0.41, "basis": "swing"}
    assert invalidation("bearish", sr) == {"level": 553.72, "atr_distance": 3.12, "basis": "52w high + swing"}


def test_a_mixed_or_unknown_trend_has_no_invalidation_level_and_never_zero():
    sr = {"nearest_support": 1.0, "atr_to_support": 1.0, "support_basis": "swing"}
    for trend in ("mixed", None, "sideways"):
        assert invalidation(trend, sr) == {"level": None, "atr_distance": None, "basis": None}


def test_a_trend_with_no_level_on_its_stop_side_has_none_not_zero():
    """A downtrend's stop side is resistance; if only a support level exists, the level is None, not support and not 0."""
    assert invalidation("bearish", {"nearest_support": 9.0, "atr_to_support": 1.0})["level"] is None


def test_decided_fields_merge_the_copies_and_the_computed_values():
    ti = _ti(hist=1.0, before=0.1, rsi_zone_adjusted="neutral", divergence="bearish_divergence", stack_order="bullish")
    sr = {"nearest_support": 512.75, "atr_to_support": 0.41, "support_basis": "swing", "atr_to_resistance": 3.12}
    out = decided_fields({}, ti, sr)
    assert out == {
        "primary_trend": "bullish",
        "nearest_level_bias": "near_support",
        "momentum_zone": "neutral",
        "momentum_divergence": "bearish",
        "momentum_direction": "improving",
        "suggested_invalidation_level": 512.75,
        "invalidation_atr_distance": 0.41,
    }
    assert decided_fields({}, {"divergence": "none", "stack_order": "mixed"}, {})["momentum_divergence"] == "none"
    assert decided_fields({}, {"divergence": "bullish_divergence"}, {})["momentum_divergence"] == "bullish"


def test_the_primary_trend_is_the_sma_stack_not_a_model_judgement():
    """KO 2026-10-04: a payload that said "Stack order: mixed" came back "bullish" in 2 of 6 replays."""
    for stack in ("bullish", "bearish", "mixed"):
        assert primary_trend({"stack_order": stack}) == stack
    assert primary_trend({"stack_order": None}) is None and primary_trend({}) is None


def test_a_mixed_stack_has_no_invalidation_level_through_the_merge():
    sr = {"nearest_support": 85.2, "atr_to_support": 0.35, "support_basis": "swing"}
    out = decided_fields({}, {"stack_order": "mixed", "divergence": "none"}, sr)
    assert out["primary_trend"] == "mixed" and out["suggested_invalidation_level"] is None


def test_nearest_level_bias_follows_the_stated_definition():
    """MSFT 2026-10-04: 0.41 ATR to support against 3.12 to resistance came back "midrange" from the model."""
    assert nearest_level_bias({"atr_to_support": 0.41, "atr_to_resistance": 3.12}) == "near_support"
    assert nearest_level_bias({"atr_to_support": 1.3, "atr_to_resistance": 0.06}) == "near_resistance"
    assert nearest_level_bias({"atr_to_support": 1.3, "atr_to_resistance": 2.0}) == "midrange"  # neither within 1 ATR
    assert nearest_level_bias({"atr_to_support": 0.9, "atr_to_resistance": 0.4}) == "near_resistance"  # both near: the closer one
    assert nearest_level_bias({"atr_to_support": 1.0, "atr_to_resistance": 1.5}) == "near_support"  # exactly 1 ATR is near


def test_a_missing_side_is_never_nearer_than_a_real_one():
    """A stock at its 52-week high has no resistance: support within 1 ATR is near_support, otherwise midrange."""
    assert nearest_level_bias({"atr_to_support": 0.5, "atr_to_resistance": None}) == "near_support"
    assert nearest_level_bias({"atr_to_support": 2.0, "atr_to_resistance": None}) == "midrange"
    assert nearest_level_bias({"atr_to_support": None, "atr_to_resistance": 0.3}) == "near_resistance"
    assert nearest_level_bias({}) == "midrange"


def test_a_recent_listing_gets_a_trend_from_the_averages_it_has():
    """Under 200 sessions there is no stack. Pass 2 would have got no trend at all for the names with the least history."""
    base = {"stack_order": None, "sma_20": 168.8, "sma_50": 168.2}
    assert primary_trend({**base, "price_vs_sma50_pct": 1.0}) == "bullish"
    assert primary_trend({**base, "price_vs_sma50_pct": -0.6}) == "mixed"  # 20 above 50 but price below 50
    assert primary_trend({"stack_order": None, "sma_20": 90.0, "sma_50": 100.0, "price_vs_sma50_pct": -10.0}) == "bearish"
    assert primary_trend({"stack_order": None, "sma_20": 110.0, "sma_50": 100.0, "price_vs_sma50_pct": -2.0}) == "mixed"
    assert primary_trend({"stack_order": None, "sma_20": 100.0, "sma_50": None, "price_vs_sma50_pct": None}) is None
    assert primary_trend({"stack_order": "bearish", "sma_20": 1.0, "sma_50": 2.0, "price_vs_sma50_pct": 5.0}) == "bearish"  # a real stack wins
