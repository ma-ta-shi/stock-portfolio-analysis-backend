"""Code-decided Technical fields: primary trend, level bias, momentum zone, divergence and direction, and the invalidation level.

The Technical Analyst used to be asked for these, and for `momentum_zone` and `momentum_divergence`, in its
interpretive fields. Measured on 51 real runs (2026-10-04): `momentum_zone` equalled the pre-classified RSI zone 51 of
51 times and `momentum_divergence` the payload's divergence 51 of 51, the invalidation level was the nearest support in
45 of 49 runs, and `momentum_direction` (unvalidated, because a model once wrote "stabilizing") came back as "bearish"
twice. The inputs decide all of them, so they are decided here and merged into the output after the model returns; the
model keeps the judgement calls (`trend_strength`, `volume_confirmation`, `confluence_score`). The level bias
(`nearest_level_bias`) joined the code-decided fields on 2026-10-05 (see `nearest_level_bias`). The primary trend followed the SMA stack in 40 of 40 non-mixed runs and was the one judgement the model made inconsistently, so it is the stack now (see `primary_trend`). `confluence_score` was considered and left with the model: a clear code rule (momentum, volume
and level proximity agreeing with the trend) matched the model's own count in only 22 of 51 runs and scored lower
(16 zeros against the model's 3), so the model's number blends judgement a rule would not reproduce; it never failed
validation and Pass 2 rarely quotes it (4 of 30 advocate outputs).

Pure functions, no network, same pattern as `sentiment_signals.py`.
"""

# A MACD histogram change over 3 sessions smaller than this many ATR is "flat". Chosen from the distribution over
# 8 tickers x 250 sessions (2,000 days): 0.05 ATR leaves about a quarter of days flat.
MOMENTUM_DEADBAND_ATR = 0.05


def momentum_direction(technical_indicators: dict) -> str | None:
    """"improving", "flat" or "deteriorating": the MACD histogram now against 3 sessions ago, in ATR. None when
    either histogram value or the ATR is missing."""
    now = technical_indicators.get("macd_histogram")
    before = technical_indicators.get("macd_histogram_3d_ago")
    atr = technical_indicators.get("atr_14")
    if now is None or before is None or not atr:
        return None
    change = (now - before) / atr
    if change > MOMENTUM_DEADBAND_ATR:
        return "improving"
    if change < -MOMENTUM_DEADBAND_ATR:
        return "deteriorating"
    return "flat"


def invalidation(primary_trend: str | None, support_resistance: dict) -> dict:
    """The level that would invalidate the trend: the nearest level on the stop side (the nearest support under an
    uptrend, the nearest resistance over a downtrend), with its distance in ATR. None for a mixed trend or when no
    level exists on that side (a stock at its 52-week high has no resistance)."""
    if primary_trend == "bullish":
        side = "support"
    elif primary_trend == "bearish":
        side = "resistance"
    else:
        return {"level": None, "atr_distance": None, "basis": None}
    return {
        "level": support_resistance.get(f"nearest_{side}"),
        "atr_distance": support_resistance.get(f"atr_to_{side}"),
        "basis": support_resistance.get(f"{side}_basis"),
    }


def primary_trend(technical_indicators: dict) -> str | None:
    """The primary trend is the SMA stack the payload already reports: "bullish" (20 > 50 > 200), "bearish" (20 < 50 <
    200) or "mixed". The model used to choose it: it followed the stack on every non-mixed stack (40 of 40 real runs)
    but on a mixed one it wrote "bullish" half the time, and the same KO payload came back "bullish" in 2 of 6
    replays and "mixed" in 4, contradicting the payload's own "Stack order: mixed".

    Under 200 sessions of history (a recent listing) there is no 200-day average and no stack. Making the trend
    code-decided removed the model's own judgement there, so Pass 2 would have received no trend at all for exactly the
    names with the least history; instead the same rule runs on the averages that exist: "bullish" when the 20-day is
    above the 50-day and price is above the 50-day, "bearish" when both are below, otherwise "mixed". None under 50
    sessions."""
    stack = technical_indicators.get("stack_order")
    if stack in ("bullish", "bearish", "mixed"):
        return stack
    sma_20, sma_50 = technical_indicators.get("sma_20"), technical_indicators.get("sma_50")
    above_50 = technical_indicators.get("price_vs_sma50_pct")
    if sma_20 is None or sma_50 is None or above_50 is None:
        return None
    if sma_20 > sma_50 and above_50 > 0:
        return "bullish"
    if sma_20 < sma_50 and above_50 < 0:
        return "bearish"
    return "mixed"


NEAR_LEVEL_ATR = 1.0  # a level this close (in ATR) is "near"


def nearest_level_bias(support_resistance: dict) -> str:
    """"near_support" or "near_resistance" when that level is within 1 ATR of price and the other side is farther (or
    absent), otherwise "midrange". The prompt used to ask the model for this with exactly this definition and it still
    departed from it in 2 of 36 replays (MSFT, 0.41 ATR to support against 3.12 to resistance, came back "midrange");
    before the definition existed it contradicted the closer level in 16 of 51 runs."""
    support, resistance = support_resistance.get("atr_to_support"), support_resistance.get("atr_to_resistance")
    if support is not None and support <= NEAR_LEVEL_ATR and (resistance is None or resistance > support):
        return "near_support"
    if resistance is not None and resistance <= NEAR_LEVEL_ATR and (support is None or support > resistance):
        return "near_resistance"
    return "midrange"


def decided_fields(interpretive_fields: dict, technical_indicators: dict, support_resistance: dict) -> dict:
    """The code-decided `interpretive_fields` for one run, from the bundle (the primary trend first: the invalidation
    level depends on it)."""
    trend = primary_trend(technical_indicators)
    level = invalidation(trend, support_resistance)
    return {
        "primary_trend": trend,
        "nearest_level_bias": nearest_level_bias(support_resistance),
        "momentum_zone": technical_indicators.get("rsi_zone_adjusted"),
        "momentum_divergence": _divergence_label(technical_indicators.get("divergence")),
        "momentum_direction": momentum_direction(technical_indicators),
        "suggested_invalidation_level": level["level"],
        "invalidation_atr_distance": level["atr_distance"],
    }


def _divergence_label(value: str | None) -> str:
    """The payload says "bearish_divergence" / "bullish_divergence" / "none"; the output vocabulary is
    bullish / bearish / none."""
    if value in ("bullish_divergence", "bullish"):
        return "bullish"
    if value in ("bearish_divergence", "bearish"):
        return "bearish"
    return "none"
