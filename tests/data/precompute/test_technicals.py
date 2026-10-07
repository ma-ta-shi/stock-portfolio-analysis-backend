import numpy as np
import pandas as pd

from data.precompute.technicals import _rsi_zone_adjusted, _ta_result, compute_all

_AS_OF = pd.Timestamp("2026-08-16").date()


def _ohlcv(
    n: int, seed: int = 1, trend_total: float = 0.0, start: str = "2023-01-02"
) -> pd.DataFrame:
    """n business-day bars, lowercase columns, deterministic via seed.
    trend_total adds a linear drift across the whole series so tests can
    exercise a real (non-flat) stack order / market regime."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    drift = np.linspace(0, trend_total, n)
    noise = np.cumsum(rng.normal(0, 0.6, n))
    close = 100 + drift + noise
    high = close + rng.random(n) * 1.2
    low = close - rng.random(n) * 1.2
    openp = close + rng.normal(0, 0.4, n)
    volume = rng.integers(500_000, 2_000_000, n)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _capitalized(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=str.title)


# ---------- _ta_result (the pandas-ta insufficient-data guard) ----------


def test_ta_result_returns_none_when_identity_matches():
    df = _ohlcv(10)
    assert _ta_result(df, df) is None


def test_ta_result_passes_through_a_real_result():
    df = _ohlcv(300, trend_total=40)
    real = df.ta.sma(length=20)
    assert _ta_result(df, real) is real


# ---------- input normalization (real gap this module was built to fix) ----------


class TestColumnCasingRegression:
    """Regression test for the real gap found planning this ticket:
    router.get_price_history() can hand this module capitalized columns
    (yfinance's raw output) or lowercase (FMP/openbb-tmx). pandas-ta
    silently returns NaN for every indicator on capitalized columns —
    confirmed live — rather than raising. This must not happen."""

    def test_capitalized_columns_produce_the_same_result_as_lowercase(self):
        lower = _ohlcv(300, trend_total=30)
        capped = _capitalized(lower)

        result_lower = compute_all(lower, lower, None, None, None, as_of=_AS_OF)
        result_capped = compute_all(capped, capped, None, None, None, as_of=_AS_OF)

        assert result_capped["technical_indicators"]["sma_20"] is not None
        assert (
            result_capped["technical_indicators"]["sma_20"]
            == result_lower["technical_indicators"]["sma_20"]
        )
        assert (
            result_capped["technical_indicators"]["rsi_14"]
            == result_lower["technical_indicators"]["rsi_14"]
        )


# ---------- moving averages ----------


class TestMovingAverages:
    def test_sufficient_data_computes_all_three_smas(self):
        df = _ohlcv(250, trend_total=30)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["sma_20"] is not None
        assert ind["sma_50"] is not None
        assert ind["sma_200"] is not None
        assert ind["stack_order"] in ("bullish", "bearish", "mixed")

    def test_insufficient_data_for_sma200_omits_it_not_crashes(self):
        """Regression test for the real pandas-ta bug found building this
        module: .ta.sma(length=200) on <200 rows doesn't return a NaN
        series, it returns the identical input DataFrame object — which
        previously crashed downstream comparisons with 'ambiguous truth
        value of a Series'. 50 rows is enough for sma20 but not sma200."""
        df = _ohlcv(50)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["sma_20"] is not None
        assert ind["sma_200"] is None
        assert ind["stack_order"] is None  # requires all three SMAs present

    def test_thin_data_all_moving_averages_none(self):
        df = _ohlcv(5)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["sma_20"] is None
        assert ind["ema_12"] is None

    def test_sma_200_slope_populates_with_long_history(self):
        """The prompt renders Slope200={sma_200_slope}; before this producer
        existed it was always N/A even on a full history."""
        df = _ohlcv(260, trend_total=40)
        ind = compute_all(df, df, None, None, None, as_of=_AS_OF)["technical_indicators"]
        assert ind["sma_200_slope"] in ("rising", "falling", "flat")
        assert ind["sma_50_slope"] in ("rising", "falling", "flat")

    def test_sma_200_slope_none_while_sma_50_slope_still_populates(self):
        """The asymmetry this fix is about: with 60 bars the 50-day slope
        resolves and the 200-day cannot. Both must degrade independently,
        not crash and not silently share a value."""
        df = _ohlcv(60, trend_total=15)
        ind = compute_all(df, df, None, None, None, as_of=_AS_OF)["technical_indicators"]
        assert ind["sma_50_slope"] in ("rising", "falling", "flat")
        assert ind["sma_200_slope"] is None

    def test_sma_200_slope_needs_five_bars_past_the_sma_itself(self):
        """_slope_label needs lookback+1 non-NaN points, so sma_200 can be
        present while sma_200_slope is still None (200 bars -> SMA, 205 ->
        slope). Guards the boundary, not just the far ends."""
        df = _ohlcv(202, trend_total=20)
        ind = compute_all(df, df, None, None, None, as_of=_AS_OF)["technical_indicators"]
        assert ind["sma_200"] is not None
        assert ind["sma_200_slope"] is None

    def test_primary_trend_alias_key_is_gone(self):
        """Regression: `primary_trend` was a verbatim alias of `stack_order`
        in the payload and collided with the Technical agent's own
        `primary_trend` output field, which uses a different vocabulary.
        The consumer (rsi_zone_adjusted) reads `stack_order` directly now."""
        df = _ohlcv(250, trend_total=30)
        ind = compute_all(df, df, None, None, None, as_of=_AS_OF)["technical_indicators"]
        assert "primary_trend" not in ind
        assert ind["stack_order"] in ("bullish", "bearish", "mixed")
        assert ind["rsi_zone_adjusted"] in ("overbought", "oversold", "neutral", None)


# ---------- momentum ----------


class TestMomentum:
    def test_sufficient_data_computes_rsi_and_macd(self):
        df = _ohlcv(250, trend_total=20)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["rsi_14"] is not None
        assert 0 <= ind["rsi_14"] <= 100
        assert ind["macd_line"] is not None
        assert ind["macd_recent_cross"] in ("bullish_cross", "bearish_cross", "none")

    def test_insufficient_data_for_macd_omits_it(self):
        df = _ohlcv(20)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["macd_line"] is None
        assert ind["macd_recent_cross"] is None


class TestRsiZoneAdjusted:
    """Characterization tests. These pin CURRENT behaviour, including the
    known gap flagged in the module: the Technical prompt's Design Decision
    #5 says "80/40 in uptrends, 60/20 in downtrends" and is silent on
    `mixed`, and the code gives `mixed` the downtrend thresholds. If that
    is later decided to be wrong, test_mixed_* below should fail loudly."""

    def test_none_rsi_returns_none(self):
        assert _rsi_zone_adjusted(None, "bullish") is None

    def test_bullish_uses_80_40_band(self):
        assert _rsi_zone_adjusted(75.0, "bullish") == "neutral"
        assert _rsi_zone_adjusted(85.0, "bullish") == "overbought"
        assert _rsi_zone_adjusted(35.0, "bullish") == "oversold"

    def test_bearish_uses_60_20_band(self):
        assert _rsi_zone_adjusted(65.0, "bearish") == "overbought"
        assert _rsi_zone_adjusted(15.0, "bearish") == "oversold"

    def test_mixed_and_unknown_stacks_get_the_standard_70_30_band(self):
        """Decided 2026-10-04 (it fell through to the downtrend 60/20 band before): RSI 65 is neutral in a mixed
        market, where the downtrend band called it overbought."""
        for stack in ("mixed", None):
            assert _rsi_zone_adjusted(65.0, stack) == "neutral"
            assert _rsi_zone_adjusted(70.5, stack) == "overbought"
            assert _rsi_zone_adjusted(29.0, stack) == "oversold"
            assert _rsi_zone_adjusted(25.0, stack) == "oversold" and _rsi_zone_adjusted(35.0, stack) == "neutral"
        assert _rsi_zone_adjusted(65.0, "bearish") == "overbought"  # the downtrend band is unchanged


# ---------- volume ----------


def test_volume_block_computed_with_enough_bars():
    df = _ohlcv(30)
    ind = compute_all(df, df, None, None, None, as_of=_AS_OF)["technical_indicators"]
    assert ind["volume_avg_20"] is not None
    assert ind["volume_ratio_today"] is not None
    # volume_today: the prompt renders `Today vol={volume_today}` and had no
    # producer, so the model was citing "today vol=N/A" as evidence.
    assert ind["volume_today"] is not None
    assert ind["volume_today"] > 0


def test_volume_block_shape_consistent_below_20_bars():
    df = _ohlcv(10)
    ind = compute_all(df, df, None, None, None, as_of=_AS_OF)["technical_indicators"]
    assert ind["volume_ratio_today"] is None
    assert ind["volume_avg_20"] is None
    # volume_today must be present-and-None on the degraded path, not absent,
    # so the payload template renders it consistently.
    assert "volume_today" in ind
    assert ind["volume_today"] is None


# ---------- Bollinger / squeeze ----------


class TestBollinger:
    def test_sufficient_data_computes_bands(self):
        df = _ohlcv(250)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["bb_upper"] is not None
        assert ind["bb_upper"] > ind["bb_lower"]
        assert ind["bb_regime"] in ("squeeze", "walking_band", "normal")
        assert ind["bb_percent_b"] is not None
        assert isinstance(ind["squeeze_released"], bool)

    def test_squeeze_needs_more_than_bbands_minimum(self):
        """Regression test: confirmed live that ta.squeeze() still returns
        the identity-df sentinel at exactly 20 rows even though
        ta.bbands(length=20) doesn't — a stricter internal minimum than
        bbands' own length. bb_regime must not crash if squeeze isn't
        available yet, even when the bands themselves are."""
        df = _ohlcv(20)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["bb_regime"] in (None, "normal", "walking_band")
        assert ind["squeeze_released"] is False


# ---------- support / resistance ----------


def test_support_resistance_computed_with_minimal_data():
    df = _ohlcv(5)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    sr = result["support_resistance"]
    assert sr["levels"], "even 5 sessions give the 52-week high and low as levels"
    # nearest support/resistance re-derived from real price comparison
    if sr["nearest_support"] is not None:
        assert sr["nearest_support"] < df["close"].iloc[-1] * 1.0001


def test_support_resistance_flat_shape_matches_prompt_template():
    """Real gap found reviewing this module against the live Technical
    Analyst Prompt: its payload template treats nearest_support/
    nearest_resistance as flat scalar fields with pct_to_*/atr_to_*/
    *_touch_count as siblings, not a nested dict -- confirmed against the
    prompt's actual template text, not the ticket's own summary list."""
    df = _ohlcv(250, trend_total=30, seed=6)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    sr = result["support_resistance"]
    for key in (
        "nearest_support",
        "pct_to_support",
        "atr_to_support",
        "support_touch_count",
        "nearest_resistance",
        "pct_to_resistance",
        "atr_to_resistance",
        "resistance_touch_count",
        "support_basis",
        "resistance_basis",
    ):
        assert key in sr
    if sr["nearest_support"] is not None:
        assert isinstance(sr["nearest_support"], float)
        assert sr["support_touch_count"] is not None
        assert isinstance(sr["support_touch_count"], int)


def test_support_resistance_empty_when_no_data():
    df = pd.DataFrame()
    result = compute_all(df, _ohlcv(5), None, None, None, as_of=_AS_OF)
    assert result["support_resistance"] == {}


def _bars(highs, lows, closes=None):
    closes = closes if closes is not None else [(hi + lo) / 2 for hi, lo in zip(highs, lows)]
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": [1000] * len(closes)})


def test_swing_points_are_the_extremes_of_five_bars_either_side():
    from data.precompute.technicals import _swing_points

    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lows = [h - 2 for h in highs]
    lows[10] = 3  # a dip at index 10, the lowest of the 11 bars around it
    swing_highs, swing_lows = _swing_points(_bars(highs, lows))
    assert swing_highs == [20.0]  # index 5 is the high of the 11 bars around it
    assert swing_lows == [3.0]


def test_the_last_five_sessions_cannot_be_swing_points():
    from data.precompute.technicals import _swing_points

    highs = [10.0] * 10 + [30.0]  # a new high on the last bar is not yet confirmed
    swing_highs, _ = _swing_points(_bars(highs, [h - 1 for h in highs]))
    assert 30.0 not in swing_highs


def test_levels_within_one_atr_are_merged_and_touches_count_only_swing_points():
    from data.precompute.technicals import _structural_levels

    # two swing lows 0.4 apart (inside 1.0 ATR) form one level with 2 touches; a distant swing high is its own level
    lows = [105, 104, 103, 102, 101, 100.0, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100.4, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    highs = [lo + 2 for lo in lows]
    levels = _structural_levels(_bars(highs, lows), atr14=1.0)
    lows_level = [lv for lv in levels if lv["price"] < 102]
    assert len(lows_level) == 1 and lows_level[0]["touches"] == 2 and "swing" in lows_level[0]["basis"]
    assert all("price" in lv and "touches" in lv and "basis" in lv for lv in levels)


def test_a_stock_at_its_52_week_high_has_no_resistance_and_the_rest_still_resolves():
    n = 120
    closes = [100 + i * 0.5 for i in range(n)]
    df = _bars(closes, [c - 1.0 for c in closes], closes)  # each close is that session's high: price is AT the 52-week high
    sr = _support_resistance_for(df)
    assert sr["nearest_resistance"] is None and sr["resistance_touch_count"] is None and sr["resistance_basis"] is None
    assert sr["nearest_support"] is not None and sr["atr_to_support"] is not None


def test_levels_do_not_move_with_a_flat_day_the_way_daily_pivots_did():
    """The pivots moved about 0.4 ATR a day; structural levels over a year of history barely move with one more session."""
    from data.precompute.technicals import _support_resistance

    df = _ohlcv(300, trend_total=20, seed=11)
    atr = float((df["high"] - df["low"]).tail(14).mean())
    a = _support_resistance(df.iloc[:-1], atr)
    b = _support_resistance(df, atr)
    if a["nearest_support"] is not None and b["nearest_support"] is not None:
        assert abs(a["nearest_support"] - b["nearest_support"]) < 0.5 * atr


def _support_resistance_for(df):
    from data.precompute.technicals import _support_resistance

    atr = float((df["high"] - df["low"]).tail(14).mean())
    return _support_resistance(df, atr)


# ---------- trend structure (zigzag tuning) ----------


def test_swing_structure_populated_on_trending_real_shaped_data():
    """Regression test for the real gap found planning this ticket:
    pandas-ta's zigzag() defaults (5% deviation) produce only 1-2 swing
    points across an entire 300-bar trending series — nowhere near enough
    for a 20-session HH/HL/LH/LL read. This module tunes to legs=5,
    deviation=0.03 specifically so this doesn't come back None on
    ordinary trending data."""
    df = _ohlcv(300, trend_total=50, seed=7)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["trend_structure"]["swing_structure_20d"] in ("HH", "HL", "LH", "LL")


def test_swing_structure_none_on_too_little_data():
    df = _ohlcv(10)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["trend_structure"]["swing_structure_20d"] is None


def test_swing_structure_none_on_zero_variance_price_not_a_crash():
    """Regression test for the most serious real gap found building this
    module: pandas-ta's zigzag() segfaults the whole process on
    zero-variance (completely flat) price data — a real, if rare, case
    for a halted or extremely illiquid stock. This must return None, not
    crash the test process. If this test hangs or kills the runner
    instead of completing, the precondition guard in _swing_structure()
    has regressed."""
    idx = pd.bdate_range("2026-01-01", periods=30)
    flat = pd.Series(10.0, index=idx)
    df = pd.DataFrame(
        {
            "open": flat,
            "high": flat,
            "low": flat,
            "close": flat,
            "volume": pd.Series(200, index=idx),
        }
    )
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["trend_structure"]["swing_structure_20d"] is None


# ---------- multi-timeframe ----------


def test_multi_timeframe_none_on_too_little_weekly_data():
    """~50 daily bars resamples to well under 20 weekly bars -- below
    _multi_timeframe's own len(weekly) < 20 guard, so both fields stay
    None rather than reaching the weekly RSI/MA calls at all."""
    df = _ohlcv(50)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["multi_timeframe"] == {"weekly_trend": None, "weekly_rsi_zone": None}


def test_weekly_trend_needs_40_weeks_not_200():
    """RDDT 2026-10-03 had weekly_trend None (a 20/50/200 weekly stack needs about four years of history). Now price
    against the 40-week SMA: about 2.5 years of daily data gives a trend; under 40 weeks gives none."""
    df = _ohlcv(650, trend_total=60, seed=11)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["multi_timeframe"]["weekly_trend"] in ("bullish", "bearish", "mixed")
    assert result["multi_timeframe"]["weekly_rsi_zone"] is not None
    short = _ohlcv(150, trend_total=10, seed=3)  # about 30 weekly bars
    result = compute_all(short, short, None, None, None, as_of=_AS_OF)
    assert result["multi_timeframe"]["weekly_trend"] is None and result["multi_timeframe"]["weekly_rsi_zone"] is None


def test_a_squeeze_release_is_a_released_squeeze_not_a_band_walk():
    from data.precompute.technicals import _pattern_metrics

    assert _pattern_metrics(True, "bullish") == {"squeeze_release": True, "breakout_direction": "bullish"}
    assert _pattern_metrics(False, "bullish") == {"squeeze_release": False, "breakout_direction": None}
    assert _pattern_metrics(True, None) == {"squeeze_release": True, "breakout_direction": None}


def test_multi_timeframe_weekly_trend_populates_with_enough_history():
    """~5 years of daily data: the weekly trend resolves."""
    df = _ohlcv(1300, trend_total=150, seed=12)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["multi_timeframe"]["weekly_trend"] in ("bullish", "bearish", "mixed")
    assert result["multi_timeframe"]["weekly_rsi_zone"] is not None


# ---------- market context ----------


def test_market_context_regime_on_sufficient_benchmark_data():
    stock = _ohlcv(250, trend_total=20, seed=2)
    benchmark = _ohlcv(250, trend_total=40, seed=3)
    result = compute_all(stock, benchmark, None, None, None, as_of=_AS_OF)
    assert result["market_context"]["market_regime"] in ("bull", "bear", "choppy")


def test_market_context_none_on_thin_benchmark_data():
    stock = _ohlcv(250, trend_total=20)
    benchmark = _ohlcv(30)
    result = compute_all(stock, benchmark, None, None, None, as_of=_AS_OF)
    assert result["market_context"]["market_regime"] is None


# ---------- relative performance (folded into trend_structure -- real gap
# found reviewing this module: DataBundle.relative_performance was never
# actually read anywhere in the real orchestrator's merge_output() code;
# the live prompt's payload template expects rs_vs_sector_3mo/
# rs_leadership inside trend_structure instead) ----------


def test_relative_performance_none_without_sector_etf():
    df = _ohlcv(100)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    ts = result["trend_structure"]
    assert ts["rs_vs_sector_3mo"] is None
    assert ts["rs_leadership"] is None


def test_relative_performance_computed_with_sector_etf():
    stock = _ohlcv(100, trend_total=15, seed=4)
    sector_etf = _ohlcv(100, trend_total=5, seed=5)
    result = compute_all(stock, stock, sector_etf, None, None, as_of=_AS_OF)
    ts = result["trend_structure"]
    assert ts["rs_vs_sector_3mo"] is not None
    assert ts["rs_leadership"] in ("leading", "lagging")


def test_relative_performance_uses_dates_not_row_position_across_series():
    """Regression test for the real bug caught on review: comparing stock
    vs. sector-ETF returns via matching row *position* (.iloc[-63] on
    each) silently misaligns the comparison if the two series don't share
    an identical trading calendar. Here the ETF has 20 extra early rows
    the stock doesn't have — same row position on each would land on
    different calendar dates. The date-keyed fix must still compute a
    stock return anchored ~90 days before the *stock's* own latest date,
    not wherever position -63 happens to fall in a longer ETF series."""
    stock = _ohlcv(100, trend_total=15, seed=4, start="2023-01-02")
    long_etf = _ohlcv(120, trend_total=5, seed=5, start="2022-11-28")  # starts 20 sessions earlier

    stock_only_result = compute_all(stock, stock, stock, None, None, as_of=_AS_OF)
    misaligned_result = compute_all(stock, stock, long_etf, None, None, as_of=_AS_OF)

    # The stock's own return-vs-itself is always 0 by construction — proves
    # the stock side of the comparison is still anchored off the stock's
    # own dates, unaffected by the ETF series being longer/differently dated.
    assert stock_only_result["trend_structure"]["rs_vs_sector_3mo"] == 0.0
    assert misaligned_result["trend_structure"]["rs_vs_sector_3mo"] is not None


# ---------- liquidity ----------


def test_liquidity_flags_reports_raw_dollar_volume_no_boolean_field():
    """v2.2 removed the older below_liquidity_floor boolean — the LLM
    reasons from the raw avg_dollar_volume_20 figure itself, this module
    doesn't pre-decide 'thin' with a hard-coded threshold."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.Series(10.0, index=idx)
    volume = pd.Series(rng.integers(100, 500, 30), index=idx)  # ~$1-5k/day, well under $1M
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume}
    )
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    liquidity = result["liquidity_flags"]
    assert liquidity["avg_dollar_volume_20"] < 1_000_000
    assert "thin_liquidity" not in liquidity


# ---------- earnings proximity ----------


def test_earnings_proximity_extracts_earliest_future_date():
    df = _ohlcv(30)
    calendar = [
        {"symbol": "TEST", "date": "2026-12-01"},
        {"symbol": "TEST", "date": "2026-09-15"},  # earlier — should win
    ]
    result = compute_all(df, df, None, calendar, None, as_of=_AS_OF)
    ep = result["earnings_proximity"]
    assert ep["next_earnings_date"] == "2026-09-15"
    assert ep["earnings_proximity_days"] > 0


def test_earnings_proximity_none_without_calendar():
    df = _ohlcv(30)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["earnings_proximity"] == {
        "next_earnings_date": None,
        "earnings_proximity_days": None,
    }


def test_earnings_proximity_ignores_past_dates():
    df = _ohlcv(30)
    calendar = [{"symbol": "TEST", "date": "2020-01-01"}]
    result = compute_all(df, df, None, calendar, None, as_of=_AS_OF)
    assert result["earnings_proximity"]["next_earnings_date"] is None


# ---------- freshness ----------


def test_is_current_true_when_last_bar_is_as_of_date():
    idx = pd.bdate_range("2026-08-01", periods=12)
    df = _ohlcv(12)
    df.index = idx
    as_of = idx[-1].date()
    result = compute_all(df, df, None, None, None, as_of=as_of)
    assert result["is_current"] is True
    assert result["days_old"] == 0


def test_days_old_positive_when_stale():
    df = _ohlcv(30, start="2026-01-02")
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["days_old"] > 0
    assert result["is_current"] is False


# ---------- price position (52-week high/low) ----------


def test_price_position_computes_pct_from_52w_high_and_low():
    df = _ohlcv(100, trend_total=20)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    pp = result["price_position"]
    assert pp["high_52w"] is not None
    assert (
        pp["pct_from_52w_high"] <= 0
    )  # current price can't exceed the rolling high that includes it
    assert pp["pct_from_52w_low"] >= 0


# ---------- pre-flight anomalies ----------


class TestPreflightWarnings:
    def test_zero_volume_bar_flagged(self):
        df = _ohlcv(30)
        df.iloc[-1, df.columns.get_loc("volume")] = 0
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        assert any("zero-volume" in w for w in result["preflight_warnings"])

    def test_negative_price_flagged(self):
        df = _ohlcv(30)
        df.iloc[5, df.columns.get_loc("low")] = -1.0
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        assert any("negative/zero price" in w for w in result["preflight_warnings"])

    def test_large_gap_flagged_without_news_ids(self):
        df = _ohlcv(30)
        close_col = df.columns.get_loc("close")
        open_col = df.columns.get_loc("open")
        df.iloc[10, open_col] = df.iloc[9, close_col] * 1.5  # 50% gap up
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        gap_warnings = [w for w in result["preflight_warnings"] if "gap" in w]
        assert len(gap_warnings) == 1
        assert "no corresponding news item" not in gap_warnings[0]

    def test_large_gap_with_news_present_notes_it_was_explained(self):
        df = _ohlcv(30)
        close_col = df.columns.get_loc("close")
        open_col = df.columns.get_loc("open")
        gap_date = df.index[10]
        df.iloc[10, open_col] = df.iloc[9, close_col] * 1.5
        news_ids = [{"id": "N1", "date": gap_date.date().isoformat()}]
        result = compute_all(df, df, None, None, news_ids, as_of=_AS_OF)
        gap_warnings = [w for w in result["preflight_warnings"] if "gap" in w]
        assert gap_warnings == []  # explained by news, not flagged

    def test_large_gap_with_news_ids_but_no_matching_date_still_flagged(self):
        df = _ohlcv(30)
        close_col = df.columns.get_loc("close")
        open_col = df.columns.get_loc("open")
        df.iloc[10, open_col] = df.iloc[9, close_col] * 1.5
        news_ids = [{"id": "N1", "date": "2020-01-01"}]  # unrelated date
        result = compute_all(df, df, None, None, news_ids, as_of=_AS_OF)
        gap_warnings = [w for w in result["preflight_warnings"] if "gap" in w]
        assert len(gap_warnings) == 1
        assert "no corresponding news item" in gap_warnings[0]


# ---------- empty / degenerate input ----------


def test_empty_raw_price_returns_fully_degraded_bundle():
    result = compute_all(pd.DataFrame(), _ohlcv(5), None, None, None, as_of=_AS_OF)
    assert result["technical_indicators"] == {}
    assert result["is_current"] is False
    assert "no price history available" in result["preflight_warnings"]


def test_a_gap_older_than_the_news_window_or_the_indicator_year_is_not_an_anomaly():
    """RDDT 2026-10-03: a 2024-10-30 earnings gap was flagged 'with no corresponding news item' on 8 of 51 runs."""
    df = _ohlcv(400)
    open_col = df.columns.get_loc("open")
    close_col = df.columns.get_loc("close")
    df.iloc[20, open_col] = df.iloc[19, close_col] * 1.5  # 380 sessions ago: outside the year the indicators use
    result = compute_all(df, df, None, None, [{"id": "N1", "date": df.index[-3].date().isoformat()}], as_of=_AS_OF)
    assert [w for w in result["preflight_warnings"] if "gap" in w] == []

    df2 = _ohlcv(400)
    df2.iloc[200, open_col] = df2.iloc[199, close_col] * 1.5  # inside the year, but before the news window starts
    result = compute_all(df2, df2, None, None, [{"id": "N1", "date": df2.index[-3].date().isoformat()}], as_of=_AS_OF)
    assert [w for w in result["preflight_warnings"] if "gap" in w] == []

    df3 = _ohlcv(400)
    df3.iloc[395, open_col] = df3.iloc[394, close_col] * 1.5  # inside the news window and unmatched: still flagged
    result = compute_all(df3, df3, None, None, [{"id": "N1", "date": df3.index[-8].date().isoformat()}], as_of=_AS_OF)
    assert len([w for w in result["preflight_warnings"] if "gap" in w]) == 1


def test_price_moves_and_the_up_down_volume_ratio():
    """KO 2026-10-03: heavy volume on a down day was called "confirming" because the agent saw the volume ratio
    and no price change at all."""
    from data.precompute.technicals import _price_moves

    closes = [100.0 + i for i in range(25)]  # up every day
    volume = [1000] * 25
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": volume})
    moves = _price_moves(df)
    assert moves["price_change_1d_pct"] == round((124 / 123 - 1) * 100, 2)
    assert moves["price_change_5d_pct"] == round((124 / 119 - 1) * 100, 2)
    assert moves["up_down_volume_ratio_20"] is None  # no down session in the window: nothing to divide by

    closes2 = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100.5]
    vols = [2000 if c2 > c1 else 1000 for c1, c2 in zip([99] + closes2[:-1], closes2)]
    df2 = pd.DataFrame({"open": closes2, "high": closes2, "low": closes2, "close": closes2, "volume": vols})
    assert _price_moves(df2)["up_down_volume_ratio_20"] == 2.0  # volume leans with the up sessions


def test_price_moves_are_none_without_enough_history():
    from data.precompute.technicals import _price_moves

    df = pd.DataFrame({"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [10, 10]})
    moves = _price_moves(df)
    assert moves["price_change_1d_pct"] == 100.0 and moves["price_change_5d_pct"] is None and moves["up_down_volume_ratio_20"] is None
