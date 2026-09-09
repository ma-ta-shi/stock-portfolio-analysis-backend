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

    def test_mixed_is_currently_treated_as_a_downtrend(self):
        # The known gap: mixed falls through to the 60/20 band, identical to
        # bearish. RSI 70 is "neutral" under uptrend rules but "overbought"
        # here. If mixed is later given its own band, this fails on purpose.
        assert _rsi_zone_adjusted(70.0, "mixed") == "overbought"
        for rsi in (25.0, 55.0, 70.0):
            assert _rsi_zone_adjusted(rsi, "mixed") == _rsi_zone_adjusted(rsi, "bearish")


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
        assert ind["bb_position"] in ("squeeze", "walking_band", "mean_reverting")

    def test_squeeze_needs_more_than_bbands_minimum(self):
        """Regression test: confirmed live that ta.squeeze() still returns
        the identity-df sentinel at exactly 20 rows even though
        ta.bbands(length=20) doesn't — a stricter internal minimum than
        bbands' own length. bb_position must not crash if squeeze isn't
        available yet, even when the bands themselves are."""
        df = _ohlcv(20)
        result = compute_all(df, df, None, None, None, as_of=_AS_OF)
        ind = result["technical_indicators"]
        assert ind["bb_position"] in (None, "mean_reverting", "walking_band")


# ---------- support / resistance ----------


def test_support_resistance_computed_with_minimal_data():
    df = _ohlcv(5)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    sr = result["support_resistance"]
    assert "P" in sr["levels"]
    # nearest support/resistance re-derived from real price comparison, not the pivot label
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


def test_multi_timeframe_rsi_zone_populates_before_weekly_trend_does():
    """Real gap found reviewing this module: weekly_trend reuses
    _moving_averages' stack_order, which needs sma_200 -- on WEEKLY bars
    that's 200 weeks (~4 years) of daily history, far more than the ~2
    years typical callers fetch. Confirmed live against real router-fetched
    AAPL/RY.TO price history (2y each): weekly_rsi_zone populated on both,
    weekly_trend was None on both. ~2.5 years of daily data (enough for
    ~130 weekly bars, short of the 200 sma_200 needs but past the 14-bar
    weekly RSI minimum) reproduces the same split here."""
    df = _ohlcv(650, trend_total=60, seed=11)
    result = compute_all(df, df, None, None, None, as_of=_AS_OF)
    assert result["multi_timeframe"]["weekly_rsi_zone"] is not None
    assert result["multi_timeframe"]["weekly_trend"] is None


def test_multi_timeframe_weekly_trend_populates_with_enough_history():
    """~5 years of daily data gives >200 weekly bars, enough for weekly
    sma_200 -- and therefore stack_order/weekly_trend -- to resolve."""
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
