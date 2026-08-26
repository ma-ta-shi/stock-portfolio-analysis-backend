import numpy as np
import pandas as pd
import pytest

from data.precompute.risk_metrics import compute_all

_BENCHMARK = "^GSPC"
_CURRENCY = "USD"


def _ohlcv(
    n: int,
    seed: int = 1,
    trend_total: float = 0.0,
    start: str = "2020-01-02",
    daily_vol: float = 0.6,
) -> pd.DataFrame:
    """n business-day bars, lowercase columns, deterministic via seed.
    trend_total adds a linear drift across the whole series. daily_vol
    controls the noise magnitude (used to build a scaled/noisy stock
    series from a benchmark series for beta tests)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    drift = np.linspace(0, trend_total, n)
    noise = np.cumsum(rng.normal(0, daily_vol, n))
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


def _from_close(close: pd.Series, seed: int = 1) -> pd.DataFrame:
    """Builds a full OHLCV frame around an arbitrary close series, so
    drawdown/recovery scenarios can be hand-engineered precisely."""
    rng = np.random.default_rng(seed)
    n = len(close)
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    openp = close
    volume = rng.integers(500_000, 2_000_000, n)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=close.index,
    )


# ---------- empty input ----------


def test_empty_price_returns_all_none_but_keys_present():
    result = compute_all(pd.DataFrame(), pd.DataFrame(), _BENCHMARK, _CURRENCY)
    assert result["benchmark"] == _BENCHMARK
    assert result["adv_currency"] == _CURRENCY
    for field in (
        "beta",
        "annualized_vol_pct",
        "max_drawdown_1yr_pct",
        "recovery_1yr_days",
        "max_drawdown_3yr_pct",
        "recovery_3yr_days",
        "adv_millions",
    ):
        assert result[field] is None


# ---------- beta ----------


def test_beta_reflects_a_known_scaling_relationship():
    benchmark = _ohlcv(300, seed=1, trend_total=20)
    scaled_returns = benchmark["close"].pct_change().dropna() * 1.5
    stock_close = 100 * (1 + scaled_returns).cumprod()
    stock_close = pd.concat([pd.Series([100.0], index=benchmark.index[:1]), stock_close])
    stock = _from_close(stock_close, seed=2)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["beta"] is not None
    assert 1.3 < result["beta"] < 1.7


def test_beta_none_below_minimum_overlapping_bars():
    stock = _ohlcv(30, seed=1)
    benchmark = _ohlcv(30, seed=2)
    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["beta"] is None


# ---------- vol/beta independence (real design fix from plan review) ----------


def test_volatility_resolves_even_when_benchmark_has_far_fewer_bars():
    stock = _ohlcv(300, seed=1, trend_total=10)
    thin_benchmark = _ohlcv(30, seed=2)  # far shorter than the stock's own history

    result = compute_all(stock, thin_benchmark, _BENCHMARK, _CURRENCY)
    assert result["annualized_vol_pct"] is not None
    assert result["beta"] is None


def test_volatility_known_variance_series():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=300)
    daily_returns = rng.normal(0, 0.01, 300)  # 1% daily stdev
    close = pd.Series(100 * (1 + daily_returns).cumprod(), index=idx)
    stock = _from_close(close, seed=7)
    benchmark = _ohlcv(300, seed=8)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    # Expected value is derived from the actual sample stdev of the
    # generated series (not the theoretical 1% population parameter) -
    # with n=300 there's real sampling variance between the two.
    expected = close.tail(253).pct_change().dropna().std() * np.sqrt(252) * 100
    assert result["annualized_vol_pct"] == pytest.approx(expected, abs=0.1)


# ---------- drawdown / recovery ----------


def test_drawdown_and_recovery_engineered_scenario():
    idx = pd.bdate_range("2024-01-02", periods=300)
    close = np.concatenate(
        [
            np.linspace(100, 100, 50),
            np.linspace(100, 60, 50),  # -40% drawdown
            np.linspace(60, 110, 200),  # recovers past the prior peak
        ]
    )
    stock = _from_close(pd.Series(close, index=idx), seed=3)
    benchmark = _ohlcv(300, seed=4)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["max_drawdown_1yr_pct"] is not None
    assert result["max_drawdown_1yr_pct"] > 30
    assert result["recovery_1yr_days"] is not None
    assert result["recovery_1yr_days"] > 0


def test_recovery_days_is_exact_not_off_by_one():
    """Regression for a real bug found on review: the recovery search
    used to skip the first qualifying post-trough bar (on the wrong
    assumption that index 0 might spuriously be the trough itself, which
    can't happen - the trough's value is strictly below peak by
    construction), silently overstating recovery_days by a trading day
    every time. Precise value asserted here, not just '> 0', since the
    original bug passed a '> 0' check without being caught.

    Needs >=~202 bars to clear the 1yr window's coverage threshold, so
    the drawdown/recovery shape is embedded in a long flat run rather
    than a bare few-bar series."""
    idx = pd.bdate_range("2023-01-02", periods=250)
    close = np.concatenate(
        [
            np.full(100, 100.0),  # flat run, bars 0-99
            [60.0],  # trough, bar 100
            np.full(9, 90.0),  # still below peak, bars 101-109
            [105.0],  # first bar that clears the prior peak of 100, bar 110
            np.full(139, 105.0),  # padding to reach 250 total bars
        ]
    )
    stock = _from_close(pd.Series(close, index=idx), seed=3)
    benchmark = _ohlcv(250, seed=4)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    expected_days = int(np.busday_count(idx[100].date(), idx[110].date()))
    assert result["recovery_1yr_days"] == expected_days


def test_recovery_on_the_very_last_bar_is_not_misreported_as_none():
    """Regression for the second half of the same bug: when exactly one
    bar qualifies as recovered, the old `len(recovered) <= 1` guard
    treated it as 'not recovered' instead of a real, confirmed recovery."""
    idx = pd.bdate_range("2023-01-02", periods=250)
    close = np.concatenate(
        [
            np.full(100, 100.0),  # flat run, bars 0-99
            [60.0],  # trough, bar 100
            np.full(148, 90.0),  # stays below peak for the rest of the series...
            [105.0],  # ...until the very last bar, the only qualifying recovery point
        ]
    )
    stock = _from_close(pd.Series(close, index=idx), seed=3)
    benchmark = _ohlcv(250, seed=4)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["recovery_1yr_days"] is not None
    assert result["recovery_1yr_days"] == int(np.busday_count(idx[100].date(), idx[249].date()))


def test_drawdown_still_in_progress_recovery_is_none():
    idx = pd.bdate_range("2024-01-02", periods=300)
    close = np.concatenate(
        [
            np.linspace(100, 100, 50),
            np.linspace(100, 60, 200),  # steady decline, never recovers
            np.linspace(60, 65, 50),
        ]
    )
    stock = _from_close(pd.Series(close, index=idx), seed=3)
    benchmark = _ohlcv(300, seed=4)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["max_drawdown_1yr_pct"] is not None
    assert result["recovery_1yr_days"] is None


def test_asymmetric_degradation_1yr_resolves_3yr_does_not():
    """The common real-world case: a recently-listed stock with ~1.5
    years of history. 1yr drawdown should resolve; 3yr needs >=756 bars
    and this fixture deliberately falls short of that."""
    stock = _ohlcv(380, seed=5, trend_total=-20)  # ~1.5 years of business days
    benchmark = _ohlcv(380, seed=6, trend_total=-15)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["max_drawdown_1yr_pct"] is not None
    assert result["max_drawdown_3yr_pct"] is None
    assert result["recovery_3yr_days"] is None


def test_drawdown_none_below_minimum_bars():
    stock = _ohlcv(10, seed=1)
    benchmark = _ohlcv(10, seed=2)
    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["max_drawdown_1yr_pct"] is None
    assert result["recovery_1yr_days"] is None
    assert result["max_drawdown_3yr_pct"] is None
    assert result["recovery_3yr_days"] is None


# ---------- ADV ----------


def test_adv_millions_known_value():
    idx = pd.bdate_range("2024-01-02", periods=25)
    close = pd.Series([10.0] * 25, index=idx)
    volume = pd.Series([1_000_000] * 25, index=idx)
    stock = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx
    )
    benchmark = _ohlcv(25, seed=2)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    # 10 * 1_000_000 = $10M/day, averaged over 20 bars -> 10.0
    assert result["adv_millions"] == 10.0
    assert result["adv_currency"] == _CURRENCY


def test_adv_millions_none_with_a_clean_but_short_series():
    """Regression for a bug introduced by the rolling-window fix itself:
    an earlier version tolerated a partial window down to half of
    _ADV_WINDOW, but .rolling(_ADV_WINDOW).mean() needs the full window's
    worth of bars to produce anything at all regardless of any separate
    length check - a clean 15-bar series (no gaps, just short) silently
    always returned None even though it looked like it should have
    cleared a '10 is enough' threshold that no longer meant anything
    under rolling(). Locking in the corrected, simpler behavior: the full
    window is required, matching technicals.py's own convention exactly."""
    idx = pd.bdate_range("2024-01-02", periods=15)
    close = pd.Series([10.0] * 15, index=idx)
    volume = pd.Series([1_000_000] * 15, index=idx)
    stock = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx
    )
    benchmark = _ohlcv(15, seed=2)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["adv_millions"] is None


def test_adv_millions_none_below_minimum_bars():
    idx = pd.bdate_range("2024-01-02", periods=5)
    close = pd.Series([10.0] * 5, index=idx)
    volume = pd.Series([1_000_000] * 5, index=idx)
    stock = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx
    )
    benchmark = _ohlcv(5, seed=2)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["adv_millions"] is None
    # a static echo of the input, unrelated to whether adv_millions computed
    assert result["adv_currency"] == _CURRENCY


def test_adv_millions_none_when_a_volume_gap_falls_inside_the_window():
    """Regression for a real bug found on review: _normalize_ohlcv only
    dropna's on open/high/low/close, not volume, so a single NaN volume
    tick (a halted session, a provider gap) can survive into the ADV
    calculation. .tail(20).mean()'s default skipna=True would silently
    average over 19 real values and report it as a clean 20-day figure;
    .rolling(20).mean() (matching technicals.py's own avg_dollar_volume_20
    pattern) correctly fails the whole window to None instead."""
    idx = pd.bdate_range("2024-01-02", periods=25)
    close = pd.Series([10.0] * 25, index=idx)
    volume = pd.Series([1_000_000] * 25, index=idx, dtype=float)
    volume.iloc[-5] = np.nan
    stock = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx
    )
    benchmark = _ohlcv(25, seed=2)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["adv_millions"] is None


def test_adv_currency_present_even_when_thin_liquidity():
    """A genuinely thin/illiquid ticker can legitimately produce a very
    small non-None adv_millions - that's a real number, not a bug."""
    idx = pd.bdate_range("2024-01-02", periods=25)
    close = pd.Series([1.0] * 25, index=idx)
    volume = pd.Series([100] * 25, index=idx)
    stock = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx
    )
    benchmark = _ohlcv(25, seed=2)

    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
    assert result["adv_millions"] is not None
    assert result["adv_millions"] < 0.01


# ---------- numpy leak regression (established pattern this session) ----------


def test_numeric_fields_are_plain_python_floats_not_numpy_scalars():
    """A plain == check doesn't catch this - np.float64 == float compares
    true by value. Matches the same class of bug already found and fixed
    in yfinance.py's get_earnings_surprises() this session."""
    stock = _ohlcv(300, seed=1, trend_total=10)
    benchmark = _ohlcv(300, seed=2, trend_total=8)
    result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)

    for field in ("beta", "annualized_vol_pct", "max_drawdown_1yr_pct", "adv_millions"):
        assert type(result[field]) is float, f"{field} leaked a numpy scalar: {type(result[field])}"


# ---------- benchmark / currency passthrough ----------


def test_benchmark_and_currency_are_echoed():
    stock = _ohlcv(300, seed=1)
    benchmark = _ohlcv(300, seed=2)
    result = compute_all(stock, benchmark, "^GSPTSE", "CAD")
    assert result["benchmark"] == "^GSPTSE"
    assert result["adv_currency"] == "CAD"


# ---------- column casing / index regression (mirrors technicals.py's own bug history) ----------


class TestColumnCasingRegression:
    def test_capitalized_columns_produce_the_same_result_as_lowercase(self):
        lower_stock = _ohlcv(300, seed=1, trend_total=15)
        lower_benchmark = _ohlcv(300, seed=2, trend_total=10)
        capped_stock = _capitalized(lower_stock)
        capped_benchmark = _capitalized(lower_benchmark)

        result_lower = compute_all(lower_stock, lower_benchmark, _BENCHMARK, _CURRENCY)
        result_capped = compute_all(capped_stock, capped_benchmark, _BENCHMARK, _CURRENCY)

        assert result_capped["beta"] is not None
        assert result_capped["beta"] == result_lower["beta"]
        assert result_capped["adv_millions"] == result_lower["adv_millions"]


class TestDateIndexRegression:
    def test_plain_date_index_is_coerced_to_datetime_index(self):
        stock = _ohlcv(300, seed=1, trend_total=10)
        benchmark = _ohlcv(300, seed=2, trend_total=5)
        stock.index = stock.index.date  # plain datetime.date objects, like openbb-tmx returns
        benchmark.index = benchmark.index.date

        result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
        assert result["beta"] is not None
        assert result["annualized_vol_pct"] is not None

    def test_tz_aware_and_tz_naive_indices_still_align(self):
        """Regression for a real bug found via live AAPL/RY.TO verification:
        yfinance returns tz-aware timestamps, openbb-tmx returns tz-naive
        ones. Before this fix, pairing a tz-aware benchmark series with a
        tz-naive stock series via pd.concat's inner join silently produced
        zero overlapping rows - beta came back None even with a full year
        of genuinely overlapping calendar dates on both sides."""
        stock = _ohlcv(300, seed=1, trend_total=10)
        benchmark = _ohlcv(300, seed=2, trend_total=8)
        benchmark.index = benchmark.index.tz_localize("US/Eastern")

        result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
        assert result["beta"] is not None

    def test_misaligned_date_ranges_align_by_date_not_position(self):
        stock = _ohlcv(300, seed=1, trend_total=10, start="2023-01-02")
        # benchmark starts 30 business days later and is shorter - a
        # positional (.iloc) comparison would silently misalign these.
        benchmark = _ohlcv(270, seed=2, trend_total=8, start="2023-02-13")

        result = compute_all(stock, benchmark, _BENCHMARK, _CURRENCY)
        assert result["beta"] is not None
