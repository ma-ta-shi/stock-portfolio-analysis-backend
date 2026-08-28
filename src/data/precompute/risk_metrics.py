"""Risk metrics for the Risk Advisor agent (Pass 2), ClickUp 86ban0wee.

Pure function — fetching/resolution of raw_price, benchmark_price,
benchmark_ticker, and currency are the caller's job (DataPipeline.prepare(),
not yet built). Consumed by build_precomputed_risk_metrics() in
"Agent Prompts/Current Prompts/Risk Advisor Agent Prompt.md" (lines
252-278) — the only real consumer, confirmed by direct read, not the
ticket's own draft design. That function reads exactly these fields:
benchmark, beta, annualized_vol_pct, max_drawdown_1yr_pct,
recovery_1yr_days, max_drawdown_3yr_pct, recovery_3yr_days, adv_millions,
adv_currency. liquidity_classification/sector_overlap_summary/portfolio_*
are deliberately NOT produced here — liquidity is an LLM judgment
contextual on implied position size (Step 5 of the prompt), and the
CORR/CONC fields need a correlation matrix (correlation.py, deferred) plus
live portfolio context that a single-stock call can't produce.

Caller contract: max_drawdown_3yr_pct/recovery_3yr_days can only resolve
if raw_price covers >=3 years of history — this module has no I/O of its
own to fetch more. Request a >=3y period from router.get_price_history()
when building this module's inputs, or the 3yr fields will silently and
permanently read None.
"""

import numpy as np
import pandas as pd

_MIN_RETURN_BARS = 60  # ~3 months of daily bars; below this a beta/vol estimate isn't statistically meaningful
_MIN_DRAWDOWN_COVERAGE = 0.8  # a "1yr"/"3yr" drawdown needs close to a full window's worth of bars,
# not just a handful - scales with window size rather than a flat constant (e.g. 20 bars would let a
# 1.5-year fixture pass as a "3yr" drawdown, which isn't a meaningful claim)
_ADV_WINDOW = 20  # matches technicals.py's own avg_dollar_volume_20 window, for cross-module consistency
_TRADING_DAYS_PER_YEAR = 252
_ONE_YEAR_BARS = 252
_THREE_YEAR_BARS = 756


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Same defense as technicals.py's _normalize_ohlcv (lowercase columns,
    coerce to DatetimeIndex, sort ascending, dropna) — duplicated here
    rather than imported, matching this codebase's convention that each
    precompute module is self-contained. Root cause is identical:
    router.get_price_history() can return capitalized (yfinance) or
    lowercase (FMP/openbb-tmx) columns, and openbb-tmx returns a plain
    object-dtype Index of datetime.date values instead of a DatetimeIndex.

    Also strips timezone info. Live-verified real bug: yfinance (a
    fallback source for both stock and benchmark price history) returns
    tz-aware timestamps (e.g. US/Eastern), while openbb-tmx returns
    tz-naive ones — when raw_price and benchmark_price come from
    different sources, pd.concat's inner join between a tz-aware and a
    tz-naive DatetimeIndex silently produces ZERO overlapping rows even
    when the underlying calendar dates genuinely match, which killed beta
    entirely for a real RY.TO/^GSPTSE run in this exact scenario."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.rename(columns=str.lower)
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    return out.dropna(subset=[c for c in ("open", "high", "low", "close") if c in out.columns])


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def _beta(stock_close: pd.Series, benchmark_close: pd.Series) -> float | None:
    stock_returns = _daily_returns(stock_close.tail(_ONE_YEAR_BARS + 1))
    benchmark_returns = _daily_returns(benchmark_close.tail(_ONE_YEAR_BARS + 1))
    # Date-keyed alignment, not positional — same class of bug already
    # fixed in technicals.py's _return_over_days: comparing two series by
    # row position assumes identical trading calendars, which isn't
    # guaranteed (data gaps, halt days, differently-fetched date ranges).
    aligned = pd.concat({"stock": stock_returns, "benchmark": benchmark_returns}, axis=1, join="inner")
    if len(aligned) < _MIN_RETURN_BARS:
        return None
    benchmark_variance = aligned["benchmark"].var()
    if not benchmark_variance:
        return None
    covariance = aligned["stock"].cov(aligned["benchmark"])
    return round(float(covariance / benchmark_variance), 2)


def _annualized_vol_pct(stock_close: pd.Series) -> float | None:
    # Deliberately independent of benchmark data/alignment — volatility is
    # a property of the stock's own return series alone. Coupling this to
    # beta's benchmark-overlap threshold would wrongly null out a
    # computable metric just because the benchmark series has gaps.
    returns = _daily_returns(stock_close.tail(_ONE_YEAR_BARS + 1))
    if len(returns) < _MIN_RETURN_BARS:
        return None
    vol = returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR) * 100
    return round(float(vol), 1) if pd.notna(vol) else None


def _max_drawdown(close: pd.Series, window_bars: int) -> tuple[float | None, int | None]:
    windowed = close.tail(window_bars)
    if len(windowed) < window_bars * _MIN_DRAWDOWN_COVERAGE:
        return None, None
    running_max = windowed.cummax()
    drawdown = (windowed - running_max) / running_max
    trough_idx = drawdown.idxmin()
    max_dd = drawdown.loc[trough_idx]
    if pd.isna(max_dd):
        return None, None
    peak_value = running_max.loc[trough_idx]
    # windowed.loc[trough_idx] is strictly < peak_value by construction (that's
    # what makes it the trough), so the trough bar itself never appears in
    # `recovered` below - no need to skip index 0. An earlier version skipped
    # it anyway, which silently overstated recovery_days by a day in the
    # normal case, and misreported a real single-bar recovery as None.
    post_trough = windowed[windowed.index > trough_idx]
    recovered = post_trough[post_trough >= peak_value]
    if recovered.empty:
        recovery_days = None
    else:
        recovery_date = recovered.index[0]
        recovery_days = int(np.busday_count(trough_idx.date(), recovery_date.date()))
    return round(float(abs(max_dd) * 100), 1), recovery_days


def _adv_millions(close: pd.Series, volume: pd.Series) -> float | None:
    if len(close) < _ADV_WINDOW:
        return None
    # .rolling(window).mean() (not .tail(window).mean()) deliberately, matching
    # technicals.py's own avg_dollar_volume_20 exactly - rolling's default
    # min_periods equals the window size, so a single NaN volume tick anywhere
    # in the trailing window (a halted session, a provider gap) correctly
    # fails the whole figure to NaN instead of .mean()'s default skipna=True
    # silently averaging over however many real values happen to remain and
    # reporting it as a clean N-day average. Live-verified: a lone NaN 5 bars
    # back produced a normal-looking 10.0 under .tail().mean() with no
    # indication one of the 20 days was actually missing.
    #
    # An earlier version of this function allowed a partial window down to
    # half of _ADV_WINDOW (via a separate _MIN_ADV_BARS constant, tolerated
    # by .tail().mean()) - that tolerance doesn't survive the switch to
    # .rolling(), which needs the full window's worth of bars to produce
    # anything at all regardless of any separate length check, and
    # technicals.py's own established convention for this same metric
    # doesn't tolerate a partial window either. Matching it exactly is
    # simpler than trying to preserve a partial-window allowance that would
    # have silently reintroduced the NaN-gap tolerance this fix removes.
    dollar_volume = (close * volume).rolling(_ADV_WINDOW).mean().iloc[-1]
    return round(float(dollar_volume / 1_000_000), 1) if pd.notna(dollar_volume) else None


def compute_all(
    raw_price: pd.DataFrame,
    benchmark_price: pd.DataFrame,
    benchmark_ticker: str,
    currency: str,
) -> dict:
    """Assembles every risk field the Risk Advisor agent consumes from raw
    OHLCV price data. Pure function — caching, fetching, and
    benchmark/currency resolution are the caller's job. See module
    docstring for the >=3y history contract on raw_price."""
    df = _normalize_ohlcv(raw_price)
    benchmark_df = _normalize_ohlcv(benchmark_price)
    # Every helper below already has its own length-based threshold that
    # degrades to None on a too-short (including zero-length) series, so a
    # missing column can just become an empty Series here rather than
    # needing a separate all-None early-return branch - one code path
    # handles "no data at all" and "not enough data" identically.
    close = df["close"] if "close" in df.columns else pd.Series(dtype=float)
    benchmark_close = benchmark_df["close"] if "close" in benchmark_df.columns else pd.Series(dtype=float)
    volume = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)

    dd_1yr, recovery_1yr = _max_drawdown(close, _ONE_YEAR_BARS)
    dd_3yr, recovery_3yr = _max_drawdown(close, _THREE_YEAR_BARS)

    return {
        "benchmark": benchmark_ticker,
        "beta": _beta(close, benchmark_close),
        "annualized_vol_pct": _annualized_vol_pct(close),
        "max_drawdown_1yr_pct": dd_1yr,
        "recovery_1yr_days": recovery_1yr,
        "max_drawdown_3yr_pct": dd_3yr,
        "recovery_3yr_days": recovery_3yr,
        "adv_millions": _adv_millions(close, volume),
        "adv_currency": currency,
    }
