"""technicals.py — full technical-analysis input contract for the
Technical Analyst agent (ClickUp 86ban0wde). Takes raw OHLCV price data
and produces every structured technical field the agent consumes, using
pandas-ta for the math. Pure computation — no LLM calls, no API calls,
per CLAUDE.md's "pipeline does the math" rule.

Doc/contract reconciliation (disclosed, not silently resolved):

- Built to the v2.2 Technical Analyst Agent Prompt, not the older Data
  Pipeline doc: `below_liquidity_floor` (bool) and `unfilled_gaps_nearby`
  are v2.2-removed fields, not produced here. Liquidity is a soft caveat
  off raw `avg_dollar_volume_20 < $1M`, no boolean field.
- `timeline` is deliberately NOT a parameter of `compute_all()`, unlike
  the ticket's literal signature. Checked the live prompt: timeline only
  drives `{timeline_instruction}`, an LLM-interpretation weighting hint
  ("short-term weights momentum... long-term emphasizes 200-day
  structure") — it doesn't change which fields get computed or with what
  window sizes anywhere in this module's design. Carrying an unused
  parameter through would be dead surface.
- First real gap found auditing the ticket against real upstream data:
  `router.get_price_history()` can hand this module three genuinely
  different raw DataFrame shapes depending on which provider actually
  served the request (yfinance uses capitalized Open/High/Low/Close/
  Volume; FMP and openbb-tmx use lowercase). Live-confirmed: pandas-ta's
  `.ta` accessor requires lowercase columns — capitalized columns don't
  raise, they silently return NaN for every indicator. `_normalize_ohlcv`
  below defends against this with a blanket `.rename(columns=str.lower)`
  rather than a provider-layer NormalizedPriceHistory type (this fix
  works regardless of source; a per-provider field-mapping type would be
  more machinery for the same result). A real provider-layer type is a
  legitimate future improvement if another consumer needs the same
  guarantee — not built here.
- Second real gap found, also not in the ticket: `ta.zigzag()`'s default
  parameters (5% deviation, 10-bar minimum) are far too insensitive for
  the ticket's own `swing_structure_20d` design (label successive zigzag
  swing pairs over a trailing 20-session window as HH/HL/LH/LL).
  Live-verified against both random-walk and strongly-trending 300-session
  synthetic series: only 1-2 non-null swing points registered in the
  *entire* 300-bar history at default settings — nowhere near enough for
  one HH/HL pair inside a 20-session window. Tuned to `legs=5,
  deviation=0.03` (3%), which produced 79 non-null points over the same
  300 bars in the trending case. First-pass, from synthetic data — same
  disclosure as every other undocumented threshold in this project's
  precompute layer (see macro_sources.py's own module docstring).
- Third real gap found while testing, not in the ticket, and the most
  serious: `ta.zigzag()` segfaults the whole Python process on
  zero-variance price data (a completely flat close over the window —
  rare in practice, but a real, reachable case for a halted or
  extremely illiquid stock). Confirmed live: the crash happens during
  interpreter cleanup *after* `zigzag()` appears to have returned a
  normal result, not synchronously inside the call — so no `try/except`
  around the call site can catch it. `_swing_structure()` checks for
  zero price variance before ever calling `zigzag()` and returns `None`
  instead — a hard precondition, not error handling after the fact,
  because there is no reliable "after the fact" here.
- Fourth real gap found on review, also not in the ticket: the first
  implementation of `_swing_structure()`'s HH/HL/LH/LL classification
  compared *adjacent* zigzag swings (a peak vs. the trough right before
  it, or vice versa) rather than same-type extrema (latest peak vs. the
  *previous* peak). Since zigzag swings alternate by construction, a peak
  is always higher than its immediately preceding trough and a trough
  always lower than its immediately preceding peak — so that comparison
  carried no real trend-structure signal, it just re-derived which type
  the latest swing already was. Fixed to compare a swing against the one
  2 positions back (same type, since types alternate), using the zigzag
  output's own sign column — the standard definition of higher-high/
  higher-low/lower-high/lower-low.
- Fifth real gap found on review: `price_position`'s 52-week high/low
  was computed from the closing price only, not the intraday `high`/`low`
  columns — understated the real high, overstated the real low, and
  didn't match what "52-week high/low" conventionally means. Fixed to
  use `high`/`low`.
- Sixth real gap found on review: `_relative_performance()` compared
  stock vs. sector-ETF 3-month returns via `.iloc[-63]` on each series —
  a raw row position, not a date. Two different series aren't guaranteed
  to share an identical trading calendar (a data gap, a halt day, a
  slightly different fetched date range), so "63 rows back" could
  silently land on different actual calendar dates for the stock vs. the
  ETF, comparing two returns as of two different days without knowing
  it. Fixed with `_return_over_days()`, a date-keyed lookback — same
  fix shape as macro_sources.py's `_value_n_days_ago`.
- `price_position` field mapping is this module's own inference, not a
  stated fact anywhere: the ticket lists "52-week high/low" as a
  computation but never says which of DataBundle's 13 technical output
  fields it lands in. `price_position` is the only field with no other
  obvious owner.
- Caching (the ticket's "4-hour TTL" note) is explicitly not this
  module's job — `compute_all()` is a pure function, like every other
  precompute module. DataPipeline.prepare() decides when to call it.
- Pre-flight gap-vs-news cross-check is optional: `news_id_assignment.py`
  is real, tested code on a separate, still-unmerged branch, not
  importable here. `news_ids` defaults to None; the raw gap anomaly still
  gets flagged without it, just without the "and no news explains it"
  qualifier. Zero-volume and negative-price anomaly detection don't
  depend on it at all and always run.
- Sector ETF mapping (for relative strength) doesn't exist anywhere else
  in the codebase (checked) and isn't built here either — `compute_all()`
  takes `sector_etf_price` as an already-resolved DataFrame, matching
  this module's own "pure function, fetching/resolution is the caller's
  job" design (see below). Caught on review: an earlier draft of this
  module built a `_SECTOR_ETF` lookup (GICS sector -> SPDR Select Sector
  ETF ticker) internally, but nothing ever called it — resolving a sector
  to a ticker is exactly the kind of resolution this module explicitly
  isn't supposed to own, so it was dead weight, not a real dependency.
  Wherever `DataPipeline.prepare()` ends up resolving `sector_etf_price`
  from a sector name is the right place for that mapping to live.
- Seventh real gap found on review, no test coverage existed for this
  group at all until now: `multi_timeframe.weekly_trend` reuses
  `_moving_averages`' `stack_order`, which requires sma_200 to resolve —
  on WEEKLY bars that's 200 *weeks*, ~4 years of daily history. The
  Technical Analyst prompt cites `weekly_trend` most heavily for the
  long-term timeline ("emphasize... weekly structure"), but confirmed
  live against real router-fetched AAPL/RY.TO history (2y each, a
  plausible real fetch length): `weekly_rsi_zone` populated on both,
  `weekly_trend` was None on both — the field silently never resolves
  unless the caller fetches materially more history than the daily
  fields need. Not fixed here (changing weekly stack_order's threshold
  would diverge from the daily definition it deliberately reuses) —
  flagged so whoever wires the caller's `raw_price` fetch length knows
  ~5y/max is needed if `weekly_trend` is meant to actually populate.
- Eighth real gap found on review, the most serious contract-vs-consumer
  mismatch found in this module: several output field names didn't match
  what the live Technical Analyst Prompt's payload template and
  `merge_output()` orchestrator snippet actually read, confirmed against
  the real prompt file, not the ticket's own (less precise) summary list.
  Fixed:
  - `atr_60_avg` was computed internally (for `volatility_regime`'s own
    classification) but never returned — the prompt needs it as a field.
  - `volatility_regime` -> `volatility_regime_derived`; `bb_mid` ->
    `bb_middle`; `trend_structure.weekly_swing_structure` ->
    `trend_structure.trend_structure_weekly`; `market_context.regime` ->
    `market_context.market_regime`.
  - `support_resistance` was nesting `level`/`distance_pct`/`atr_multiple`
    inside `nearest_support`/`nearest_resistance` dicts. The prompt's own
    payload template treats these as flat scalar fields — literally
    `Support={nearest_support} ({pct_to_support}, {atr_to_support} ATR,
    touches={support_touch_count})` — so `_support_resistance()` now
    returns a flat shape with `pct_to_support`/`atr_to_support` as
    siblings, plus new `support_touch_count`/`resistance_touch_count`
    fields (sessions in the trailing ~6 months where price came within
    0.5 ATR of the level — a first-pass tolerance, not specified
    anywhere, same disclosure as the zigzag deviation tuning above).
  - `rs_vs_sector_3mo`/`rs_leadership` lived in their own top-level
    `relative_performance` dict — but `DataBundle.relative_performance`
    was never actually read anywhere in the real orchestrator's
    `merge_output()` code; the prompt's template expects both fields
    inside `trend_structure` instead. Folded `_relative_performance()`'s
    output into `_trend_structure()` (now takes `sector_etf_price`) and
    dropped `relative_performance` as its own `compute_all()`/`DataBundle`
    field.
  - `pattern_metrics`'s `squeeze_release`/`breakout_direction` shape still
    doesn't match the prompt's `detected_patterns[]`/`pattern_target` —
    left as-is; this is the already-disclosed MVP-proxy scope-down above,
    not a naming bug, and closing it means real geometric pattern
    recognition, explicitly out of scope for this module.
"""

from datetime import date

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401  (registers the .ta DataFrame accessor)

_ZIGZAG_LEGS = 5
_ZIGZAG_DEVIATION = 0.03  # 3% — see module docstring on why not pandas-ta's 5% default


# ---------- input normalization ----------


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns (pandas-ta requires this — see module docstring)
    and sort ascending by date with NaNs dropped. None of the provider
    get_price_history() implementations document or guarantee ordering,
    and pandas-ta's rolling/window functions are position-based: an
    out-of-order DataFrame produces silently wrong indicators, not an
    error. Same class of bug already fixed twice elsewhere in this
    codebase (macro_sources.py's _sorted_dropna, stats_canada.py's
    date-keyed lookups).

    Also coerces the index to a real DatetimeIndex — confirmed live that
    openbb-tmx's get_price_history() (the CA primary source) returns a
    plain object-dtype Index of `datetime.date` values, not a
    DatetimeIndex. pandas-ta's own pivots() only warns about this
    ("[!] Pivots requires an ordered DatetimeIndex"), but pandas' own
    .resample('W') (used for multi_timeframe) raises a hard TypeError on
    it — confirmed live, this crashed a real CA-ticker run before this
    fix. yfinance/FMP already return a real DatetimeIndex; pd.to_datetime
    is a no-op for them and a real fix for openbb-tmx."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.rename(columns=str.lower)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.dropna(subset=[c for c in ("open", "high", "low", "close") if c in out.columns])


# ---------- pandas-ta insufficient-data guard ----------


def _ta_result(df: pd.DataFrame, result: pd.Series | pd.DataFrame):
    """pandas-ta's length-parameterized indicators (sma/ema/rsi/atr/bbands/
    macd — confirmed live, all of them) don't return a NaN-filled result
    when there isn't enough data for the requested length; they silently
    return the exact same input DataFrame object unchanged. Not
    documented anywhere. Detected via identity (`result is df`), which is
    precise — confirmed live this is the actual return, not just a
    same-shaped coincidence. Treat as 'insufficient data' (None), not as
    if the raw OHLCV frame were the indicator's output — using it as-is
    would silently read open/high/low/volume values as if they were an
    SMA/RSI/whatever."""
    return None if result is df else result


def _last_or_none(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else None


# ---------- moving averages ----------


def _slope_label(series: pd.Series, lookback: int = 5) -> str | None:
    if series is None or len(series.dropna()) < lookback + 1:
        return None
    today, past = series.iloc[-1], series.iloc[-1 - lookback]
    if pd.isna(today) or pd.isna(past) or past == 0:
        return None
    change_pct = (today - past) / past * 100
    if change_pct > 0.1:
        return "rising"
    if change_pct < -0.1:
        return "falling"
    return "flat"


def _moving_averages(df: pd.DataFrame) -> dict:
    close = df["close"]
    sma20 = _ta_result(df, df.ta.sma(length=20))
    sma50 = _ta_result(df, df.ta.sma(length=50))
    sma200 = _ta_result(df, df.ta.sma(length=200))
    ema12 = _ta_result(df, df.ta.ema(length=12))
    ema26 = _ta_result(df, df.ta.ema(length=26))

    s20, s50, s200 = _last_or_none(sma20), _last_or_none(sma50), _last_or_none(sma200)
    stack_order = None
    if s20 is not None and s50 is not None and s200 is not None:
        if s20 > s50 > s200:
            stack_order = "bullish"
        elif s20 < s50 < s200:
            stack_order = "bearish"
        else:
            stack_order = "mixed"

    e12, e26 = _last_or_none(ema12), _last_or_none(ema26)
    price = close.iloc[-1]

    def _price_vs(sma_last: float | None) -> float | None:
        if sma_last is None or sma_last == 0:
            return None
        return round((price - sma_last) / sma_last * 100, 2)

    return {
        "sma_20": round(s20, 4) if s20 is not None else None,
        "sma_50": round(s50, 4) if s50 is not None else None,
        "sma_200": round(s200, 4) if s200 is not None else None,
        "ema_12": round(e12, 4) if e12 is not None else None,
        "ema_26": round(e26, 4) if e26 is not None else None,
        "stack_order": stack_order,
        "primary_trend": stack_order,  # same signal, referenced by name elsewhere (rsi_zone_adjusted)
        "sma_50_slope": _slope_label(sma50) if sma50 is not None else None,
        "price_vs_sma20_pct": _price_vs(s20),
        "price_vs_sma50_pct": _price_vs(s50),
        "price_vs_sma200_pct": _price_vs(s200),
    }


# ---------- momentum ----------


def _macd_recent_cross(macd_line: pd.Series, signal_line: pd.Series, bars: int = 10) -> str | None:
    diff = (macd_line - signal_line).dropna()
    if len(diff) < 2:
        return None
    window = diff.iloc[-bars:] if len(diff) >= bars else diff
    sign_changes = np.sign(window).diff().dropna()
    crosses = sign_changes[sign_changes != 0]
    if crosses.empty:
        return "none"
    last_cross_sign = np.sign(window.loc[crosses.index[-1]])
    return "bullish_cross" if last_cross_sign > 0 else "bearish_cross"


def _divergence(close: pd.Series, rsi: pd.Series, lookback: int = 20) -> str | None:
    c = close.dropna()
    r = rsi.dropna()
    if len(c) < lookback + 1 or len(r) < lookback + 1:
        return None
    c_win, r_win = c.iloc[-lookback:], r.iloc[-lookback:]
    price_new_high = c_win.iloc[-1] >= c_win.max()
    price_new_low = c_win.iloc[-1] <= c_win.min()
    rsi_new_high = r_win.iloc[-1] >= r_win.max()
    rsi_new_low = r_win.iloc[-1] <= r_win.min()
    if price_new_high and not rsi_new_high:
        return "bearish_divergence"
    if price_new_low and not rsi_new_low:
        return "bullish_divergence"
    return "none"


def _rsi_zone_adjusted(rsi14: float | None, primary_trend: str | None) -> str | None:
    if rsi14 is None or pd.isna(rsi14):
        return None
    uptrend = primary_trend == "bullish"
    if uptrend:
        if rsi14 > 80:
            return "overbought"
        if rsi14 < 40:
            return "oversold"
        return "neutral"
    if rsi14 > 60:
        return "overbought"
    if rsi14 < 20:
        return "oversold"
    return "neutral"


def _momentum(df: pd.DataFrame, primary_trend: str | None) -> dict:
    close = df["close"]
    rsi14 = _ta_result(df, df.ta.rsi(length=14))
    macd = _ta_result(df, df.ta.macd(fast=12, slow=26, signal=9))

    macd_line = macd_signal = macd_hist = None
    if macd is not None:
        macd_cols = list(macd.columns)
        macd_line = macd[[c for c in macd_cols if c.startswith("MACD_")][0]]
        macd_signal = macd[[c for c in macd_cols if c.startswith("MACDs_")][0]]
        macd_hist = macd[[c for c in macd_cols if c.startswith("MACDh_")][0]]

    rsi_last = _last_or_none(rsi14)
    line_last = _last_or_none(macd_line)
    signal_last = _last_or_none(macd_signal)
    hist_last = _last_or_none(macd_hist)

    return {
        "rsi_14": round(rsi_last, 2) if rsi_last is not None else None,
        "rsi_zone_adjusted": _rsi_zone_adjusted(rsi_last, primary_trend),
        "macd_line": round(line_last, 4) if line_last is not None else None,
        "macd_signal": round(signal_last, 4) if signal_last is not None else None,
        "macd_histogram": round(hist_last, 4) if hist_last is not None else None,
        "macd_recent_cross": _macd_recent_cross(macd_line, macd_signal)
        if macd is not None
        else None,
        "divergence": _divergence(close, rsi14) if rsi14 is not None else None,
    }


# ---------- volume ----------


def _volume(df: pd.DataFrame) -> dict:
    volume = df["volume"]
    if len(volume.dropna()) < 20:
        return {"volume_avg_20": None, "volume_ratio_today": None}
    avg20 = volume.rolling(20).mean().iloc[-1]
    today = volume.iloc[-1]
    ratio = round(today / avg20, 2) if avg20 else None
    return {
        "volume_avg_20": round(avg20, 0) if pd.notna(avg20) else None,
        "volume_ratio_today": ratio,
    }


# ---------- Bollinger Bands ----------


def _bollinger(df: pd.DataFrame) -> dict:
    close = df["close"]
    if len(close.dropna()) < 20:
        return {
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "bb_width_pct": None,
            "bb_position": None,
        }

    bb = _ta_result(df, df.ta.bbands(length=20, std=2))
    if bb is None:
        return {
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "bb_width_pct": None,
            "bb_position": None,
        }
    cols = list(bb.columns)
    upper = bb[[c for c in cols if c.startswith("BBU_")][0]]
    mid = bb[[c for c in cols if c.startswith("BBM_")][0]]
    lower = bb[[c for c in cols if c.startswith("BBL_")][0]]

    width_pct = (upper - lower) / mid * 100
    today_width = width_pct.iloc[-1]

    trailing = width_pct.iloc[-126:] if len(width_pct.dropna()) >= 20 else width_pct
    width_percentile = None
    if pd.notna(today_width) and trailing.dropna().shape[0] >= 20:
        width_percentile = (trailing.dropna() <= today_width).mean() * 100

    # squeeze() needs more history than bbands' own length=20 to produce a
    # real result (confirmed live: still returns the identity-df sentinel
    # at exactly 20 rows) — guard separately, don't assume the bbands
    # guard above covers it.
    sqz = _ta_result(df, df.ta.squeeze())
    sqz_on = (
        bool(sqz["SQZ_ON"].iloc[-1])
        if sqz is not None and "SQZ_ON" in sqz.columns and pd.notna(sqz["SQZ_ON"].iloc[-1])
        else False
    )

    walking = False
    if len(close) >= 3:
        recent = close.iloc[-3:]
        recent_upper = upper.iloc[-3:]
        recent_lower = lower.iloc[-3:]
        walking = bool((recent > recent_upper).all() or (recent < recent_lower).all())

    if width_percentile is not None and width_percentile < 20 and sqz_on:
        position = "squeeze"
    elif walking:
        position = "walking_band"
    else:
        position = "mean_reverting"

    return {
        "bb_upper": round(upper.iloc[-1], 4) if pd.notna(upper.iloc[-1]) else None,
        "bb_middle": round(mid.iloc[-1], 4) if pd.notna(mid.iloc[-1]) else None,
        "bb_lower": round(lower.iloc[-1], 4) if pd.notna(lower.iloc[-1]) else None,
        "bb_width_pct": round(today_width, 2) if pd.notna(today_width) else None,
        "bb_width_percentile": round(width_percentile, 1) if width_percentile is not None else None,
        "bb_position": position,
    }


# ---------- ATR / volatility ----------


def _volatility(df: pd.DataFrame) -> dict:
    atr14 = _ta_result(df, df.ta.atr(length=14)) if len(df) >= 14 else None
    if atr14 is None:
        return {"atr_14": None, "atr_60_avg": None, "volatility_regime_derived": None}
    atr_last = _last_or_none(atr14)
    atr60_avg = atr14.rolling(60).mean().iloc[-1] if len(atr14.dropna()) >= 60 else None
    atr60_avg_val = float(atr60_avg) if atr60_avg is not None and pd.notna(atr60_avg) else None

    regime = None
    if atr_last is not None and atr60_avg_val is not None and atr60_avg_val:
        ratio = atr_last / atr60_avg_val
        regime = "expanding" if ratio > 1.2 else "contracting" if ratio < 0.8 else "normal"

    return {
        "atr_14": round(atr_last, 4) if atr_last is not None else None,
        "atr_60_avg": round(atr60_avg_val, 4) if atr60_avg_val is not None else None,
        "volatility_regime_derived": regime,
    }


# ---------- support / resistance ----------


_TOUCH_TOLERANCE_ATR = 0.5
_TOUCH_LOOKBACK = 126  # ~6 months of trading sessions


def _touch_count(
    df: pd.DataFrame, level: float | None, column: str, atr14: float | None
) -> int | None:
    """Sessions in the trailing ~6 months where the relevant intraday
    extreme (low for support, high for resistance) came within 0.5 ATR of
    the level -- a first-pass tolerance/window choice, not specified
    anywhere (same disclosure as every other undocumented threshold in
    this module, e.g. the zigzag deviation)."""
    if level is None or not atr14:
        return None
    window = df.iloc[-_TOUCH_LOOKBACK:] if len(df) > _TOUCH_LOOKBACK else df
    tolerance = atr14 * _TOUCH_TOLERANCE_ATR
    return int((window[column] - level).abs().le(tolerance).sum())


def _support_resistance(df: pd.DataFrame, atr14: float | None) -> dict:
    # Flat shape (nearest_support/pct_to_support/atr_to_support/
    # support_touch_count as sibling keys, not a nested dict) matches the
    # real Technical Analyst prompt's payload template literally
    # ("Support={nearest_support} ({pct_to_support}, {atr_to_support} ATR,
    # touches={support_touch_count})") -- confirmed against the live
    # prompt text, not the ticket's own (less precise) field-list summary.
    empty = {
        "levels": {},
        "nearest_support": None,
        "pct_to_support": None,
        "atr_to_support": None,
        "support_touch_count": None,
        "nearest_resistance": None,
        "pct_to_resistance": None,
        "atr_to_resistance": None,
        "resistance_touch_count": None,
    }
    pivots = _ta_result(df, df.ta.pivots()) if len(df) >= 2 else None
    if pivots is None:
        return empty
    latest = pivots.iloc[-1]
    price = df["close"].iloc[-1]

    levels = {col.split("_")[-1]: latest[col] for col in pivots.columns if pd.notna(latest[col])}
    supports = sorted((lvl for lvl in levels.values() if lvl < price), reverse=True)
    resistances = sorted(lvl for lvl in levels.values() if lvl > price)
    support = supports[0] if supports else None
    resistance = resistances[0] if resistances else None

    def _pct_to(level: float | None) -> float | None:
        return round((price - level) / price * 100, 2) if level is not None else None

    def _atr_to(level: float | None) -> float | None:
        # abs(price - level) / atr14, not a percent round-trip (caught on
        # review: distance_pct / 100 * price algebraically simplifies
        # straight back to price - level, adding nothing).
        return round(abs(price - level) / atr14, 2) if level is not None and atr14 else None

    return {
        "levels": {k: round(v, 4) for k, v in levels.items()},
        "nearest_support": round(support, 4) if support is not None else None,
        "pct_to_support": _pct_to(support),
        "atr_to_support": _atr_to(support),
        "support_touch_count": _touch_count(df, support, "low", atr14),
        "nearest_resistance": round(resistance, 4) if resistance is not None else None,
        "pct_to_resistance": _pct_to(resistance),
        "atr_to_resistance": _atr_to(resistance),
        "resistance_touch_count": _touch_count(df, resistance, "high", atr14),
    }


# ---------- trend structure ----------


def _swing_structure(df: pd.DataFrame, window: int = 20) -> str | None:
    if len(df) < window:
        return None
    # Hard precondition, not a try/except: confirmed live that pandas-ta's
    # zigzag() segfaults the whole process on zero-variance price data
    # (a real, if rare, case — a halted or completely illiquid stock).
    # The crash happens during cleanup after zigzag appears to return
    # successfully, not synchronously inside the call, so no exception
    # handler can catch it — the only real fix is never calling it on
    # degenerate input in the first place.
    if df["close"].std() == 0 or df["high"].max() == df["low"].min():
        return None
    zz = df.ta.zigzag(legs=_ZIGZAG_LEGS, deviation=_ZIGZAG_DEVIATION)
    val_col = [c for c in zz.columns if c.startswith("ZIGZAGv_")][0]
    sign_col = [c for c in zz.columns if c.startswith("ZIGZAGs_")][0]
    points = zz[[val_col, sign_col]].dropna().iloc[-window:]
    points = points[points.index >= df.index[-window]]

    # Real bug caught on review: comparing adjacent swings (a peak vs. the
    # trough right before it, or vice versa) is close to tautological —
    # zigzag swings alternate by construction, so a peak is *always*
    # higher than its immediately preceding trough and a trough *always*
    # lower than its immediately preceding peak. That comparison carries
    # no real HH/HL/LH/LL signal. Standard market-structure classification
    # compares same-type extrema — the latest peak against the *previous*
    # peak (2 swings back, since types alternate), not against the trough
    # in between. Needs 3 points minimum, not 2.
    if len(points) < 3:
        return None
    latest_val, latest_sign = points[val_col].iloc[-1], points[sign_col].iloc[-1]
    prev_same_type_val = points[val_col].iloc[-3]
    is_peak = latest_sign > 0
    if is_peak:
        return "HH" if latest_val > prev_same_type_val else "LH"
    return "HL" if latest_val > prev_same_type_val else "LL"


def _trend_structure(df: pd.DataFrame, sector_etf_price: pd.DataFrame | None) -> dict:
    # _swing_structure() already guards its own minimum length internally
    # (including the empty-DataFrame case) — no need to re-check here
    # before calling it, and the daily call above never did.
    daily = _swing_structure(df, window=20)
    weekly = _swing_structure(_resample_weekly(df), window=20)
    rel = _relative_performance(df, sector_etf_price)
    return {
        "swing_structure_20d": daily,
        "trend_structure_weekly": weekly,
        "rs_vs_sector_3mo": rel["rs_vs_sector_3mo"],
        "rs_leadership": rel["rs_leadership"],
    }


# ---------- multi-timeframe ----------


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample("W").agg(agg).dropna(subset=["close"])


def _multi_timeframe(df: pd.DataFrame) -> dict:
    weekly = _resample_weekly(df)
    if len(weekly) < 20:
        return {"weekly_trend": None, "weekly_rsi_zone": None}
    weekly_ma = _moving_averages(weekly)
    weekly_rsi = _ta_result(weekly, weekly.ta.rsi(length=14))
    rsi_last = _last_or_none(weekly_rsi)
    return {
        "weekly_trend": weekly_ma["stack_order"],
        "weekly_rsi_zone": _rsi_zone_adjusted(rsi_last, weekly_ma["stack_order"]),
    }


# ---------- pattern metrics (MVP proxy, not full geometric recognition) ----------


def _pattern_metrics(bb_position: str | None, swing_structure_20d: str | None) -> dict:
    """Ship the ticket's MVP proxy — squeeze-release + zigzag-swing-direction
    — not named pattern types with measured-move targets. No pandas-ta
    coverage exists for multi-bar chart-pattern recognition (confirmed
    during the ticket's own earlier reconciliation pass); building a
    custom geometric detector is real, open-ended, failure-prone work
    out of scope here."""
    squeeze_release = bb_position == "walking_band"
    breakout_direction = None
    if swing_structure_20d in ("HH", "HL"):
        breakout_direction = "bullish"
    elif swing_structure_20d in ("LL", "LH"):
        breakout_direction = "bearish"
    return {
        "squeeze_release": squeeze_release,
        "breakout_direction": breakout_direction if squeeze_release else None,
    }


# ---------- market context (regime, off the benchmark) ----------


def _market_context(benchmark_df: pd.DataFrame) -> dict:
    bdf = _normalize_ohlcv(benchmark_df)
    sma200 = _ta_result(bdf, bdf.ta.sma(length=200)) if len(bdf) >= 200 else None
    if sma200 is None:
        return {"market_regime": None}
    slope = _slope_label(sma200)
    price = bdf["close"].iloc[-1]
    s200 = _last_or_none(sma200)
    if s200 is None:
        return {"market_regime": None}
    if price > s200 and slope == "rising":
        regime = "bull"
    elif price < s200 and slope == "falling":
        regime = "bear"
    else:
        regime = "choppy"
    return {"market_regime": regime}


# ---------- relative strength ----------


def _return_over_days(series: pd.Series, days: int) -> float | None:
    """% change over ~days calendar days, found by date, not row
    position. Real bug caught on review: comparing two different series
    (stock vs. sector ETF) via `.iloc[-63]` on each assumes both have
    identical trading calendars, which isn't guaranteed — a data gap, a
    halt day, or a slightly different fetched date range would silently
    misalign 'the row 63 positions back' to different actual calendar
    dates on each side, comparing two returns as of two different days
    without knowing it. Same class of bug already fixed elsewhere in
    this codebase via date-keyed lookups instead of positional ones
    (macro_sources.py's _value_n_days_ago, stats_canada.py's exact-refPer
    matching)."""
    if series.empty:
        return None
    cutoff = series.index[-1] - pd.Timedelta(days=days)
    eligible = series[series.index <= cutoff]
    if eligible.empty:
        return None
    past = eligible.iloc[-1]
    if past == 0:
        return None
    return (series.iloc[-1] / past - 1) * 100


def _relative_performance(df: pd.DataFrame, sector_etf_df: pd.DataFrame | None) -> dict:
    if sector_etf_df is None or sector_etf_df.empty:
        return {"rs_vs_sector_3mo": None, "rs_leadership": None}
    etf = _normalize_ohlcv(sector_etf_df)
    stock_return = _return_over_days(df["close"], 90)
    sector_return = _return_over_days(etf["close"], 90)
    if stock_return is None or sector_return is None:
        return {"rs_vs_sector_3mo": None, "rs_leadership": None}
    rs = round(stock_return - sector_return, 2)
    return {"rs_vs_sector_3mo": rs, "rs_leadership": "leading" if rs > 0 else "lagging"}


# ---------- liquidity ----------


def _liquidity_flags(df: pd.DataFrame) -> dict:
    # Real inconsistency caught on review: this module's own docstring
    # states liquidity is v2.2's "soft caveat off raw avg_dollar_volume_20
    # < $1M (no boolean field)" — a deliberate removal of the older
    # below_liquidity_floor flag. A `thin_liquidity` boolean lived here
    # until this fix, which was exactly the kind of pre-decided flag the
    # docstring says not to produce. The LLM reasons from the raw number.
    if len(df) < 20:
        return {"avg_dollar_volume_20": None}
    dollar_vol = (df["close"] * df["volume"]).rolling(20).mean().iloc[-1]
    return {"avg_dollar_volume_20": round(dollar_vol, 0) if pd.notna(dollar_vol) else None}


# ---------- earnings proximity ----------


def _earnings_proximity(earnings_calendar: list[dict] | None, as_of: date) -> dict:
    if not earnings_calendar:
        return {"next_earnings_date": None, "earnings_proximity_days": None}
    future_dates = []
    for row in earnings_calendar:
        raw = row.get("date")
        if not raw:
            continue
        try:
            d = pd.Timestamp(raw).date()
        except (ValueError, TypeError):
            continue
        if d >= as_of:
            future_dates.append(d)
    if not future_dates:
        return {"next_earnings_date": None, "earnings_proximity_days": None}
    next_date = min(future_dates)
    trading_days = np.busday_count(as_of, next_date)
    return {
        "next_earnings_date": next_date.isoformat(),
        "earnings_proximity_days": int(trading_days),
    }


# ---------- freshness ----------


def _freshness(df: pd.DataFrame, as_of: date) -> dict:
    if df.empty:
        return {"is_current": False, "days_old": 0}
    last_bar_date = df.index[-1].date()
    trading_days_old = int(np.busday_count(last_bar_date, as_of))
    return {"is_current": trading_days_old <= 0, "days_old": max(trading_days_old, 0)}


# ---------- price position (52-week high/low) ----------


def _price_position(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {
            "high_52w": None,
            "low_52w": None,
            "pct_from_52w_high": None,
            "pct_from_52w_low": None,
        }
    # Real bug caught on review: 52-week high/low must come from the
    # intraday high/low columns, not the closing price — using close
    # alone systematically understates the real high and overstates the
    # real low, and doesn't match what any real consumer of "52-week
    # high/low" means by the term.
    window = df.iloc[-252:]
    high_52w, low_52w = window["high"].max(), window["low"].min()
    price = df["close"].iloc[-1]
    # `!= 0`, not truthiness (`if high_52w else`) — caught on review: a
    # bare truthiness check treats a literal $0 high/low as "no data"
    # (falsy) rather than as a real, if degenerate, value — inconsistent
    # with the explicit zero-checks used everywhere else in this file
    # (_price_vs, _return_over_days).
    return {
        "high_52w": round(high_52w, 4),
        "low_52w": round(low_52w, 4),
        "pct_from_52w_high": round((price - high_52w) / high_52w * 100, 2)
        if high_52w != 0
        else None,
        "pct_from_52w_low": round((price - low_52w) / low_52w * 100, 2) if low_52w != 0 else None,
    }


# ---------- pre-flight anomalies ----------


def _preflight_warnings(df: pd.DataFrame, news_ids: list[dict] | None) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        return warnings

    zero_volume_days = int((df["volume"] == 0).sum()) if "volume" in df.columns else 0
    if zero_volume_days:
        warnings.append(f"{zero_volume_days} zero-volume session(s) in the price history")

    non_positive = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if non_positive:
        warnings.append(f"{non_positive} session(s) with a negative/zero price")

    if len(df) >= 2:
        prior_close = df["close"].shift(1)
        gap_pct = (df["open"] - prior_close) / prior_close * 100
        gap_days = df.index[gap_pct.abs() > 25]
        if len(gap_days):
            news_dates = set()
            if news_ids:
                news_dates = {
                    pd.Timestamp(item["date"]).date() for item in news_ids if item.get("date")
                }
            for gap_date in gap_days:
                d = gap_date.date()
                if news_ids and d in news_dates:
                    continue
                qualifier = "" if news_ids is None else " with no corresponding news item"
                warnings.append(f">25% gap on {d.isoformat()}{qualifier}")

    return warnings


# ---------- entry point ----------


def compute_all(
    raw_price: pd.DataFrame,
    benchmark_price: pd.DataFrame,
    sector_etf_price: pd.DataFrame | None,
    earnings_calendar: list[dict] | None,
    news_ids: list[dict] | None = None,
    as_of: date | None = None,
) -> dict:
    """Assembles every technical field the Technical Analyst agent
    consumes from raw OHLCV price data. Pure function — caching, fetching,
    and sector->ETF/benchmark resolution are the caller's job."""
    as_of_date = as_of if as_of is not None else pd.Timestamp.now().date()
    df = _normalize_ohlcv(raw_price)

    if df.empty:
        empty_freshness = {"is_current": False, "days_old": 0}
        return {
            "technical_indicators": {},
            "support_resistance": {},
            "trend_structure": {},
            "multi_timeframe": {},
            "pattern_metrics": {},
            "market_context": {},
            "liquidity_flags": {},
            "earnings_proximity": _earnings_proximity(earnings_calendar, as_of_date),
            "price_position": {},
            "is_current": empty_freshness["is_current"],
            "days_old": empty_freshness["days_old"],
            "preflight_warnings": ["no price history available"],
        }

    ma = _moving_averages(df)
    momentum = _momentum(df, ma["primary_trend"])
    volume = _volume(df)
    bollinger = _bollinger(df)
    volatility = _volatility(df)

    technical_indicators = {**ma, **momentum, **volume, **bollinger, **volatility}

    support_resistance = _support_resistance(df, volatility["atr_14"])
    trend_structure = _trend_structure(df, sector_etf_price)
    multi_timeframe = _multi_timeframe(df)
    pattern_metrics = _pattern_metrics(
        bollinger["bb_position"], trend_structure["swing_structure_20d"]
    )
    market_context = _market_context(benchmark_price)
    liquidity_flags = _liquidity_flags(df)
    earnings_proximity = _earnings_proximity(earnings_calendar, as_of_date)
    freshness = _freshness(df, as_of_date)
    price_position = _price_position(df)
    preflight_warnings = _preflight_warnings(df, news_ids)

    return {
        "technical_indicators": technical_indicators,
        "support_resistance": support_resistance,
        "trend_structure": trend_structure,
        "multi_timeframe": multi_timeframe,
        "pattern_metrics": pattern_metrics,
        "market_context": market_context,
        "liquidity_flags": liquidity_flags,
        "earnings_proximity": earnings_proximity,
        "price_position": price_position,
        "is_current": freshness["is_current"],
        "days_old": freshness["days_old"],
        "preflight_warnings": preflight_warnings,
    }
