"""Pass 1 — Technical Analyst runner (v2.2).

Production port of `simulation/runners/pass1_technical_analyst.py` (86bbuhjup).
NOT a clean port -- field-by-field notes, verified against
`data/precompute/technicals.py::compute_all()` directly:

- Technical Analyst's DataBundle coverage is genuinely GOOD -- almost every
  harness fixture field has a real, matching precompute field, including
  several the harness's own runner had marked "Not present in this fixture's
  schema" and hardcoded to `"N/A"` (`rs_leadership`, `rsi_zone_adjusted`,
  `weekly_rsi_zone`): all three are real fields on `trend_structure`/
  `technical_indicators`/`multi_timeframe` today. The harness fixture is
  simply older than the current precompute module -- this port uses the real
  values, not the harness's stale placeholders.
- `earnings_proximity_days`/`next_earnings_date` live on `bundle.earnings_proximity`
  for real (not `fundamental_data`, where the harness fixture put them) --
  cross-agent field-location difference, not a data gap.
- `avg_dollar_volume_20` lives on `bundle.liquidity_flags` directly, matching
  the harness fixture's own field name and meaning exactly.
- `beta`/`ytd_return_pct`: same as pass1_stock_researcher.py/
  pass1_fundamental_analyst.py -- beta comes from `bundle.risk_metrics`, no
  precompute module anywhere computes ytd_return_pct (genuine gap, omitted).
- `volume_trend` (harness, a labeled trend) has no real equivalent --
  `technical_indicators.volume_ratio_today` (today's volume / 20d average, a
  ratio not a label) is the real, closest field and is rendered instead.
- `pattern_confirmed`/`confluence_score` (harness fixture): no precompute
  field produces either. `confluence_score` in particular reads like an LLM
  interpretive judgment synthesizing TREND/MOMO/VOL/SR/PAT, the same
  "give raw data, let the LLM interpret" pattern as Fundamental's
  guidance_vs_consensus -- omitted from input, not fabricated.

86bbummwp Tier 1 additions:
- `data_coverage_line`: now built from the same `field_presence` map
  `build_user_message()` already computes for `input_field_coverage`, via the
  shared `render_data_coverage_line()` helper -- was hardcoded to
  "Data coverage: standard." on every run before this.
- `data_warnings`: now wired to `bundle.preflight_warnings` (real, computed by
  `precompute/technicals.py`'s own anomaly detector -- zero-volume sessions,
  negative/zero prices, unexplained >25% gaps -- and previously discarded
  after being persisted onto DataBundle for no live reader).
- The retry loop's validator now also enforces the earnings-proximity and
  thin-volume caveats the user message already tells the model are
  mandatory/required (see `earnings_flag`/`volume_flag` below) -- previously
  requested in the prompt but never mechanically checked.

86bbummwp Tier 2 additions -- D6's mechanical flags, now real, stored fields
on `agent_outputs`, not just rendered into the prompt:
- `data_coverage`: the same `field_presence` map, reshaped into
  `{"present": [...], "absent": [...]}` via `to_data_coverage()`.
- `anomalies`: `bundle.preflight_warnings` persisted as-is -- the same list
  already rendered into `data_warnings` above, now also a queryable,
  structured fact about this run instead of only ever a prompt string.
- `stale_data`: real threshold on `bundle.days_old`, not `bundle.is_current`
  -- checked `_freshness()`'s own definition directly
  (`precompute/technicals.py`): `is_current` is `trading_days_old <= 0`,
  strictly "does the data include the most recent trading day," which would
  false-positive on ordinary fetch-timing variance (e.g. fetched the morning
  before the new bar posts). `days_old > 5` (roughly a trading week) is a
  meaningful threshold for a genuinely daily series; `is_current` is not used
  here at all.
"""
from functools import partial

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import render_data_coverage_line, render_data_warnings, to_data_coverage
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import (
    validate_earnings_proximity_caveat,
    validate_technical_analyst,
    validate_thin_volume_caveat,
)
from data.schemas.data_bundle import DataBundle

_STALE_PRICE_DAYS = 5

# Only 4 fields, all worth a coverage-line mention directly -- unlike
# Fundamental Analyst's 15-key missing_fields map, no coarse/granular split
# needed here.
_COVERAGE_GAP_SENTENCES = {
    "earnings_proximity": "no earnings calendar data available",
    "weekly_timeframe": "no weekly timeframe data available (insufficient price history)",
    "sector_relative_strength": "no sector relative strength data available",
    "support_resistance": "no support/resistance levels could be resolved",
}


def _data_coverage_line(field_presence: dict[str, bool]) -> str:
    return render_data_coverage_line(field_presence, _COVERAGE_GAP_SENTENCES)


def _stale_data(days_old: int) -> list[str]:
    """86bbummwp Tier 2 -- see the module docstring for why this uses
    `days_old` and not `is_current`."""
    return ["price"] if days_old > _STALE_PRICE_DAYS else []


def _fmt(v, suffix: str = ""):
    return "N/A" if v is None else f"{v}{suffix}"


def _validate_with_caveats(
    output: dict,
    earnings_days: int | None,
    avg_dollar_vol: float | None,
    material_absent: list[str],
    anomalies: list[str],
    stale_data: list[str],
) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp Tier 1c, extended by the follow-on
    confidence/data-quality coupling rule) -- merges the base schema check
    with the mandatory-caveat checks `call_with_validation()` (agents/base.py)
    has no way to receive extra context for on its own: its `validator`
    callable takes exactly one positional arg (`result`), so every extra
    param here is closed over via `partial()` at the call site instead. The
    earnings/thin-volume checks already reflect a requirement the real prompt
    already tells the model about (`earnings_flag`/`volume_flag` in
    build_user_message() above) -- this only makes an already-visible
    instruction mechanically enforced, not a new one. The new confidence/
    data-quality rule is the backstop for a real, live case (Macro Economist
    claiming `analysis_confidence: "high"` while its own stale_data flag
    showed 4 stale series) -- see `validate_confidence_requires_caveat_when_flagged`'s
    own docstring.

    `earnings_days`/`avg_dollar_vol` can be `None` when the underlying data
    itself is absent (no earnings calendar entry, no liquidity data) -- both
    caveat checks require a real number to compare against a threshold, so
    `None` skips that specific check the same way "value present but above
    threshold" does (there's no proximity/thin-volume condition to flag when
    the fact needed to evaluate it isn't known either).
    """
    passed, errors = validate_technical_analyst(output)
    if earnings_days is not None:
        ep_passed, ep_errors = validate_earnings_proximity_caveat(output, earnings_days)
        passed = passed and ep_passed
        errors = errors + ep_errors
    if avg_dollar_vol is not None:
        vol_passed, vol_errors = validate_thin_volume_caveat(output, avg_dollar_vol)
        passed = passed and vol_passed
        errors = errors + vol_errors
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=output.get("analysis_confidence") == "high",
        material_absent=material_absent,
        anomalies=anomalies,
        stale_data=stale_data,
    )
    return passed and cq_passed, errors + cq_errors


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """Returns (rendered user message, field presence map) -- the second
    element is 86bbwachy Phase 4's own new addition. Unlike
    pass1_stock_researcher.py's per-block helpers, this agent's whole
    payload is one f-string with ~30 individually-formatted metrics --
    restructuring every one into its own RenderedField-returning helper
    would be a much bigger, riskier change for little real benefit (most
    of these are plain indicator/quote passthroughs, virtually always
    present once a ticker resolves at all). Presence is computed directly
    from the same underlying bundle values _fmt() already reads below, not
    re-derived from the rendered text afterward -- only for the fields
    confirmed (against technicals.py's own compute_all()) to have a real,
    meaningful sometimes-absent case: weekly timeframe data needs >=20
    weeks of history, sector relative strength needs a resolvable sector
    ETF, earnings proximity needs a real calendar, support/resistance
    needs enough price history for pivots to resolve at all.
    """
    ctx = bundle.context
    company_info = bundle.company_info
    ti = bundle.technical_indicators
    sr = bundle.support_resistance
    ts = bundle.trend_structure
    mtf = bundle.multi_timeframe
    pm = bundle.pattern_metrics
    pp = bundle.price_position
    liq = bundle.liquidity_flags
    ep = bundle.earnings_proximity

    avg_dollar_vol = liq.get("avg_dollar_volume_20")
    avg_dollar_vol_str = f"${avg_dollar_vol:,.0f}" if avg_dollar_vol is not None else "N/A"
    earnings_days = ep.get("earnings_proximity_days")

    earnings_flag = ""
    if earnings_days is not None and earnings_days <= 5:
        earnings_flag = f"\n⚠️ EARNINGS IN {earnings_days} DAY(S) — mandatory earnings proximity caveat required."

    volume_flag = ""
    if avg_dollar_vol is not None and avg_dollar_vol < 1_000_000:
        volume_flag = f"\n⚠️ THIN VOLUME: avg_dollar_volume_20=${avg_dollar_vol:,.0f} (<$1M) — soft thin-volume caveat required."

    text = f"""{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}{earnings_flag}{volume_flag}

EARNINGS PROXIMITY: {_fmt(earnings_days, ' days')}

PRICE DATA:
  Current: {bundle.price_info.get('current_price')} {bundle.stock.currency}
  52w High: {_fmt(pp.get('high_52w'))} | 52w Low: {_fmt(pp.get('low_52w'))}
  % from 52w high: {_fmt(pp.get('pct_from_52w_high'), '%')} | % from 52w low: {_fmt(pp.get('pct_from_52w_low'), '%')}
  Beta: {_fmt(bundle.risk_metrics.get('beta'))}

VOLUME (VOL):
  Avg daily volume (20d): {_fmt(ti.get('volume_avg_20'))} shares
  Avg dollar volume (20d): {avg_dollar_vol_str}
  Volume today: {_fmt(ti.get('volume_today'))} | Volume ratio vs 20d avg: {_fmt(ti.get('volume_ratio_today'))}

TREND (TREND):
  Stack order: {_fmt(ti.get('stack_order'))} | SMA20/50/200 slope: {_fmt(ti.get('sma_50_slope'))}/{_fmt(ti.get('sma_200_slope'))}
  Price vs SMA20/50/200: {_fmt(ti.get('price_vs_sma20_pct'), '%')}/{_fmt(ti.get('price_vs_sma50_pct'), '%')}/{_fmt(ti.get('price_vs_sma200_pct'), '%')}
  Swing structure (20d): {_fmt(ts.get('swing_structure_20d'))} | Weekly: {_fmt(ts.get('trend_structure_weekly'))}
  RS vs sector (3mo): {_fmt(ts.get('rs_vs_sector_3mo'), '%')} | Leadership: {_fmt(ts.get('rs_leadership'))}
  Weekly trend: {_fmt(mtf.get('weekly_trend'))}

MOMENTUM (MOMO):
  RSI(14): {_fmt(ti.get('rsi_14'))} | RSI zone (adjusted): {_fmt(ti.get('rsi_zone_adjusted'))}
  MACD line/signal/histogram: {_fmt(ti.get('macd_line'))}/{_fmt(ti.get('macd_signal'))}/{_fmt(ti.get('macd_histogram'))}
  MACD recent cross: {_fmt(ti.get('macd_recent_cross'))} | Divergence: {_fmt(ti.get('divergence'))}
  Weekly RSI zone: {_fmt(mtf.get('weekly_rsi_zone'))}

SUPPORT / RESISTANCE (SR):
  Nearest support: {_fmt(sr.get('nearest_support'))} ({_fmt(sr.get('pct_to_support'), '%')}, {_fmt(sr.get('atr_to_support'))} ATR, touches={_fmt(sr.get('support_touch_count'))})
  Nearest resistance: {_fmt(sr.get('nearest_resistance'))} ({_fmt(sr.get('pct_to_resistance'), '%')}, {_fmt(sr.get('atr_to_resistance'))} ATR, touches={_fmt(sr.get('resistance_touch_count'))})

VOLATILITY (VOLA):
  ATR(14): {_fmt(ti.get('atr_14'))} | ATR(60d avg): {_fmt(ti.get('atr_60_avg'))}
  Volatility regime: {_fmt(ti.get('volatility_regime_derived'))}

PATTERN (PAT):
  Bollinger position: {_fmt(ti.get('bb_position'))} | Width pct: {_fmt(ti.get('bb_width_pct'), '%')}
  Squeeze release: {pm.get('squeeze_release', False)} | Breakout direction: {_fmt(pm.get('breakout_direction'))}

MARKET REGIME:
  {_fmt(bundle.market_context.get('market_regime'))}"""

    field_presence = {
        "earnings_proximity": earnings_days is not None,
        "weekly_timeframe": mtf.get("weekly_trend") is not None,
        "sector_relative_strength": ts.get("rs_leadership") is not None,
        # nearest_support/nearest_resistance always go missing together --
        # _support_resistance() returns its whole empty dict as one unit
        # when pivots can't resolve at all -- so checking one is a
        # sufficient, accurate proxy for the pair, unlike Stock
        # Researcher's dividend yield/payout ratio (genuinely independent).
        "support_resistance": sr.get("nearest_support") is not None,
    }
    return text, field_presence


class TechnicalAnalystRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "TECH"
        ctx = bundle.context
        sr = bundle.support_resistance
        ts = bundle.trend_structure
        mtf = bundle.multi_timeframe
        ti = bundle.technical_indicators
        ep = bundle.earnings_proximity
        liq = bundle.liquidity_flags
        earnings_days = ep.get("earnings_proximity_days")
        avg_dollar_vol = liq.get("avg_dollar_volume_20")
        user_msg, field_presence = build_user_message(bundle)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        # 86bbummwp Tier 2 -- D6's mechanical flags, set here for the same
        # reason as last_field_coverage above (persisted regardless of
        # whether the LLM call itself succeeds).
        self.last_data_coverage = to_data_coverage(field_presence, _COVERAGE_GAP_SENTENCES)
        self.last_anomalies = list(bundle.preflight_warnings)
        self.last_stale_data = _stale_data(bundle.days_old)
        system_prompt = fill(
            load_template("technical_analyst"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": _data_coverage_line(field_presence),
                "data_warnings": render_data_warnings(self.last_anomalies, self.last_stale_data),
                "memory_brief": "",
                "earnings_proximity_days": str(ep.get("earnings_proximity_days", "N/A")),
                "nearest_support": str(sr.get("nearest_support", "N/A")),
                "nearest_resistance": str(sr.get("nearest_resistance", "N/A")),
                "weekly_trend": str(mtf.get("weekly_trend", "N/A")),
                "rs_leadership": str(ts.get("rs_leadership", "N/A")),
                "rsi_zone_adjusted": str(ti.get("rsi_zone_adjusted", "N/A")),
                "volatility_regime_derived": str(ti.get("volatility_regime_derived", "N/A")),
                "weekly_rsi_zone": str(mtf.get("weekly_rsi_zone", "N/A")),
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(
                _validate_with_caveats,
                earnings_days=earnings_days,
                avg_dollar_vol=avg_dollar_vol,
                material_absent=self.last_data_coverage["absent"],
                anomalies=self.last_anomalies,
                stale_data=self.last_stale_data,
            ),
            max_tokens=3500,
            temperature=0.3,
        )
