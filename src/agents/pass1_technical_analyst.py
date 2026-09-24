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
"""
from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.validators.pass1 import validate_technical_analyst
from data.schemas.data_bundle import DataBundle


def _fmt(v, suffix: str = ""):
    return "N/A" if v is None else f"{v}{suffix}"


def build_user_message(bundle: DataBundle) -> str:
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

    return f"""{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
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


class TechnicalAnalystRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "TECH"
        ctx = bundle.context
        sr = bundle.support_resistance
        ts = bundle.trend_structure
        mtf = bundle.multi_timeframe
        ti = bundle.technical_indicators
        ep = bundle.earnings_proximity
        user_msg = build_user_message(bundle)
        system_prompt = fill(
            load_template("technical_analyst"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": "Data coverage: standard.",
                "data_warnings": "",
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
            validate_technical_analyst,
            max_tokens=3500,
            temperature=0.3,
        )
