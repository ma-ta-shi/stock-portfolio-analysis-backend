**Agent**: Technical Analyst
**Status:** Complete for Phase 1
**Phase:** Phase 1
**Pass**: Pass 1 (Research & Analysis)
**Role**: Price-action analysis — trend, momentum, volume confirmation, support/resistance, volatility regime, and chart patterns
**Prompt Version**: v2.2
**Target Model**: `qwen3.6:35b-a3b` via Ollama (non-thinking mode — verify `enable_thinking:false` fully suppresses reasoning on this model; see Agent Design Patterns Pattern 10 caveat)
**Token Budget**: \~390 system \| \~470 data \| \~430 output \| \~1,290 total
---
# Design Decisions
Mirrors the Fundamental Analyst architecture for consistency across Pass 1 agents. See Agent Design Patterns for the canonical patterns this prompt applies. The audience for this system is Canadian retail investors with portfolios \< \$250k CAD across TFSA / RRSP / Trading / general accounts; the design choices below reflect that.
1. **Pre-computed indicators only.** The data pipeline computes every indicator (SMAs, EMAs, RSI, MACD, ATR, Bollinger, volume ratios, S/R levels, pattern detections). The LLM interprets — it never recomputes from OHLCV.
2. **LLM produces interpretive fields only** (Pattern 1). Orchestrator passes through all numeric indicator values from `DataBundle`. The LLM produces only fields requiring judgment — trend classification, signal quality, pattern conviction.
3. **Hybrid reliability scoring** (Pattern 2). Orchestrator pre-computes `base_reliability_score` from mechanical checks (history depth, data freshness, indicator availability). LLM adjusts ±15pts with justification in caveats.
4. **Orchestrator-derived quality enums.** `data_quality_assessment`, `data_completeness`, and `data_freshness` are computed by the orchestrator. The LLM produces only `analysis_confidence` and `caveats`.
5. **Trend-adjusted RSI zones.** Fixed-threshold RSI (70/30) generates false bearish signals in strong uptrends. The orchestrator pre-classifies `rsi_zone_adjusted` using 80/40 in uptrends and 60/20 in downtrends. The LLM consumes the classification, not the raw threshold.
6. **Momentum split into zone + direction.** Two separate fields — `momentum_zone` (oversold/neutral/overbought) and `momentum_direction` (improving/flat/deteriorating) — so the model can distinguish “overbought but still strengthening” from “overbought and rolling over.”
7. **Confluence as a structured field.** `confluence_score` (0–3, count of independent dimensions — momentum, volume, S/R-proximity — agreeing with `primary_trend`) is an LLM output. Pass 2 can filter on it without parsing prose.
8. **Invalidation level as structured output.** `suggested_invalidation_level` (price below which the technical thesis breaks) is an interpretive field so Pass 2 can read it directly instead of parsing it out of narrative.
9. **6-step CoT.** TREND → MOMENTUM → VOLUME+CONFIRMATION → LEVELS+VOLATILITY → PATTERN → SYNTHESIS.
10. **Bounded output arrays** (Pattern 7). `key_factors` 1–4 items, `risks` 1–3 items. Sentiment values are `positive|negative|neutral` per the base contract.
11. **Citation tokens** (Pattern 8). 7 tokens: `TREND`, `MOMO`, `VOL`, `SR`, `VOLA`, `PAT`, `REGIME`. (v2.2 removed `MTF` after dropping intraday context — weekly is now cited inline as part of TREND.) Every `key_factors`/`risks` evidence field must begin with one of these tokens.
12. **Mandatory earnings caveat.** TA setups within 5 trading days of earnings are gambling. Orchestrator flags `earnings_proximity_days`; prompt requires an earnings caveat and caps `analysis_confidence` at medium when ≤5.
13. **Soft thin-volume caveat.** Below \~\$1M/day average dollar volume, prompt requires a thin-volume caveat. Retail at this portfolio scale rarely encounters meaningfully illiquid TSX/NYSE/NASDAQ names, so this is a soft guard rather than a hard mandatory-caveat-or-fail rule (changed from v2.1’s `below_liquidity_floor` machinery).
14. **Conditional injection by timeline** (Pattern 4). Short-term weights momentum, breakouts, RSI extremes; medium-term balances trend with momentum quality; long-term emphasizes 200-day position and weekly/monthly structure.
15. **Account injection is light.** TA is largely account-agnostic. The four account types (tfsa/rrsp/trading/general) produce slightly different invalidation framing, not different analysis.
16. **Sector injection** (Pattern 9). Stock’s sector is in the header so the model can make sector-appropriate judgments (REIT volatility norms differ from biotech norms).
17. **Daily + weekly only.** v2.2 drops intraday (1h) timeframe entirely — retail at this scale doesn’t manage entries off 1h bars even on short-term setups. Daily is primary; weekly is injected as a compact pre-classified summary inside the TREND block.
18. **Volatility regime pre-derived.** `volatility_regime_derived` (expanding/contracting/normal) is computed by the orchestrator from `atr_14` vs `atr_60_avg`.
19. **Anomaly flags flow to structured output** (Pattern 5). Suspicious data (zero-volume bars, gap \>25% without news flag, negative prices) adjusts `reliability_score` and appears in `caveats`.
20. **Brief evidence citations** (Pattern 8). `key_factors`/`risks` evidence uses compact references (e.g., “MOMO: RSI=72, zone=overbought”), not sentences.
21. **Non-thinking mode** (Pattern 10). `enable_thinking: false` for deterministic JSON output. On `qwen3.6:35b-a3b`, verify this actually suppresses reasoning output before trusting it (see Pattern 10 caveat) — `qwen3:14b`'s clean suppression is not guaranteed to carry over.
22. **No buy/sell recommendations.** Pass 2 synthesizes; Pass 1 describes and assesses.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- Daily price history — openbb-tmx `.equity.price.historical` (CA) / FMP `/historical-price-full` (US)
- Technical indicators — pandas-ta, computed locally (CA) / pandas-ta, computed locally (US)
- Benchmark (TSX) — yfinance `^GSPTSE` history (CA)
- Benchmark (S&P 500) — FMP `/historical-price-full/^GSPC` (US)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- Trend structure — SMA 20/50/200 with stack order and slopes, 20-session swing structure, weekly trend structure (HH/HL), sector relative strength (3mo) and leadership classification
- Momentum — RSI-14 plus trend-adjusted zone classification (80/40 in uptrends, 60/20 in downtrends), MACD line/signal/histogram with recent cross, price-vs-momentum divergence flag
- Volume confirmation — today's volume vs 20-day average (ratio), 20-day average dollar volume for the liquidity/earnings-proximity gate
- Volatility regime — ATR-14 vs ATR-60 average, pre-classified as expanding/contracting/normal; Bollinger Band upper/mid/lower, width percentile, and position
- Support/resistance — nearest support and resistance levels in % and ATR-multiple distance, with historical touch counts at each
- Pattern detection — detected chart patterns with confidence scores (≥0.6 used) and measured-move target if confirmed
- Weekly context — pre-classified weekly trend and weekly RSI zone, injected inline into the TREND block
- Earnings proximity and broad market regime (bull/bear/choppy) — gate mandatory caveats and provide index-level context
- `base_reliability_score` — computed by the orchestrator from bar count, core-indicator availability, data freshness, and anomaly count, before the LLM call
---
# System Prompt
Invoke with `enable_thinking: false`. The `build_messages()` method renders this template.

**Runtime prompt:** see [`prompts/technical_analyst/v2.2.txt`](../../prompts/technical_analyst/v2.2.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`):
<table header-row="true">
<colgroup>
<col>
<col width="570">
</colgroup>
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks). Weight RSI extremes, MACD crosses, breakouts/breakdowns, volume spikes. Daily trend matters; weekly is context. Ignore 200-day unless price is within 3% of it.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Balance trend (50-day slope, weekly higher highs) with momentum quality. Pattern completion + measured-move targets matter. Watch for 50/200 SMA crosses.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). Emphasize 200-day SMA position and slope, weekly structure, multi-year S/R, major pattern completions. Ignore RSI noise <2 weeks. Volume regime over months matters more than any single session.`</td>
</tr>
</table>
**Account** (inject based on `AnalysisContext.account_type`):
<table header-row="true">
<colgroup>
<col>
<col width="570">
</colgroup>
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Long-hold bias typical; frame invalidation as a thesis-break price, not a day-trade stop.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. Long-hold bias typical; weight primary trend over short-term signals when synthesizing.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading. Shorter-horizon framing acceptable. Invalidation level still expressed as a concrete price.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General. Note which signals matter for swing vs position vs investor framing.`</td>
</tr>
</table>
---
# Output Schema (LLM-produced fields only)
The orchestrator populates numeric passthrough fields and derived enum fields from `DataBundle`. The LLM produces **only** the interpretive fields below.
```javascript
{
  "assessment_summary": "≤80 words. Most important technical finding first.",
  "reliability_score": 0-100,
  "analysis_confidence": "high|medium|low",
  "caveats": ["specific data gaps, anomalies, signal conflicts, earnings proximity, thin volume"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "positive|negative|neutral", "evidence": "brief citation starting with a valid block token"}
  ],
  "risks": [
    {"risk": "name", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "brief citation starting with a valid block token"}
  ],
  "narrative": "120-180 word analysis following steps 1-6. Reference specific indicator values.",
  "interpretive_fields": {
    "primary_trend": "uptrend|downtrend|sideways",
    "trend_strength": "strong|moderate|weak",
    "momentum_zone": "oversold|neutral|overbought",
    "momentum_direction": "improving|flat|deteriorating",
    "momentum_divergence": "bullish|bearish|none",
    "volume_confirmation": "confirming|diverging|inconclusive",
    "nearest_level_bias": "near_support|near_resistance|midrange",
    "pattern_signal": "bullish|bearish|neutral|none",
    "pattern_confirmed": false,
    "confluence_score": 0,
    "suggested_invalidation_level": 0.0
  }
}
```
**v2.2 contract notes:**
- Evidence fields must begin with a valid block token (`TREND`, `MOMO`, `VOL`, `SR`, `VOLA`, `PAT`, `REGIME`).
- `pattern_confirmed` (bool) replaces v2.1’s `pattern_stage` (3-state enum).
- `MTF` token removed.
## Fields the orchestrator populates (NOT produced by LLM)
**Numeric passthrough**, sourced from `DataBundle`'s real field groups (fixed 2026-08-07, contract-vs-consumer audit — the flat list below previously named a `TechnicalDataBundle` type that never existed on the schema; every field named here lives inside one of `DataBundle`'s dict fields, not as a top-level attribute):
- `bundle.technical_indicators`: `sma_20`, `sma_50`, `sma_200`, `ema_12`, `ema_26`, `rsi_14`, `rsi_zone_adjusted`, `macd_line`, `macd_signal`, `macd_histogram`, `macd_recent_cross`, `atr_14`, `atr_60_avg`, `volatility_regime_derived`, `bb_upper`, `bb_middle`, `bb_lower`, `bb_width_pct`, `bb_width_percentile`, `bb_position`, `volume_avg_20`, `volume_ratio_today`
- `bundle.support_resistance`: `nearest_support`, `nearest_resistance`, `support_touch_count`, `resistance_touch_count`
- `bundle.price_position`: `pct_from_52w_high`, `pct_from_52w_low`
- `bundle.trend_structure`: `swing_structure_20d`, `trend_structure_weekly`, `rs_vs_sector_3mo`, `rs_leadership`
- `bundle.multi_timeframe`: `weekly_trend`, `weekly_rsi_zone`
- `bundle.pattern_metrics`: `detected_patterns[]`, `pattern_target`
- `bundle.earnings_proximity`: `earnings_proximity_days`
- `bundle.liquidity_flags`: `avg_dollar_volume_20`
- `bundle.market_context`: `market_regime`
**Removed in v2.2:** `obv`, `obv_trend`, `distribution_days_25`, `accumulation_days_25`, `index_distribution_days_25`, `intraday_trend`, `multi_timeframe_alignment_derived`, `unfilled_gaps_nearby`, `below_liquidity_floor`, `pattern_stage`.
**Derived enums** (orchestrator computes):
- `data_quality_assessment`: from `reliability_score` — ≥70=“high”, 40-69=“medium”, 20-39=“low”, \<20=“insufficient”
- `reliability_factors.data_completeness`: from indicator-availability count (0 missing=“complete”, 1-2=“mostly_complete”, 3-4=“partial”, 5+=“minimal”)
- `reliability_factors.data_freshness`: from last bar timestamp vs market close (0-1 market days=“current”, 2-5=“slightly_stale”, 6-10=“stale”, 10+=“very_stale”)
**Metadata**: `agent_name`, `agent_pass`, `analysis_context`, `data_sources_used`.
---
# Orchestrator Pre-computation: `base_reliability_score`
```python
def compute_base_reliability(bundle: DataBundle) -> tuple[int, str]:
    # Fixed 2026-08-07 (contract-vs-consumer audit): `bundle.daily_bars`
    # bar-count check removed entirely — raw price history has no home
    # anywhere in DataBundle by design (the pipeline pre-computes every
    # indicator; raw OHLCV never reaches the LLM layer, so there's nothing
    # to expose a bar count from without adding a new schema field).
    # Revisit if this turns out to matter once precompute/technicals.py
    # exists. `bundle.sma_20` etc. below are now dict-key reads —
    # DataBundle.technical_indicators is a dict, not flat attributes.
    score = 100
    reasons = []

    # v2.2: dropped OBV from core indicators
    core_indicators = [
        bundle.technical_indicators["sma_20"], bundle.technical_indicators["sma_50"],
        bundle.technical_indicators["rsi_14"], bundle.technical_indicators["macd_line"],
        bundle.technical_indicators["atr_14"], bundle.technical_indicators["bb_width_pct"],
    ]
    null_count = sum(1 for i in core_indicators if i is None)
    if null_count >= 3:
        score -= 30; reasons.append(f"{null_count}/6 core indicators missing")
    elif null_count >= 2:
        score -= 15; reasons.append(f"{null_count}/6 core indicators missing")
    elif null_count >= 1:
        score -= 5; reasons.append(f"{null_count}/6 core indicators missing")

    if not bundle.is_current:
        if bundle.days_old > 5:
            score -= 25; reasons.append(f"last bar{bundle.days_old} market days old")
        elif bundle.days_old > 2:
            score -= 10; reasons.append(f"last bar{bundle.days_old} market days old")

    anomaly_count = len(bundle.preflight_warnings)
    if anomaly_count >= 3:
        score -= 15; reasons.append(f"{anomaly_count} data anomalies")
    elif anomaly_count >= 1:
        score -= 5; reasons.append(f"{anomaly_count} data anomaly")

    # Fixed: bundle.weekly_summary had no contract equivalent anywhere —
    # multi_timeframe is the real DataBundle field (dict, holds
    # weekly_trend/weekly_rsi_zone); empty-dict check is the equivalent
    # "no weekly context" signal.
    if not bundle.multi_timeframe:
        score -= 5; reasons.append("no weekly context")

    reason_str = "; ".join(reasons) if reasons else "all indicators present and current"
    return (max(0, score), reason_str)
```
---
# User Message (Data Payload)
Block labels match citation tokens (`TREND`, `MOMO`, `VOL`, `SR`, `VOLA`, `PAT`, `REGIME`).
```plain text
{canonical_ticker} ({company_name}) | {sector} | {primary_exchange} | {currency} | as of {data_timestamp}
Price: {current_price} | Prev close: {prev_close} | Day chg: {day_change_pct}
52w: {low_52w}-{high_52w} | %from 52wH: {pct_from_52w_high} | %from 52wL: {pct_from_52w_low}
Liquidity: avg $vol(20d)={avg_dollar_volume_20}
Earnings: next={next_earnings_date} | days_until={earnings_proximity_days}

TREND (daily + weekly):
  SMA20={sma_20} SMA50={sma_50} SMA200={sma_200}
  Stack={sma_stack_order} Slope50={sma_50_slope} Slope200={sma_200_slope}
  Price vs SMA20={pct_vs_sma_20} vs SMA50={pct_vs_sma_50} vs SMA200={pct_vs_sma_200}
  Swing structure (20 sessions): {swing_structure_20d}
  Weekly trend structure (HH/HL): {trend_structure_weekly}
  Weekly trend: {weekly_trend} | Weekly RSI zone: {weekly_rsi_zone}
  Sector RS (3mo): {rs_vs_sector_3mo} | leading_or_lagging={rs_leadership}

MOMO:
  RSI14={rsi_14} | trend-adjusted zone={rsi_zone_adjusted}
  MACD={macd_line} Sig={macd_signal} Hist={macd_histogram}
  MACD cross last 10 days: {macd_recent_cross}
  Divergence (price vs RSI/MACD): {price_momentum_divergence}

VOL:
  Today vol={volume_today} 20d avg={volume_avg_20} Ratio={volume_ratio_today}
  Avg dollar volume (20d): {avg_dollar_volume_20}

SR + VOLA (ATR is the unit):
  ATR14={atr_14} ATR60avg={atr_60_avg} regime={volatility_regime_derived}
  BB upper={bb_upper} mid={bb_middle} lower={bb_lower} width%ile={bb_width_percentile} position={bb_position}
  Support={nearest_support} ({pct_to_support}, {atr_to_support} ATR, touches={support_touch_count})
  Resistance={nearest_resistance} ({pct_to_resistance}, {atr_to_resistance} ATR, touches={resistance_touch_count})

PAT (use only if confidence ≥0.6):
  Detected: {detected_patterns_with_confidence}
  Measured move target (if confirmed): {pattern_target}

REGIME: {market_regime}  (broad-index context: bull/bear/choppy)

{data_pipeline_warnings}
```
**v2.2 changes:** TREND block now contains weekly trend + weekly RSI zone inline (replacing the separate MTF block). VOL block dropped OBV, distribution-day, and accumulation-day lines. SR+VOLA dropped unfilled-gap-proximity line. Liquidity simplified to avg dollar volume only.
---
# Orchestrator Merge Logic
```python
def merge_output(llm_response: dict, bundle: DataBundle,
                 context: AnalysisContext, base_score: int) -> dict:
    derived = derive_quality_enums(base_score, bundle)
    ifields = llm_response["interpretive_fields"]
    return {
        "agent_name": "technical_analyst",
        "agent_pass": "pass1",
        "analysis_context": {
            "account_type": context.account_type,
            "timeline": context.timeline,
            "stock_id": str(context.stock_id),
            "canonical_ticker": context.canonical_ticker,
        },
        "assessment_summary": llm_response["assessment_summary"],
        "reliability_score": llm_response["reliability_score"],
        "reliability_factors": {
            "data_completeness": derived["data_completeness"],
            "data_freshness": derived["data_freshness"],
            "analysis_confidence": llm_response["analysis_confidence"],
            "caveats": llm_response["caveats"],
        },
        "key_factors": llm_response["key_factors"],
        "risks": llm_response["risks"],
        "data_quality_assessment": derived["data_quality_assessment"],
        "narrative": llm_response["narrative"],
        "data_sources_used": derive_sources(bundle),
        "structured_data": {
            "technical_indicators": {
                **bundle.technical_indicators,
                "momentum_zone": ifields["momentum_zone"],
                "momentum_direction": ifields["momentum_direction"],
                "momentum_divergence": ifields["momentum_divergence"],
                "volume_confirmation": ifields["volume_confirmation"],
            },
            "trend_structure": {
                **bundle.trend_structure,
                "primary_trend": ifields["primary_trend"],
                "trend_strength": ifields["trend_strength"],
            },
            "support_resistance": {
                **bundle.support_resistance,
                "nearest_level_bias": ifields["nearest_level_bias"],
            },
            "liquidity_flags": {
                **bundle.liquidity_flags,
            },
            "pattern_metrics": {
                **bundle.pattern_metrics,
                "pattern_signal": ifields["pattern_signal"],
                "pattern_confirmed": ifields["pattern_confirmed"],
            },
            "weekly_context": {
                "weekly_trend": bundle.multi_timeframe["weekly_trend"],
                "weekly_rsi_zone": bundle.multi_timeframe["weekly_rsi_zone"],
            },
            "confluence_score": ifields["confluence_score"],
            "suggested_invalidation_level": ifields["suggested_invalidation_level"],
        },
    }
```
**Restructured 2026-08-07 (contract-vs-consumer audit):** the old `trend_metrics`/`momentum_metrics`/`volume_metrics`/`sr_metrics`/`volatility_metrics` five-way output grouping came from the phantom `TechnicalDataBundle` type and didn't map 1:1 onto `DataBundle`'s real ten fields — splitting `technical_indicators` across separate momentum/volume/volatility buckets would have meant manually enumerating which indicator keys go where. Simpler and more honest: `structured_data` now mirrors `DataBundle`'s real field names directly (`technical_indicators`, `trend_structure`, `support_resistance`, `liquidity_flags`, `pattern_metrics`, `weekly_context`) — one bucket per real source field, no manual key-splitting. `pass2_view` below is unaffected: it's a separate, already-flat structure that was never nested under the old grouping.
**v2.1 → v2.2 change (unrelated, kept for history):** `multi_timeframe` block replaced with simpler `weekly_context`. `pattern_stage` replaced with `pattern_confirmed`.
---
# Pass 2 Consumption View
Pass 2 agents (Bull, Bear, Risk, Tax) receive a flat `pass2_view` dict produced by the orchestrator’s `compress_pass1_for_advocate()`, NOT the full `structured_data`.
```python
pass2_view = {
    "primary_trend": str,
    "trend_strength": str,
    "momentum_zone": str,
    "momentum_direction": str,
    "momentum_divergence": str,
    "volume_confirmation": str,
    "nearest_level_bias": str,
    "pattern_signal": str,
    "pattern_confirmed": bool,                 # v2.2: replaces pattern_stage
    "confluence_score": int,
    "suggested_invalidation_level": float,
    "nearest_support": float|None,
    "nearest_resistance": float|None,
    "rsi_zone_adjusted": str,
    "volatility_regime_derived": str,
    "weekly_trend": str,                       # v2.2: replaces multi_timeframe_alignment_derived
}
```
**v2.2 changes:** `multi_timeframe_alignment_derived` removed; `weekly_trend` exposed directly. `pattern_confirmed` boolean replaces `pattern_stage`. **Bull/Bear/Risk/Tax payload templates need a one-line update** to read `weekly_trend` instead of `multi_timeframe_alignment_derived` and `pattern_confirmed` instead of `pattern_stage`.
---
# How the System Prompt and User Message Interact
- **Citation token consistency**: System prompt defines exact tokens and the user message uses them as block labels. Validator enforces tokens in LLM output.
- **Every CoT reference has a corresponding field**: If a step mentions an indicator, it must appear in the user message, even if “N/A”.
- **Variable redundancy**: Ticker, company name, sector, timeline, account type appear in both messages.
- **Non-thinking mode**: Set `enable_thinking: false` in the Ollama API call. Verify on `qwen3.6:35b-a3b` that this suppresses reasoning as reliably as it did on `qwen3:14b` (see Pattern 10 caveat).
- **Schema at end of system prompt**: `qwen3.6:35b-a3b`'s much larger native context (\~262K, vs `qwen3:14b`'s 32K) and reduced positional bias make this placement effective.
- **Single exchange**: `[{role: "system", content: ...}, {role: "user", content: ...}]`. Retries append to the user message.
- **Mandatory-caveat triggers in header**: `earnings_proximity_days` and `avg_dollar_volume_20` appear above all indicator sections.
---
# Prompt Variable Injection Reference
<table header-row="true">
<tr>
<td>Placeholder</td>
<td>Source</td>
<td>Notes</td>
</tr>
<tr>
<td>`{canonical_ticker}`</td>
<td>`DataBundle.stock.ticker`</td>
<td>e.g., “AAPL” or “RY.TO” — field is named `.ticker` on `StockRef`, not `.canonical_ticker`</td>
</tr>
<tr>
<td>`{company_name}`</td>
<td>`bundle.company_info["name"]`</td>
<td>Full company name</td>
</tr>
<tr>
<td>`{sector}`</td>
<td>`bundle.company_info["sector"]`</td>
<td>Enables sector-aware judgment</td>
</tr>
<tr>
<td>`{timeline}`</td>
<td>`AnalysisContext.timeline`</td>
<td>Always set; defaults to `medium_term`</td>
</tr>
<tr>
<td>`{account_type}`</td>
<td>`AnalysisContext.account_type`</td>
<td>`tfsa`, `rrsp`, `trading`, or `general`</td>
</tr>
<tr>
<td>`{timeline_instruction}`</td>
<td>Conditional injection table</td>
<td>Selected by `build_messages()`</td>
</tr>
<tr>
<td>`{account_instruction}`</td>
<td>Conditional injection table</td>
<td>Selected by `build_messages()`</td>
</tr>
<tr>
<td>`{base_reliability_score}`</td>
<td>`compute_base_reliability()`</td>
<td>Orchestrator pre-computed, 0–100</td>
</tr>
<tr>
<td>`{reliability_reason}`</td>
<td>`compute_base_reliability()`</td>
<td>Human-readable reason string</td>
</tr>
<tr>
<td>`{rsi_zone_adjusted}`</td>
<td>`bundle.technical_indicators["rsi_zone_adjusted"]`</td>
<td>Pre-classified using 80/40 in uptrends, 60/20 in downtrends — nested dict key</td>
</tr>
<tr>
<td>`{earnings_proximity_days}`</td>
<td>`bundle.earnings_proximity["earnings_proximity_days"]`</td>
<td>Trading days until next earnings release — nested dict key</td>
</tr>
<tr>
<td>`{volatility_regime_derived}`</td>
<td>`bundle.technical_indicators["volatility_regime_derived"]`</td>
<td>`expanding`/`contracting`/`normal` from ATR14 vs ATR60 — nested dict key</td>
</tr>
<tr>
<td>`{weekly_trend}`</td>
<td>`bundle.multi_timeframe["weekly_trend"]`</td>
<td>Pre-classified weekly trend — nested dict key, not a top-level attribute</td>
</tr>
<tr>
<td>`{weekly_rsi_zone}`</td>
<td>`bundle.multi_timeframe["weekly_rsi_zone"]`</td>
<td>Pre-classified weekly RSI zone — nested dict key, not a top-level attribute</td>
</tr>
<tr>
<td>`{rs_leadership}`</td>
<td>`bundle.trend_structure["rs_leadership"]`</td>
<td>leading/lagging vs sector ETF — nested dict key</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory` query</td>
<td>Empty string if no prior analysis or below decay threshold</td>
</tr>
<tr>
<td>`{data_warnings}`</td>
<td>Pre-flight validation</td>
<td>Empty string if clean</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
<td>The LLM-only schema above</td>
</tr>
<tr>
<td>Data payload placeholders</td>
<td>`bundle.*`</td>
<td>null → “N/A”</td>
</tr>
</table>
---
# Validation Rules
Enforced by `src/agents/validators.py`. Failure triggers retry with error appended to user message.
1. **JSON parseable** (strip markdown fences if present).
2. **Required fields**: `assessment_summary`, `reliability_score`, `analysis_confidence`, `caveats`, `key_factors`, `risks`, `narrative`, `interpretive_fields`.
3. **`reliability_score`**: integer 0–100, within ±15 of `base_reliability_score`.
4. **`analysis_confidence`**: one of `[high, medium, low]`.
5. **`caveats`**: non-empty array of strings.
6. **`key_factors`**: 1–4 items, each with `factor`, `importance`, `sentiment`, `evidence`. `sentiment` must be one of `[positive, negative, neutral]`. `evidence` must start with one of `TREND`, `MOMO`, `VOL`, `SR`, `VOLA`, `PAT`, `REGIME`. **v2.2: MTF removed.**
7. **`risks`**: 1–3 items, each with `risk`, `severity`, `likelihood`, `evidence`. Same evidence-token rule as `key_factors`.
8. **`narrative`**: 720–1080 characters (\~120–180 words).
9. **`interpretive_fields.primary_trend`**: one of `[uptrend, downtrend, sideways]`.
10. **`interpretive_fields.trend_strength`**: one of `[strong, moderate, weak]`.
11. **`interpretive_fields.momentum_zone`**: one of `[oversold, neutral, overbought]`.
12. **`interpretive_fields.momentum_direction`**: one of `[improving, flat, deteriorating]`.
13. **`interpretive_fields.momentum_divergence`**: one of `[bullish, bearish, none]`.
14. **`interpretive_fields.volume_confirmation`**: one of `[confirming, diverging, inconclusive]`.
15. **`interpretive_fields.nearest_level_bias`**: one of `[near_support, near_resistance, midrange]`.
16. **`interpretive_fields.pattern_signal`**: one of `[bullish, bearish, neutral, none]`.
17. **`interpretive_fields.pattern_confirmed`**: boolean. **v2.2: replaces pattern_stage enum check.**
18. **`interpretive_fields.confluence_score`**: integer 0–3.
19. **`interpretive_fields.suggested_invalidation_level`**: float. Directional sanity check — if `primary_trend=uptrend`, must be below current price; if `downtrend`, must be above; if `sideways`, any value within ±3 ATR is acceptable.
20. **Earnings-proximity caveat**: if `earnings_proximity_days ≤ 5`, `caveats` must contain at least one string matching `/earn|EPS|quarterly/i` AND `analysis_confidence` must be `medium` or `low`.
21. **Thin-volume caveat**: if `bundle.liquidity_flags["avg_dollar_volume_20"] < 1_000_000`, `caveats` SHOULD contain at least one string matching `/liquid|thin volume|volume/i`. **v2.2: soft check (warn, not fail) — replaces the hard ****`below_liquidity_floor`**** rule from v2.1.**
**Removed in v2.2:** the v2.1 mandatory `below_liquidity_floor`-driven validation rule.
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember: evidence in key_factors and risks must start with a valid block token (TREND, MOMO, VOL, SR, VOLA, PAT, REGIME). sentiment values are positive/negative/neutral. pattern_confirmed is a boolean.
```
---
# Memory Brief Schema
Schema (\~90 tokens):
```plain text
Prior technical analysis ({days_since} days ago, timeline={prior_timeline}, score={composite_score}/100):
  primary_trend_then: {prior_primary_trend} (strength: {prior_trend_strength})
  momentum_then: zone={prior_momentum_zone} direction={prior_momentum_direction}
  confluence_then: {prior_confluence_score}/3
  outcome: {return_pct if scored else "not yet scored"}
  what_changed_since: {short_list_of_price_action_events}
```
**v2.2 change:** dropped `key_levels_then` (prior support/resistance) — the model just needs the prior call and what changed since, not the prior levels themselves.
**Timeline-scaled decay**: memory is included if **\<14 days old at short_term, \<60 days at medium_term, \<180 days at long_term**.
Memory reconciliation: prior memory is a prior to update, not a new source of truth. Current price action is authoritative. Prior memory is NOT a citable source for new claims.
---
# Accuracy Brief (Post-20 Predictions)
Injected into the system prompt before the output schema, \~80 tokens.
```plain text
YOUR ACCURACY: {N} predictions scored. Bias: {bias_description}. Adjust: {calibration_instruction}. Strongest pattern: {best_pattern_type}. Weakest pattern: {worst_pattern_type}.
```
---
# Minimum Data Threshold
At least 60 daily bars + current price. Below threshold: `base_reliability_score` \<20, `data_quality_assessment` auto-set to `insufficient`. At \<20 bars, the orchestrator skips the LLM call entirely and returns a stub AgentOutput flagging insufficient data.
---
# Model Configuration
- `num_ctx`: 8192
- `enable_thinking`: false
- VRAM: \~18-22GB at Q4_K_M (MoE, \~3B active params; partial CPU offload into system RAM expected)
- Expected latency: unverified estimate — re-measure after deployment (was \~7–9s per stock at \~60 tok/s on the prior `qwen3:14b`; `qwen3.6:35b-a3b` will likely be slower per-token due to partial offload)
---
# Testing Checklist
- [ ] `qwen3.6:35b-a3b` non-thinking mode produces valid JSON matching the v2.2 schema (including `pattern_confirmed` boolean and 7-token citation grammar), with no leaked `<think>` reasoning tokens.
- [ ] `<think>` tags absent from output.
- [ ] `reliability_score` within ±15 of `base_reliability_score`.
- [ ] Narrative 120–180 words (720–1080 chars) and references actual indicator values.
- [ ] `interpretive_fields` use correct enums — `momentum_zone` AND `momentum_direction` both present, `pattern_confirmed` is a bool.
- [ ] `confluence_score` is an integer 0–3 and matches the dimensions described in the narrative.
- [ ] `suggested_invalidation_level` is on the correct side of current price per `primary_trend`.
- [ ] 1–4 `key_factors` and 1–3 `risks`.
- [ ] Every `key_factors` and `risks` evidence field begins with a valid block token (`TREND`, `MOMO`, `VOL`, `SR`, `VOLA`, `PAT`, `REGIME`). **No ****`MTF`**** token in v2.2 output.**
- [ ] `sentiment` values in `key_factors` are `positive|negative|neutral` only.
- [ ] No `pattern_stage`, `obv`, `distribution_days`, `intraday_trend`, `unfilled_gaps`, or `below_liquidity_floor` field references in output (v2.2 removal compliance).
- [ ] Confluence (2+ agreeing dimensions) called out explicitly when present.
- [ ] Model adapts focus by timeline: short-term ignores 200-day, long-term ignores RSI noise.
- [ ] Test with: AAPL (clean uptrend), a recent breakdown stock, a sideways consolidation, a \<60-bar stock (insufficient threshold), a \<20-bar stock (orchestrator-skip path), a stock within 5 days of earnings (mandatory caveat), a sub-\$1M-avg-dollar-volume stock (soft thin-volume caveat).
- [ ] Test with anomalous data (gap \>25%, zero-volume bar) — appears in caveats AND reduces `reliability_score`.
- [ ] `data_quality_assessment` is orchestrator-derived, not LLM-produced.
- [ ] **Phase-narration leakage test**: narrative does not contain “Step 1:”, “First, the trend…”, or similar CoT-recap phrasing.
- [ ] **pass2_view shape test**: `weekly_trend` present (not `multi_timeframe_alignment_derived`); `pattern_confirmed` present (not `pattern_stage`); no `obv`/`distribution_days` fields.
- [ ] Retry rate \<10% across 50 test stocks.
---
# Open Questions / Future Improvements
- **Few-shot example**: Test without one first. If retry rate \>10%, add a partial example (\~100 tokens).
- **Restoring a 7th CoT step for level confluence**: If S/R analysis keeps getting collapsed into trend during testing, split it back out.
- **Technical-specific memory decay**: After 20+ scored predictions, review whether Technical’s memory relevance window should be shorter than the shared 14/60/180 thresholds.
- **Sector-specific anomaly thresholds**: The 25% gap anomaly threshold is one-size-fits-all. Small-cap Canadian biotech / juniors may legitimately gap 30%+ on news. Consider sector-conditional thresholds.
- **Pattern detector calibration**: Currently uses a single 0.6 confidence cutoff. Post-feedback, track win rate by pattern type and adjust per-pattern thresholds.
- **Re-introducing OBV / distribution-day detection** if a future paid-tier or institutional-grade audience emerges that would benefit. Currently filed as not-applicable to the \< \$250k retail target.
**Removed from v2.1 Open Questions:** thinking-mode experiment (latency cost not worth marginal narrative-quality gain at retail-watchlist fan-out scale); intraday analyst as a separate agent (intraday cut entirely from the retail target — no longer relevant).
---
## Downstream impact summary from v2.2
<table header-row="true">
<tr>
<td>Component</td>
<td>Change</td>
<td>Effort</td>
</tr>
<tr>
<td>`TechnicalDataBundle` (historical name at the time of this v2.2 changelog entry — the type was never built under this name and was superseded by `DataBundle`, see the 2026-08-07 notes above)</td>
<td>Drop fields: `obv`, `obv_trend`, `distribution_days_25`, `accumulation_days_25`, `index_distribution_days_25`, `intraday_trend`, `unfilled_gaps_nearby`, `below_liquidity_floor`, `pattern_stage`. Add: `pattern_confirmed: bool`.</td>
<td>Small — schema + pipeline plumbing.</td>
</tr>
<tr>
<td>Data pipeline pre-computation</td>
<td>Stop computing distribution/accumulation days, OBV, 1h timeframe classification, gap-proximity scan. ATR / SMA / RSI / MACD / BB / volume / S-R / pattern detection / sector RS unchanged.</td>
<td>Medium-positive — fewer indicators to compute means lower API/compute cost.</td>
</tr>
<tr>
<td>Orchestrator (`compute_base_reliability`)</td>
<td>No-op — the function already keys off SMA/RSI/MACD/ATR/BB/OBV. Drop the OBV check from `core_indicators`.</td>
<td>Trivial.</td>
</tr>
<tr>
<td>Orchestrator merge logic</td>
<td>Drop merged-in fields for OBV / distribution / intraday / pattern_stage. Update `pass2_view` shape.</td>
<td>Small.</td>
</tr>
<tr>
<td>Validators</td>
<td>Drop MTF from citation-token regex (now 7 tokens). Drop `below_liquidity_floor`-driven mandatory caveat. Drop `pattern_stage` enum check. Add `pattern_confirmed: bool` check.</td>
<td>Small.</td>
</tr>
<tr>
<td>Pass 2 view contract</td>
<td>Remove `multi_timeframe_alignment_derived` (or set to a 2-state daily-weekly enum). Bull/Bear/Risk/Tax payload templates need a one-line update.</td>
<td>Small but coordinated — matches the v2.1 cross-agent-review pattern.</td>
</tr>
<tr>
<td>Memory brief</td>
<td>Drop prior support/resistance lines.</td>
<td>Trivial.</td>
</tr>
</table>
**Net effect:** roughly 27% reduction in total token usage per call, \~10–15% reduction in data-pipeline compute (no OBV / distribution-day / 1h-bar work), and a meaningfully simpler Pass 2 contract.

_Source: Notion — Project Stock Picker → Agent Prompts → Technical Analyst Agent Prompt. Downloaded 2026-07-02._
