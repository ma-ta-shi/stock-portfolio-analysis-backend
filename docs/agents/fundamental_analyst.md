**Agent**: Fundamental Analyst
**Status:** Complete for Phase 1
**Phase:** Phase 1
**Pass**: Pass 1 (Research & Analysis)
**Role**: Quantitative financial analysis — valuation, growth, profitability, balance sheet health, and peer comparison
**Prompt Version**: v5.0
**Target Model**: `qwen3.6:35b-a3b` via Ollama (non-thinking mode — verify `enable_thinking:false` fully suppresses reasoning on this model; see Agent Design Patterns Pattern 10 caveat)
**Audience**: Canadian retail investors with total portfolio values under \$250k CAD, holding stocks across TFSA / RRSP / Trading / General accounts. Institutional-grade complexity that doesn’t pay rent at this scale has been trimmed.
**Token Budget**: \~420 system prompt \| \~500 data input \| \~530 expected output \| \~1,450 total
---
# Design Decisions & Rationale
Built from [Agents Overview](https://www.notion.so/Agents-Overview-325ea06798238004a793fc1720637698?pvs=21) and [Technical Design](https://www.notion.so/Technical-Design-325ea067982380cb8374da20e14c8923?pvs=21). Key decisions:
1. **Pre-computed metrics only**: The pipeline does the math, the LLM does the thinking.
2. **LLM produces interpretive fields only**: The orchestrator passes through all numeric metrics from the DataBundle. The LLM produces only fields requiring judgment.
3. **Hybrid reliability scoring**: Orchestrator pre-computes `base_reliability_score` from mechanical checks. LLM adjusts ±15pts with justification in caveats.
4. **Orchestrator-derived fields**: `data_quality_assessment` and `reliability_factors.data_completeness`/`data_freshness` are computed by the orchestrator. The LLM produces only `analysis_confidence` and `caveats`.
5. **Bounded output arrays**: `key_factors` limited to 2-4 items, `risks` limited to 1-3 items.
6. **6-step CoT** (was 7 in v4.1): PROFITABILITY includes earnings quality (FCF/NI check). SYNTHESIS includes peer context. The standalone DATA SANITY step from v4.1 was removed in v5.0 — orchestrator’s `preflight_warnings`, `base_reliability_score`, and `derive_quality_enums()` already handle anomaly detection upstream; the model’s only job is to surface anomalies in `caveats` and adjust `reliability_score`, which the existing rules already mandate.
7. **Conditional injection**: Timeline, account, and **sector-specific valuation metrics** (new in v5.0) are injected based on context rather than hardcoded into the base prompt.
8. **Single prompt, non-thinking mode**: `/no_think` / `enable_thinking:false` is intended to produce deterministic JSON; verify this holds on `qwen3.6:35b-a3b` (see Pattern 10 caveat) rather than assuming parity with `qwen3:14b`'s behavior.
9. **Sector-aware anomaly detection**: Sector injected into system prompt for context-appropriate threshold judgments.
10. **Anomaly flags flow to caveats**: Anomalies adjust `reliability_score` and appear in `caveats`, ensuring downstream visibility.
11. **Evidence fields are brief citations, not sentences**: e.g., “VAL: P/E=25 vs median=18, 39% premium”.
12. **Citation token grammar**: Evidence fields must start with one of `VAL`, `GROWTH`, `PROF`, `BAL`, `DIV`, `PEER`, `ANALYST`. Cross-agent grounding consistency.
13. **Retail-investor scope** (new in v5.0): Target users are self-directed Canadian retail investors with portfolios under \$250k CAD optimizing TFSA/RRSP/Trading accounts. Institutional-grade complexity that doesn’t change retail decisions has been removed.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- Income statement — yfinance `income_stmt` + `quarterly_income_stmt` (CA) / edgartools `get_financials().income_statement()` + quarterly via XBRLS (US)
- Balance sheet — yfinance `balance_sheet` + `quarterly_balance_sheet` (CA) / edgartools `financials.balance_sheet()` (US)
- Cash flow — yfinance `cashflow` + `quarterly_cashflow` (CA) / edgartools `financials.cashflow_statement()` (US)
- Financial ratios — computed locally from yfinance statements + `.info` (CA) / computed from edgartools statements, with FMP `/ratios` for cross-check (US)
- Analyst estimates — openbb-tmx `.equity.estimates.consensus` (CA) / FMP `/analyst-estimates` (US)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- All numeric valuation/growth/profitability/balance-sheet/dividend metrics (`VAL`, `GROWTH`, `PROF`, `BAL`, `DIV` blocks) — pre-computed passthrough from the DataBundle; the LLM never recomputes a ratio
- Sector peer medians (`PEER` block) — P/E, P/B, revenue growth, margins, ROE, D/E, plus conditional `EV/EBITDA` for capital-intensive sectors (Energy, Utilities, Real Estate, Materials, Industrials)
- `base_reliability_score` — computed from core-metric null counts (0/8–8/8), financials staleness in days, pre-flight anomaly count, and peer-data sufficiency (\<2 peers penalized); LLM only adjusts ±15pts with justification in caveats
- Derived quality enums (`data_quality_assessment`, `reliability_factors.data_completeness`, `reliability_factors.data_freshness`) — computed entirely by the orchestrator from `base_reliability_score` and null/staleness counts, not by the LLM
- `memory_brief` — prior analysis summary (\~120 tokens), timeline-scaled decay (\<14 days short-term, \<60 days medium-term, \<180 days long-term); v4.1 `health_rating` values (`fortress`/`distressed`) mapped to the current 3-tier enum on read
- `data_warnings` — pre-flight validation findings; empty string if clean
---
# System Prompt (v5.0)
Invoke with `enable_thinking: false`. The `build_messages()` method renders this template.

**Runtime prompt:** see [`prompts/fundamental_analyst/v5.0.txt`](../../prompts/fundamental_analyst/v5.0.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`):
<table header-row="true">
<colgroup>
<col>
<col width="568">
</colgroup>
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks): weight earnings momentum, surprises, guidance, fwd P/E.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo): balance valuation with growth. PEG + margin trend key.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs): emphasize 3yr CAGR, ROE consistency, balance sheet, dividends. Ignore quarterly noise.`</td>
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
<td>`TFSA. Gains tax-free. Foreign dividends 15% WHT, non-recoverable.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US dividends WHT-exempt (treaty). Weight income sustainability.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading. 50% capital gains inclusion. Canadian eligible dividends get tax credit.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General. Note which metrics matter most for each account type (TFSA/RRSP/Trading).`</td>
</tr>
</table>
**Sector-specific valuation** (new in v5.0; inject based on `DataBundle.company_info["sector"]`):
<table header-row="true">
<colgroup>
<col width="342">
<col width="358">
</colgroup>
<tr>
<td>Sector</td>
<td>`{sector_specific_valuation_instruction}`</td>
</tr>
<tr>
<td>`Energy`, `Utilities`, `Real Estate` (REITs), `Materials`, `Industrials`</td>
<td>`Also weigh EV/EBITDA — primary valuation metric for capital-intensive sectors.`</td>
</tr>
<tr>
<td>All other sectors (Technology, Financials, Consumer, Healthcare, Communication Services, etc.)</td>
<td>`""` (empty string — no injection)</td>
</tr>
</table>
When the sector match injects EV/EBITDA, the orchestrator also includes `EV/EBITDA` lines in the user-message `VAL` and `PEER` blocks (see User Message section). When no match, both lines are omitted.
---
# Output Schema (v5.0 — LLM-produced fields only)
The orchestrator populates numeric passthrough fields and derived enum fields from the DataBundle. The LLM produces **only** the interpretive fields below.
```javascript
{
  "assessment_summary": "≤80 words. Most important finding first.",
  "reliability_score": 0-100,
  "analysis_confidence": "high|medium|low",
  "caveats": ["specific data gaps, anomalies, or qualitative concerns"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "positive|negative|neutral", "evidence": "brief citation starting with a valid block token"}
  ],
  "risks": [
    {"risk": "name", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "brief citation starting with a valid block token"}
  ],
  "narrative": "120-180 word analysis following steps 1-6. Reference specific numbers.",
  "interpretive_fields": {
    "pe_vs_sector_avg": "undervalued|fair|overvalued",
    "margin_trend": "expanding|stable|contracting",
    "health_rating": "healthy|adequate|stressed",
    "guidance_vs_consensus": "above|inline|below|not_available",
    "dividend_sustainability": "strong|adequate|at_risk|not_applicable",
    "peer_comparison_summary": "1-2 sentences comparing to peers"
  }
}
```
### Fields the orchestrator populates (NOT produced by LLM)
**Numeric passthrough** from DataBundle: All metric fields (pe_ratio through consecutive_years_paid). Note that `roic` and `dcf_implied_value`/`dcf_upside_pct` are no longer referenced by the prompt or `pass2_view` — see Migration Notes for retention guidance in DataBundle.
**Derived enums** (orchestrator computes):
- `data_quality_assessment`: Derived from `reliability_score` — ≥70=“high”, 40-69=“medium”, 20-39=“low”, \<20=“insufficient”
- `reliability_factors.data_completeness`: Derived from null count in core fields
- `reliability_factors.data_freshness`: Derived from `data_bundle.data_freshness` timestamps
**Metadata**: `agent_name`, `agent_pass`, `analysis_context`, `data_sources_used`
---
# Orchestrator Pre-computation: `base_reliability_score`
Unchanged from v3.0.
```python
def compute_base_reliability(data_bundle: DataBundle) -> tuple[int, str]:
    score = 100
    reasons = []

    core_fields = [pe_ratio, forward_pe, revenue_growth_yoy, gross_margin,
                   operating_margin, net_margin, debt_to_equity, free_cash_flow]
    null_count = sum(1 for f in core_fields if f is None)
    if null_count >= 5:
        score -= 40; reasons.append(f"{null_count}/8 core metrics missing")
    elif null_count >= 3:
        score -= 25; reasons.append(f"{null_count}/8 core metrics missing")
    elif null_count >= 1:
        score -= 10; reasons.append(f"{null_count}/8 core metrics missing")

    days_old = (now - data_bundle.data_freshness["financials"]).days
    if days_old > 180:
        score -= 25; reasons.append(f"financials{days_old}d old")
    elif days_old > 90:
        score -= 15; reasons.append(f"financials{days_old}d old")
    elif days_old > 45:
        score -= 5; reasons.append(f"financials{days_old}d old")

    anomaly_count = len(data_bundle.preflight_warnings)
    if anomaly_count >= 3:
        score -= 15; reasons.append(f"{anomaly_count} data anomalies")
    elif anomaly_count >= 1:
        score -= 5; reasons.append(f"{anomaly_count} data anomaly")

    if not data_bundle.peer_metrics or len(data_bundle.peer_metrics) < 2:
        score -= 10; reasons.append("limited peer data")

    reason_str = "; ".join(reasons) if reasons else "all data present and current"
    return (max(0, score), reason_str)
```
**Orchestrator also derives** (since v4.0):
```python
def derive_quality_enums(base_score: int, data_bundle: DataBundle) -> dict:
    if base_score >= 70: dqa = "high"
    elif base_score >= 40: dqa = "medium"
    elif base_score >= 20: dqa = "low"
    else: dqa = "insufficient"

    null_count = count_nulls(data_bundle, CORE_FIELDS)
    if null_count == 0: dc = "complete"
    elif null_count <= 2: dc = "mostly_complete"
    elif null_count <= 4: dc = "partial"
    else: dc = "minimal"

    days_old = (now - data_bundle.data_freshness["financials"]).days
    if days_old <= 45: df = "current"
    elif days_old <= 90: df = "slightly_stale"
    elif days_old <= 180: df = "stale"
    else: df = "very_stale"

    return {"data_quality_assessment": dqa, "data_completeness": dc, "data_freshness": df}
```
---
# User Message (Data Payload)
Block labels in the payload match the citation tokens the LLM uses (`VAL`, `GROWTH`, `PROF`, `BAL`, `DIV`, `ANALYST`, `PEER`).
The `EV/EBITDA` line in `VAL` and the `EV/EBITDA={sector_median_ev_ebitda}` line in `PEER` are **conditional**: included only when the sector is in the capital-intensive list (Energy, Utilities, Real Estate/REITs, Materials, Industrials). Otherwise omitted entirely. `DCF` and `ROIC` lines from v4.1 are removed.
```plain text
{canonical_ticker} ({company_name}) | {sector} | {primary_exchange} | {currency} | as of {data_timestamp}
Price: {current_price} | MCap: {market_cap_formatted} | 52w: {low_52w}-{high_52w}

VAL:
  P/E={pe_ratio} FwdPE={forward_pe} P/B={pb_ratio} P/S={ps_ratio} PEG={peg_ratio}
  {ev_ebitda_line}            # conditional — see above

GROWTH:
  RevYoY={revenue_growth_yoy} Rev3yrCAGR={revenue_growth_3yr_cagr}
  EPSYoY={eps_growth_yoy} Surprises={earnings_surprises} Guidance={guidance_vs_consensus}

PROF:
  GrossM={gross_margin} OpM={operating_margin} NetM={net_margin}
  ROE={roe} Trend={margin_trend} FCF/NI={fcf_to_net_income}

BAL:
  D/E={debt_to_equity} Current={current_ratio} IntCov={interest_coverage}
  FCF={free_cash_flow} Cash={cash_position}

DIV:
  Yield={dividend_yield} Payout={payout_ratio} 5yrGrowth={dividend_growth_5yr}
  Type={dividend_type} YrsPaid={consecutive_years_paid}

ANALYST: Rating={consensus_rating} Target={avg_price_target} Count={num_analysts}

PEER (sector median):
  P/E={sector_median_pe} P/B={sector_median_pb}
  {peer_ev_ebitda_line}       # conditional — see above
  RevGrowth={sector_median_rev_growth} GrossM={sector_median_gross_margin}
  OpM={sector_median_op_margin} ROE={sector_median_roe} D/E={sector_median_de}

{peer_data_block}

{data_pipeline_warnings}
```
Where:
```python
ev_ebitda_line = f"  EV/EBITDA={ev_ebitda}" if sector in CAPITAL_INTENSIVE else ""
peer_ev_ebitda_line = f"  EV/EBITDA={sector_median_ev_ebitda}" if sector in CAPITAL_INTENSIVE else ""
CAPITAL_INTENSIVE = {"Energy", "Utilities", "Real Estate", "Materials", "Industrials"}
```
---
# Orchestrator Merge Logic
Updated from v4.1 to (a) reflect new `health_rating` enum, (b) drop ROIC/DCF references from `pass2_view` and `structured_data` projections. The DataBundle itself can still carry `roic`, `dcf_implied_value`, `dcf_upside_pct` if the data pipeline produces them — they’re simply not surfaced to the LLM or downstream Pass 2 agents.
```python
def merge_output(llm_response: dict, data_bundle: DataBundle,
                 context: AnalysisContext, base_score: int) -> dict:
    derived = derive_quality_enums(base_score, data_bundle)
    return {
        "agent_name": "fundamental_analyst",
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
        "data_sources_used": derive_sources(data_bundle),
        "structured_data": {
            "valuation_metrics": {
                **data_bundle.valuation_metrics,           # ROIC/DCF excluded by projection layer
                "pe_vs_sector_avg": llm_response["interpretive_fields"]["pe_vs_sector_avg"],
            },
            "growth_metrics": {
                **data_bundle.growth_metrics,
                "guidance_vs_consensus": llm_response["interpretive_fields"]["guidance_vs_consensus"],
            },
            "profitability_metrics": {
                **data_bundle.profitability_metrics,       # ROIC excluded
                "margin_trend": llm_response["interpretive_fields"]["margin_trend"],
            },
            "balance_sheet_health": {
                **data_bundle.balance_sheet_metrics,
                "health_rating": llm_response["interpretive_fields"]["health_rating"],
            },
            "dividend_info": {
                **data_bundle.dividend_info,
                "dividend_sustainability": llm_response["interpretive_fields"]["dividend_sustainability"],
            },
            "peer_comparison_summary": llm_response["interpretive_fields"]["peer_comparison_summary"],
        },
    }
```
---
# Pass 2 Consumption View
`pass2_view` updated from v4.1: `roic` removed. Other fields unchanged.
```python
pass2_view = {
    "valuation_vs_sector": str,              # structured_data.valuation_metrics.pe_vs_sector_avg
    "pe_ratio": float|None,
    "sector_pe_median": float|None,
    "forward_pe": float|None,
    "peg_ratio": float|None,
    "revenue_growth_yoy": float|None,
    "revenue_growth_3yr_cagr": float|None,
    "guidance_vs_consensus": str,
    "gross_margin": float|None,
    "operating_margin": float|None,
    "net_margin": float|None,
    "roe": float|None,
    "margin_trend": str,
    "fcf_to_net_income": float|None,
    "debt_to_equity": float|None,
    "interest_coverage": float|None,
    "free_cash_flow": float|None,
    "health_rating": str,                    # values: healthy|adequate|stressed
    "dividend_yield": float|None,
    "payout_ratio": float|None,
    "dividend_sustainability": str,
    "consensus_rating": str|None,
    "average_price_target": float|None,
    "peer_comparison_summary": str,
}
```
The envelope fields Pass 2 sees alongside `pass2_view` — `reliability_score`, `data_quality_assessment`, `caveats`, `assessment_summary`, truncated `narrative`, top-2 `key_factors`, top-2 `risks` — are defined in the Orchestration Engine’s Pass 1 compression spec.
**Pass 2 prompt impact**: Bull/Bear/Risk/Tax payload templates that reference `roic` or `health_rating ∈ {fortress, distressed}` will need parallel updates during the cross-agent alignment pass. The `health_rating` collapse is a breaking change for any agent that branches on `fortress` or `distressed` literals.
---
# How the System Prompt and User Message Interact
- **Citation token consistency**: System prompt defines `VAL`, `GROWTH`, etc.; user message block labels are `VAL:`, `GROWTH:`, etc.
- **Every CoT reference has a corresponding field**: If a step mentions a metric, it must appear in the user message, even if “N/A”. When EV/EBITDA is omitted by sector conditional, step 1’s instruction is also pruned (`{sector_specific_valuation_instruction}` is empty), preserving this invariant.
- **Variable redundancy**: Ticker, company name, timeline, account type appear in both messages.
- **Non-thinking mode**: Set `enable_thinking: false` in the Ollama API call. Verify on `qwen3.6:35b-a3b` that this suppresses reasoning as reliably as it did on `qwen3:14b` (see Pattern 10 caveat).
- **Schema at end of system prompt**: `qwen3.6:35b-a3b`'s much larger native context (\~262K, vs `qwen3:14b`'s 32K) and reduced positional bias make this placement effective.
- **Single exchange**: `[{role: "system", content: ...}, {role: "user", content: ...}]`. Retries append to user message.
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
<td>`DataBundle.stock.canonical_ticker`</td>
<td>e.g., “AAPL” or “RY.TO”</td>
</tr>
<tr>
<td>`{company_name}`</td>
<td>`DataBundle.company_info["name"]`</td>
<td>Full company name</td>
</tr>
<tr>
<td>`{sector}`</td>
<td>`DataBundle.company_info["sector"]`</td>
<td>Drives both anomaly detection and EV/EBITDA conditional</td>
</tr>
<tr>
<td>`{timeline}`</td>
<td>`AnalysisContext.timeline`</td>
<td>Always set; defaults to “medium_term”</td>
</tr>
<tr>
<td>`{account_type}`</td>
<td>`AnalysisContext.account_type`</td>
<td>“tfsa”, “rrsp”, “trading”, or “general”</td>
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
<td>`{sector_specific_valuation_instruction}`</td>
<td>Conditional injection table</td>
<td>New in v5.0; empty for non-capital-intensive sectors</td>
</tr>
<tr>
<td>`{base_reliability_score}`</td>
<td>`compute_base_reliability()`</td>
<td>Orchestrator pre-computed, 0-100</td>
</tr>
<tr>
<td>`{reliability_reason}`</td>
<td>`compute_base_reliability()`</td>
<td>Human-readable reason string</td>
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
<td>`{peer_data_block}`</td>
<td>`DataBundle.peer_metrics`</td>
<td>Compact: `PEER MSFT: P/E=32.1 P/B=12.4 ...`</td>
</tr>
<tr>
<td>Metric placeholders</td>
<td>`DataBundle.*_metrics`</td>
<td>null → “N/A”; ROIC and DCF placeholders no longer rendered</td>
</tr>
</table>
---
# Validation Rules
Enforced by `src/agents/validators.py`. Failure triggers retry with error appended to user message.
1. **JSON parseable** (strip markdown fences/preamble if present)
2. **Required fields**: `assessment_summary`, `reliability_score`, `analysis_confidence`, `caveats`, `key_factors`, `risks`, `narrative`, `interpretive_fields`
3. **reliability_score**: integer 0-100, within ±15 of `base_reliability_score`
4. **analysis_confidence**: one of `[high, medium, low]`
5. **caveats**: non-empty array
6. **key_factors**: 2-4 items, each with `factor`, `importance`, `sentiment` (one of `[positive, negative, neutral]`), `evidence`. Evidence must start with one of `VAL`, `GROWTH`, `PROF`, `BAL`, `DIV`, `PEER`, `ANALYST`.
7. **risks**: 1-3 items, each with `risk`, `severity`, `likelihood`, `evidence`. Same evidence-token rule as `key_factors`.
8. **narrative**: 720-1080 characters (\~120-180 words).
9. **interpretive_fields.pe_vs_sector_avg**: one of `[undervalued, fair, overvalued]`
10. **interpretive_fields.margin_trend**: one of `[expanding, stable, contracting]`
11. **interpretive_fields.health_rating**: one of `[healthy, adequate, stressed]`. **Changed in v5.0** — was 5-tier (`fortress|healthy|adequate|stressed|distressed`).
12. **interpretive_fields.guidance_vs_consensus**: one of `[above, inline, below, not_available]`
13. **interpretive_fields.dividend_sustainability**: one of `[strong, adequate, at_risk, not_applicable]`
14. **interpretive_fields.peer_comparison_summary**: non-empty string
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember: evidence in key_factors and risks must start with a valid block token (VAL, GROWTH, PROF, BAL, DIV, PEER, ANALYST). sentiment values are positive/negative/neutral. health_rating is healthy/adequate/stressed (no fortress/distressed).
```
---
# Memory Brief Schema
Schema (\~120 tokens):
```plain text
Prior fundamental analysis ({days_since} days ago, timeline={prior_timeline}, score={composite_score}/100):
  pe_vs_sector_then: {prior_pe_vs_sector}
  margin_trend_then: {prior_margin_trend}
  health_rating_then: {prior_health_rating}
  guidance_vs_consensus_then: {prior_guidance}
  dividend_sustainability_then: {prior_dividend_sustainability}
  top_factors_then: [{top_2_factor_names}]
  top_risks_then: [{top_2_risk_names}]
  outcome: {return_pct if scored else "not yet scored"}
  what_changed_since: {short_list_of_material_events}
```
**Timeline-scaled decay**: memory is included if **\<14 days old at short_term, \<60 days at medium_term, \<180 days at long_term**.
**Reconciliation logic**: prior memory is a prior to update, not a new source of truth. If current metrics confirm the prior conclusion, keep it and say so briefly. If current metrics contradict (e.g., prior `health_rating=healthy` and current BAL shows interest coverage fallen below 2x), override and explain. Prior memory is NOT a citable source for new claims — only current payload blocks (`VAL`, `GROWTH`, `PROF`, `BAL`, `DIV`, `PEER`, `ANALYST`) can be cited.
**v5.0 migration note for ****`health_rating_then`**: prior records may contain `fortress` or `distressed` from v4.1 outputs. The memory loader should map `fortress → healthy` and `distressed → stressed` on read so prior values align with the current 3-tier enum without rewriting historical records.
---
# Accuracy Brief (Post-20 Predictions)
Injected into system prompt before output schema. \~80 tokens.
```plain text
YOUR ACCURACY: {N} predictions scored. Bias: {bias_description}. Adjust: {calibration_instruction}. Strongest: {best_stock_type}. Weakest: {worst_stock_type}.
```
---
# Minimum Data Threshold
From [Agents Overview](https://www.notion.so/Agents-Overview-325ea06798238004a793fc1720637698?pvs=21): At least 2 quarters of income statement data + current stock price.
Below threshold: `base_reliability_score` will be \<20. Orchestrator sets `data_quality_assessment` to “insufficient” automatically.
---
# Testing Checklist
- [ ] Does `qwen3.6:35b-a3b` in non-thinking mode produce valid JSON matching the v5.0 schema, with no leaked `<think>` reasoning tokens?
- [ ] Does thinking mode accidentally activate? (Check for `<think>` tags in output)
- [ ] Does `reliability_score` stay within ±15 of `base_reliability_score`?
- [ ] Does the narrative stay within 120-180 words (720-1080 chars) and reference actual data points?
- [ ] Does the orchestrator merge produce a valid full `AgentOutput` matching Agents Overview schema?
- [ ] Do `interpretive_fields` use correct enum values (including the new 3-tier `health_rating`)?
- [ ] **Health rating enum test**: model never produces `fortress` or `distressed`. Validator 11 rejects both.
- [ ] Does the model produce exactly 2-4 key_factors and 1-3 risks (not fewer, not more)?
- [ ] **Citation token test**: Every `key_factors` and `risks` evidence field begins with one of `VAL`, `GROWTH`, `PROF`, `BAL`, `DIV`, `PEER`, `ANALYST`.
- [ ] Are `sentiment` values in `key_factors` restricted to `positive|negative|neutral`?
- [ ] Are evidence fields brief citations (not full sentences)?
- [ ] Does the model correctly handle null/N/A metrics without hallucinating values?
- [ ] Are peer comparisons quantified (premium/discount percentage)?
- [ ] Does the model avoid buy/sell recommendations?
- [ ] Does the model adapt analysis focus based on timeline instruction?
- [ ] Does the model adapt dividend emphasis based on account instruction?
- [ ] Does `dividend_sustainability` align with the dividend data?
- [ ] **Phase-narration leakage test**: narrative does not contain “Step 1:”, “First, valuation…”, or similar CoT-recap phrasing.
- [ ] **pass2_view shape test**: the orchestrator’s compression produces every key listed in the Pass 2 Consumption View section with correct types. `roic` is NOT present.
- [ ] **Sector-conditional EV/EBITDA test**: VAL block contains `EV/EBITDA` line for `RY.TO` (Real Estate? Financials — should NOT have it) and CNQ.TO (Energy — SHOULD have it). System prompt step 1 mirrors the conditional. Spot-check `XEG.TO` constituents (Energy = present) vs `SHOP.TO` (Technology = absent).
- [ ] **DCF/ROIC absence test**: VAL block does NOT contain `DCF=` or `DCFupside=`. PROF block does NOT contain `ROIC=`. Step 1 does not mention DCF; step 3 does not mention ROIC.
- [ ] Test with 3 fixture stocks: AAPL (US large-cap, no EV/EBITDA), RY.TO (Canadian financial, no EV/EBITDA), SHOP.TO (Canadian growth, no EV/EBITDA), CNQ.TO or ENB.TO (Canadian energy, EV/EBITDA injected).
- [ ] Test with sparse data (many N/A) — does orchestrator-derived `data_quality_assessment` correctly show “insufficient”?
- [ ] Test with anomalous data (P/E=5000, negative revenue) — anomalies appear in `caveats` even without dedicated DATA SANITY step?
- [ ] **Memory enum migration test**: load a v4.1 memory record with `health_rating_then=fortress`; verify loader maps to `healthy` before injection.
---
# Token Budget Verification
<table header-row="true">
<colgroup>
<col>
<col>
<col>
<col width="345">
</colgroup>
<tr>
<td>Component</td>
<td>v4.1</td>
<td>v5.0</td>
<td>Change</td>
</tr>
<tr>
<td>System prompt</td>
<td>\~490 tokens</td>
<td>\~420 tokens</td>
<td>-70 (DCF/ROIC removed from steps; CoT collapsed 7→6; timeline+account prose tightened; health_rating enum shorter)</td>
</tr>
<tr>
<td>User message</td>
<td>\~550 tokens</td>
<td>\~500 tokens</td>
<td>-50 (DCF and ROIC lines removed; EV/EBITDA conditional saves \~10 for non-capital-intensive sectors)</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~550 tokens</td>
<td>\~530 tokens</td>
<td>-20 (`health_rating` enum shorter; narrative bound unchanged but slightly less metric surface to recite)</td>
</tr>
<tr>
<td>**Total**</td>
<td>**\~1,590 tokens**</td>
<td>**\~1,450 tokens**</td>
<td>**-140 tokens (\~9%)**</td>
</tr>
</table>
**Model configuration**:
- `num_ctx`: 8192 (sufficient, conserves VRAM)
- `enable_thinking`: false (verify suppression is clean on `qwen3.6:35b-a3b` — see Pattern 10 caveat)
- VRAM: \~18-22GB at Q4_K_M (MoE, \~3B active params; partial CPU offload into system RAM expected)
- Expected latency: unverified estimate — re-measure after deployment (was \~7-9s per stock at \~60 tok/s on the prior `qwen3:14b`; `qwen3.6:35b-a3b` will likely be slower per-token due to partial offload)
---
# Migration Notes (v4.1 → v5.0)
**DataBundle**: No required schema change. The pipeline can still compute and store `roic`, `dcf_implied_value`, `dcf_upside_pct` if cheap to retain — they’re simply unused by this agent. If FMP API budget is tight, dropping these from the data fetch saves provider calls.
**Memory store (****`StockAnalysisMemory`****)**: existing rows with `health_rating ∈ {fortress, distressed}` should be left as-is on disk. The memory loader maps `fortress → healthy` and `distressed → stressed` at read time (see Memory Brief Schema migration note). No backfill needed.
**Pass 2 prompts (Bull, Bear, Risk, Tax)**: reference `pass2_view.roic` or `pass2_view.health_rating ∈ {fortress, distressed}` literals must be updated during the cross-agent alignment pass. Bull/Bear narrative templates that praised “fortress balance sheets” should generalize to “healthy” or use a derived signal (e.g., interest coverage \> 10x).
**CIO synthesis prompt**: same `health_rating` and `roic` review applies. DCF inputs from Pass 1 disappear — the CIO should not be asked to weigh DCF upside, only the metrics still surfaced in `pass2_view`.
**Portfolio Optimizer (Health Assessor / Opportunity Ranker / Action Synthesizer)**: review for any reliance on Fundamental’s prior 5-tier health_rating granularity or ROIC. Likely minor.
---
# Open Questions / Future Improvements
- **Cross-agent alignment pass**: v5.0 is intentionally a single-agent change. Run the same retail-scope review on Macro Economist, Sentiment Analyst, Stock Researcher, Technical Analyst, Bull/Bear/Risk/Tax, and CIO before locking the changes — at minimum to update Pass 2 templates that reference `roic` or the old 5-tier `health_rating`.
- **Few-shot example**: `qwen3:14b` had strong JSON adherence; re-establish this for `qwen3.6:35b-a3b` rather than assuming it carries over. Test without one first. If retry rate exceeds 10%, add a partial example (\~100 tokens).
- **Thinking mode experiment**: Run 10 stocks in thinking mode (strip `<think>` before parsing). If narrative quality is noticeably deeper, the \~2-3x latency may be worth it since this agent is the fastest in the pipeline.
- **Entity anonymization (P4)**: Current rationale for not anonymizing is documented in the Agent Design Patterns anonymization principle.
- **Sector list refinement**: the capital-intensive sector set (Energy, Utilities, Real Estate, Materials, Industrials) is GICS-aligned but could be tuned. Telecom (Communication Services) is borderline — capital-intensive but P/E often the more-quoted metric. Defer until cross-agent pass.
- **12-month prediction checkpoint**: Feedback engine needs this for upper-range medium-term predictions.
- **Restore the 5-tier ****`health_rating`****?**: If post-launch backtest shows `fortress` and `distressed` carry signal Pass 2 agents wanted, restore. Hold for now.
- **Accuracy brief effectiveness**: Monitor whether `qwen3.6:35b-a3b` adjusts behavior based on the \~80 token brief.

_Source: Notion — Project Stock Picker → Agent Prompts → Fundamental Analyst Agent Prompt. Downloaded 2026-07-02._
