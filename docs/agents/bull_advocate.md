**Agent**: Bull Case Advocate
**Status:** Draft — v1.6 retail-simplification pass
**Phase:** Phase 1
**Pass**: Pass 2 (Perspective & Strategy)
**Role**: Build the strongest possible bullish thesis for this stock given the analysis context. Deliberately biased — this agent’s job is persuasion grounded in data, not balance. **Audience: an individual Canadian retail investor with a portfolio under \$250k CAD across TFSA/RRSP/Trading accounts.**
**Target Model**: **Cloud-primary: Claude Sonnet 4.6** (thinking mode OFF). Local fallback: `qwen3.6:35b-a3b` via Ollama, accepting materially degraded quality.
**Audience**: Canadian retail investors with total portfolio values under \$250k CAD, holding stocks across TFSA / RRSP / Trading / General accounts. Institutional-grade complexity that doesn’t pay rent at this scale has been trimmed.
**Token Budget**: \~700 system prompt \| \~4,000 data input (target — see caveat in Token Budget section) \| \~1,500 expected output \| \~6,200 total. v1.6 trims \~100 tokens off the v1.5 system prompt by reducing Rule 7 patterns, merging Rules 22/23, and collapsing CoT to 5 steps.
---
# Design Decisions & Rationale
Built following the [Agent Design Patterns](https://www.notion.so/Agent-Design-Patterns-33fea067982380e6a055d3ed74c8eada?pvs=21) validated through the Fundamental Analyst and Stock Researcher implementations.
1. **Advocate, not judge — with tightened honesty controls at low confidence.** This agent’s `recommendation` is hard-coded to `"bullish"` by validation. The adversarial pair (bull + bear) works only if each side actually advocates. The hard-coded stance creates a fabrication pressure at low confidence; we mitigate via (a) Rule 7’s high-bar frame requirement, and (b) Rule 22’s verbatim-quoting requirement at confidence \< 35. v1.6 reduces Rule 7 from five frames to three retail-relevant ones — the dropped frames (special-situation arbitrage, hidden-asset / sum-of-parts) almost never apply to a sub-\$250k Canadian retail watchlist of TSX large/mid-caps and US blue chips, and offering them as exit ramps mostly served to give the model thesis patterns the user can’t actually act on.
2. **Pass 2 input = compressed Pass 1, not raw DataBundle.** The orchestrator’s Pass 1 compression pass produces \~500–800 token summaries per Pass 1 agent. Citation tokens refer to Pass 1 agents (`RSRCH`, `FUND`, `TECH`, `SENT`, `MACRO`), not raw sources.
3. **Reliability-aware weighting is mandatory.** Treat `reliability_score >= 70` inputs as trustworthy, 40-69 with caution, \<40 as unreliable. If the bull thesis rests primarily on a \<40-reliability input, cap `confidence` accordingly and call this out in `weakest_point`.
4. **Researcher archetype as evidence, not constraint.** The Researcher’s `thesis_archetype` is an input signal, not a constraint. The bull picks its own archetype and the orchestrator logs disagreement as a signal for the CIO.
5. **Positive grounding directive, no entity anonymization.** Pass 2 on Sonnet 4.6 uses real ticker and company name — Pass 2 inputs are already interpretive and historical analogies require real names.
6. **5-step CoT** (v1.6 — was 6 in v1.5). Each step maps to at least one structured output field. Steps: thesis construction (with timeline/account context baked in), catalyst identification, valuation argument + market-misread, historical analogy, self-critique. The old step 6 (context adaptation) folded into step 1 because the timeline/account framing already runs through every step via the injected blocks — keeping it as a separate step was redundant.
7. **Bounded output arrays** (Pattern 7). `core_arguments` 1–3, `catalysts` 1–4, `market_misreads` 1–3, `key_factors` 2–4, `thesis_risks` 1–3.
8. **Cited evidence required on every argument.** Each `core_argument`, `catalyst`, and `market_misread` entry must cite at least one Pass 1 agent ID. Compact citations (`"FUND: FCF yield 6.2% vs sector 3.1%"`).
9. **Context adaptation via conditional injection.** Timeline and account_type blocks are injected from lookup tables. The account block is substantive: a TFSA bull looks meaningfully different from an RRSP bull.
10. **Self-critique is structured output, not narrative.** `weakest_point` is required. v1.6: `weakest_point_type` enum reduced to 4 values (`negative_catalyst | adverse_fundamental | data_gap | no_clear_invalidator`) — dropped `analogy_mismatch` because Rule 4’s strong default to `none_found` makes analogy-driven invalidations a near-zero category.
11. **Thinking mode OFF on Sonnet 4.6** for JSON stability. Latency target: \~5–8s cloud.
12. **Memory — advocate reads, doesn’t defer, but MUST flag shared-assumption risk** (Rule 23). # v1.6c: corrected from 24→23 (renumbered in v1.6 when old Rules 22+23 merged). If the prior bull call was scored wrong AND the current `bull_archetype` matches the prior archetype, `weakest_point` must explicitly acknowledge the prior failure and name the load-bearing assumption that’s shared.
13. **Numeric grounding consolidated into a single rule** (v1.6). The old Rule 22 (numeric plausibility check) and Rule 23 (verbatim quoting at low confidence) were doing related work and have been merged into a single Rule 22. Behavior is unchanged; the rule is shorter and the awkward “Rule 22 isn’t a real guardrail” hedge has moved to design notes.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- No direct API calls — consumes Pass 1 agent outputs (this agent doesn't call external data providers; its evidence base is entirely Pass 1's prior work)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload):
- Compressed Pass 1 outputs for all five agents (RSRCH, FUND, TECH, SENT, MACRO) via `compress_pass1_for_advocate()` — each block \~500–800 tokens, comprising `assessment_summary`, `reliability_score`, `data_quality_assessment`, `caveats`, full `structured_data`, and `narrative` truncated to 300 tokens
- Pass 1 reliability warnings (`{pass1_reliability_warnings}`) — flags agents scoring \<70 ("use with caution") or \<40 ("treat as unreliable"), built by `build_pass1_reliability_warnings()`
- Researcher's `thesis_archetype` (`{researcher_thesis_archetype}`), extracted from `researcher_output.pass2_view.thesis_archetype` — falls back to `\"unclassified\"` if the Researcher failed; provided as an input signal only, not a constraint
- Timeline and account-type instruction blocks (`{timeline_instruction}`, `{account_instruction}`) injected from conditional lookup tables based on `AnalysisContext.timeline` / `account_type`
- Prior-analysis memory brief (`{memory_brief}`) from `StockAnalysisMemory` — prior bull confidence, thesis summary, and scored outcome — included only within timeline-scaled decay windows (30/90/180 days for short/medium/long term), else empty
- Accuracy brief and Winning Patterns brief from the feedback engine — both omitted entirely until ≥20 scored predictions exist
- Missing-agent flags (`{missing_agents_list}`) — any Pass 1 agent that failed or fell below its minimum data threshold is replaced with a "NOT AVAILABLE" placeholder the advocate is barred from citing
- Total compressed-payload target \~4,000 tokens (5 Pass 1 blocks × \~700 tokens avg, plus context header)
---
# System Prompt
Invoke with `enable_thinking: false`. The `build_messages()` method renders this template.

**Runtime prompt:** see [`prompts/bull_advocate/v1.6.txt`](../../prompts/bull_advocate/v1.6.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`):
<table header-row="true">
<colgroup>
<col>
<col width="1179">
</colgroup>
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks). Weight near-term catalysts: earnings, product launches, technical breakouts, event-driven upside. Sentiment and Technical inputs should carry more weight than Macro. A strong short-term bull case is catalyst-rich and has supportive tape.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Balance catalysts with business quality and sector positioning. Fundamental trajectory and management execution matter. Multi-quarter earnings momentum and emerging secular themes are the strongest medium-term bull signals.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). Weight durable moats, secular tailwinds, compounding growth, and management track record. De-emphasize single-quarter catalysts unless they structurally change the thesis. A strong long-term bull case is a compounder story, not a trade.`</td>
</tr>
</table>
**Account** (inject based on `AnalysisContext.account_type`):
<table header-row="true">
<colgroup>
<col>
<col width="1180">
</colgroup>
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Gains and Canadian dividends tax-free. Emphasize COMPOUND CAPITAL APPRECIATION over multi-year horizons — this is where TFSA room is most valuable. For US-listed holdings, note that 15% US dividend WHT is NOT recoverable in a TFSA, so dividend-heavy US stocks are a sub-optimal TFSA fit; argue for growth/capital-gain-dominant theses.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US dividends WHT-exempt under treaty. The bull case can LEAN INTO DIVIDEND YIELD AND GROWTH for US dividend payers (REITs, telecoms, financials) as part of total return. Explicitly argue yield + dividend growth trajectory: "a 3% yield growing at 8%/year is a powerful compounder in tax-deferred space." For Canadian holdings, dividend tax credit is lost inside RRSP — prefer capital appreciation theses for Canadian names.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Non-registered (Trading) account. Capital gains taxed at 50% inclusion. Eligible Canadian dividends receive the dividend tax credit, which makes Canadian dividend payers a tax-efficient fit here when TFSA/RRSP room is constrained. Near-term catalysts and sentiment shifts are valid bull signals, but DO NOT assume rapid in-and-out trading — at retail scale, fees and the 30-day superficial loss rule erode short-holding-period theses. If your thesis depends on quick exits, flag this explicitly in weakest_point.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General analysis — no specific account. Produce a bull case that could apply across account types. Surface in context_aware_strongest_argument which account type this thesis fits best (e.g., "strongest in TFSA due to pure capital-gain nature").`</td>
</tr>
</table>
---
# Output Schema (LLM-produced fields)
The orchestrator populates metadata and derived context fields. The LLM produces the interpretive fields below.
```javascript
{
  "recommendation": "bullish",
  "confidence": 0-100,
  "thesis_summary": "2-3 sentences. The bull case in its most compressed form. Must reference the timeline and account context.",
  "strongest_argument": "1-2 sentences. The single most compelling bull argument. Must cite a Pass 1 agent ID.",
  "weakest_point": "1-2 sentences. What would INVALIDATE the bull thesis — a negative catalyst, adverse fundamental, or data gap. Must cite a Pass 1 agent ID or explicitly state 'no Pass 1 input contradicts this thesis' if genuinely true.",
  "weakest_point_type": "negative_catalyst|adverse_fundamental|data_gap|no_clear_invalidator",
  "caveats": ["specific data gaps, reliability flags, or archetype-reclassification notes"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "positive", "evidence": "PASS1_ID: compact citation"}
  ],
  "thesis_risks": [
    {"risk": "risk TO THIS BULL THESIS (not general risk to stock)", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "PASS1_ID: compact citation"}
  ],
  "narrative": "250-400 words (stop at 400 words). Persuasive bull case, grounded, references specific Pass 1 findings. This is the document the CIO reads.",
  "structured_data": {
    "core_arguments": [
      {
        "argument": "specific falsifiable claim",
        "supporting_pass1_agents": ["RSRCH", "FUND"],
        "evidence": "compact citation(s)",
        "strength": "primary|secondary|tertiary"
      }
    ],
    "catalysts": [
      {
        "catalyst": "specific near/medium-term event",
        "timing": "within_4_weeks|1_to_3_months|3_to_12_months|1_plus_years",
        "probability": "high|medium|low",
        "supporting_pass1_agents": ["TECH"],
        "evidence": "compact citation"
      }
    ],
    "valuation_argument": {
      "claim": "why current price undervalues the stock",
      "supporting_pass1_agents": ["FUND"],
      "evidence": "compact citation"
    },
    "market_misreads": [
      {
        "misread": "what the market is underweighting, ignoring, or mispricing",
        "why_this_persists": "1 sentence on why this inefficiency exists",
        "supporting_pass1_agents": ["SENT", "RSRCH"],
        "evidence": "compact citation"
      }
    ],
    "historical_analogy": {
      "company": "real company name, or 'none_found'",
      "period": "e.g., '2015-2018' or 'n/a' if none_found",
      "situation_match": "1-2 sentences on what makes this a valid analogy",
      "key_differences": "1 sentence on what differs — this is where analogies break",
      "outcome_cited": "what happened to the analogy company — return, time frame",
      "analogy_fit": "strong|moderate|weak|none"
    },
    "thesis_archetype_alignment": {
      "researcher_archetype": "injected from Pass 1 RSRCH — reference only",
      "bull_archetype": "bull's own choice from the controlled vocabulary",
      "alignment_with_researcher": "agrees|disagrees",
      "disagreement_reason": "non-empty only if disagrees, else empty string"
    },
    "context_aware_strongest_argument": "for this {timeline} + {account_type} combo, which single aspect matters most",
    "asymmetry_assessment": "one plain-language sentence: what does the bull case look like if it works, and what's the cost if it doesn't"
  }
}
```
### Array bounds (enforced by validator)
<table header-row="true">
<tr>
<td>Field</td>
<td>Min</td>
<td>Max</td>
</tr>
<tr>
<td>`caveats`</td>
<td>0</td>
<td>5</td>
</tr>
<tr>
<td>`key_factors`</td>
<td>2</td>
<td>4</td>
</tr>
<tr>
<td>`thesis_risks` (risks to the thesis)</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>`core_arguments`</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`catalysts`</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`market_misreads`</td>
<td>1</td>
<td>3</td>
</tr>
</table>
> **Empirical baseline (2026-05-15 simulation run):** `core_arguments` max raised from 3 → 4 because the model consistently produced 4 well-supported arguments on multi-thesis stocks. Primary-strength requirement loosened from "exactly 1" to "at least 1" for the same reason — multi-archetype bull theses sometimes carry two co-equal primaries (e.g., quality + cyclical).
`caveats` has a 0 floor because a clean, high-reliability dataset may not require caveats. `thesis_risks` has a ≥1 floor because every bull thesis has at least one load-bearing assumption — if the agent can’t name what would invalidate its thesis, the thesis isn’t falsifiable and therefore isn’t useful.
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name` (`"bull_case_advocate"`), `agent_pass` (`"pass2"`), `analysis_context`, `data_sources_used`, `data_quality_assessment`
- **`data_quality_assessment`**: Derived from the average reliability_score of the Pass 1 inputs: `>=70 avg and min >= 40` = `"high"`, `>=50 avg` = `"medium"`, `>=30 avg` = `"low"`, else `"insufficient"`.
- **`recommendation`**: Hard-coded to `"bullish"` by the orchestrator (defense in depth — validator also enforces).
- **`reliability_factors`**** intentionally omitted**: Pass 2 agents do not produce `reliability_factors`. Reliability is already encoded in the Pass 1 `reliability_score` inputs — the bull’s `data_quality_assessment` captures the net picture.
---
# Orchestrator Pre-computation: Pass 1 Compression & Reliability Summary
Before invoking the bull advocate, the orchestrator compresses each Pass 1 output and computes a reliability summary for injection.
```python
def compress_pass1_for_advocate(pass1_outputs: list[AgentOutput]) -> dict:
    """Compresses each Pass 1 agent output to ~500-800 tokens and builds
    the advocate's reliability summary."""
    compressed = {}
    reliability_warnings = []

    for output in pass1_outputs:
        agent_id = AGENT_ID_MAP[output.agent_name]  # RSRCH, FUND, TECH, SENT, MACRO
        compressed[agent_id] = {
            "assessment_summary": output.assessment_summary,
            "reliability_score": output.reliability_score,
            "data_quality_assessment": output.data_quality_assessment,
            "caveats": output.reliability_factors.caveats,
            "structured_data": output.structured_data,
            "narrative": truncate_tokens(output.narrative, 300),
        }

        if output.reliability_score < 40:
            reliability_warnings.append(
                f"{agent_id} reliability_score={output.reliability_score} — treat as unreliable"
            )
        elif output.reliability_score < 70:
            reliability_warnings.append(
                f"{agent_id} reliability_score={output.reliability_score} — use with caution"
            )

    return {"compressed_outputs": compressed, "reliability_warnings": reliability_warnings}
```
Note: `build_pass1_reliability_warnings()` is defined in the Orchestration Engine doc (canonical implementation). It returns an all-clear string when all agents \>=70, or a pipe-separated warning line when any agent is below threshold. This prompt doc previously contained a conflicting local implementation — that has been removed.
# Always follow the Orchestration Engine version.
**Researcher archetype extraction**: The orchestrator reads `researcher_output.pass2_view.thesis_archetype` and injects it into the STOCK-SPECIFIC CONTEXT block as `{researcher_thesis_archetype}`. If the Researcher failed or returned `insufficient_data`, the orchestrator sets `researcher_thesis_archetype = "unclassified"`. # v1.6c: corrected from structured_data — always use pass2_view (stable contract).
---
# User Message (Data Payload)
Pass 2 does NOT receive raw filings, transcripts, or news — those were digested by Pass 1.
```javascript
{ticker} ({company_name}) | {sector} | {primary_exchange} | {currency}
As of: {data_timestamp} | Timeline: {timeline} | Account: {account_type}

PASS 1 — STOCK RESEARCHER (RSRCH):
  reliability_score: {rsrch.reliability_score}/100| quality: {rsrch.data_quality_assessment}
  assessment_summary: {rsrch.assessment_summary}
  thesis_archetype: {rsrch.structured_data.thesis_archetype}
  competitive_position: {rsrch.structured_data.competitive_position}
  moat: overall={rsrch.structured_data.moat_assessment.overall_moat_durability} trend={rsrch.structured_data.moat_assessment.moat_trend}
  management: {rsrch.structured_data.management_assessment}
  growth_drivers: {rsrch.structured_data.growth_drivers}
  competitive_threats: {rsrch.structured_data.competitive_threats}
  recent_developments: {rsrch.structured_data.recent_developments}
  caveats: {rsrch.caveats}
  narrative(key points): {rsrch.narrative_truncated_300}

PASS 1 — FUNDAMENTAL ANALYST(FUND):
  reliability_score: {fund.reliability_score}/100 | quality: {fund.data_quality_assessment}
  assessment_summary: {fund.assessment_summary}
  valuation: {fund.structured_data.valuation_assessment}
  key_metrics: P/E={fund.pe_ratio} vs sector {fund.sector_pe_median} | FCF_yield={fund.fcf_yield} | ROE={fund.roe} | D/E={fund.debt_to_equity}
  growth_trajectory: {fund.structured_data.growth_trajectory}
  profitability_trend: {fund.structured_data.profitability_trend}
  balance_sheet_health: {fund.structured_data.balance_sheet_health}
  dividend_sustainability: {fund.structured_data.dividend_sustainability}
  caveats: {fund.caveats}
  narrative (key points): {fund.narrative_truncated_300}

PASS 1 — TECHNICAL ANALYST (TECH):
  reliability_score: {tech.reliability_score}/100| quality: {tech.data_quality_assessment}
  assessment_summary: {tech.assessment_summary}
  trend: {tech.structured_data.trend_direction}
  momentum: {tech.structured_data.momentum_regime}
  key_levels: support={tech.structured_data.key_support}| resistance={tech.structured_data.key_resistance}
  volume_pattern: {tech.structured_data.volume_pattern}
  technical_signal: {tech.structured_data.aggregate_signal}
  caveats: {tech.caveats}
  narrative(key points): {tech.narrative_truncated_300}

PASS 1 — SENTIMENT ANALYST(SENT):
  reliability_score: {sent.reliability_score}/100 | quality: {sent.data_quality_assessment}
  assessment_summary: {sent.assessment_summary}
  news_sentiment: {sent.structured_data.news_sentiment}
  narrative_momentum: {sent.structured_data.narrative_momentum}
  crowded_positioning: {sent.structured_data.positioning_assessment}
  analyst_consensus: {sent.structured_data.analyst_consensus} | avg_target: {sent.structured_data.avg_price_target}
  caveats: {sent.caveats}
  narrative (key points): {sent.narrative_truncated_300}

PASS 1 — MACRO ECONOMIST (MACRO):
  reliability_score: {macro.reliability_score}/100| quality: {macro.data_quality_assessment}
  assessment_summary: {macro.assessment_summary}
  overall_macro_environment: {macro.structured_data.overall_macro_environment}
  sector_cycle_position: {macro.structured_data.sector_cycle_position}
  rate_impact: {macro.structured_data.interest_rate_environment.impact_on_stock}| direction: {macro.structured_data.interest_rate_environment.current_direction}
  inflation_impact: {macro.structured_data.inflation_environment.impact_on_stock}| trend: {macro.structured_data.inflation_environment.trend}
  growth_outlook: {macro.structured_data.economic_growth.outlook}| impact: {macro.structured_data.economic_growth.impact_on_stock}
  currency_impact: {macro.structured_data.currency_impact.impact_on_stock}| trend: {macro.structured_data.currency_impact.cad_usd_trend}
  volatility_regime: {macro.structured_data.volatility_regime}
  sector_tailwinds: {macro.structured_data.sector_tailwinds}
  sector_headwinds: {macro.structured_data.sector_headwinds}
  commodity_context: {macro.structured_data.commodity_context}
  # v1.6c: updated to Macro v1.2 field names. Removed stale v1.0 fields
  #(sector_macro_posture, rate_environment_impact, fx_exposure_assessment)
  #         and geopolitical_factors(removed from Macro v1.2 structured output).
  caveats: {macro.caveats}
  narrative(key points): {macro.narrative_truncated_300}

RESEARCHER ARCHETYPE FOR THIS STOCK: {researcher_thesis_archetype}

DATA GAPS / MISSING PASS 1 AGENTS:
  {missing_agents_list}
```
**Payload notes**:
- If a Pass 1 agent failed or fell below its minimum data threshold, its block is replaced with: `PASS 1 — {AGENT} (ID): NOT AVAILABLE — reason: {reason}`. The bull must not cite that agent.
- Agent IDs (`RSRCH`, `FUND`, `TECH`, `SENT`, `MACRO`) are stable across retries.
- Each Pass 1 block targets 500-800 tokens. Total payload target \~4,000 tokens.
---
# Orchestrator Merge Logic
```python
def merge_output_bull_advocate(llm_response: dict, pass1_outputs: list[AgentOutput],
                               context: AnalysisContext) -> dict:
    # Compute data_quality_assessment from Pass 1 reliability scores
    scores = [o.reliability_score for o in pass1_outputs if o is not None]
    if not scores:
        dqa = "insufficient"
    else:
        avg = sum(scores) / len(scores)
        min_score = min(scores)
        if avg >= 70 and min_score >= 40: dqa = "high"
        elif avg >= 50: dqa = "medium"
        elif avg >= 30: dqa = "low"
        else: dqa = "insufficient"

    # Hard-code recommendation (defense in depth; validator also enforces)
    llm_response["recommendation"] = "bullish"

    return {
        "agent_name": "bull_case_advocate",
        "agent_pass": "pass2",
        "analysis_context": {
            "account_type": context.account_type,
            "timeline": context.timeline,
            "stock_id": str(context.stock_id),
            "canonical_ticker": context.canonical_ticker,
        },
        "recommendation": "bullish",
        "confidence": llm_response["confidence"],
        "thesis_summary": llm_response["thesis_summary"],
        "strongest_argument": llm_response["strongest_argument"],
        "weakest_point": llm_response["weakest_point"],
        "weakest_point_type": llm_response["weakest_point_type"],
        "caveats": llm_response["caveats"],  # v1.6c: was omitted — LLM-produced caveats were silently dropped from merged output.
        "key_factors": llm_response["key_factors"],
        "thesis_risks": llm_response["thesis_risks"],
        "data_quality_assessment": dqa,
        "narrative": llm_response["narrative"],
        "structured_data": llm_response["structured_data"],
        "data_sources_used": [p.agent_name for p in pass1_outputs if p is not None],
    }
```
---
# Validation Rules
Enforced by `src/agents/validators.py`. Failure triggers retry with error list appended to user message.
1. **JSON parseable** (strip markdown fences/preamble if present).
2. **Required fields**: `recommendation`, `confidence`, `thesis_summary`, `strongest_argument`, `weakest_point`, `weakest_point_type`, `caveats`, `key_factors`, `thesis_risks`, `narrative`, `structured_data`.
3. **recommendation**: Must equal `"bullish"` exactly. If the LLM returns “neutral” or “bearish”, validation fails with: `"You are the Bull Case Advocate — recommendation must be 'bullish'. If the bull case is weak, return bullish with a low confidence score and an honest weakest_point."`
4. **confidence**: Integer 0–100.
5. **thesis_summary**: Non-empty, 80–500 characters, must reference the timeline or account_type.
6. **strongest_argument**: Non-empty, 40–400 characters, must contain at least one Pass 1 agent ID token (`RSRCH`, `FUND`, `TECH`, `SENT`, `MACRO`).
7. **weakest_point + weakest_point_type (paired validation)**:
	- `weakest_point`: Non-empty, 40–400 characters.
		- `weakest_point_type`: Must be one of `negative_catalyst | adverse_fundamental | data_gap | no_clear_invalidator`.
		- If `weakest_point_type` is `no_clear_invalidator`, confidence must be \>= 70.
		- If `weakest_point_type` is NOT `no_clear_invalidator`, `weakest_point` must contain at least one Pass 1 agent ID token.
		- If `confidence >= 80`, `weakest_point_type` must NOT be `no_clear_invalidator`.
		- **Validator does NOT attempt to check whether the cited Pass 1 finding is genuinely a bull-thesis weakness or a bear rebuttal dressed as weakness.** That semantic check is unenforceable given only the text string. Correctness is evaluated retrospectively by the Feedback Analyst.
8. **caveats**: 0–5 items. If the payload included a reliability_warnings block (any Pass 1 agent \<70), caveats must contain at least one entry explicitly acknowledging at least one flagged agent.
9. **key_factors**: 2–4 items. Every entry must have `sentiment: "positive"`. Evidence must start with a valid Pass 1 agent ID token followed by `:`.
10. **thesis_risks**: 1–3 items. Each evidence field must start with a valid Pass 1 agent ID token OR the phrase `"thesis assumption:"` (for load-bearing assumptions not tied to specific Pass 1 findings).
11. **narrative**: 250–400 words (stop at 400 words; \~1,800–2,800 characters). Must reference at least 3 distinct Pass 1 agent IDs. Empirical 2,800-char ceiling supersedes the prior 2,400 limit — see Array bounds note. Length is enforced; the validator performs a sentence-boundary auto-trim in `base.py` before retrying, so narratives that overshoot by ≤1 sentence are repaired in-place rather than re-rolled.
12. **structured_data.core_arguments**: 1–4 items. Each requires `argument` (non-empty), `supporting_pass1_agents` (non-empty list of valid IDs), `evidence` (non-empty, with Pass 1 agent ID), `strength` (`primary`, `secondary`, or `tertiary`). At least one entry must be `strength: "primary"`.
13. **structured_data.catalysts**: 1–4 items. At least one catalyst’s `timing` must be consistent with the analysis `timeline`:
	- `short_term` → at least one catalyst with `timing: "within_4_weeks"` or `"1_to_3_months"`
		- `medium_term` → at least one catalyst with timing in `within_4_weeks`, `1_to_3_months`, or `3_to_12_months`
		- `long_term` → at least one catalyst with timing in `3_to_12_months` or `1_plus_years`
14. **structured_data.valuation_argument**: all sub-fields non-empty. Preferred (not required) Pass 1 agent: `FUND`.
15. **structured_data.market_misreads**: 1–3 items, all sub-fields non-empty.
16. **structured_data.historical_analogy**: Required field, but `company == "none_found"` is the expected default. If `company == "none_found"`, all other sub-fields must be empty/n/a and `analogy_fit == "none"`. If `company != "none_found"`, `analogy_fit` must be `strong` or `moderate` — a `weak` analogy is inadmissible.
17. **structured_data.thesis_archetype_alignment**: `researcher_archetype` matches the injected value exactly. `bull_archetype` is one of the 5 controlled vocabulary archetypes: `secular_grower | dividend_compounder | cyclical_recovery | quality_compounder | value_trap_candidate`. # v1.6c: corrected from stale ‘8 archetypes’ to Researcher v1.3 5-value set (dropped: turnaround, disrupted_incumbent, special_situation). `alignment_with_researcher` is `"agrees"` if `bull_archetype == researcher_archetype`, else `"disagrees"`. If `"disagrees"`, `disagreement_reason` must be non-empty AND contain at least one Pass 1 agent ID. If `"agrees"`, `disagreement_reason` is the empty string.
18. **structured_data.context_aware_strongest_argument**: Non-empty, must reference the timeline OR account_type.
19. **structured_data.asymmetry_assessment**: Non-empty, plain-language. No format requirement (free sentence).
20. **Reliability-coupling rule**: If 3+ Pass 1 inputs had `reliability_score < 40`, `confidence` must be `<= 45`.
21. **Confidence-weakness coupling**: If `confidence >= 80`, `weakest_point_type` must NOT be `no_clear_invalidator`.
22. **Numeric grounding (consolidated v1.6)** — combines the old Rules 22 and 23:
	- **Numeric plausibility (always-on)**: Scan the narrative and all evidence fields for numeric tokens matching `\d+\.?\d*%` or `\$\d+`. Every such token must appear somewhere in the input payload (exact-string match). Missing tokens trigger a warning and a single retry.
		- **Verbatim quoting (low confidence only)**: If `confidence < 35`, every `structured_data.core_arguments[].evidence` field must additionally contain an EXACT substring from the Pass 1 payload (copied verbatim, not paraphrased), wrapped in double quotes within the evidence string. Example valid form: `"FUND: \"D/E=8.2, interest coverage 1.1x\" — equity as call option on deleveraging"`. Example invalid: `"FUND: debt extremely high, call option dynamics"` (paraphrased). Validator checks via substring match against the concatenated payload. Case-sensitive. Whitespace-tolerant.
		- **Documented limits**: The plausibility check only catches raw numeric hallucinations, not fabricated reasoning chains connecting real numbers in false ways; ratios like `3:1` are not caught by the regex; short numeric tokens may produce coincidental matches. The plausibility check is a cheap sanity guard, NOT a fabrication guardrail. The verbatim-quoting subrule is the real low-confidence guardrail.
23. **Shared-assumption memory rule (formerly Rule 24)**: If the memory brief is non-empty AND `if_scored_prior_was == "incorrect"` AND `prior_thesis_archetype == llm_response.structured_data.thesis_archetype_alignment.bull_archetype`, then `weakest_point` MUST contain BOTH: (a) an explicit reference to the prior failure (one of the phrases: `"prior bull case"`, `"prior thesis"`, `"previously wrong"`, `"repeated"`, `"replay"`, or equivalent), AND (b) a named load-bearing assumption that the current thesis shares with the prior one. Validator compares the archetype strings directly. If the memory brief is empty or the prior was scored correct or the archetypes differ, this rule is inert. **Failure mode this prevents**: the bull constructs the same structurally flawed thesis on the same stock run after run because the advocate frame encourages it to ignore its own history.
---
# Retry Prompt Injection
```javascript
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember:
- recommendation must be "bullish" — you are the advocate, not a judge
- every claim must cite a Pass 1 agent ID (RSRCH, FUND, TECH, SENT, MACRO)
- at least one catalyst must be timed consistent with the analysis timeline (timing enum: within_4_weeks | 1_to_3_months | 3_to_12_months | 1_plus_years)
- historical_analogy must be a real named company or "none_found"
- weakest_point must identify what would INVALIDATE the bull thesis (negative catalyst, adverse fundamental, or data gap) — not rebut the bear case
- weakest_point_type must be one of: negative_catalyst | adverse_fundamental | data_gap | no_clear_invalidator
- thesis_risks entries are things that would INVALIDATE the bull case
- if confidence >= 80, weakest_point_type must NOT be "no_clear_invalidator"
```
---
# Memory Brief Schema
The `{memory_brief}` placeholder is populated from `StockAnalysisMemory` if the stock has been analyzed before.
```javascript
Prior Bull Case ({days_since} days ago, timeline={prior_timeline}):
  confidence_then: {prior_confidence}/100
  thesis_summary_then: {prior_thesis_summary}
  strongest_argument_then: {prior_strongest_argument}
  outcome: {return_pct} over {holding_period_days} days(scored: yes|no|not_yet)
  if_scored_prior_was: correct|incorrect|partially_correct
  what_changed_since: {short_list_of_material_events}
```
**Timeline-scaled decay**: Included if \<30 days old at short_term, \<90 days at medium_term, \<180 days at long_term. Below threshold, the brief is omitted entirely. # v1.6c: replaced vague ‘full-weight/reduced/marginal’ language with explicit thresholds matching Researcher v1.3 and Sentiment v1.2c cross-agent standard.
**How the advocate uses memory**: The bull reads the prior bull case but is NOT required to temper its current case because the prior was wrong. HOWEVER, per Rule 23, if the prior was scored wrong AND the current `bull_archetype` matches the prior archetype, `weakest_point` MUST explicitly acknowledge the prior failure and identify the load-bearing assumption being replayed.
**Empty case**: If no prior analysis exists or memory is below marginal-weight threshold, `{memory_brief}` renders as empty string.
---
# Accuracy & Winning Patterns Briefs (Post-20 Predictions)
Both briefs are omitted entirely when N \< 20 scored predictions.
**Accuracy brief** (\~80 tokens):
```javascript
YOUR ACCURACY: {N} scored bull predictions. Directional accuracy: {X}%. For this sector ({sector}): {X}%. For this timeline ({timeline}): {X}%. Known bias: {e.g., "you tend to over-confident on small-cap tech during risk-on regimes (+7pts vs actual)"}. Adjustment: {e.g., "consider reducing confidence by 5-10 points for small-cap tech in current regime"}.
```
**Winning patterns brief** (\~120 tokens):
```javascript
YOUR WINNING PATTERNS:
Most accurate recent bull call: {ticker} on {date} — confidence {X}, actual return {Y}%.
- Key insight that worked: {the reasoning that proved correct}
- Reasoning pattern that consistently works for you: {e.g., "when FUND shows FCF/net-income divergence improving AND RSRCH shows management capital return, your bull calls hit 74% directionally"}
- Your strongest stock characteristics: {e.g., "large-cap cash-generative names with improving capital efficiency"}
Double down on these patterns when you see them in the current analysis.
```
---
# Minimum Data Threshold
Per Agents Overview Gate 1: at least 3 of 5 Pass 1 agents must have `reliability_score >= 30`, and either FUND or RSRCH must be among them.
**Bull-advocate-specific behavior below threshold**: If Gate 1 fails, the orchestrator does NOT invoke the bull advocate at all — it short-circuits to a stub output with `confidence: 0`, `thesis_summary: "insufficient data for bull thesis"`, and the run is marked `completed_with_warnings`.
**Edge case — Gate 1 passes but one of FUND/RSRCH specifically is unavailable**: The bull advocate IS invoked. It must acknowledge the missing agent in `caveats` and cap `confidence <= 55`.
---
# Prompt Variable Injection Reference
<table header-row="true">
<tr>
<td>Placeholder</td>
<td>Source</td>
<td>Notes</td>
</tr>
<tr>
<td>`{ticker}`</td>
<td>`AnalysisContext.canonical_ticker`</td>
<td>Real ticker (NOT anonymized at Pass 2)</td>
</tr>
<tr>
<td>`{company_name}`</td>
<td>`DataBundle.company_info["name"]`</td>
<td>Real name (NOT anonymized at Pass 2)</td>
</tr>
<tr>
<td>`{sector}`</td>
<td>`DataBundle.company_info["sector"]`</td>
<td></td>
</tr>
<tr>
<td>`{timeline}`</td>
<td>`AnalysisContext.timeline`</td>
<td>short_term / medium_term / long_term</td>
</tr>
<tr>
<td>`{account_type}`</td>
<td>`AnalysisContext.account_type`</td>
<td>tfsa / rrsp / trading / general</td>
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
<td>`{researcher_thesis_archetype}`</td>
<td>`researcher_output.pass2_view.thesis_archetype`</td>
<td>Falls back to `"unclassified"` if Researcher failed. **v1.6c: corrected from structured_data — always use pass2_view (stable contract). Injected in STOCK-SPECIFIC CONTEXT, not stable prefix.**</td>
</tr>
<tr>
<td>`{pass1_reliability_warnings}`</td>
<td>`build_pass1_reliability_warnings()`</td>
<td>Rendered from Pass 1 reliability summary</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory` query</td>
<td>Empty string if none / below marginal weight</td>
</tr>
<tr>
<td>`{accuracy_brief}`</td>
<td>Feedback engine</td>
<td>Empty string if N \< 20 scored predictions</td>
</tr>
<tr>
<td>`{winning_patterns_brief}`</td>
<td>Feedback engine</td>
<td>Empty string if N \< 20 scored predictions</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
<td>The schema above</td>
</tr>
<tr>
<td>Pass 1 block placeholders</td>
<td>Compressed `AgentOutput` for each Pass 1 agent</td>
<td>Missing agent → “NOT AVAILABLE — reason: \{reason\}”</td>
</tr>
</table>
---
# Testing Checklist
- Does Claude Sonnet 4.6 with `enable_thinking: false` produce valid JSON matching the schema?
- Does the model return `recommendation: "bullish"` even when Pass 1 is uniformly negative?
- Does the model return a low `confidence` (\<30) when the bull case is genuinely weak, rather than inflating to 50+?
- Does the model avoid clustering `confidence` at 65-75? Test with 5 stocks across the spectrum and verify confidence spreads broadly.
- Does every `key_factors`, `thesis_risks`, `core_arguments`, `catalysts`, and `market_misreads` evidence field cite a valid Pass 1 agent ID?
- Does the narrative reference at least 3 distinct Pass 1 agents?
- Does `historical_analogy` default to `"none_found"` for the majority of stocks?
- **Archetype independence test**: Pass a stock where Researcher classified as `value_trap_candidate`. Does the bull exercise independent judgment?
- **Timeline catalyst test**: With `timeline=short_term`, does at least one catalyst have timing `within_4_weeks` or `1_to_3_months`? With `timeline=long_term`, does at least one catalyst have timing `3_to_12_months` or `1_plus_years`? (Validation Rule 13)
- **Reliability coupling test**: Pass a payload where 3 of 5 Pass 1 agents have reliability \< 40. Does `confidence <= 45`? (Validation Rule 20)
- **Confidence-weakness coupling test**: Pass a strong-bull payload. If the model returns confidence \>= 80, is `weakest_point_type != no_clear_invalidator`? (Validation Rule 21)
- **weakest_point_type distribution test**: Run on 10 stocks across the confidence range. Confirm the four-value enum varies meaningfully — not always the same value. Failure modes: model always returns `no_clear_invalidator` to dodge the constraint, or always `data_gap` as a generic escape hatch.
- **Dual-signal confidence independence test**: Run Bull and Bear on a genuinely dual-signal stock. Confirm both advocates can return moderate-to-high confidence simultaneously (e.g., Bull 60 + Bear 55).
- **Account adaptation test**: Pass a US dividend stock (e.g., JNJ) through TFSA vs RRSP. Do the bull cases differ meaningfully — TFSA warns on WHT drag / favors capital gains, RRSP leans into dividend compounding?
- **Trading account test (v1.6)**: Pass a Canadian dividend payer through `account_type=trading`. Does the bull cite the dividend tax credit as a tax-efficiency argument when relevant? Conversely, if the thesis depends on a fast in-and-out, is this flagged in `weakest_point` per the new account block?
- **Retail-scale test (v1.6)**: Pass an illiquid micro-cap or a name where the natural thesis is special-situation/sum-of-parts. Does the bull either (a) decline the thesis with low confidence and a frank `weakest_point`, or (b) frame the case in one of the three retained Rule 7 frames? Failure mode: bull invents an institutional-flavored thesis (private placements, activist play, etc.) that retail can’t act on.
- **Weak-bull honesty test (critical)**: Pass a high-quality bear setup. Does the model (a) return low confidence (\< 35), (b) frame arguments as one of the THREE high-bar bull frames per Rule 7, (c) quote verbatim substrings in `core_arguments[].evidence` per Rule 22, (d) produce a frank `weakest_point` with a valid `weakest_point_type` enum value?
- **Missing agent test**: Drop SENT from the payload. Does the model avoid citing SENT and still produce a valid output?
- **Thesis risks test**: Are the `thesis_risks` entries genuinely risks to the BULL THESIS, not just general risks to the stock?
- **Numeric grounding test (Rule 22, v1.6 consolidated)**: Grep the narrative for numbers; verify every number appears in the payload. On low-confidence runs, also confirm `core_arguments[].evidence` contains at least one double-quoted verbatim substring.
- **Shared-assumption memory test (Rule 23)**: Construct a memory brief with `if_scored_prior_was: incorrect` and `prior_thesis_archetype: quality_compounder`. Run the bull on a fixture stock where the current payload also suggests `quality_compounder`. Confirm (a) the bull chooses `bull_archetype: quality_compounder`, (b) `weakest_point` contains BOTH a prior-failure reference phrase AND a named shared load-bearing assumption, (c) confidence is NOT forced down by the rule.
- **Local fallback test** (`qwen3.6:35b-a3b`): How much quality degradation? Expect shallower arguments, weaker analogies, more generic catalysts — re-establish this baseline for the new model rather than assuming it matches `qwen3:14b`'s prior behavior.
- **Fixture stocks**: NVDA (strong bull, secular grower), NFLX (moderate bull, quality compounder in maturity), AQN (weak bull, dividend compounder with recent cut), a struggling retailer (near-impossible bull — should return confidence \< 25 honestly)
---
# Token Budget
<table header-row="true">
<tr>
<td>Component</td>
<td>Budget</td>
<td>Notes</td>
</tr>
<tr>
<td>System prompt</td>
<td>\~700 tokens</td>
<td>v1.6: Rule 7 reduced to 3 patterns, Rules 22/23 merged, CoT to 5 steps. \~100-token reduction vs v1.5.</td>
</tr>
<tr>
<td>User message (compressed Pass 1)</td>
<td>\~4,000 tokens</td>
<td>5 Pass 1 blocks × \~700 tokens avg + context header + missing-agent flags</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~1,500 tokens</td>
<td>Structured data + narrative (\~250-400 words) + thesis fields</td>
</tr>
<tr>
<td>Total</td>
<td>\~6,200 tokens (unverified)</td>
<td>See caveat below</td>
</tr>
</table>
**⚠️ Token budget caveat**: The \~4,000 token target for the compressed Pass 1 payload is unmeasured. Each Pass 1 block includes full `structured_data` JSON, `assessment_summary`, `caveats`, truncated `narrative`, plus YAML-ish label overhead. The realistic payload size is likely 5,500–6,500 tokens. **Action before v1 testing**: run `compress_pass1_for_advocate()` on 3 fixture stocks and measure the token count empirically.
**Model configuration (cloud-primary)**:
- Model: `claude-sonnet-4-6`
- `max_tokens`: **5000** (empirical min safe value from 2026-05-15 simulation; the prior 2000 caused silent JSON truncation on every run. Set to min safe + 20% headroom per the Orchestration Engine LLM Call Parameters section.)
- `temperature`: 0.3
- Thinking mode: OFF
- JSON output: prefilled assistant turn with `{` to eliminate markdown fences (see Orchestration Engine § LLM Call Parameters).
- **Prompt caching**: Stable content (role framing, rules 1-8, CoT) precedes the variable block. Cacheable prefix is \~550–650 tokens (v1.6 — slightly smaller than v1.5).
- Expected latency: \~5–8s per stock
**Model configuration (local fallback)**:
- Model: `qwen3.6:35b-a3b` via Ollama
- `num_ctx`: 8192
- `enable_thinking`: false (verify suppression is clean on this model — see Agent Design Patterns Pattern 10 caveat)
- Expected latency: unverified estimate, re-measure after deployment (was \~30s per stock on the prior `qwen3:14b`)
- Expected quality degradation: materially weaker argumentation, generic analogies, formulaic catalysts.
**Cost per stock**: \~\$0.003-0.005 at Sonnet 4.6 pricing (pre-cache).
---
# Open Questions / Future Improvements
- **Bull–Bear shared input vs. differentiated input**: Currently both advocates receive the same compressed Pass 1 payload. Decision for v1: shared payload, let role framing drive divergence. Revisit if outputs are too similar after 20+ runs.
- **Historical analogy quality control**: A later improvement is a post-hoc scoring pass.
- **Asymmetry assessment quantification (post-Phase 2)**: v1.6 reverted `asymmetry_assessment` to plain-language free text. Once FUND produces base/bull/bear scenario ranges (Phase 2), the bull can quote a specific ratio. Re-introduce ratio formality only when the underlying scenario data exists — not before.
- **Memory-bull interaction**: Rule 23 (formerly 24) covers same-archetype repeats. Known gaps: (a) archetype-dodge across runs; (b) partial-correct silence; (c) cross-timeline invisibility. No fix proposed for v1 — document and monitor.
- **Controlled-vocabulary catalyst types**: `catalyst` is free text. A later improvement could use a controlled vocabulary.
- **Temperature sweep**: temp=0.3 is the v1.1 starting point. After first batch, A/B test 0.2 vs 0.3 vs 0.5.
- **Re-introducing dropped Rule 7 frames if needed (v1.6 watch-item)**: v1.6 dropped `special_situation` and `hidden_asset / sum_of_parts` from Rule 7 on the assumption that they almost never apply to a sub-\$250k Canadian retail watchlist. If post-launch the Feedback Analyst surfaces stocks where the bull would have benefited from one of these frames (e.g., a TSX-listed conglomerate with an unrecognized sum-of-parts story), re-introduce selectively rather than reverting the full v1.5 framework.
---
# Related Documents
- Agent Design Patterns — the validated patterns this prompt applies
- Stock Researcher Agent Prompt — Pass 1 reference; source of `thesis_archetype`
- Fundamental Analyst Agent Prompt — the agent FUND refers to
- Bear Case Advocate Agent Prompt — adversarial counterpart
- Agents Overview — Pass 2 output contract, Gate 1/Gate 2 rules
- Accuracy & Winning patterns briefs — injected post-20-prediction calibration
- Orchestration Engine — Pass 1 compression, memory decay

_Source: Notion — Project Stock Picker → Agent Prompts → Bull Case Advocate Agent Prompt. Downloaded 2026-07-02._
