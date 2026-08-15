**Agent**: Bear Case Advocate
**Status:** Draft v2 — simplified for Canadian retail (\<\$250k) audience
**Phase:** Phase 1
**Pass**: Pass 2 (Perspective & Strategy)
**Role**: Build the strongest possible bearish thesis for this stock given the analysis context. Deliberately biased — this agent’s job is persuasion grounded in data, not balance. The adversarial counterpart to the Bull Case Advocate.
**Target Model**: Cloud-primary: `claude-sonnet-4-6` (thinking mode OFF). Local fallback: `qwen3.6:35b-a3b` via Ollama.
**Audience**: Canadian retail investors with total portfolio values under \$250k CAD, holding stocks across TFSA / RRSP / Trading / General accounts. Institutional-grade complexity that doesn’t pay rent at this scale has been trimmed.
**Token Budget**: \~550 system prompt \| \~4,000 data input \| \~1,300 expected output \| \~5,850 total
**Changes (v2c)**:
- Fix A+H: Rule 3 — removed `{researcher_thesis_archetype}` from stable prefix (caching bug); archetype injected in STOCK-SPECIFIC CONTEXT; vocabulary corrected to Researcher v1.3 5-value set (dropped: turnaround, disrupted_incumbent, special_situation)
- Fix B: `caveats` added to `merge_output_bear_advocate()` return dict (was silently dropped)
- Fix C: Injection table: `structured_data.thesis_archetype` → `pass2_view.thesis_archetype` (stable contract — pass2_view is the correct source)
- Fix D: Memory decay thresholds: vague “60-day” language → explicit \<30d/\<90d/\<180d (matches Researcher v1.3 and cross-agent standard)
- Fix H (×3): Design Decision #4, output schema `bear_archetype` description, Validation Rule 16 — all updated from 8-value to 5-value vocabulary
**Design asymmetries (intentional)**:
- **Calibration brief**: Bear uses a single `{calibration_brief}` variable. Bull and Risk Advisor use two separate variables (`{accuracy_brief}` + `{winning_patterns_brief}`). The single combined brief is intentional — Bear's prior-failure pattern is simpler to diagnose (directional miss vs. correct) and doesn't benefit from the separate winning-patterns signal that Bull uses to reinforce high-confidence constructive cases. The injection layer must supply one variable for Bear and two for Bull/Risk Advisor.
- **`weakest_point_type`**: Bear has `weakest_point` as required free text but no `weakest_point_type` enum (Bull has both). Intentional — Bear's invalidation cases (positive catalyst, supportive fundamental, data gap) are fewer and more obvious at this audience scale; the type enum adds validation overhead without commensurate signal.
- **Rule 21 prior-failure trigger**: Bear's trigger is `same bear_archetype OR overlap in primary core_argument themes` (broader than Bull's pure archetype match). Intentional — bearish arguments (valuation, earnings risk) recur across different archetypes more naturally than bull theses, so the OR condition is needed to catch replays that dodge an archetype change.
---
# Design Decisions & Rationale
Built following the Agent Design Patterns and Pass 2 Considerations. v2 simplifies v1 by removing complexity calibrated for institutional-style risk frameworks and short-positioning that is irrelevant to Canadian retail investors with \<\$250k portfolios.
1. **Advocate, not judge — with honesty controls at low confidence.** `recommendation` is hard-coded to `"bearish"` by validation. The Bear’s failure mode is overstating mundane risks as existential threats. Mitigated via (a) high-bar framing at `confidence < 35` (Rule 7) and (b) numeric grounding at low confidence (Rule 19).
2. **Pass 2 input = compressed Pass 1.** Five compressed Pass 1 summaries plus context. Citations refer to Pass 1 agents (`RSRCH`, `FUND`, `TECH`, `SENT`, `MACRO`).
3. **Reliability-aware weighting is mandatory.** Standard tiers: `>=70` trustworthy, 40-69 caution, \<40 unreliable. Bear-specific: a bear case built primarily on low-reliability SENT data is especially suspect — prefer FUND/RSRCH-grounded bear theses.
4. **Researcher archetype as evidence, not constraint.** Bear picks its own archetype from the same 5-value vocabulary as the Researcher (secular_grower \| dividend_compounder \| cyclical_recovery \| quality_compounder \| value_trap_candidate). # v2c: corrected from 8-value (dropped: turnaround, disrupted_incumbent, special_situation — Researcher v1.3 no longer produces these). Disagreement is captured in a single free-text `disagreement_note`.
5. **6-step CoT.** Each step maps to a structured field. Steps: thesis construction, downside triggers, valuation/market-misread, tail-risk assessment, self-critique, context adaptation.
6. **Bounded output arrays** (Pattern 7). `core_arguments` 1-3, `downside_triggers` 1-4, `market_misreads` 1-3, `key_factors` 2-4, `thesis_risks` 1-3.
7. **Cited evidence required on every argument** (Pattern 8). Each `core_argument`, `downside_trigger`, `market_misread` entry cites at least one Pass 1 agent ID.
8. **Context adaptation via conditional injection** (Pattern 4). Timeline and account_type blocks injected from lookup tables.
9. **Self-critique is structured.** `weakest_point` is required free-text, must cite a Pass 1 agent ID, must identify what would *invalidate* the bear thesis (positive catalyst, supportive fundamental, data gap) — not a bull-case rebuttal.
10. **Thinking mode OFF** for JSON stability.
11. **Memory — advocate reads, doesn’t defer.** If a prior bear case on this stock was scored wrong and the current thesis is directionally similar, `weakest_point` must briefly acknowledge the prior failure (Rule 18).
12. **Advocacy is for avoidance, not shorting.** Bear case translates to “sell” or “avoid.” Shorting in registered accounts is not generally permitted; this is captured at the role-framing level rather than repeated in every account block.
13. **Tail risk consideration is required as a forced rule-out.** On fundamentally sound stocks, the expected output is `tail_risk_level: negligible` — that is the correct answer. The feature’s value is the required consideration, not always-present alarming content.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- No direct API calls — consumes Pass 1 agent outputs (this agent doesn't call external data providers itself)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload):
- Compressed Pass 1 outputs for all 5 agents (RSRCH, FUND, TECH, SENT, MACRO) via `compress_pass1_for_advocate()` — shared with the Bull Case Advocate. Each block carries `assessment_summary`, `reliability_score`, `data_quality_assessment`, `caveats`, full `structured_data`, and `narrative` truncated to 300 tokens. Budgeted at \~4,000 tokens total (\~700 tokens/agent avg).
- `{pass1_reliability_warnings}` — auto-generated flags for any Pass 1 agent scoring 40-69 ("use with caution") or \<40 ("treat as unreliable").
- `{researcher_thesis_archetype}` — Stock Researcher's archetype classification (`pass2_view.thesis_archetype`), injected in STOCK-SPECIFIC CONTEXT; falls back to "unclassified" if Researcher failed.
- `{memory_brief}` — prior bear case on this stock (confidence, thesis, outcome, scored correct/incorrect) if within timeline-scaled decay window (\<30d short_term / \<90d medium_term / \<180d long_term); empty otherwise.
- `{calibration_brief}` — single \~150-token combined accuracy + pattern brief from the feedback engine; empty string until N \>= 20 scored bear predictions.
- Gate 1 minimum data threshold enforced upstream: at least 3 of 5 Pass 1 agents must have `reliability_score >= 30`, including FUND or RSRCH — otherwise the orchestrator skips invoking this agent entirely.
---
# System Prompt
Invoke with `enable_thinking: false`. Stable content precedes the `--- STOCK-SPECIFIC CONTEXT ---` separator for prompt caching.

**Runtime prompt:** see [`prompts/bear_advocate/v2.txt`](../../prompts/bear_advocate/v2.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`):
<table header-row="true">
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks). Weight near-term event-driven downside: earnings misses, guidance cuts, technical breakdowns, sentiment reversals, sector rotation. TECH and SENT carry more weight than MACRO. A strong short-term bear case is trigger-rich with a deteriorating tape.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Balance event-driven triggers with structural deterioration. Multi-quarter earnings momentum loss, margin compression, competitive share loss. Dividend sustainability matters if the stock is held partly for yield.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). Weight structural impairment: moat erosion, secular decline, competitive disruption, permanent capital loss risk. De-emphasize single-quarter misses unless they signal structural damage.`</td>
</tr>
</table>
**Account** (inject based on `AnalysisContext.account_type`):
<table header-row="true">
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Gains and Canadian dividends are tax-free, BUT capital losses cannot offset gains elsewhere — a loss in TFSA permanently destroys contribution room. This AMPLIFIES the bear case for speculative or impaired names: the downside is permanent loss of tax-sheltered compounding capacity. Argue for avoidance of any name where tail risk or permanent impairment is elevated.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US dividends are WHT-exempt under treaty, so RRSP is often used for dividend-compounding theses. The PRIMARY bear case angle for RRSP holdings is therefore DIVIDEND SUSTAINABILITY RISK. If FUND flags high payout ratio, declining FCF, earnings coverage deterioration, or rising leverage threatening the dividend, argue this forcefully — a dividend cut destroys both the income thesis AND capital value (typical reaction: -15 to -30% on announcement). For non-dividend holdings, argue standard structural impairment.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading (non-registered) account. Capital losses are deductible against capital gains (50% inclusion). Even moderate-conviction bear calls are actionable because loss-harvesting mitigates the cost of being wrong. Near-term catalysts, technical breakdowns, and sentiment reversals are fair game.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General analysis — no specific account. Produce a bear case applicable across account types. In context_aware_strongest_argument, identify which account type this bear thesis matters MOST in (e.g., "most damaging in TFSA given permanent capital loss risk, least actionable in Trading given loss-harvest availability").`</td>
</tr>
</table>
---
# Output Schema (LLM-produced fields)
The orchestrator populates metadata and derived fields. The LLM produces:
```json
{
  "recommendation": "bearish",
  "confidence": 0,
  "thesis_summary": "2-3 sentences. The bear case in compressed form. Must reference timeline and account context.",
  "strongest_argument": "1-2 sentences. The single most compelling bear argument. Must cite a Pass 1 agent ID.",
  "weakest_point": "1-2 sentences. What would INVALIDATE the bear thesis — a positive catalyst, supportive fundamental, or data gap. Must cite a Pass 1 agent ID, or explicitly state 'no Pass 1 input supports continued ownership' if genuinely true.",
  "caveats": ["specific data gaps, reliability flags, archetype-reclassification notes"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "negative", "evidence": "PASS1_ID: compact citation"}
  ],
  "thesis_risks": [
    {"risk": "thing that would INVALIDATE the bear case — a positive catalyst or supportive indicator (NOT a general risk to the stock)", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "PASS1_ID: compact citation"}
  ],
  "narrative": "250-400 words (stop at 400 words). Persuasive bear case, grounded, references specific Pass 1 findings. The document the CIO reads.",
  "structured_data": {
    "core_arguments": [
      {
        "argument": "specific falsifiable claim",
        "supporting_pass1_agents": ["RSRCH", "FUND"],
        "evidence": "compact citation(s)",
        "strength": "primary|secondary|tertiary"
      }
    ],
    "downside_triggers": [
      {
        "trigger": "specific event that could drive the stock lower",
        "timing": "within_4_weeks|1_to_3_months|3_to_12_months|1_to_3_years|3_plus_years",
        "probability": "high|medium|low",
        "severity": "high|medium|low",
        "supporting_pass1_agents": ["TECH"],
        "evidence": "compact citation"
      }
    ],
    "valuation_argument": {
      "claim": "why current price overvalues the stock OR why risk/reward is unfavorable",
      "supporting_pass1_agents": ["FUND"],
      "evidence": "compact citation"
    },
    "market_misreads": [
      {
        "misread": "what the market is overweighting, ignoring, or mispricing on the UPSIDE",
        "why_this_persists": "1 sentence on why this narrative premium exists",
        "supporting_pass1_agents": ["SENT", "RSRCH"],
        "evidence": "compact citation"
      }
    ],
    "tail_risk_assessment": {
      "tail_risk_level": "elevated|moderate|negligible",
      "scenario": "1-2 sentences describing the structural/operational/regulatory scenario, or empty string if negligible",
      "triggering_event": "short free-text description of the named trigger, or empty string if negligible",
      "supporting_pass1_agents": ["FUND"],
      "evidence": "compact citation, or empty string if negligible"
    },
    "thesis_archetype_alignment": {
      "researcher_archetype": "injected from Pass 1 RSRCH — reference only",
      "bear_archetype": "bear's own choice from the 5-value controlled vocabulary: secular_grower | dividend_compounder | cyclical_recovery | quality_compounder | value_trap_candidate",  # v2c: corrected from 8-value to Researcher v1.3 5-value set
      "agrees_with_researcher": true,
      "disagreement_note": "non-empty only if disagrees, else empty string"
    },
    "context_aware_strongest_argument": "for this {timeline} + {account_type} combo, which single aspect matters most",
    "asymmetry_assessment": "downside/upside asymmetry as the bear sees it (e.g., '2:1 downside over 12 months per FUND scenario analysis — dividend cut would add ~20% additional drawdown')"
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
<td>`thesis_risks`</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>`core_arguments`</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>`downside_triggers`</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`market_misreads`</td>
<td>1</td>
<td>3</td>
</tr>
</table>
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name` (`"bear_case_advocate"`), `agent_pass` (`"pass2"`), `analysis_context`, `data_sources_used`, `data_quality_assessment`
- **`data_quality_assessment`**: Derived from Pass 1 reliability scores: `>=70 avg and min >= 40` = `"high"`, `>=50 avg` = `"medium"`, `>=30 avg` = `"low"`, else `"insufficient"`.
- **`recommendation`**: Hard-coded to `"bearish"` (defense in depth — validator also enforces).
---
# Orchestrator Pre-computation: Pass 1 Compression
Identical to the Bull’s. The compressed Pass 1 payload is shared between Bull and Bear.
```python
def compress_pass1_for_advocate(pass1_outputs: list[AgentOutput]) -> dict:
    """Shared between Bull and Bear advocates."""
    compressed = {}
    reliability_warnings = []

    for output in pass1_outputs:
        agent_id = AGENT_ID_MAP[output.agent_name]
        compressed[agent_id] = {
            "assessment_summary": output.assessment_summary,
            "reliability_score": output.reliability_score,
            "data_quality_assessment": output.data_quality_assessment,
            "caveats": output.reliability_factors.caveats,
            "structured_data": output.structured_data,
            "narrative": truncate_tokens(output.narrative, 300),
        }

        if output.reliability_score < 40:
            reliability_warnings.append(f"{agent_id} reliability_score={output.reliability_score} — treat as unreliable")
        elif output.reliability_score < 70:
            reliability_warnings.append(f"{agent_id} reliability_score={output.reliability_score} — use with caution")

    return {"compressed_outputs": compressed, "reliability_warnings": reliability_warnings}
```
---
# Orchestrator Merge Logic
```python
def merge_output_bear_advocate(llm_response: dict, pass1_outputs: list[AgentOutput],
                               context: AnalysisContext) -> dict:
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

    llm_response["recommendation"] = "bearish"

    return {
        "agent_name": "bear_case_advocate",
        "agent_pass": "pass2",
        "analysis_context": {
            "account_type": context.account_type,
            "timeline": context.timeline,
            "stock_id": str(context.stock_id),
            "canonical_ticker": context.canonical_ticker,
        },
        "recommendation": "bearish",
        "confidence": llm_response["confidence"],
        "thesis_summary": llm_response["thesis_summary"],
        "strongest_argument": llm_response["strongest_argument"],
        "weakest_point": llm_response["weakest_point"],
        "caveats": llm_response["caveats"],  # v2c: was omitted — LLM-produced caveats were silently dropped from merged output.
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
2. **Required fields**: `recommendation`, `confidence`, `thesis_summary`, `strongest_argument`, `weakest_point`, `caveats`, `key_factors`, `thesis_risks`, `narrative`, `structured_data`.
3. **recommendation**: Must equal `"bearish"` exactly. Failure message: *“You are the Bear Case Advocate — recommendation must be ‘bearish’. If the bear case is weak, return bearish with a low confidence score and an honest weakest_point.”*
4. **confidence**: Integer 0-100.
5. **thesis_summary**: Non-empty, 80-500 characters, must reference the timeline or account_type.
6. **strongest_argument**: Non-empty, 40-400 characters, must contain at least one Pass 1 agent ID token (`RSRCH`, `FUND`, `TECH`, `SENT`, `MACRO`).
7. **weakest_point**: Non-empty, 40-400 characters, must contain at least one Pass 1 agent ID token OR the explicit phrase `"no Pass 1 input supports continued ownership"`.
8. **caveats**: 0-5 items. If the payload included a reliability_warnings block (any Pass 1 agent \<70), caveats must contain at least one entry acknowledging at least one flagged agent.
9. **key_factors**: 2-4 items. Every entry must have `sentiment: "negative"`. Evidence must start with a valid Pass 1 agent ID token followed by `:`.
10. **thesis_risks**: 1-3 items. Each evidence field must start with a valid Pass 1 agent ID token OR the phrase `"thesis assumption:"`.
11. **narrative**: 250–400 words (stop at 400 words; \~1,800–2,800 characters). Must reference at least 3 distinct Pass 1 agent IDs. Empirical 2,800-char ceiling (raised from 2,400 after 2026-05-15 simulation — multi-agent citation requirements naturally produced 2,600–2,800-char narratives). Length is enforced; the validator performs a sentence-boundary auto-trim in `base.py` before retrying, so narratives that overshoot by ≤1 sentence are repaired in-place rather than re-rolled.
12. **structured_data.core_arguments**: 1-3 items. Each requires `argument`, `supporting_pass1_agents` (non-empty), `evidence` (with Pass 1 agent ID), `strength` (`primary|secondary|tertiary`). Exactly one entry must be `strength: "primary"`.
13. **structured_data.downside_triggers**: 1-4 items. At least one trigger’s `timing` must be consistent with the analysis `timeline`:
	- `short_term` → at least one with `within_4_weeks` or `1_to_3_months`
		- `medium_term` → at least one with `within_4_weeks`, `1_to_3_months`, or `3_to_12_months`
		- `long_term` → at least one with `3_to_12_months`, `1_to_3_years`, or `3_plus_years`
14. **structured_data.valuation_argument**: all sub-fields non-empty. Preferred Pass 1 agent: `FUND`.
15. **structured_data.market_misreads**: 1-3 items, all sub-fields non-empty.
16. **structured_data.thesis_archetype_alignment**: `researcher_archetype` matches injected value exactly. `bear_archetype` is one of the `5 controlled vocabulary values`: `secular_grower | dividend_compounder | cyclical_recovery | quality_compounder | value_trap_candidate`. # v2c: corrected from stale ‘8 vocabulary values’ to Researcher v1.3 5-value set (dropped: turnaround, disrupted_incumbent, special_situation). `agrees_with_researcher` is `true` if `bear_archetype == researcher_archetype`, else `false`. If `false`, `disagreement_note` must be non-empty AND contain at least one Pass 1 agent ID. If `true`, `disagreement_note` is the empty string.
17. **structured_data.context_aware_strongest_argument**: Non-empty, must reference the timeline OR account_type.
18. **structured_data.asymmetry_assessment**: Non-empty.
19. **Reliability-coupling rule**: If 3+ Pass 1 inputs had `reliability_score < 40`, `confidence` must be `<= 45`.
20. **Numeric grounding at low confidence**: If `confidence < 35`, every `structured_data.core_arguments[].evidence` field must contain at least one numeric token (matches `\d+\.?\d*%`, `\$\d+`, `\d+:\d+`, or a 4-digit year). Prevents fabricated bearish reasoning unmoored from data.
21. **Prior-failure acknowledgment**: If memory brief is non-empty AND `if_scored_prior_was == "incorrect"` AND the current bear thesis is directionally similar to the prior (same `bear_archetype` OR overlap in primary `core_argument` themes flagged by orchestrator), `weakest_point` must contain one of the acknowledgment phrases: `"prior bear case"`, `"prior thesis"`, `"previously wrong"`, `"repeated"`, `"replay"`, or `"earlier call"`. Validator only checks for phrase presence; semantic correctness is reviewed by the Feedback Analyst.
22. **Tail risk assessment completeness**: `tail_risk_level` must be `elevated|moderate|negligible`. If `elevated` or `moderate`: `scenario`, `triggering_event`, `supporting_pass1_agents`, and `evidence` must all be non-empty, and `evidence` must cite a Pass 1 agent ID. If `negligible`: `scenario`, `triggering_event`, and `evidence` must be empty strings. Inverts effort asymmetry: escalation is structurally harder than `negligible`.
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember:
- recommendation must be "bearish" — you are the advocate, not a judge
- every claim must cite a Pass 1 agent ID (RSRCH, FUND, TECH, SENT, MACRO)
- at least one downside_trigger must be timed consistent with the analysis timeline
- weakest_point must identify what would invalidate the BEAR thesis (positive catalyst, supportive fundamental, or data gap) — not a bull-case rebuttal
- thesis_risks entries are things that would INVALIDATE the bear case — positive catalysts or supportive fundamentals
- tail_risk is structural/operational/regulatory impairment — NOT valuation; valuation belongs in valuation_argument
- tail_risk_level: negligible is the expected answer for most fundamentally sound stocks
```
---
# Memory Brief Schema
The `{memory_brief}` placeholder is populated from `StockAnalysisMemory` if the stock has been analyzed before.
```plain text
Prior Bear Case ({days_since} days ago, timeline={prior_timeline}):
  confidence_then: {prior_confidence}/100
  thesis_summary_then: {prior_thesis_summary}
  strongest_argument_then: {prior_strongest_argument}
  outcome: {return_pct} over {holding_period_days} days (scored: yes|no|not_yet)
  if_scored_prior_was: correct|incorrect|partially_correct
  what_changed_since: {short_list_of_material_events}
```
**Timeline-scaled decay**: Included if \<30 days old at short_term, \<90 days at medium_term, \<180 days at long_term. Below threshold, the brief is omitted entirely. # v2c: replaced vague ‘60-day’ language with explicit thresholds matching Researcher v1.3 and cross-agent standard (\<30d/\<90d/\<180d).
**How the advocate uses memory**: The Bear reads the prior case but is NOT required to temper its current case because the prior was wrong. Per Rule 21, if the prior was scored wrong AND current thesis is directionally similar, `weakest_point` must briefly acknowledge it.
---
# Calibration Brief (Post-20 Predictions)
Single \~150-token brief replacing the previous accuracy + winning-patterns split. Omitted when N \< 20.
```plain text
YOUR CALIBRATION ({N} scored bear predictions):
- Directional accuracy: {X}% overall, {Y}% in this sector ({sector}), {Z}% on this timeline ({timeline}).
- Most accurate recent call: {ticker} on {date} — confidence {C}, actual return {R}%. Key insight: {short reasoning that proved correct}.
- Pattern that consistently works: {e.g., "FUND payout ratio >100% + RSRCH management credibility erosion → 78% directional hits"}.
- Known bias / adjustment: {e.g., "you over-flag dividend-cut risk in consumer staples (+8pts vs actual); reduce confidence by 5-10 points for this pattern in current regime"}.
```
---
# Minimum Data Threshold
Per Gate 1: at least 3 of 5 Pass 1 agents must have `reliability_score >= 30`, and either FUND or RSRCH must be among them. If Gate 1 fails, the orchestrator does NOT invoke the bear advocate — stub output with `confidence: 0`, `thesis_summary: "insufficient data for bear thesis"`, run marked `completed_with_warnings`. If Gate 1 passes but FUND/RSRCH specifically is unavailable, invoke the bear advocate but cap `confidence <= 55` and acknowledge in `caveats`.
---
# How the System Prompt and User Message Interact
- **Every CoT step maps to a structured field**: Step 1 → `core_arguments`; Step 2 → `downside_triggers`; Step 3 → `valuation_argument` + `market_misreads`; Step 4 → `tail_risk_assessment`; Step 5 → `strongest_argument` + `weakest_point`; Step 6 → `context_aware_strongest_argument` + `thesis_summary`.
- **No entity anonymization at Pass 2**: Real ticker and company name throughout.
- **Thinking mode OFF**: `enable_thinking: false`.
- **Schema at end of system prompt**: Consistent with Bull and Researcher.
- **Single exchange**: retries append to user message.
- **Cacheable prefix structure**: all stable content precedes `-- STOCK-SPECIFIC CONTEXT ---`.
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
<td>Real ticker</td>
</tr>
<tr>
<td>`{company_name}`</td>
<td>`DataBundle.company_info["name"]`</td>
<td>Real name</td>
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
<td>Falls back to `"unclassified"` if Researcher failed. **v2c: corrected from structured_data — always use pass2_view (stable contract). Injected in STOCK-SPECIFIC CONTEXT, not stable prefix.**</td>
</tr>
<tr>
<td>`{pass1_reliability_warnings}`</td>
<td>`build_pass1_reliability_warnings()`</td>
<td></td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory` query</td>
<td>Empty string if none / below marginal</td>
</tr>
<tr>
<td>`{calibration_brief}`</td>
<td>Feedback engine</td>
<td>Empty string if N \< 20</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
<td></td>
</tr>
<tr>
<td>Pass 1 block placeholders</td>
<td>Compressed `AgentOutput` per Pass 1 agent</td>
<td>Missing → “NOT AVAILABLE — reason: \{reason\}”</td>
</tr>
</table>
---
# Testing Checklist
- Valid JSON from Sonnet 4.6 with `enable_thinking: false`?
- `recommendation: "bearish"` even when Pass 1 is uniformly positive?
- Low `confidence` (\<30) when bear case is genuinely weak (clean balance sheet, positive momentum)?
- Confidence spread broadly — not clustered at 65-75?
- Every `key_factors`, `thesis_risks`, `core_arguments`, `downside_triggers`, `market_misreads` evidence field cites a valid Pass 1 agent ID?
- `narrative` references at least 3 distinct Pass 1 agent IDs?
- **Archetype independence test**: Researcher classified as `quality_compounder` — does the Bear exercise independent judgment?
- **Timeline trigger test**: `timeline=short_term` → at least one `downside_trigger` with timing `within_4_weeks` or `1_to_3_months`?
- **Reliability coupling test**: 3 of 5 Pass 1 agents with reliability \< 40 → `confidence <= 45`?
- **Account adaptation test**: Dividend stock with deteriorating payout → RRSP bear case leans into dividend-cut cascade; Trading allows more technical framing?
- **Weak-bear honesty test**: Clean bull setup → (a) `confidence < 35`, (b) high-bar bear frames per Rule 6, (c) numeric tokens present in `core_arguments[].evidence` per Rule 20, (d) `weakest_point` identifies what would invalidate the bear thesis?
- **thesis_risks framing test**: `thesis_risks` entries are genuinely things that would invalidate the bear case (positive catalysts, supportive fundamentals) — not general risks?
- **Tail risk test**: Sound compounder → `tail_risk_level: negligible`, scenario/trigger/evidence empty. Leveraged cyclical → `elevated` or `moderate` with named trigger and Pass 1 citation. Valuation arguments alone should NOT produce `elevated` or `moderate`.
- **Tail risk / valuation separation**: Rich valuation but clean balance sheet → `negligible`, with valuation captured in `valuation_argument` and `asymmetry_assessment`.
- **Prior-failure acknowledgment test (Rule 21)**: Memory brief with `if_scored_prior_was: incorrect` and similar current thesis → `weakest_point` contains an acknowledgment phrase.
- **Citation asymmetry test**: Run Bull and Bear on 5 mixed-signal stocks. Bull should over-represent FUND/RSRCH positive signals; Bear should over-represent TECH/SENT negative signals. Symmetric distribution suggests advocate framing isn’t driving divergence.
- **Fixture stocks**: Overvalued high-multiple name (strong bear, valuation-reset); leveraged cyclical at peak (use `cyclical_recovery` archetype, note bearish direction); dividend aristocrat with safe payout (weak bear — `confidence < 25`, `tail_risk: negligible`); NVDA at rich multiple (medium bear, priced-for-perfection).
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
<td>\~550 tokens</td>
<td>CoT + 7 rules + injection blocks + schema</td>
</tr>
<tr>
<td>User message (compressed Pass 1)</td>
<td>\~4,000 tokens</td>
<td>5 Pass 1 blocks × \~700 tokens avg + context header. Shared with Bull. Measure against fixture stocks before v1 testing.</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~1,300 tokens</td>
<td>\~250 tokens lighter than v1 (no historical_analogy, no weakest_point_type, simpler tail_risk, simpler archetype block)</td>
</tr>
<tr>
<td>Total</td>
<td>\~5,850 tokens</td>
<td></td>
</tr>
</table>
**Model configuration (cloud-primary)**:
- Model: `claude-sonnet-4-6`
- `max_tokens`: **5000** (empirical min safe value from 2026-05-15 simulation; the prior 1800 caused silent JSON truncation. Set to min safe + 20% headroom per the Orchestration Engine LLM Call Parameters section.)
- `temperature`: 0.3
- Thinking mode: OFF
- JSON output: prefilled assistant turn with `{` to eliminate markdown fences (see Orchestration Engine § LLM Call Parameters).
- Prompt caching: stable content precedes `-- STOCK-SPECIFIC CONTEXT ---`; cacheable prefix \~400-500 tokens
- Expected latency: \~5-8s
**Model configuration (local fallback)**:
- Model: `qwen3.6:35b-a3b`, `num_ctx`: 8192, `enable_thinking`: false (verify suppression is clean on this model — see Agent Design Patterns Pattern 10 caveat)
- Expected latency: unverified estimate, re-measure after deployment (was \~30s on the prior `qwen3:14b`); quality degradation vs Sonnet: re-establish baseline for this model rather than assuming it matches `qwen3:14b`'s prior behavior
---

_Source: Notion — Project Stock Picker → Agent Prompts → Bear Case Advocate Agent Prompt. Downloaded 2026-07-02._
