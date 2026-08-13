**Agent**: Risk Advisor<br>**Status**: Draft v1.1 — simplified for retail Canadian investor scope<br>**Phase**: Phase 2<br>**Pass**: Pass 2 (Perspective & Strategy)
**Role**: Quantify the range of outcomes for `{ticker}` for the specified timeline and account — specific downside scenarios, their probability, their magnitude, and an appropriate position size given them. Independent of Bull/Bear: advocates argue direction; the Risk Advisor quantifies the distribution and identifies where things go really wrong.
**Target user profile**: Individual Canadian retail investor, total portfolio \< \$250k CAD, holding in TFSA / RRSP / Trading. Stop-loss machinery, deep liquidity analysis, and exotic concentration math are out of scope at this size.
**Target Model**: Local-primary `qwen3.6:35b-a3b` via Ollama (`enable_thinking: false` — verify this fully suppresses reasoning on this model; see Agent Design Patterns Pattern 10 caveat). Cloud fallback `claude-sonnet-4-6`. (Note 2026-07-01: the retry-rate/quality-monitor-triggered promotion described below is no longer planned as an automated mechanism — model assignment is a fixed upfront choice per agent. Treat the quality signals as things to check manually before launch, not an automatic runtime rule.)
**Token Budget**: \~700 system \| \~4,000 compressed Pass 1 \| \~250 pre-computed metrics \| \~1,200 expected output \| \~6,150 total. Headroom against `num_ctx: 8192` ≈ 2,000 tokens.
---
# Design Decisions (condensed)
1. **Analyst, not advocate.** No `recommendation` field. Output is `risk_reward_ratio: favorable | neutral | unfavorable` plus a structured `risk_profile`. CIO consumes this as a *constraint* on directional advocacy.
2. **Pre-computed metrics are orchestrator-injected, not LLM-produced.** The data pipeline computes beta, volatility, max drawdown, ADV. The orchestrator stamps `beta` and `max_drawdown_1yr` directly into the final output object. The LLM only produces interpretation strings (`beta_interpretation`, `max_drawdown_interpretation`). This removes a whole class of passthrough validation failures.
3. **Reliability-aware weighting.** Pass 1 `reliability_score` ≥70 trustworthy, 40–69 caution, \<40 unreliable. SENT-grounded “sentiment reversal” scenarios admissible only when SENT reliability ≥60.
4. **Researcher archetype is descriptive only.** Risk Advisor does not pick its own archetype. References the Researcher’s `thesis_archetype` to inform scenario construction but does not echo it back. Ignore entirely when archetype is `unclassified`.
5. **Positive grounding, no anonymization.** Real ticker and name throughout. Every claim cites a Pass 1 agent ID or pre-computed metric token.
6. **6-step CoT.** Each step maps to ≥1 structured field. Steps: volatility, drawdown, downside scenarios, portfolio fit, sizing + stop, sanity + synthesis.
7. **Bounded output arrays.** `downside_scenarios` 2–4 (≥2 floor — single-scenario risk profile is structurally inadequate). `key_factors` 2–4. `caveats` 0–5. `data_sanity_flags` 0–3.
8. **Cited evidence required.** Compact citations using Pass 1 IDs (`RSRCH FUND TECH SENT MACRO`) or metric tokens (`BETA VOL DD LIQ CORR CONC`).
9. **Generic-scenario guard is soft + retrospective.** Real-time validators enforce trigger-citation (Rule 11) and distinct-impact (Rule 12). The actual generic-scenario check is a Feedback Analyst monitor logging trigger phrasings across 20 runs. A 14B model on this task will produce some generic scenarios — be clear-eyed.
10. **Position size is X–Y%; stop-loss is null in most retail cases.** For TFSA/RRSP medium/long-term (the dominant case for this user profile), `stop_loss_suggestion: null` is enforced. For Trading + short-term, must be a number grounded in TECH support or ATR. All other combos: null.
11. **Portfolio context is conditional.** When provided: `correlation_to_existing_portfolio` and `concentration_risk` are substantive. When absent: default to `unknown` and `not_applicable`. Simple rule for `concentration_risk`: any single position \>10% of portfolio = `high`.
12. **Self-critique via ****`data_sanity_flags`****, not ****`weakest_point`****.** No thesis to critique. Surface 0–3 anomalies in pre-computed metrics. Clean data returns zero flags — that is the correct answer.
13. **Memory: read, don’t defer; flag scenario reuse.** If prior was scored wrong AND current scenarios reuse prior verbatim (\>0.85 token-Jaccard similarity), `caveats` must acknowledge the prior miss.
14. **Thinking mode OFF** for JSON stability. On `qwen3.6:35b-a3b`, verify this actually suppresses reasoning output before trusting it (see Pattern 10 caveat) — `qwen3:14b`'s clean suppression is not guaranteed to carry over.
15. **Groundedness semantics differ from advocates'.** `groundedness_score` reflects how well-grounded the *risk profile* is, NOT directional conviction. CIO must consume this differently than advocate confidences. Renamed from `confidence` in v1.1 to prevent semantics collision with Bull/Bear advocates.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- Daily price history — openbb-tmx `.equity.price.historical` (CA) / FMP `/historical-price-full` (US)
- TSX benchmark history — yfinance `^GSPTSE` (CA) / — (US)
- S&P 500 benchmark history — — (CA) / FMP `/historical-price-full/^GSPC` (US)
- Beta, volatility, drawdown, Sharpe — computed locally from prices (CA) / same (US)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- `precomputed_risk_metrics` block (`build_precomputed_risk_metrics()`): BETA vs benchmark, annualized VOL, DD 1yr/3yr with recovery days, LIQ (ADV in CAD/USD), CORR sector overlap, optional CONC (portfolio concentration) — all pre-formatted strings, \~250 tokens
- `beta` and `max_drawdown_1yr` injected directly into the final output object by the orchestrator — the LLM never produces or echoes these numerics
- Compressed Pass 1 agent outputs (RSRCH, FUND, TECH, SENT, MACRO) with per-agent `reliability_score`, \~4,000 tokens, shared with Bull/Bear
- Researcher's `thesis_archetype` (via `pass2_view`, falls back to `"unclassified"`) — descriptive context only, not echoed as a field
- `portfolio_context` (PortfolioSnapshot summary or empty string) — drives CORR/CONC conditional logic
- `memory_brief` — prior Risk Advisor run from `StockAnalysisMemory`, timeline-scaled decay
- `pass1_reliability_warnings`, `accuracy_brief`, `winning_patterns_brief` — feedback-engine strings, last two omitted when N\<20
---
# System Prompt
Invoke with `enable_thinking: false`. Stable content precedes `--- STOCK-SPECIFIC CONTEXT ---`.

**Runtime prompt:** see [`prompts/risk_advisor/v1.1.txt`](../../prompts/risk_advisor/v1.1.txt)
## Conditional Injection Reference
**Timeline**:
<table header-row="true">
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks). Emphasize daily volatility (VOL), gap risk, and explicit stop-loss only on Trading accounts. Position sizing TIGHTER. Downside scenarios should be event-driven (earnings miss, guidance cut, technical breakdown).`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Balance event-driven scenarios with multi-quarter structural risk. Drawdown tolerance moderate. Stop-loss null for TFSA/RRSP.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). Emphasize permanent capital loss risk, drawdown tolerance, and recovery time. Wider tolerance on volatility — quality compounder with 25% historical drawdowns and reliable recovery is appropriate. Stop-loss null. Downside scenarios should be structural (moat erosion, secular decline, leverage spiral) not event-driven.`</td>
</tr>
</table>
**Account**:
<table header-row="true">
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Permanent capital loss has AMPLIFIED cost — losses cannot offset gains elsewhere AND permanently reduce tax-sheltered compounding capacity. Position sizing ERRS TIGHTER for any name with elevated tail risk. Stop-loss null. Risk-reward synthesis treats permanent-loss scenarios as more costly than equivalent magnitude in Trading.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US dividends WHT-exempt; commonly used for dividend-compounding holdings. Risk profile must explicitly assess DIVIDEND-STREAM RELIABILITY for income holdings — if FUND flags payout-ratio risk, declining FCF, or earnings coverage deterioration, surface a dividend-cut cascade scenario (-15 to -30% on announcement is the historical pattern). Stop-loss null. Sizing reflects long-term dividend-stream weight.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading (non-registered). Active management appropriate. For short_term timeline, stop_loss_suggestion is a numeric price level grounded in TECH support or ATR. Loss-harvesting (50% inclusion) reduces the cost of being wrong. Downside scenarios may include short-horizon event triggers.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General — no specific account. Summarise which risk dimensions are amplified in each account type: permanent loss cost in TFSA, dividend-stream reliability in RRSP, loss-harvesting lever in Trading. position_size_recommendation applies to all three; stop_loss_suggestion null.`</td>
</tr>
</table>
---
# Output Schema (LLM-produced fields)
Orchestrator populates metadata, derived context, AND injects `beta` and `max_drawdown_1yr` directly from pre-computed metrics — the LLM does NOT produce those numeric fields. The LLM produces interpretation strings only.
```json
{
  "groundedness_score": 0-100,
  "thesis_summary": "2-3 sentences. Risk profile compressed. Must reference timeline OR account_type AND at least one of: BETA, VOL, DD.",
  "strongest_signal": "1-2 sentences. The dominant downside scenario or sanity flag. Must cite a Pass 1 agent ID or metric token.",
  "caveats": ["specific data gaps, reliability flags, or memory-reuse acknowledgments"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "negative|neutral", "evidence": "PASS1_ID_OR_METRIC: compact citation"}
  ],
  "narrative": "300-500 words (stop at 500 words). Quantitative risk profile, grounded, references specific Pass 1 findings AND pre-computed metrics.",
  "risk_profile": {
    "volatility_assessment": "very_high|high|moderate|low",
    "beta_interpretation": "1-2 sentences. Plain language connecting beta to timeline and account.",
    "max_drawdown_interpretation": "1-2 sentences. Includes recovery time and trigger if known. Cite Pass 1 if cause is identifiable.",
    "downside_scenarios": [
      {
        "scenario": "1 sentence specific outcome (NOT generic)",
        "probability": "high|medium|low",
        "estimated_impact_pct": <number, typically negative>,
        "trigger": "specific Pass 1 finding or metric driving the scenario, with named mechanism",
        "timeline": "within_4_weeks|1_to_3_months|3_to_12_months|1_to_3_years|3_plus_years"
      }
    ],
    "concentration_risk": "high|medium|low|not_applicable",
    "liquidity_risk": "high|medium|low",
    "correlation_to_existing_portfolio": "high|medium|low|unknown",
    "risk_reward_ratio": "favorable|neutral|unfavorable",
    "position_size_recommendation": "X-Y% format string (e.g., '1-2%', '3-5%')",
    "stop_loss_suggestion": <number price level OR null>,
    "data_sanity_flags": [
      "1-sentence flag describing the anomaly and what to verify"
    ]
  }
}
```
### Array bounds (validator-enforced)
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
<td>`risk_profile.downside_scenarios`</td>
<td>2</td>
<td>4</td>
</tr>
<tr>
<td>`risk_profile.data_sanity_flags`</td>
<td>0</td>
<td>3</td>
</tr>
</table>
### Fields the orchestrator populates
- **Metadata**: `agent_name` (“risk_advisor”), `agent_pass` (“pass2”), `analysis_context`, `data_sources_used`, `data_quality_assessment`.
- **Pre-computed numerics**: `risk_profile.beta` (2-decimal) and `risk_profile.max_drawdown_1yr` (1-decimal) injected directly from the data pipeline. LLM does not produce or echo these.
- **`data_quality_assessment`**: from Pass 1 reliability scores. \>=70 avg AND min \>=40 = “high”; \>=50 avg = “medium”; \>=30 avg = “low”; else “insufficient”.
- **`recommendation`** intentionally omitted: Risk Advisor is not directional. CIO reads `risk_reward_ratio` as analytical equivalent.
- **`reliability_factors`** intentionally omitted: Pass 2 agents do not produce these.
- **`weakest_point`** intentionally omitted: no thesis to invalidate. `data_sanity_flags` is the analogous self-critique mechanism.
---
# Pre-computed Risk Metrics Pipeline
```python
def build_precomputed_risk_metrics(stock_id: UUID, ctx: AnalysisContext,
                                   portfolio: PortfolioSnapshot | None) -> str:
    """
    Assembles the {precomputed_risk_metrics} block injected into user message.
    All values computed deterministically by the data pipeline.
    Currency note: ADV reported in CAD-equivalent for TSX names, USD for US names —
    block must indicate which.
    """
    metrics = risk_pipeline.compute(stock_id, benchmark=benchmark_for(ctx))
    block = [
        f"BETA (vs{metrics.benchmark}):{metrics.beta:.2f}",
        f"VOL (annualized,{ctx.timeline}-relevant lookback):{metrics.annualized_vol_pct:.1f}%",
        f"DD 1yr:{metrics.max_drawdown_1yr_pct:.1f}% (recovery:{metrics.recovery_1yr_days}d)",
        f"DD 3yr:{metrics.max_drawdown_3yr_pct:.1f}% (recovery:{metrics.recovery_3yr_days}d)",
        f"LIQ avg daily volume: ${metrics.adv_millions:.1f}M{metrics.adv_currency}",
        f"CORR sector overlap:{metrics.sector_overlap_summary}",
    ]
    if portfolio is not None:
        block.append(
            f"CONC portfolio concentration: "
            f"sector_weight={metrics.portfolio_sector_weight_pct:.1f}%, "
            f"single_position_max={metrics.portfolio_max_single_position_pct:.1f}%, "
            f"position_count={portfolio.position_count}"
        )
    return "\n".join(block)
```
The orchestrator passes `precomputed_metrics.beta` and `precomputed_metrics.max_drawdown_1yr_pct` directly into the merged output’s `risk_profile.beta` and `risk_profile.max_drawdown_1yr` — the LLM does not produce or validate these numeric fields.
---
# Orchestrator Merge Logic
```python
def merge_output_risk_advisor(llm_response: dict, pass1_outputs: list[AgentOutput],
                              precomputed_metrics: PrecomputedRiskMetrics,
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

    # Inject pre-computed numerics. LLM does NOT produce these.
    risk_profile = llm_response["risk_profile"]
    risk_profile["beta"] = round(precomputed_metrics.beta, 2)
    risk_profile["max_drawdown_1yr"] = round(precomputed_metrics.max_drawdown_1yr_pct, 1)

    # Strip forbidden fields silently.
    for forbidden in ("recommendation", "risk_archetype", "bull_archetype",
                      "bear_archetype", "context_aware_strongest_argument"):
        llm_response.pop(forbidden, None)
        risk_profile.pop(forbidden, None)

    return {
        "agent_name": "risk_advisor",
        "agent_pass": "pass2",
        "analysis_context": {
            "account_type": context.account_type,
            "timeline": context.timeline,
            "stock_id": str(context.stock_id),
            "canonical_ticker": context.canonical_ticker,
        },
        "groundedness_score": llm_response["groundedness_score"],
        "thesis_summary": llm_response["thesis_summary"],
        "strongest_signal": llm_response["strongest_signal"],
        "caveats": llm_response["caveats"],
        "key_factors": llm_response["key_factors"],
        "narrative": llm_response["narrative"],
        "risk_profile": risk_profile,
        "data_quality_assessment": dqa,
        "data_sources_used": [p.agent_name for p in pass1_outputs if p is not None]
                             + ["precomputed_risk_pipeline"],
    }
```
---
# Validation Rules
Enforced by `src/agents/validators.py`. Failure triggers retry with error list appended.
1. **JSON parseable** (strip markdown fences/preamble if present).
2. **Required fields**: `groundedness_score`, `thesis_summary`, `strongest_signal`, `caveats`, `key_factors`, `narrative`, `risk_profile`. Within `risk_profile`: `volatility_assessment`, `beta_interpretation`, `max_drawdown_interpretation`, `downside_scenarios`, `concentration_risk`, `liquidity_risk`, `correlation_to_existing_portfolio`, `risk_reward_ratio`, `position_size_recommendation`, `stop_loss_suggestion`, `data_sanity_flags`. (Note: `beta` and `max_drawdown_1yr` are orchestrator-injected, not LLM-required.)
3. **No forbidden fields**: `recommendation`, `risk_archetype`, `bull_archetype`, `bear_archetype`, `context_aware_strongest_argument` stripped silently. Logs prompt-regression warning.
4. **groundedness_score**: integer 0-100. Anchors groundedness of the *risk profile*, not directional conviction.
5. **thesis_summary**: 80-500 chars, must reference timeline OR account_type, AND at least one of BETA, VOL, DD.
6. **strongest_signal**: 40-400 chars, must contain at least one valid citation token.
7. **caveats**: 0-5 items. If reliability_warnings block is present (any Pass 1 \<70), caveats must contain at least one entry acknowledging at least one flagged agent.
8. **key_factors**: 2-4 items. `sentiment` is `negative|neutral` only (Risk Advisor does not produce `positive` factors). Evidence must start with a valid citation token followed by `:`.
9. **narrative**: 300–500 words (stop at 500 words; \~1,800–3,200 characters). Must reference \>=2 distinct Pass 1 agent IDs AND \>=2 distinct metric tokens. **Fallback** when 3+ Pass 1 inputs have reliability \<40: requirement relaxes to \>=4 distinct metric tokens, no Pass 1 ID requirement. Empirical 3,200-char ceiling (raised from 2,000 after 2026-05-15 simulation — combined Pass 1 + metric-token citation requirements naturally produced 2,800–3,200-char narratives). Length is validator-enforced; `base.py` auto-trims to the last sentence boundary when overshoot is small.
10. **downside_scenarios**: 2-4 items. Per scenario: scenario + probability + estimated_impact_pct + trigger + timeline all required. `trigger` must contain \>=1 valid citation token. At least one scenario’s `timeline` must be consistent with the analysis timeline:
	- `short_term` -\> at least one within_4_weeks or 1_to_3_months
		- `medium_term` -\> at least one in within_4_weeks, 1_to_3_months, or 3_to_12_months
		- `long_term` -\> at least one in 3_to_12_months, 1_to_3_years, or 3_plus_years
11. **Distinct-impact rule**: no two scenarios have estimated_impact_pct within +/-2pct of each other.
12. **Portfolio context conditional fields**:
	- portfolio_context provided AND CORR/CONC entry exists -\> correlation and concentration must be high\|medium\|low.
		- portfolio_context provided BUT no CORR/CONC entry -\> correlation may be unknown AND caveats must reference missing data.
		- portfolio_context absent -\> correlation = unknown AND concentration = not_applicable.
13. **risk_reward_ratio soft consistency check** (warning, no retry): scenario-weighted-impact = sum(probability_weight \* estimated_impact_pct) where weights are \{high: 0.7, medium: 0.4, low: 0.15\}. If `unfavorable` AND weighted-impact \> -8%, log warning. If `favorable` AND weighted-impact \< -20%, log warning.
14. **position_size_recommendation format**: regex `^\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?%$`. Examples valid: “1-2%”, “3-5%”, “0.5-1.5%”. Inadmissible: “small”, “5%”, “\~3%”.
15. **stop_loss_suggestion strict conditional**:
	- `account_type` in \{tfsa, rrsp\} AND `timeline` in \{medium_term, long_term\} -\> MUST be null. Numeric value triggers retry.
		- `account_type` == “trading” AND `timeline` == “short_term” -\> MUST be a number in \[0, 1.5 \* current_price\].
		- All other combinations -\> null.<br>Out-of-range numerics trigger retry.
16. **Memory reuse rule**: if memory brief is non-empty AND `if_scored_prior_was == "incorrect"` AND any current scenario string matches a prior scenario at \>0.85 token-Jaccard similarity, caveats MUST contain an entry referencing the prior miss.
17. **Reliability-coupling rule**: if 3+ Pass 1 inputs had reliability \<40, `groundedness_score` must be \<=50 AND \>=1 caveat references the data quality limitation.
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember:
- Risk Advisor produces NO directional recommendation — no "bullish" or "bearish" anywhere
- Risk Advisor does NOT produce an archetype field of any kind
- Do NOT produce `beta` or `max_drawdown_1yr` numeric fields — those are orchestrator-injected
- Every downside_scenarios.trigger must cite a Pass 1 agent ID or metric token AND name a specific mechanism
- No two scenarios may have estimated_impact_pct within +/-2pct of each other
- At least one downside scenario must be timed consistent with the analysis timeline
- position_size_recommendation must be in X-Y% format
- stop_loss_suggestion: TFSA/RRSP medium/long-term -> null. Trading short-term -> numeric. Otherwise null.
- If portfolio_context is empty, correlation_to_existing_portfolio = "unknown" AND concentration_risk = "not_applicable"
- data_sanity_flags is empty for clean inputs — do NOT invent flags
```
---
# Memory Brief Schema
`{memory_brief}` is populated from `StockAnalysisMemory` if the stock has a prior Risk Advisor run.
```plain text
Prior Risk Profile ({days_since} days ago, timeline={prior_timeline}):
  prior_volatility_assessment: {prior_assessment}
  prior_downside_scenarios:
    - scenario: {prior_scenario_1}; probability: {p1}; impact: {i1}%; occurred: yes|no|partial|not_yet
    - scenario: {prior_scenario_2}; probability: {p2}; impact: {i2}%; occurred: yes|no|partial|not_yet
  prior_position_size_recommendation: {prior_size}
  prior_risk_reward_ratio: {prior_ratio}
  outcome_observed: {return_pct} over {holding_period_days} days; max_drawdown_observed: {observed_dd}%
  if_scored_prior_was: correct|incorrect|partially_correct
```
`occurred` enum:
- `yes` — realized within +/-5pct of estimated_impact_pct AND named trigger observed
- `partial` — within +/-5-15pct OR trigger observed but magnitude differed OR magnitude matched but trigger differed
- `no` — opposite-signed OR \> +/-15pct off magnitude AND trigger not observed
- `not_yet` — holding still in flight
Timeline-scaled decay: memory included if \<30 days old at short_term, \<90 days at medium_term, \<180 days at long_term. Matches Researcher v1.3 and Sentiment v1.2c thresholds. Below threshold, brief is omitted. # v1.1c: replaced vague ‘full-weight/reduced/marginal’ description with explicit day thresholds.
Rule 16 (memory reuse): inert when memory empty or prior was scored correct.
---
# Accuracy & Winning Patterns Briefs (Post-20 Predictions)
Both briefs omitted when N \< 20. Per \[Accuracy & Winning Patterns Briefs\] notion doc.
**Accuracy brief** (\~80 tokens):
```plain text
YOUR ACCURACY: {N} scored risk profiles. Scenario-hit rate: {X}%. Position-sizing accuracy: {Y}% (too tight on {A}, too wide on {B}). Known bias: {description}. Adjustment: {description}.
```
**Winning patterns brief** (\~120 tokens):
```plain text
YOUR WINNING PATTERNS:
Most accurate recent risk call: {ticker} on {date} — flagged scenario {S} with probability {P}, occurred with realized drawdown {D}%.
- Pattern that consistently works: {description}
- Sizing patterns that work: {description}
Double down on these patterns.
```
---
# Minimum Data Threshold
Gate 1: at least 3 of 5 Pass 1 agents must have `reliability_score >= 30`, AND pre-computed metrics block must contain valid BETA, VOL, DD 1yr. If Pass 1 fails Gate 1: stub output with `groundedness_score: 0`, `thesis_summary: "insufficient pass 1 data for risk profile"`. If pre-computed metrics fail: orchestrator does NOT invoke Risk Advisor — agent requires deterministic risk metrics by construction.
**Pre-computed-metric independence**: Risk Advisor can produce useful output with WEAK Pass 1 IF pre-computed metrics are strong. Confidence capped at 60, caveats list Pass 1 gaps, scenarios rely on metric-grounded triggers. Rule 9’s relaxed citation requirement applies (\>=4 distinct metric tokens, no Pass 1 ID requirement).
---
# Prompt Variable Injection Reference
<table header-row="true">
<tr>
<td>Placeholder</td>
<td>Source</td>
</tr>
<tr>
<td>`{ticker}`</td>
<td>`AnalysisContext.canonical_ticker`</td>
</tr>
<tr>
<td>`{company_name}`</td>
<td>`DataBundle.company_info["name"]`</td>
</tr>
<tr>
<td>`{sector}`</td>
<td>`DataBundle.company_info["sector"]`</td>
</tr>
<tr>
<td>`{timeline}`</td>
<td>`AnalysisContext.timeline` (short_term / medium_term / long_term)</td>
</tr>
<tr>
<td>`{account_type}`</td>
<td>`AnalysisContext.account_type` (tfsa / rrsp / trading)</td>
</tr>
<tr>
<td>`{timeline_instruction}`</td>
<td>Conditional injection table</td>
</tr>
<tr>
<td>`{account_instruction}`</td>
<td>Conditional injection table</td>
</tr>
<tr>
<td>`{researcher_thesis_archetype}`</td>
<td>`researcher_output.pass2_view.thesis_archetype` (falls back to `"unclassified"`). **v1.1c: corrected from structured_data — always use pass2_view (stable contract). Injected in STOCK-SPECIFIC CONTEXT block, not in stable prefix.**</td>
</tr>
<tr>
<td>`{portfolio_context}`</td>
<td>`PortfolioSnapshot` summary or empty string</td>
</tr>
<tr>
<td>`{precomputed_risk_metrics}`</td>
<td>`build_precomputed_risk_metrics()`</td>
</tr>
<tr>
<td>`{pass1_reliability_warnings}`</td>
<td>`build_pass1_reliability_warnings()`</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory` query (Risk Advisor profile)</td>
</tr>
<tr>
<td>`{accuracy_brief}`</td>
<td>Feedback engine (empty when N\<20)</td>
</tr>
<tr>
<td>`{winning_patterns_brief}`</td>
<td>Feedback engine (empty when N\<20)</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
</tr>
<tr>
<td>Pass 1 block placeholders</td>
<td>Compressed `AgentOutput` per Pass 1 agent</td>
</tr>
</table>
---
# Testing Checklist
- Valid JSON from `qwen3.6:35b-a3b` with `enable_thinking: false`, and no leaked `<think>` reasoning tokens
- No `recommendation` or archetype field present (validator strips silently + logs)
- LLM does NOT produce `beta` or `max_drawdown_1yr` (orchestrator-injected)
- `narrative` references \>=2 distinct Pass 1 IDs AND \>=2 distinct metric tokens
- **Pass-1-weak / metrics-strong fallback test**: 3-of-5 Pass 1 below 40 -\> narrative satisfies relaxed \>=4-metric-token requirement, confidence \<=60, caveats reference gaps
- **Distinct-impact test** (Rule 11): 5 stocks, no two scenarios within +/-2pct
- **Generic-scenario manual grade**: 10 stocks scored on SPECIFIC / FALSIFIABLE / GROUNDED. Validator does not enforce — manual review only.
- **Scenario quality monitor**: across first 20 runs, distinct-trigger ratio \>=60%; flag for cloud promotion if not
- **Timeline trigger test** (Rule 10): timeline=short_term -\> at least one scenario within_4_weeks or 1_to_3_months
- **Position-size format test** (Rule 14): “small”/“5%”/“\~3%” trigger retry; “1-2%”/“3-5%”/“0.5-1.5%” pass
- **Stop-loss conditional test** (Rule 15): tfsa+long_term returning a number -\> retry (now strict); trading+short_term returning null -\> retry; numeric \> 1.5x current price -\> retry
- **Portfolio conditional test** (Rule 12): three sub-cases pass
- **Reliability coupling test** (Rule 17): 3-of-5 reliability \<40 -\> groundedness_score \<=50 AND caveat present
- **Account adaptation test**: dividend stock with deteriorating payout -\> RRSP surfaces dividend-cut cascade scenario at -15 to -30%; TFSA tightens sizing; Trading short-term adds numeric stop_loss
- **Data sanity flag distribution**: clean datasets produce ZERO flags on majority. Over-flagging = adjust prompt or escalate to cloud.
- **`unclassified`**** archetype test**: narrative does NOT reference “given the unclassified classification”
- **Risk-reward soft-warning test** (Rule 13): four -5% low-prob scenarios + unfavorable -\> warning; two -25% high-prob + favorable -\> warning
- **Memory reuse test** (Rule 16): prior was incorrect, current scenario \>0.85 similar -\> caveats acknowledges prior miss
- **Retry-context-exhaustion test**: force 2 consecutive validation failures -\> measure context size; \>7,500 tokens flags for cloud promotion
- **Fixture stocks**: clean dividend aristocrat (favorable); leveraged cyclical at peak (1-2% sizing); thin small-cap (liquidity_risk: high); rich-multiple growth name (multiple-reset scenario with named mechanism, not generic); unclassified-archetype stock
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
<td>6 CoT steps + 10 rules + injection blocks + schema. \~200 tokens lighter than v1.0 — `beta`/`max_drawdown_1yr` removed from LLM output, passthrough rules collapsed, simplified concentration logic.</td>
</tr>
<tr>
<td>User message — compressed Pass 1</td>
<td>\~4,000 tokens</td>
<td>Shared with Bull/Bear.</td>
</tr>
<tr>
<td>User message — pre-computed risk metrics</td>
<td>\~250 tokens</td>
<td>BETA + VOL + DD + LIQ + CORR + optional CONC. Plus \~250 tokens portfolio_context when present.</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~1,200 tokens</td>
<td>Smaller than v1.0 — interpretations only, no numeric passthrough; `context_aware_strongest_argument` removed.</td>
</tr>
<tr>
<td>Total (first call)</td>
<td>\~6,150 tokens</td>
<td>Headroom against `num_ctx: 8192`: \~2,000 tokens.</td>
</tr>
<tr>
<td>Total (after first retry)</td>
<td>\~6,800-7,400 tokens</td>
<td>Retries append validation errors + previous response. Headroom \~800-1,400 tokens — safer than v1.0.</td>
</tr>
</table>
**Operational notes**:
1. Measure actual compressed Pass 1 output across 3 fixture stocks. If average per-Pass-1 compression exceeds 700 tokens, tighten compression budget to 500 tokens/agent.
2. Commit to cloud promotion if retry-failure rate exceeds 15% across first 20 runs.
3. Promotion criteria: scenario quality monitor flags \<60% distinct-trigger ratio across 20 runs OR retry-failure rate \>15%.
**Model configuration (local-primary)**: `qwen3.6:35b-a3b` \| `num_ctx: 8192` \| `temperature: 0.3` \| `enable_thinking: false` (verify suppression is clean — see Pattern 10 caveat) \| \~36s latency on RTX 4070 Ti Super (unverified estimate, re-measure after deployment) \| JSON validity rate not yet measured for this model — re-establish baseline (was \~90-95% first-response on the prior `qwen3:14b`).
**Model configuration (cloud fallback)**: `claude-sonnet-4-6` \| `max_tokens: 5000` (empirical min safe value from 2026-05-15 simulation; the prior 2000 caused silent JSON truncation. Set to min safe + 20% headroom per Orchestration Engine § LLM Call Parameters) \| `temperature: 0.3` \| thinking OFF \| JSON output prefilled with `{` per the same section \| \~5-8s latency.
---
# Open Questions / Future Improvements
- **Quantitative scenario probabilities**: convert `probability` from 3-value ordinal to numeric (0.05-0.35 low, 0.35-0.65 medium, 0.65-0.95 high) once data pipeline produces FUND scenario distributions. Enables Brier-score calibration in feedback engine.
- **Scenario-similarity threshold (Rule 16)**: 0.85 token-Jaccard is a guess. Tune after 20+ runs; raise to 0.90 if false positives, shift to LLM-judge similarity if paraphrased reuse passes.
- **Cyclical-recovery / cyclical-peak archetype reference**: update Step 3 if Researcher’s archetype vocabulary expands.
(Cut from v1.0: pre-computed scenario library, structured stop-loss-as-ATR-multiple object, pre-computed metric reliability scoring, probability-impact joint calibration, `risk_profile_invalidators` mirror field — all deferred or out of scope for retail v1.)
---
# Pending Propagations
1. **CIO Agent Prompt — Risk Advisor consumption note**: (a) `risk_reward_ratio` is the analytical equivalent of an advocate `recommendation`; (b) `position_size_recommendation` and `stop_loss_suggestion` are CIO-actionable; (c) Risk Advisor `groundedness_score` reflects risk-profile groundedness, NOT directional conviction; (d) `key_factors[].sentiment: neutral` is admissible from this agent.
2. **CIO Agent Prompt — downside_scenarios consumption**: read as probability-weighted distribution input, NOT directional thesis. CIO weighs Bull/Bear against Risk Advisor scenario distribution.
3. **Bear Case Advocate — ****`tail_risk_assessment`**** boundary**: Bear’s tail_risk is a forced rule-out per advocacy framing; Risk Advisor’s downside_scenarios are the calibrated distribution. Mismatch (e.g., Bear flags `elevated`, Risk Advisor’s max scenario is -10%) is itself signal.
4. **Data Pipeline / Risk Metrics module**: confirm pipeline emits BETA, VOL, DD (1yr/3yr), LIQ (with currency tag), CORR, optional CONC. Document gate behavior if any are missing.
5. **Memory schema — StockAnalysisMemory.risk_profile_history**: add field capturing prior downside_scenarios strings, position_size_recommendation, risk_reward_ratio, observed-vs-predicted scoring for Rule 16.
6. **Feedback Analyst — Risk Advisor scoring rules**: scenario-hit rate (per `occurred` enum), position-sizing accuracy, distinct-trigger ratio across runs.
(Cut from v1.0 pending list: Bull risk_profile reference note, Agents Overview row update, Orchestration Engine dual-input documentation — those are bookkeeping, not blocking.)

_Source: Notion — Project Stock Picker → Agent Prompts → Risk Advisor Agent Prompt. Downloaded 2026-07-02._
