**Agent**: Chief Investment Officer (CIO)<br>**Status**: Draft v1.1<br>**Phase**: Phase 1<br>**Pass**: Synthesis (runs after Pass 2)<br>**Role**: Synthesize four strategic perspectives (Bull, Bear, Risk, Tax) into a single actionable recommendation for a Canadian retail investor. The CIO is the final decision-maker — its output is what the user sees.<br>**Target Model**: **`claude-sonnet-4-6`**** (primary)** with `claude-opus-4-6` as optional escalation for shadow CIO A/B comparison and edge-case high-conflict runs. Thinking mode OFF for JSON stability. The CIO task is heavily scaffolded by structured Pass 2 inputs and explicit decision rules, so the quality gap between frontier-tier cloud models compresses substantially; Sonnet 4.6's structured-output reliability is well-suited to the schema. No local fallback by design — small local models lack the context-handling fidelity for synthesis. Aligns with the LLM Routing rules in `CLAUDE.md`.<br>**Audience**: Canadian retail investors with portfolios under \$250k CAD across TFSA / RRSP / Trading. Output should be readable in 2–3 minutes.<br>**Token Budget**: \~2,500 system prompt \| \~7,000–9,000 compressed Pass 2 + Pass 1 summaries \| \~1,800 expected output \| \~12,000 total
---
# Design Decisions & Rationale
The CIO is the sole synthesis-pass agent. It consumes all Pass 2 outputs and resolves disagreement into a single recommendation.
1. **Synthesis, not advocacy.** The CIO does not pick a side. It reads four perspectives as inputs with different semantics and produces the clearest defensible recommendation given the total evidence.
2. **Input semantics differ by agent.** Bull and Bear `confidence` mean directional conviction. Risk Advisor and Tax Strategist `groundedness_score` mean data quality, NOT direction. Their directional proxies are `risk_profile.risk_reward_ratio` and `tax_profile.tax_efficiency_for_account`. Treating groundedness as direction is a hard semantic error.
3. **Risk Advisor is a constraint, not a vote.** `position_size_recommendation` and `stop_loss_suggestion` are forwarded directly into the final recommendation. `downside_scenarios` are a probability-weighted distribution — NOT a bear thesis.
4. **Tail-risk cross-check is required.** Bear's `tail_risk_assessment` and Risk Advisor's `downside_scenarios` are independently derived; mismatch is signal, not error. Surface divergence in both directions.
5. **Tax is a directional constraint, not a verdict.** `tax_efficiency_for_account: unfavorable` can shave 3–7 confidence points when annual drag is substantial. `cross_account_recommendation` is surfaced verbatim but does not override the placement.
6. **Disagreement score drives confidence ceiling.** `consensus`: full range. `mild_dissent`: note dissent. `split_decision`: hard cap at 65. `high_conflict`: outlook must be neutral or one of the "somewhat" range.
7. **Cold-start cap.** Orchestrator caps `confidence ≤ 65` until 20+ scored predictions exist. The CIO is told the cap is in force so its narrative tone matches the displayed confidence. `confidence_uncapped` is stored for the feedback engine.
8. **Parallel predictions are internal calibration data.** The CIO produces 1-month and 3-month directional calls reusing the same reasoning. They feed the prediction-accumulation engine; the frontend de-emphasizes them relative to the primary recommendation. Both predictions are always produced regardless of timeline — a 1-month call on a short-term stock is still useful calibration data.
9. **Feedback-informed weighting.** When agent accuracy briefs are available (N ≥ 20), the CIO calibrates per-agent weighting. Influence is recorded in `accuracy_influence_note`.
10. **Reflexion brief on prior scored analyses.** The CIO receives a diagnosis of what the prior call got wrong. Use it to avoid repeating the same structural error.
11. **Outlook scale, not buy/sell.** `stock_outlook` is the directional field consumed by the Portfolio Optimization System. Five values: `bullish` / `somewhat_bullish` / `neutral` / `somewhat_bearish` / `bearish`.
12. **Cloud-only model routing.** The local model class used elsewhere in this system (currently `qwen3.6:35b-a3b`, previously `qwen3:14b`) cannot reliably synthesize 12K tokens of competing views at the quality bar this task needs. CIO runs cloud-only by design — this isn't specific to any one local model and shouldn't be revisited without re-testing.
---
# Input Semantics Reference
<table header-row="true">
<tr>
<td>Agent</td>
<td>Directional field</td>
<td>What it means</td>
<td>Direction proxy</td>
</tr>
<tr>
<td>Bull Advocate</td>
<td>`confidence` (0–100)</td>
<td>Directional conviction UP</td>
<td>(same field)</td>
</tr>
<tr>
<td>Bear Advocate</td>
<td>`confidence` (0–100)</td>
<td>Directional conviction DOWN</td>
<td>(same field)</td>
</tr>
<tr>
<td>Risk Advisor</td>
<td>`groundedness_score` (0–100)</td>
<td>Risk-profile data quality, NOT direction</td>
<td>`risk_profile.risk_reward_ratio`</td>
</tr>
<tr>
<td>Tax Strategist</td>
<td>`groundedness_score` (0–100)</td>
<td>Tax-profile data quality, NOT direction</td>
<td>`tax_profile.tax_efficiency_for_account`</td>
</tr>
</table>
**Rule**: Only Bull and Bear `confidence` enter the disagreement-score calculation. Risk and Tax `groundedness_score` are NEVER averaged with advocate confidence and NEVER read as directional signal.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- No direct API calls — consumes Pass 2 agent outputs (this agent doesn't call external data providers itself)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload):
- Pass 2 outputs from all four agents — Bull Advocate, Bear Advocate, Risk Advisor, Tax Strategist — formatted verbatim by `format_advocate_for_cio()` / `format_risk_advisor_for_cio()` / `format_tax_strategist_for_cio()` (\~1,200–1,500 tokens each)
- Pass 1 assessment summaries (RSRCH, FUND, TECH, SENT, MACRO) compressed to \~100 tokens/agent via `format_pass1_summaries_for_cio()` — reference/fact-check only, not for new analysis
- Disagreement score and classification (`consensus` / `mild_dissent` / `split_decision` / `high_conflict`), pre-computed by `DisagreementCalculator` from Bull/Bear `confidence` only
- Cold-start cap active flag — `true` when fewer than 20 scored predictions exist, so the CIO's narrative tone matches the capped confidence the user will see
- Reflexion brief (prior scored predictions, timeline-scaled decay) or Memory brief (prior unscored analysis) — whichever applies; empty if neither
- Agent accuracy briefs, recency-weighted, injected only once N ≥ 20 scored predictions exist
- Stock header (ticker, company name, sector, timeline, account type) plus the corresponding timeline/account instruction blocks
---
# System Prompt
Invoke with `enable_thinking: false`. Stable content precedes the `--- STOCK-SPECIFIC CONTEXT ---` separator for prompt caching.

**Runtime prompt:** see [`prompts/cio/v1.1.txt`](../../prompts/cio/v1.1.txt)
---
# Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`):
<table header-row="true">
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1–4 wks). Central question: is the near-term risk/reward favorable for entry now? Weight near-term catalysts, technicals, event risk, sentiment. Macro context matters less. Sizing reflects 1–4 week hold with a defined exit trigger.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1–12 mo). Balance directional thesis with execution risk and data quality. Multi-quarter earnings momentum, competitive positioning, and macro cycle matter. Sizing reflects 3–12 month horizon.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1–5+ yrs). Weight business quality, moat durability, management track record, secular tailwinds. Single-quarter misses are noise unless they signal structural damage. Sizing reflects multi-year compounding.`</td>
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
<td>`TFSA. Capital gains and Canadian dividends tax-free. US dividend WHT 15% non-recoverable. Permanent capital loss destroys contribution room — amplifies bear case for elevated tail-risk names. Frame around long-run tax-free compounding and permanent-loss asymmetry.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US dividend WHT-exempt under treaty. No withdrawals (locked-in for retirement). Canadian eligible-dividend DTC lost at withdrawal. Frame around tax-deferred compounding and income vs. capital-gain composition at withdrawal.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading (non-registered, taxable). Capital gains at 50% inclusion. Capital losses deductible (indefinite carry-forward). Loss-harvesting is a lever; superficial-loss rule applies (30-day window). Near-term catalysts more actionable here. Frame around after-tax risk-adjusted return.`</td>
</tr>
</table>
The previous `general` branch is removed — this system always runs against a specific account context. If a watchlist or screening mode is later added, define an explicit injection variant rather than reintroducing a fallback.
---
# Output Schema (LLM-produced fields)
Orchestrator populates metadata, applies the cold-start cap, and forwards `stop_loss_suggestion` from Risk Advisor.
```json
{
  "stock_outlook": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish",
  "confidence": 0,
  "expected_return_tier": "strong_outperform|outperform|market_perform|underperform|strong_underperform",
  "position_sizing_recommendation": "X-Y% (adopted from Risk Advisor or explicitly justified deviation)",
  "thesis_summary": "2-3 sentences. CIO's synthesized view. References timeline and account_type.",
  "synthesis_narrative": "250-400 words. Primary user-facing deliverable. Plain language for a Canadian retail investor. Engages both advocates, addresses risk and tax.",
  "what_would_change_my_mind": "1-2 sentences naming the specific evidence that would flip the outlook.",
  "bull_case_assessment": {
    "verdict": "accepted|partially_accepted|rejected",
    "engagement": "1-2 sentences explicitly addressing bull.strongest_argument — agree, refute, or qualify.",
    "weight_applied": "high|medium|low",
    "weight_reason": "1 sentence: why this weight (data quality, accuracy brief, archetype concern, etc.)"
  },
  "bear_case_assessment": {
    "verdict": "accepted|partially_accepted|rejected",
    "engagement": "1-2 sentences explicitly addressing bear.strongest_argument — agree, refute, or qualify.",
    "weight_applied": "high|medium|low",
    "weight_reason": "1 sentence"
  },
  "risk_profile_summary": {
    "risk_reward_consumed_as": "favorable|neutral|unfavorable",
    "position_size_source": "risk_advisor_adopted|risk_advisor_adjusted|default_conservative",
    "sizing_justification": "1 sentence — adoption confirmation OR explicit deviation reason",
    "tail_risk_cross_check": "aligned|bear_elevated_risk_advisor_mild|bear_mild_risk_advisor_elevated|insufficient_data",
    "tail_risk_note": "1 sentence on the cross-check finding — empty string when aligned"
  },
  "tax_summary": {
    "tax_efficiency_consumed_as": "favorable|neutral|unfavorable",
    "tax_impact_on_recommendation": "strengthens|neutral|weakens",
    "cross_account_note": "1-2 sentences. Surface cross_account_recommendation if present (note: cross-account moves apply to new capital, not transfer of existing positions); else confirm current account is appropriate. References account_type."
  },
  "key_decision_factors": [
    {
      "factor": "name",
      "direction": "bullish|bearish|neutral",
      "source": "bull|bear|risk|tax|pass1",
      "evidence": "1 sentence concrete citation"
    }
  ],
  "accuracy_influence_note": "1-2 sentences on how accuracy briefs influenced weighting. Empty string if briefs not present.",
  "dissent_note": "1-2 sentences on significant dissenting signals the investor should know about. Empty string if consensus or mild_dissent with no material dissenters.",
  "parallel_predictions": {
    "1_month": { "direction": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish", "confidence": 0 },
    "3_month": { "direction": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish", "confidence": 0 }
  }
}
```
### Expected Return Tier — relative anchors
<table header-row="true">
<tr>
<td>Tier</td>
<td>Meaning</td>
</tr>
<tr>
<td>`strong_outperform`</td>
<td>Materially ahead of market over the analysis timeline (anchor: \>15%)</td>
</tr>
<tr>
<td>`outperform`</td>
<td>Modestly ahead of market (anchor: 8–15%)</td>
</tr>
<tr>
<td>`market_perform`</td>
<td>Broadly tracking market (anchor: −5% to +8%)</td>
</tr>
<tr>
<td>`underperform`</td>
<td>Modestly behind market (anchor: −5% to −15%)</td>
</tr>
<tr>
<td>`strong_underperform`</td>
<td>Materially behind market (anchor: \>15% downside)</td>
</tr>
</table>
Until FUND produces price scenarios (Phase 2), `expected_return_tier` is a relative-performance call, not a price target. The frontend should label it accordingly to avoid retail users treating it as a forecast.
### Array Bounds (validator-enforced)
<table header-row="true">
<tr>
<td>Field</td>
<td>Min</td>
<td>Max</td>
</tr>
<tr>
<td>`key_decision_factors`</td>
<td>3</td>
<td>6</td>
</tr>
</table>
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name` (`"cio"`), `agent_pass` (`"synthesis"`), `analysis_context`, `pass2_agent_status`
- **`confidence`**: Orchestrator applies cold-start cap: `final_confidence = min(llm_confidence, 65)` when `predictions_count < 20`. `confidence_uncapped` stored separately. The CIO is told whether the cap is active so its narrative tone matches the displayed confidence.
- **`stop_loss_suggestion`**: Orchestrator forwards `risk_advisor.risk_profile.stop_loss_suggestion` directly — LLM does not produce this field.
- **`disagreement_score`** and **`disagreement_classification`**: From `AnalysisRun` (pre-computed by `DisagreementCalculator`).
- **`provisional_cap_applied`**: Boolean — `true` when the cap was binding.
---
# Orchestrator Input Assembly
Before invoking the CIO, the orchestrator assembles the payload in this order: stock header → Pass 2 outputs → Pass 1 summaries.
```python
def build_cio_input(
    pass2_outputs: dict[str, AgentOutput],
    pass1_outputs: list[AgentOutput],
    context: AnalysisContext,
    run: AnalysisRun,
    cold_start_cap_active: bool,
) -> str:
    header = (
        f"{context.canonical_ticker} ({context.company_name}) | {context.sector} | "
        f"Timeline: {context.timeline} | Account: {context.account_type}\n"
        f"Disagreement score: {run.disagreement_score} | Classification: {run.disagreement_class}\n"
        f"Cold-start cap active: {cold_start_cap_active}\n"
    )
    bull_block  = format_advocate_for_cio(pass2_outputs.get("bull_case_advocate"), "BULL")
    bear_block  = format_advocate_for_cio(pass2_outputs.get("bear_case_advocate"), "BEAR")
    risk_block  = format_risk_advisor_for_cio(pass2_outputs.get("risk_advisor"))
    tax_block   = format_tax_strategist_for_cio(pass2_outputs.get("tax_strategist"))
    pass1_block = format_pass1_summaries_for_cio(pass1_outputs)
    return "\n\n".join([header, bull_block, bear_block, risk_block, tax_block, pass1_block])

def format_advocate_for_cio(output: AgentOutput | None, label: str) -> str:
    if output is None or output.status != "completed":
        reason = output.error_detail if output else "agent not run"
        return f"--- {label} CASE ADVOCATE: NOT AVAILABLE (reason: {reason}) ---"

    sd  = output.structured_data or {}
    tar = sd.get("tail_risk_assessment", {})  # Bear only; empty for Bull
    arch = sd.get("thesis_archetype_alignment", {})
    arch_key  = "bull_archetype" if label == "BULL" else "bear_archetype"
    align_key = "alignment_with_researcher" if label == "BULL" else "agrees_with_researcher"

    return f"""--- {label} CASE ADVOCATE ---
confidence: {output.confidence}/100  <- directional conviction
data_quality: {output.data_quality_assessment}
thesis_summary: {output.thesis_summary}
strongest_argument: {output.strongest_argument}
weakest_point: {output.weakest_point}
caveats: {output.caveats}
narrative: {output.narrative}
core_arguments: {sd.get("core_arguments", [])}
archetype: researcher={arch.get("researcher_archetype")} | {label.lower()}={arch.get(arch_key)} | alignment={arch.get(align_key)}
tail_risk_level: {tar.get("tail_risk_level", "n/a (bull)")} | scenario: {tar.get("scenario", "")}
asymmetry_assessment: {sd.get("asymmetry_assessment", "")}"""

def format_risk_advisor_for_cio(output: AgentOutput | None) -> str:
    if output is None or output.status != "completed":
        return "--- RISK ADVISOR: NOT AVAILABLE ---"
    rp = output.risk_profile or {}
    scenarios = "\n".join(
        f"  - {s['scenario']} | prob: {s['probability']} | impact: {s['estimated_impact_pct']}% | trigger: {s['trigger']}"
        for s in rp.get("downside_scenarios", [])
    )
    return f"""--- RISK ADVISOR ---
groundedness_score: {output.groundedness_score}/100  <- data quality, NOT direction
risk_reward_ratio: {rp.get("risk_reward_ratio")}  <- directional proxy
data_quality: {output.data_quality_assessment}
thesis_summary: {output.thesis_summary}
strongest_signal: {output.strongest_signal}
volatility_assessment: {rp.get("volatility_assessment")} | beta: {rp.get("beta")} | max_drawdown_1yr: {rp.get("max_drawdown_1yr")}%
position_size_recommendation: {rp.get("position_size_recommendation")}  <- CIO-actionable
stop_loss_suggestion: {rp.get("stop_loss_suggestion")}  <- orchestrator passes through
downside_scenarios (probability-weighted distribution — NOT a directional thesis):
{scenarios}
concentration_risk: {rp.get("concentration_risk")} | liquidity_risk: {rp.get("liquidity_risk")}
data_sanity_flags: {rp.get("data_sanity_flags", [])}"""

def format_tax_strategist_for_cio(output: AgentOutput | None) -> str:
    if output is None or output.status != "completed":
        return "--- TAX STRATEGIST: NOT AVAILABLE ---"
    tp = output.tax_profile or {}
    cross = tp.get("cross_account_recommendation")
    cross_text = (
        f"For new capital, {cross['better_account']}: {cross['reasoning']} "
        f"(drag_delta: {cross['drag_delta_pct']}%/yr)"
        if cross else "null — current account is appropriate"
    )
    return f"""--- TAX STRATEGIST ---
groundedness_score: {output.groundedness_score}/100  <- data quality, NOT direction
tax_efficiency_for_account: {tp.get("tax_efficiency_for_account")}  <- indirect directional signal
account_fit_score: {tp.get("account_fit_score")}
data_quality: {output.data_quality_assessment}
thesis_summary: {output.thesis_summary}
strongest_signal: {output.strongest_signal}
dividend_classification: {tp.get("dividend_classification")} | yield: {tp.get("dividend_yield_pct")}% | after_tax_yield: {tp.get("effective_after_tax_yield_pct")}% | annual_drag: {tp.get("annual_tax_drag_pct")}%/yr
wht_interpretation: {tp.get("wht_interpretation")}
cross_account_recommendation: {cross_text}
key_tax_risks: {tp.get("key_tax_risks", [])}"""

def format_pass1_summaries_for_cio(pass1_outputs: list[AgentOutput]) -> str:
    """Pass 1 assessment summaries — fact-check only, ~100 tokens per agent."""
    ABBREV = {
        "stock_researcher": "RSRCH", "fundamental_analyst": "FUND",
        "technical_analyst": "TECH", "sentiment_analyst": "SENT",
        "macro_economist": "MACRO",
    }
    lines = ["--- PASS 1 SUMMARIES (reference / fact-check only — do not introduce new findings) ---"]
    for o in pass1_outputs:
        abbrev = ABBREV.get(o.agent_name if o else "", "?")
        if o and o.status == "completed":
            lines.append(f"{abbrev} (reliability {o.reliability_score}): {o.assessment_summary}")
        else:
            lines.append(f"{abbrev}: UNAVAILABLE")
    return "\n".join(lines)
```
---
# Orchestrator Merge Logic
```python
def merge_output_cio(
    llm_response: dict,
    pass2_outputs: dict[str, AgentOutput],
    context: AnalysisContext,
    run: AnalysisRun,
    provisional_cap: int = 65,
    predictions_count: int = 0,
) -> dict:
    raw_confidence   = llm_response["confidence"]
    cap_applied      = predictions_count < 20 and raw_confidence > provisional_cap
    final_confidence = min(raw_confidence, provisional_cap) if predictions_count < 20 else raw_confidence

    # Stop-loss passes through from Risk Advisor — orchestrator is authoritative
    risk = pass2_outputs.get("risk_advisor")
    stop_loss = risk.risk_profile.get("stop_loss_suggestion") if risk and risk.risk_profile else None

    # Strip any forbidden directional fields the LLM may hallucinate
    for forbidden in ("recommendation", "buy_sell_hold", "action"):
        llm_response.pop(forbidden, None)

    return {
        "agent_name":                     "cio",
        "agent_pass":                     "synthesis",
        "analysis_context": {
            "account_type":               context.account_type,
            "timeline":                   context.timeline,
            "stock_id":                   str(context.stock_id),
            "canonical_ticker":           context.canonical_ticker,
        },
        "stock_outlook":                  llm_response["stock_outlook"],
        "confidence":                     final_confidence,
        "confidence_uncapped":            raw_confidence,
        "provisional_cap_applied":        cap_applied,
        "expected_return_tier":           llm_response["expected_return_tier"],
        "position_sizing_recommendation": llm_response["position_sizing_recommendation"],
        "stop_loss_suggestion":           stop_loss,
        "thesis_summary":                 llm_response["thesis_summary"],
        "synthesis_narrative":            llm_response["synthesis_narrative"],
        "what_would_change_my_mind":      llm_response["what_would_change_my_mind"],
        "bull_case_assessment":           llm_response["bull_case_assessment"],
        "bear_case_assessment":           llm_response["bear_case_assessment"],
        "risk_profile_summary":           llm_response["risk_profile_summary"],
        "tax_summary":                    llm_response["tax_summary"],
        "key_decision_factors":           llm_response["key_decision_factors"],
        "accuracy_influence_note":        llm_response["accuracy_influence_note"],
        "dissent_note":                   llm_response["dissent_note"],
        "parallel_predictions":           llm_response["parallel_predictions"],
        "disagreement_score":             run.disagreement_score,
        "disagreement_classification":    run.disagreement_class,
        "pass2_agent_status": {
            name: output.status for name, output in pass2_outputs.items()
        },
    }
```
---
# Validation Rules
Enforced by `src/agents/validators.py`. Failure triggers retry with error list appended.
1. **JSON parseable** (strip markdown fences if present).
2. **Required fields**: all schema fields above must be present.
3. **No forbidden fields**: `recommendation`, `buy_sell_hold`, `action` stripped silently. Logs prompt-regression warning.
4. **`stock_outlook`**: one of the 5 enum values.
5. **`confidence`**: integer 0–100 (orchestrator applies cap post-validation).
6. **`expected_return_tier`**: one of the 5 enum values.
7. **`position_sizing_recommendation`**** format**: regex `^\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?%$`.
8. **`thesis_summary`**: 80–500 chars.
9. **Genuine advocate engagement is enforced by rule 10 via the structured ****`bull_case_assessment.engagement`**** and ****`bear_case_assessment.engagement`**** fields (≥ 40 chars each, explicitly addressing each advocate's ****`strongest_argument`****).**
10. **Assessment fields**: `bull_case_assessment.verdict`, `bear_case_assessment.verdict` ∈ enum; `engagement` ≥ 40 chars; `weight_applied` ∈ enum.
11. **`risk_profile_summary.risk_reward_consumed_as`**: must match `risk_advisor.risk_profile.risk_reward_ratio`. Mismatch → retry with error message.
12. **`risk_profile_summary.tail_risk_cross_check`**: ∈ enum.
13. **`tax_summary.tax_efficiency_consumed_as`**: must match `tax_strategist.tax_profile.tax_efficiency_for_account`. Mismatch → retry.
14. **`key_decision_factors`**: 3–6 items. Each requires `factor`, `direction` ∈ enum, `source` ∈ enum, `evidence` non-empty.
15. **`parallel_predictions`**: both `1_month` and `3_month` direction ∈ enum, confidence 0–100.
16. **Disagreement-confidence coupling (hard)**: `split_decision` AND `confidence > 65` → retry. `high_conflict` AND `stock_outlook ∉ {neutral, somewhat_bullish, somewhat_bearish}` → retry. (No narrative-substring bypass — the cap is hard.)
17. **Outlook/confidence consistency**: `stock_outlook == "neutral"` AND `confidence > 70` → retry. `stock_outlook ∈ {bullish, bearish}` AND `confidence < 50` → retry.
18. **Numeric fabrication guard**: numeric tokens in `synthesis_narrative` and `key_decision_factors[].evidence` must appear in the Pass 2 input payload. Missing → warning + single retry.
19. **Missing advocate handling**: Bull absent OR Bear absent → `confidence` capped at 50 by validator. Both absent → `stock_outlook: neutral`, `confidence: 0`.
20. **Parallel-prediction consistency**: 1-month or 3-month direction whose position differs from primary `stock_outlook` by more than two on the 5-point scale → require explicit reasoning citation in `synthesis_narrative` (e.g., named short-term catalyst). Otherwise retry.
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Reminders:
- stock_outlook ∈ {bullish, somewhat_bullish, neutral, somewhat_bearish, bearish} — NOT buy/sell/hold
- synthesis_narrative must engage both bull.strongest_argument AND bear.strongest_argument
- position_sizing_recommendation must be X-Y% (adopt from Risk Advisor or state deviation reason)
- risk_reward_consumed_as must match risk_advisor.risk_profile.risk_reward_ratio
- tax_efficiency_consumed_as must match tax_strategist.tax_profile.tax_efficiency_for_account
- split_decision → confidence ≤ 65 (hard cap; no narrative bypass)
- high_conflict → stock_outlook ∈ {neutral, somewhat_bullish, somewhat_bearish}
- key_decision_factors: 3–6 items, each with factor, direction, source, evidence
- Do NOT produce a "recommendation" field
```
---
# Reflexion Brief Schema
Injected when the same stock has been analyzed before AND at least one prior prediction has been scored. When prior analysis exists but is NOT yet scored, the Memory Brief is used instead.
```plain text
PRIOR ANALYSIS REFLEXION ({ticker}):
Last analysis ({days_since} days ago, timeline={prior_timeline}, account={prior_account}):
  CIO recommendation: {prior_stock_outlook} | confidence: {prior_confidence}/100
  Expected return tier: {prior_expected_return_tier}
  Actual outcome at {checkpoint_label}: {actual_return_pct}% over {holding_days} days
  Prior was: correct|incorrect|partially_correct
  Primary error source: {e.g., "bull advocate overconfident on rate sensitivity"}
  Primary error detail: {1-2 sentence diagnosis from Feedback Analyst}
  Lesson: {specific learnable correction}
  What changed since: {events — earnings, price move, macro shift, news}

Use this history to calibrate. If the same risk factors are present and unchanged, weight them more heavily. If conditions have materially changed per "what changed since", note the divergence.
```
---
# Agent Accuracy Briefs Schema
Injected when N ≥ 20 scored predictions. Empty string otherwise.
```plain text
AGENT ACCURACY BRIEFS ({N} scored predictions):

BULL ADVOCATE:
  Overall directional accuracy: {X}% | This sector: {Y}% | This timeline: {Z}%
  Recent trend: improving|stable|declining
  Notable pattern: {e.g., "Overweights momentum on tech — discount bull confidence 5–10pts on momentum-driven tech calls"}

BEAR ADVOCATE:
  Overall directional accuracy: {X}% | This sector: {Y}% | This timeline: {Z}%
  Recent trend: improving|stable|declining
  Notable pattern: {e.g., "Over-flags dividend risk in consumer staples — bear confidence on dividend-risk theses 40% accurate vs 58% overall"}

RISK ADVISOR:
  Scenario-hit rate: {X}% (scenarios realized within ±5pct of estimated_impact_pct)
  Position-sizing accuracy: {Y}%
  Notable pattern: {e.g., "Underestimates drawdown on leveraged cyclicals — widen downside scenarios by 5–10pct when archetype is cyclical_recovery"}

TAX STRATEGIST:
  Account-fit accuracy: {X}% (drag_delta_pct realized within ±0.30pct annually)
  Cross-account acted-on rate: {Y}%
  Notable pattern: {e.g., "WHT drag on US REITs in TFSA consistently 0.2–0.4pct underestimated — add buffer"}
```
---
# Memory Brief Schema (Pre-scoring Case)
Injected when a prior CIO analysis exists but has NOT been scored yet.
```plain text
PRIOR CIO ANALYSIS ({ticker}, {days_since} days ago):
  Prior recommendation: {prior_stock_outlook} | confidence: {prior_confidence}/100
  Prior thesis summary: {prior_thesis_summary}
  Events since: {events — price change, earnings, macro shifts, notable news}
  Scoring status: not_yet_scored
```
**Timeline-scaled decay (applies to both Reflexion Brief and Memory Brief)**: Included if \<30 days old at `short_term`, \<90 days at `medium_term`, \<180 days at `long_term`. Below threshold, the brief is omitted. Matches the cross-agent standard (Researcher v1.3, Sentiment v1.2c, Macro v1.2c, Bull v1.6c, Bear v2c, Risk Advisor v1.1c).
---
# Minimum Data Threshold
**Gate 2**: both Bull AND Bear must complete (mandatory). Risk and Tax failures are warnings.
<table header-row="true">
<tr>
<td>Scenario</td>
<td>CIO behavior</td>
</tr>
<tr>
<td>Both advocates available</td>
<td>Full synthesis</td>
</tr>
<tr>
<td>Bull only</td>
<td>Proceed; `confidence ≤ 50`; note asymmetric input in `dissent_note`</td>
</tr>
<tr>
<td>Bear only</td>
<td>Proceed; `confidence ≤ 50`; note asymmetric input in `dissent_note`</td>
</tr>
<tr>
<td>Both advocates absent</td>
<td>`stock_outlook: neutral`, `confidence: 0`, narrative notes inability to synthesize</td>
</tr>
<tr>
<td>Risk Advisor absent</td>
<td>`position_size_recommendation: "1-2%"` default; `position_size_source: "default_conservative"`</td>
</tr>
<tr>
<td>Tax Strategist absent</td>
<td>`tax_efficiency_consumed_as: "unknown"`; proceed; note in `dissent_note`</td>
</tr>
</table>
---
# Prompt Variable Injection Reference
<table header-row="true">
<tr>
<td>Placeholder</td>
<td>Source</td>
<td>Notes</td>
</tr>
<tr>
<td>`{ticker}`, `{company_name}`, `{sector}`</td>
<td>`AnalysisContext` • `DataBundle.company_info`</td>
<td>Real names — not anonymized</td>
</tr>
<tr>
<td>`{timeline}`, `{account_type}`</td>
<td>`AnalysisContext`</td>
<td></td>
</tr>
<tr>
<td>`{timeline_instruction}`, `{account_instruction}`</td>
<td>Conditional injection table</td>
<td></td>
</tr>
<tr>
<td>`{disagreement_score}`, `{disagreement_classification}`</td>
<td>`AnalysisRun`</td>
<td>Pre-computed</td>
</tr>
<tr>
<td>`{cold_start_cap_active}`</td>
<td>Orchestrator</td>
<td>`true` when `predictions_count < 20`</td>
</tr>
<tr>
<td>`{reflexion_brief}`</td>
<td>`StockAnalysisMemory` • Feedback Analyst</td>
<td>Empty if first analysis OR prior not yet scored</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory`</td>
<td>Empty if no prior analysis</td>
</tr>
<tr>
<td>`{agent_accuracy_briefs}`</td>
<td>Feedback engine</td>
<td>Empty if N \< 20</td>
</tr>
<tr>
<td>`{pass1_summaries}`</td>
<td>`format_pass1_summaries_for_cio()`</td>
<td>\~100 tokens per agent</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
<td></td>
</tr>
<tr>
<td>Pass 2 blocks</td>
<td>`format_*_for_cio()` per agent</td>
<td>Missing → "NOT AVAILABLE — reason: \{reason\}"</td>
</tr>
</table>
---
# Testing Checklist
- Valid JSON from Sonnet 4.6 with `enable_thinking: false`?
- No `recommendation` / `buy_sell_hold` / `action` field?
- `synthesis_narrative` 250–400 words and engages both advocates?
- `what_would_change_my_mind` populated and concrete?
- `position_sizing_recommendation` in X-Y%, adopted or explicitly justified?
- `risk_reward_consumed_as` matches Risk Advisor exactly?
- `tax_efficiency_consumed_as` matches Tax Strategist exactly?
- **Disagreement test (split)**: `confidence ≤ 65`?
- **Disagreement test (high)**: `stock_outlook` in neutral range?
- **Tail risk cross-check (elevated/mild)**: surfaced with explanatory note?
- **Tail risk cross-check (mild/elevated)**: surfaced with explanatory note?
- **Archetype disagreement**: surfaced in narrative?
- **Input semantics**: high `groundedness_score` with `unfavorable` ratio NOT treated as bullish?
- **Cold-start cap**: 78 raw → 65 displayed, narrative tone matches calibrated read?
- **Reflexion integration**: prior incorrect call + same risk factors → narrative references the lesson?
- **Accuracy brief**: low sector accuracy → `accuracy_influence_note` reflects discount?
- **Missing advocate**: Bear absent → `confidence ≤ 50`, dissent flagged?
- **Missing Risk Advisor**: default sizing applied?
- **Parallel predictions**: populated; consistent with primary unless explicit divergence reason?
- **Tax unfavorable + neutral directional**: CIO leans neutral or surfaces cross-account note?
- **Numeric fabrication guard**: all numbers in narrative trace to Pass 2 payload?
- **Fixture stocks**: (a) bull consensus → bullish/high; (b) bear consensus → bearish/high; (c) quality-at-rich-valuation → split_decision/somewhat_bullish/≤65; (d) high conflict → neutral range; (e) TFSA + US dividend stock with `cross_account_recommendation` to RRSP **for new capital** (not transfer of existing TFSA position) → cross-account note prominent.
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
<td>\~2,500 tokens</td>
<td>CoT + 6 rules + injection blocks + schema</td>
</tr>
<tr>
<td>Pass 2 outputs</td>
<td>\~6,500–8,000 tokens</td>
<td>Bull \~1,500 + Bear \~1,200 + Risk \~900 + Tax \~1,000 + formatting</td>
</tr>
<tr>
<td>Pass 1 summaries</td>
<td>\~500 tokens</td>
<td>\~100 per agent</td>
</tr>
<tr>
<td>Context + briefs</td>
<td>\~500–1,000 tokens</td>
<td>Disagreement, reflexion, accuracy briefs</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~1,800 tokens</td>
<td>Narrative 250–400 words + structured fields</td>
</tr>
<tr>
<td>**Total**</td>
<td>**\~12,000 tokens**</td>
<td>Within Sonnet/Opus 4.6 limits</td>
</tr>
</table>
**Model configuration**:
- Primary: `claude-sonnet-4-6` (per CLAUDE.md). Optional escalation: `claude-opus-4-6` for shadow CIO A/B comparison and high-stakes edge cases.
- `max_tokens`: **5000** (empirical min safe value from 2026-05-15 simulation; the prior 2,500 caused silent JSON truncation. Set to min safe + 20% headroom per the Orchestration Engine LLM Call Parameters section.)
- `temperature`: 0.3
- Thinking mode: OFF
- JSON output: prefilled assistant turn with `{` to eliminate markdown fences (see Orchestration Engine § LLM Call Parameters).
- Prompt caching: stable system prefix (\~1,500 tokens) cached per deploy
- Expected latency: \~8–12s per stock (Sonnet); \~10–18s (Opus escalation)
- Cost per stock: \~\$0.05–0.08 (Sonnet) / \~\$0.15–0.25 (Opus escalation). 50-stock weekly batch: \~\$2.50–4/week (Sonnet) at steady state — re-confirm with latest published Anthropic pricing before launch.
---
# Implementation Files
- `src/agents/cio.py` — CIO agent implementation
- `src/agents/cio_formatter.py` — `format_advocate_for_cio()`, `format_risk_advisor_for_cio()`, `format_tax_strategist_for_cio()`, `format_pass1_summaries_for_cio()`
- `src/orchestrator/engine.py` — `run_cio()`, `merge_output_cio()`
- `src/feedback/reflexion.py` — reflexion brief construction
---
# Pending Propagations
1. **Portfolio Optimization System** (Milestone 2+): consumes `stock_outlook`, `confidence`, `expected_return_tier`, and a future `return_components.net_expected_return_pct`. Define `return_components` shape when the Portfolio Health Assessor is scoped.
2. **Shadow CIO** (Milestone 2+): receives the same Pass 2 payload with a contrarian weighting variant. Stores output in `shadow_predictions`. Does not influence user-facing recommendation.
	> **Status as of 2026-05-15:** not implemented in either the production orchestrator or the Phase 1 simulation harness. No `pass4_shadow_cio.py` runner exists yet. The 2026-05-15 end-to-end simulation produced primary CIO output only — no shadow predictions have been generated. The Feedback Learning system's 20-prediction shadow vs primary win-rate comparison cannot begin accumulating data until this is built. Suggested as the first additive step of Milestone 2 backend work (same Pass 2 input shape, separate bear-biased system prompt, separate `shadow_predictions` storage path). The bear-biased prompt variant still needs to be written and reviewed before implementation.
3. **Feedback Analyst — CIO scoring rules**: directional accuracy (`stock_outlook` vs actual), return-tier accuracy, confidence calibration (Brier score per timeline), reflexion-lesson uptake.
4. **Semantic narrative-engagement validator**: replace the substring check in validation rule 9 with a small classifier (or short Sonnet call) that scores whether each side's `strongest_argument` was meaningfully addressed.
5. **Account_type general / watchlist mode**: if a non-account screening mode is added, define an explicit injection variant (header, instruction, schema branch) rather than a fallback.
6. **Sonnet vs Opus A/B fixture test**: once the pipeline is working, run both models on a fixed set of 10–20 fixture stocks. Score on (a) directional consistency between runs, (b) blind narrative-quality read, (c) first-try validation pass rate. The Shadow CIO is a natural place to run the *other* model class as the contrarian variant. If Sonnet is within 5–10% of Opus on these dimensions, the cost gap (\~5×) is not justified for the primary CIO; if not, escalate.
---
# Open Questions
1. **`expected_return_tier`**** grounding (HIGH-VALUE PHASE 1.5 CANDIDATE — do not forget once Phase 1 is running)**
	Currently the CIO produces the tier as a vibes-based judgment with no underlying valuation anchor — the percentage brackets in the schema are decoration, not commitments.
	**This is the single highest-leverage upgrade for the calibration system in the long run**
	because grounded numerical predictions enable Brier-score-equivalent calibration, scenario-hit-rate metrics, and tier-accuracy measurement — all of which are dramatically richer signal than the direction-only data you get from the 5-value outlook alone. The compounding effect over 6–12 months of pipeline operation is large.
	**Recommended graduated path:**
	- **Phase 1 (now)**: Ship with the current vibes-based tier. Frontend labels `expected_return_tier` as "relative to market" (e.g., "Expected to outperform the market") rather than displaying the percentage anchor brackets. The brackets stay in the schema as calibration documentation but never reach the user. Avoids implying false precision while we get the pipeline running.
	- **Phase 1.5 (\~1 week of work, schedule once Phase 1 has a few weeks of stable operation)**: Extend FUND output to include a single base-case target price anchored to FMP analyst consensus (LLM may deviate ±10% with named methodology reason in a `methodology` field) plus separate qualitative bull catalysts and bear catalysts lists. Orchestrator computes expected return from the base case and maps to tier. Frontend updates to show "based on consensus-anchored target". Captures \~70% of the calibration value with minimal hallucination surface area.
	- **Phase 2**: Add explicit bull / base / bear targets with probability weights. Probabilities derived from the LLM's read of four-agent inputs, calibrated against accumulated Phase 1.5 outcome data.
	**What NOT to do**
	: do not ask the LLM to perform original DCF or original valuation. Models confabulate WACC, terminal growth, multiples, etc. — this is exactly the failure mode where they produce confidently-wrong financial precision. The discipline is: anchor numbers to public data (FMP analyst consensus, peer multiples, FRED rates) and let the LLM do qualitative adjustment, not primitive valuation.
	**Decision criterion for promoting Phase 1.5 work**
	: once Phase 1 has produced \~4 weeks of pipeline-stable analyses without major schema or orchestration changes, the lite version becomes the highest-priority calibration enhancement. Tracked separately from Phase 2 Portfolio Optimizer work — these are independent workstreams.
2. **Archetype disagreement weight**: currently surfaced narratively but not penalized in confidence. Future: −5pt confidence adjustment when archetypes differ by ≥2 positions, capped.
3. **Parallel-prediction visibility**: confirm with frontend that 1-month and 3-month calls are de-emphasized relative to the primary outlook (e.g., shown only on a "calibration" expanded view), to avoid retail confusion when they diverge from primary.
4. **Confidence-weighting of accuracy briefs**: currently narrative ("discount this advocate"). Future: inject `weight_modifier: 0.75` into Pass 2 blocks once N supports reliable per-sector estimates.

_Source: Notion — Project Stock Picker → Agent Prompts → Chief Investment Officer (CIO) Agent Prompt. Downloaded 2026-07-02._
