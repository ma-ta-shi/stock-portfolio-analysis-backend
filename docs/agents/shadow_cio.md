# 👤 Shadow CIO (calibration agent) Agent Prompt

**Agent**: Shadow CIO (calibration agent)<br>**Status**: Draft v1.0<br>**Phase**: Phase 1 (ships with primary CIO — calibration data must accumulate from launch)<br>**Pass**: Synthesis (parallel, non-blocking — runs alongside primary CIO finalization)<br>**Role**: Produce a deliberately bear-biased outlook for every CIO run, stored to `shadow_predictions` for calibration against the primary CIO's outlook. Never user-facing.<br>**Target Model**: **`qwen3.6:35b-a3b`**** via Ollama (primary)** with **`claude-haiku-4-5-20251001`**** (fallback)**. Local-primary is acceptable here — output is short (~1,000 tokens), the schema is a strict subset of the CIO's, and the calibration signal is averaged across many runs so individual-run noise is tolerable. Haiku 4.5 is the cheapest cloud option that respects strict-schema output. Thinking mode OFF (verify `enable_thinking:false` fully suppresses reasoning on this model — see Agent Design Patterns Pattern 10 caveat). See Design Decision #6 for why CIO's cloud-only rationale does not transfer.<br>**Audience**: None. Output is internal data — written to `shadow_predictions`, never rendered to the user.<br>**Token Budget**: ~1,200 system prompt | ~8,000 compressed Pass 2 + Pass 1 summaries | ~1,000 expected output | ~10,200 total

---

# Design Decisions & Rationale

The Shadow CIO is a calibration instrument, not a second opinion shown to the user. Its purpose is to detect directional bias in the primary CIO over time by providing a fixed bear-biased comparison point on every run.

1. **Bear-biased weighting is the only intentional difference from the primary CIO.** Identical inputs, identical scaffolding, near-identical schema — only the system-prompt weighting differs. Introducing any other difference (different input set, different timeline framing, different account treatment) contaminates the calibration signal and makes primary-vs-shadow comparisons uninterpretable.
2. **Shadow output is a strict subset of CIO output.** No `synthesis_narrative`, no `key_decision_factors`, no `bull_case_assessment` / `bear_case_assessment` / `risk_profile_summary` / `tax_summary` objects, no `position_sizing_recommendation`, no `stop_loss_suggestion`. The shadow only needs to produce enough to be scored apples-to-apples against the primary on the *directional* call (outlook, confidence, return tier, brief thesis, 1m/3m parallel predictions). Less output = lower cost, faster latency, smaller hallucination surface area.
3. **No cold-start cap.** The primary CIO's `confidence` is capped at 65 for the first 20 scored predictions to avoid showing the user inflated confidence on an unvalidated system. Shadow `confidence` is NOT capped — the comparison engine compares `primary.confidence_uncapped` against `shadow.confidence`, and applying the cap to both would erase the signal we are trying to measure. The shadow is never shown to a user, so calibrated-vs-uncalibrated narrative tone is moot.
4. **Risk and Tax remain constraints, not directional votes.** The bear weighting applies to Bull/Bear `confidence` and to Risk's `downside_scenarios` (worst-case weighted up). It does NOT invert `tax_efficiency_for_account`, flip `risk_reward_ratio`, or treat `groundedness_score` as a directional prior. These are factual properties, not confidence-bearing signals.
5. **Disagreement-score caps still apply.** `split_decision` → `confidence ≤ 65`. `high_conflict` → `stock_outlook ∈ {neutral, somewhat_bullish, somewhat_bearish}`. These are mechanical floors that exist to prevent overconfidence in genuinely contested cases; bear bias does not bypass them. (Under bear bias, `high_conflict` more often resolves to `somewhat_bearish` than `somewhat_bullish` — that's the intended effect, not a bypass.)
6. **Local model is acceptable; CIO's "cloud-only" rationale does not transfer.** The primary CIO is cloud-only because user-facing synthesis at ~2,500-token output demands frontier-tier reasoning fidelity. The shadow produces ~1,000 tokens of structured-schema output and is averaged across many runs for the calibration signal — individual-run quality matters less. `qwen3.6:35b-a3b` at an estimated ~60s latency (unverified, re-measure after deployment) is acceptable for non-blocking calibration; Haiku 4.5 (~3–5s, ~$0.005/run) covers Ollama outages. If steady-state shadow-vs-primary win-rate analysis shows the local model is systematically too noisy, escalate to Haiku as primary in v1.1.
7. **Non-blocking and isolated from user output.** The shadow runs in parallel with the primary CIO's orchestrator merge. If it fails, the primary CIO output is still delivered to the user. Failures are logged but do NOT retry-storm — sparse early calibration data is acceptable.
8. **Aggregate metrics drive feedback into the primary CIO.** Shadow win-rate by sector, regime, and timeline is computed by the Feedback Analyst and surfaced into the primary CIO's `{agent_accuracy_briefs}` injection (when N ≥ 20 scored). See "Scoring & Feedback Loop" below.
9. **High-divergence flag is computed in orchestrator merge, not by the LLM.** When primary and shadow `stock_outlook` differ by more than 2 positions on the 5-point scale, the orchestrator records `high_divergence: true` on the AnalysisRun. The Feedback Analyst surfaces historical accuracy on high-divergence runs into the primary CIO's next-run accuracy brief. The LLM does not see the primary's output and cannot know divergence at write-time.
10. **Cold-start cap status is informational only.** The shadow is told whether the primary's cap is in force (so its `thesis_summary` tone can match if it cites prior calibration) but does not apply the cap to its own confidence.
11. **Same accuracy briefs and reflexion briefs as primary.** Shadow receives `{reflexion_brief}`, `{memory_brief}`, and `{agent_accuracy_briefs}` identically to primary. The bear weighting is a fixed system-prompt directive; underlying agent learnings flow to both pipelines equally so the comparison stays apples-to-apples. (See Open Question #1 — the alternative of holding the shadow's information substrate constant was considered and rejected.)
12. **No buy/sell/hold; same 5-value outlook scale as primary.** The shadow uses the identical `stock_outlook` enum so the comparison engine can compute distance trivially.

---

## Input & Sources

**Raw data inputs** (from Agent Data Mapping):
- No direct API calls — consumes Pass 2 agent outputs, identical to CIO's inputs (this agent doesn't call external data providers)

**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload):
- Bull Advocate and Bear Advocate Pass 2 outputs (`confidence`, `strongest_argument`), reused via `format_advocate_for_cio()` verbatim from CIO
- Risk Advisor output (`risk_profile.risk_reward_ratio`, `downside_scenarios`, `groundedness_score`), via `format_risk_advisor_for_cio()`
- Tax Strategist output (`tax_profile.tax_efficiency_for_account`, `groundedness_score`), via `format_tax_strategist_for_cio()`
- Pass 1 agent summaries (reference-only, for fact-checking), via `format_pass1_summaries_for_cio()`
- Pre-computed `disagreement_score` and `disagreement_classification` from `AnalysisRun` — shared with primary CIO
- `cold_start_cap_active` flag — informational only; shadow does not apply the cap to its own confidence
- `{reflexion_brief}`, `{memory_brief}`, `{agent_accuracy_briefs}` — identical learning substrate to primary, for calibration parity
- Bear-bias calibration override injected directly in the system prompt (2× Bear confidence, 0.5× Bull confidence, worst-case-weighted downside scenarios) — this is the agent's key differentiator from CIO; not a data input, but a fixed prompt directive applied on top of the same inputs

---

# Input Semantics Reference

Identical to the primary CIO. Bear weighting affects how the LLM *combines* these signals, not how the fields are interpreted.

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
<td>Directional conviction UP (apply 0.5× under shadow weighting)</td>
<td>(same field)</td>
</tr>
<tr>
<td>Bear Advocate</td>
<td>`confidence` (0–100)</td>
<td>Directional conviction DOWN (apply 2× under shadow weighting)</td>
<td>(same field)</td>
</tr>
<tr>
<td>Risk Advisor</td>
<td>`groundedness_score` (0–100)</td>
<td>Data quality, NOT direction (weighting unchanged)</td>
<td>`risk_profile.risk_reward_ratio`</td>
</tr>
<tr>
<td>Tax Strategist</td>
<td>`groundedness_score` (0–100)</td>
<td>Data quality, NOT direction (weighting unchanged)</td>
<td>`tax_profile.tax_efficiency_for_account`</td>
</tr>
</table>

**Rule**: As with the primary CIO, only Bull and Bear `confidence` enter the disagreement-score calculation. Risk and Tax `groundedness_score` are NEVER averaged with advocate confidence and NEVER read as directional signal — bear bias does not change semantics.

---

# System Prompt

Invoke with `enable_thinking: false`. Stable content precedes the `--- STOCK-SPECIFIC CONTEXT ---` separator for prompt caching.

**Runtime prompt:** see [`prompts/shadow_cio/v1.0.txt`](../../prompts/shadow_cio/v1.0.txt)

---

# Conditional Injection Reference

Identical to the primary CIO. Reused verbatim — calibration parity requires the shadow to see the same framing for timeline and account as primary.

**Timeline** (inject based on `AnalysisContext.timeline`):

<table header-row="true">
<colgroup>
<col>
<col width="904">
</colgroup>
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
<colgroup>
<col>
<col width="913">
</colgroup>
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

---

# Output Schema (LLM-produced fields)

Orchestrator populates metadata and computes `primary_cio_outlook_distance` / `high_divergence` post-hoc.

```json
{
  "stock_outlook": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish",
  "confidence": 0,
  "expected_return_tier": "strong_outperform|outperform|market_perform|underperform|strong_underperform",
  "thesis_summary": "100-300 chars. Bear-weighted read. References timeline and account_type.",
  "bear_weighting_application_note": "1-2 sentences on whether and how the override changed the read vs a neutrally-weighted reading.",
  "what_would_change_my_mind": "1-2 sentences naming the specific evidence that would flip the outlook to net-bullish.",
  "parallel_predictions": {
    "1_month": { "direction": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish", "confidence": 0 },
    "3_month": { "direction": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish", "confidence": 0 }
  }
}
```

### Expected Return Tier — relative anchors

Same anchors as the primary CIO. The shadow's tier is scored separately against actual relative performance.

<table header-row="true">
<tr>
<td>Tier</td>
<td>Meaning</td>
</tr>
<tr>
<td>`strong_outperform`</td>
<td>Materially ahead of market (anchor: >15%)</td>
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
<td>Materially behind market (anchor: >15% downside)</td>
</tr>
</table>

### Fields the orchestrator populates (NOT produced by LLM)

- **Metadata**: `agent_name` (`"shadow_cio"`), `agent_pass` (`"synthesis"`), `analysis_context`, `pass2_agent_status`.
- **`disagreement_score`** and **`disagreement_classification`**: from `AnalysisRun` (pre-computed by `DisagreementCalculator`, shared with primary CIO).
- **`primary_cio_outlook_distance`**: integer 0–4. Position distance on the 5-point scale between primary `stock_outlook` and shadow `stock_outlook`. Computed at merge time after both outputs are available.
- **`high_divergence`**: boolean — `true` when `primary_cio_outlook_distance > 2`.
- **NO ****`confidence_uncapped`**** field**: shadow `confidence` is already uncapped by design.
- **NO ****`stop_loss_suggestion`****, ****`position_sizing_recommendation`**: not produced; not needed for calibration.

---

# Orchestrator Input Assembly

The shadow re-uses the primary CIO's formatters verbatim — `format_advocate_for_cio()`, `format_risk_advisor_for_cio()`, `format_tax_strategist_for_cio()`, `format_pass1_summaries_for_cio()`. Calibration parity requires identical input formatting. Only the system prompt and the output schema differ.

```python
def build_shadow_cio_input(
    pass2_outputs: dict[str, AgentOutput],
    pass1_outputs: list[AgentOutput],
    context: AnalysisContext,
    run: AnalysisRun,
    cold_start_cap_active: bool,
) -> str:
    # Header notes the cap is informational only — shadow does not apply it.
    header = (
        f"{context.canonical_ticker} ({context.company_name}) | {context.sector} | "
        f"Timeline: {context.timeline} | Account: {context.account_type}\n"
        f"Disagreement score: {run.disagreement_score} | Classification: {run.disagreement_class}\n"
        f"Primary CIO cold-start cap active (informational only — do NOT apply to shadow): "
        f"{cold_start_cap_active}\n"
    )
    bull_block  = format_advocate_for_cio(pass2_outputs.get("bull_case_advocate"), "BULL")
    bear_block  = format_advocate_for_cio(pass2_outputs.get("bear_case_advocate"), "BEAR")
    risk_block  = format_risk_advisor_for_cio(pass2_outputs.get("risk_advisor"))
    tax_block   = format_tax_strategist_for_cio(pass2_outputs.get("tax_strategist"))
    pass1_block = format_pass1_summaries_for_cio(pass1_outputs)
    return "\n\n".join([header, bull_block, bear_block, risk_block, tax_block, pass1_block])
```

---

# Orchestrator Merge Logic

```python
def merge_output_shadow_cio(
    llm_response: dict,
    pass2_outputs: dict[str, AgentOutput],
    context: AnalysisContext,
    run: AnalysisRun,
    primary_cio_outlook: str | None = None,
) -> dict:
    # Strip any forbidden directional fields the LLM may hallucinate
    for forbidden in ("recommendation", "buy_sell_hold", "action"):
        llm_response.pop(forbidden, None)

    outlook_distance = None
    high_divergence  = False
    if primary_cio_outlook is not None:
        outlook_distance = _outlook_distance(primary_cio_outlook, llm_response["stock_outlook"])
        high_divergence  = outlook_distance > 2

    return {
        "agent_name":                       "shadow_cio",
        "agent_pass":                       "synthesis",
        "analysis_context": {
            "account_type":                 context.account_type,
            "timeline":                     context.timeline,
            "stock_id":                     str(context.stock_id),
            "canonical_ticker":             context.canonical_ticker,
        },
        "stock_outlook":                    llm_response["stock_outlook"],
        "confidence":                       llm_response["confidence"],          # NOT capped
        "expected_return_tier":             llm_response["expected_return_tier"],
        "thesis_summary":                   llm_response["thesis_summary"],
        "bear_weighting_application_note":  llm_response["bear_weighting_application_note"],
        "what_would_change_my_mind":        llm_response["what_would_change_my_mind"],
        "parallel_predictions":             llm_response["parallel_predictions"],
        "disagreement_score":               run.disagreement_score,
        "disagreement_classification":      run.disagreement_class,
        "primary_cio_outlook_distance":     outlook_distance,
        "high_divergence":                  high_divergence,
        "pass2_agent_status": {
            name: output.status for name, output in pass2_outputs.items()
        },
    }


_OUTLOOK_ORDER = {
    "bullish": 0, "somewhat_bullish": 1, "neutral": 2,
    "somewhat_bearish": 3, "bearish": 4,
}

def _outlook_distance(primary: str, shadow: str) -> int:
    """Position distance on the 5-point scale (0–4)."""
    return abs(_OUTLOOK_ORDER[primary] - _OUTLOOK_ORDER[shadow])
```

**Scheduling**: primary and shadow are launched in parallel by the orchestrator after Pass 2 completes. The merge step waits for primary CIO completion to compute `primary_cio_outlook_distance`; if shadow finishes first, its raw response is buffered. If shadow fails or times out, primary is still delivered to the user — the merge writes `shadow_predictions` with `status="failed"` and the reason, and the run continues.

---

# Validation Rules

Enforced by `src/agents/validators.py::validate_shadow_cio()`. Failure triggers retry with error list appended.

1. **JSON parseable** (strip markdown fences if present).
2. **Required fields**: all schema fields above must be present.
3. **No forbidden fields**: `recommendation`, `buy_sell_hold`, `action` stripped silently; logs prompt-regression warning.
4. **`stock_outlook`**: one of the 5 enum values.
5. **`confidence`**: integer 0–100.
6. **`expected_return_tier`**: one of the 5 enum values.
7. **`thesis_summary`**: 100–300 chars.
8. **`bear_weighting_application_note`**: 30–300 chars. Must be non-trivial — empty string or single-token responses fail.
9. **`what_would_change_my_mind`**: 30–300 chars.
10. **Disagreement-confidence coupling (hard)**: `split_decision` AND `confidence > 65` → retry. `high_conflict` AND `stock_outlook ∉ {neutral, somewhat_bullish, somewhat_bearish}` → retry.
11. **Outlook/confidence consistency**: `stock_outlook == "neutral"` AND `confidence > 70` → retry. `stock_outlook ∈ {bullish, bearish}` AND `confidence < 50` → retry.
12. **Data-quality gate**: if both advocates have `confidence < 40` AND both Risk and Tax have `groundedness_score < 40`, shadow `confidence` MUST NOT exceed 45 → retry.
13. **Parallel-prediction enum check**: `1_month.direction` and `3_month.direction` ∈ outlook enum; confidences 0–100.
14. **Missing advocate handling**: Bull absent OR Bear absent → `confidence` capped at 50 by validator. Both absent → `stock_outlook: neutral`, `confidence: 0`, run marked degraded (still written to `shadow_predictions` for completeness).
15. **No cold-start cap check**: validator explicitly does NOT cap shadow confidence at 65. If a cap is observed in output and primary's cold-start status was `true`, log a `shadow_self_capped` warning but accept (the LLM may have mirrored primary's tone — undesirable but not a hard error in v1.0).
16. **Numeric fabrication guard**: numeric tokens in `thesis_summary` must appear in the Pass 2 input payload. Missing → warning + single retry.

---

# Retry Prompt Injection

```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Reminders:
- stock_outlook ∈ {bullish, somewhat_bullish, neutral, somewhat_bearish, bearish} — NOT buy/sell/hold
- split_decision → confidence ≤ 65 (hard cap)
- high_conflict → stock_outlook ∈ {neutral, somewhat_bullish, somewhat_bearish}
- thesis_summary 100–300 chars
- bear_weighting_application_note: state explicitly whether the override changed the read
- DO NOT apply the cold-start cap (that's the primary CIO's job)
- DO NOT produce a "recommendation" field
```

---

# Scoring & Feedback Loop

The shadow's value is realized at prediction-resolution time, when both primary and shadow predictions are scored against actual market outcomes. Implemented by `src/feedback/shadow_comparison.py` and surfaced into the primary CIO's `{agent_accuracy_briefs}` injection.

**Stored per ****`shadow_predictions`**** row at resolution time**:
- `actual_return_pct` at the relevant checkpoint
- `primary_outcome`: `correct` / `partially_correct` / `incorrect` (against primary CIO's call)
- `shadow_outcome`: same, against shadow's call
- `divergence_winner`: `primary` / `shadow` / `tie` / `na` (na when same outlook)
- `market_regime` at resolution time: `bull` / `bear` / `sideways` / `volatile` (from the cross-system regime tagger)

**Aggregate metrics computed nightly** (rolling 30/90-day windows once N supports it):
- Primary vs shadow win rate (overall, per sector, per timeline, per regime)
- Average `confidence` divergence between primary and shadow
- High-divergence-flag accuracy: hit rate of primary's call when `high_divergence == true`
- Bear-weighting effectiveness: does the shadow add signal when primary is bullish-dominant? (compare shadow-correct cases to a sector-baseline "always bear" strawman)

**Feedback into the primary CIO**:

When N ≥ 20 scored shadow predictions exist with sufficient regime coverage, the Feedback Analyst injects findings into the primary CIO's `{agent_accuracy_briefs}`:

```plain text
SHADOW-CIO CALIBRATION ({N} scored shadow predictions):
  Primary win rate: {X}% | Shadow win rate: {Y}% | Tie: {Z}%
  Bias indicator: {primary_too_bullish | primary_too_bearish | well_calibrated}
  Notable patterns:
    - {e.g., "Shadow outperforms in tech sector — apply additional skepticism to bull arguments on tech names"}
    - {e.g., "Shadow dominates in bear regimes — discount bull confidence by 5–10pts when regime is bear or volatile"}
  High-divergence flag accuracy: {X}% (when primary and shadow disagree by >2 positions, primary correct {X}% of the time vs {Y}% baseline)
```

**Critical**: this brief is not labeled "from the Shadow CIO" to the primary — the primary CIO is told "shadow vs primary win rate suggests you are systematically too bullish on tech", framed as calibration evidence, not as a competing opinion to defer to.

---

# Minimum Data Threshold

Same Gate-2 logic as primary CIO. Bull and Bear are both mandatory; Risk/Tax failures are warnings.

<table header-row="true">
<colgroup>
<col>
<col width="434">
</colgroup>
<tr>
<td>Scenario</td>
<td>Shadow CIO behavior</td>
</tr>
<tr>
<td>Both advocates available</td>
<td>Full bear-weighted synthesis</td>
</tr>
<tr>
<td>Bull only</td>
<td>Proceed; `confidence ≤ 50`; `bear_weighting_application_note` flags missing bear input as a degraded calibration</td>
</tr>
<tr>
<td>Bear only</td>
<td>Proceed; `confidence ≤ 50`; note one-sided input</td>
</tr>
<tr>
<td>Both advocates absent</td>
<td>`stock_outlook: neutral`, `confidence: 0`, written to `shadow_predictions` with `status="degraded"`</td>
</tr>
<tr>
<td>Risk Advisor absent</td>
<td>Proceed without scenario-weighting input; note in `bear_weighting_application_note`</td>
</tr>
<tr>
<td>Tax Strategist absent</td>
<td>Proceed; tax was never directional for shadow anyway</td>
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
<td>`AnalysisContext`  • `DataBundle.company_info`</td>
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
<td>Same content as primary CIO</td>
</tr>
<tr>
<td>`{disagreement_score}`, `{disagreement_classification}`</td>
<td>`AnalysisRun`</td>
<td>Pre-computed; shared with primary</td>
</tr>
<tr>
<td>`{cold_start_cap_active}`</td>
<td>Orchestrator</td>
<td>**Informational only for shadow** — do not apply</td>
</tr>
<tr>
<td>`{reflexion_brief}`</td>
<td>`StockAnalysisMemory`  • Feedback Analyst</td>
<td>Same as primary; shadow uses identical learning substrate</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory`</td>
<td>Same as primary</td>
</tr>
<tr>
<td>`{agent_accuracy_briefs}`</td>
<td>Feedback engine</td>
<td>Same as primary. Includes shadow-vs-primary findings when N ≥ 20.</td>
</tr>
<tr>
<td>`{pass1_summaries}`</td>
<td>`format_pass1_summaries_for_cio()`</td>
<td>Reused verbatim from primary</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template (shadow-specific subset)</td>
<td>Different from primary — shorter</td>
</tr>
<tr>
<td>Pass 2 blocks</td>
<td>`format_*_for_cio()` per agent</td>
<td>Reused verbatim from primary</td>
</tr>
</table>

---

# Testing Checklist

- [ ] Valid JSON from `qwen3.6:35b-a3b` with `enable_thinking: false`, and no leaked `<think>` reasoning tokens?
- [ ] Same valid JSON from Haiku 4.5 fallback?
- [ ] No `recommendation` / `buy_sell_hold` / `action` field?
- [ ] `thesis_summary` 100–300 chars?
- [ ] `bear_weighting_application_note` populated and explicitly states whether override changed the read?
- [ ] `what_would_change_my_mind` concrete and bear-flip-oriented (names what would make shadow more bullish)?
- [ ] **Bear bias test (clearly bullish stock)**: shadow lands one notch less bullish than primary (e.g., primary `bullish` → shadow `somewhat_bullish`)?
- [ ] **Bear bias test (clearly bearish stock)**: shadow agrees with primary (override is consistent with evidence)?
- [ ] **Bear bias test (genuinely ambiguous stock)**: shadow lands `somewhat_bearish` or `neutral` where primary lands `somewhat_bullish` or `neutral`?
- [ ] **Disagreement test (split)**: `confidence ≤ 65`?
- [ ] **Disagreement test (high)**: `stock_outlook` in neutral range?
- [ ] **No cold-start cap**: when `cold_start_cap_active=true`, shadow `confidence` can still exceed 65?
- [ ] **Data-quality gate**: low-groundedness all-around → `confidence ≤ 45`?
- [ ] **High-divergence flag**: orchestrator merge sets `high_divergence: true` when primary/shadow outlook distance > 2?
- [ ] **Missing advocate**: Bear absent → `confidence ≤ 50`, note in application note?
- [ ] **Parallel predictions**: populated; consistent with primary outlook direction-of-bias unless explicit reason?
- [ ] **Numeric fabrication guard**: all numbers in thesis trace to Pass 2 payload?
- [ ] **Non-blocking**: shadow timeout/failure does NOT block primary CIO output to user?
- [ ] **Storage**: shadow output written to `shadow_predictions`, not `cio_outputs`?
- [ ] **Fixture stocks** (run alongside primary CIO fixtures): (a) bull consensus → shadow one notch less bullish at lower confidence; (b) bear consensus → shadow agrees with primary; (c) quality-at-rich-valuation → shadow likely `somewhat_bearish` where primary is `somewhat_bullish`; (d) high conflict → shadow lands in neutral range with bear lean.

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
<td>~1,200 tokens</td>
<td>Shorter than primary (no detailed advocate-assessment guidance, no narrative spec)</td>
</tr>
<tr>
<td>Pass 2 outputs</td>
<td>~6,500–7,500 tokens</td>
<td>Reuses primary CIO's formatters verbatim</td>
</tr>
<tr>
<td>Pass 1 summaries</td>
<td>~500 tokens</td>
<td>~100 per agent</td>
</tr>
<tr>
<td>Context + briefs</td>
<td>~500–1,000 tokens</td>
<td>Same as primary</td>
</tr>
<tr>
<td>Expected output</td>
<td>~1,000 tokens</td>
<td>Strict subset of primary's output</td>
</tr>
<tr>
<td>**Total**</td>
<td>**~10,200 tokens**</td>
<td>Well within `qwen3.6:35b-a3b` and Haiku 4.5 context windows</td>
</tr>
</table>

**Model configuration**:
- Primary: `qwen3.6:35b-a3b` via Ollama (local). Fallback on Ollama outage: `claude-haiku-4-5-20251001`.
- `max_tokens`: 1,200
- `temperature`: 0.3 (same as primary CIO — bear bias comes from the prompt, not from increased temperature)
- Thinking mode: OFF (verify suppression is clean on this model — see Agent Design Patterns Pattern 10 caveat)
- Prompt caching: stable system prefix cached per deploy (Haiku side; local side caching depends on Ollama version)
- Expected latency: ~60s local (unverified estimate — `qwen3.6:35b-a3b` on RTX 4070 Ti Super at ~10K context, re-measure after deployment), ~3–5s Haiku fallback
- Cost per stock: ~$0 local, ~$0.003–0.008 Haiku fallback (per current Anthropic pricing — re-confirm before launch)
- Steady-state weekly cost at 50-stock batch with Haiku fallback only: ~$0.20–0.40 (assuming local is primary path and fallback is rare)

---

# Implementation Files

- `src/agents/shadow_cio.py` — Shadow CIO agent implementation
- `src/agents/cio_formatter.py` — `format_advocate_for_cio()`, `format_risk_advisor_for_cio()`, `format_tax_strategist_for_cio()`, `format_pass1_summaries_for_cio()` reused verbatim from primary CIO
- `src/agents/validators.py` — `validate_shadow_cio()` (subset of CIO validators)
- `src/orchestrator/engine.py` — `run_shadow_cio()` launched in parallel with primary CIO finalization; `merge_output_shadow_cio()`
- `src/feedback/shadow_comparison.py` — primary-vs-shadow aggregate scoring, surfaces findings into `{agent_accuracy_briefs}`
- `src/models/shadow_predictions.py` — ORM model for `shadow_predictions` table

---

# Pending Propagations

1. **`shadow_predictions`**** table schema**: define ORM model alongside this prompt — mirrors `cio_outputs` shape but with the schema-subset fields above plus `primary_cio_outlook_distance`, `high_divergence`, `status` (`completed` | `degraded` | `failed`), and resolution-time fields (`actual_return_pct`, `primary_outcome`, `shadow_outcome`, `divergence_winner`, `market_regime`).
2. **Feedback Analyst — shadow scoring rules**: directional accuracy of shadow vs primary, divergence-winner attribution, regime-segmented win rates, bias-indicator computation (when does shadow systematically outperform primary?).
3. **Sonnet vs Opus vs Haiku A/B for shadow**: CIO doc notes the shadow is a natural place to run a different model class as the contrarian variant. v1.0 spec is `qwen3.6:35b-a3b` + Haiku fallback for cost; an A/B against Sonnet 4.6 (~10–20 fixture stocks) at month 3–4 will tell us whether the cheaper model produces enough calibration signal or whether the cost upgrade is justified.
4. **Shadow regime-stratification**: once N ≥ 50 scored shadow predictions, segment shadow win rate by regime (bull/bear/sideways/volatile). Use this to drive regime-conditional bull/bear weighting in the *primary* CIO's accuracy brief.
5. **High-divergence-flag downstream effects**: define how the Portfolio Optimizer (Milestone 2+) should treat positions where the primary's recent analysis was a high-divergence run. Candidate rule: reduce position-size confidence by 1 notch.

---

# Open Questions

1. **Information substrate parity vs fixed-baseline shadow**: v1.0 ships with the shadow receiving the same `{reflexion_brief}` / `{memory_brief}` / `{agent_accuracy_briefs}` as the primary, so both evolve as the system learns. The alternative — a fixed-baseline shadow that sees only static inputs — would provide a cleaner long-term bias signal but a worse short-term decision quality. **Decision**: ship v1.0 with parity. Revisit at month 6 if the shadow's win rate is drifting in a way that makes the calibration signal hard to interpret.
2. **Shadow self-capping observation**: validation rule 15 logs (but does not retry) when the shadow appears to cap itself at 65 during the cold-start period. Watch the rate of this in early runs — if it's frequent (>15% of runs), the prompt may need to be more emphatic about the no-cold-start-cap rule, or the temperature may need to be raised slightly to encourage the model to commit.
3. **Should the shadow produce ****`expected_return_tier`**** before primary does, or after?** Currently they're independent. If the Phase 1.5 `expected_return_tier` grounding (CIO Open Question #1) lands, the shadow should consume the same FUND base-case target — applying its bear weighting to the *qualitative adjustment* layer, not to the anchored number itself. Re-spec when Phase 1.5 is scoped.
4. **Shadow CIO confidence calibration vs primary**: Brier-score comparison between primary and shadow is the cleanest way to measure who is better calibrated. Add to Feedback Analyst scope once the prediction-resolution pipeline is live.
5. **Failure-mode mirroring**: If `qwen3.6:35b-a3b` and Sonnet 4.6 (primary CIO) systematically fail in the same ways on the same inputs (e.g., both miss the same kind of bear catalyst), the shadow's independence is illusory. Run a periodic "shadow vs primary residual correlation" check — if correlation is high, the calibration is degraded and we should escalate the shadow model class.

---

_Source: Notion — Project Stock Picker → Agent Prompts → Shadow CIO (calibration agent) Agent Prompt. Downloaded 2026-07-02._
