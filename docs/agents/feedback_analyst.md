# Feedback Analyst (meta-agent) Agent Prompt

**Agent**: Feedback Analyst (meta-agent)<br>**Status**: Draft v1.1<br>**Phase**: Phase 3 (core learning system)<br>**Pass**: Post-scoring (runs after a prediction reaches a scoring checkpoint — not part of the analysis pipeline)<br>**Role**: Diagnose why a scored prediction was right or wrong. Attribute error to specific agent(s), classify the error, produce per-agent lessons, and surface candidate cross-prediction patterns. Output is consumed by the paired Judge, the `diagnostic_reports` table, and downstream by the pattern-detection engine and learning journal.<br>**Target Model**: **`claude-opus-4-6`** (primary). `claude-sonnet-4-6` permitted as cost-saving fallback for sample sizes under 50. Thinking mode ON for diagnosis (reasoning quality dominates; latency is offline-batch and uncritical). The paired Judge runs on `claude-sonnet-4-6` with thinking OFF.<br>**Audience**: Internal — the Judge agent (first), then the pattern-detection engine and learning journal. Not user-facing. Tone is technical and evidence-grounded; no retail-investor framing.<br>**Token Budget**: ~3,500 system prompt | ~10,000–14,000 compressed analysis context + outcome + siblings | ~2,000 expected output | ~16,500–19,500 total

---

# Design Decisions & Rationale

The Feedback Analyst is the meta-agent that turns scored predictions into learnable signal. It is not part of the analysis pipeline. It runs after a prediction is scored, ingests the original analysis context plus the actual outcome, and emits a structured diagnosis. Its quality is the single largest determinant of learning-system signal-to-noise.

1. **Attribution, not adjudication.** The Feedback Analyst is *retrospective*. It does not re-run analysis or argue with the original agents. It reads what each agent said, observes what happened, and assigns responsibility for the gap.
2. **Specific over general.** Every diagnosis must name a *specific agent* in `primary_error_source` and a *specific error type* in `primary_error_type`. Generic phrasing ("the system missed", "the analysis was off") degrades pattern aggregation downstream. The pattern detector clusters on these fields.
3. **Evidence-grounded, not plausibility-grounded.** Every claim about what an agent said must be traceable to that agent's original output. The Judge is adversarial and cross-references agent outputs verbatim; post-hoc rationalization that sounds plausible but doesn't connect cause to outcome is the primary failure mode.
4. **Counterfactuals must be actionable.** `what_would_have_changed_the_outcome` is the seed of a future prompt modification. It must name (a) the specific reasoning change, (b) the agent(s) that change applies to, and (c) why the changed reasoning would have flipped or recalibrated the recommendation. "The system should have been more cautious" fails this test.
5. **Pattern flags are hypotheses, not claims.** A `pattern_flags` entry asserts "this case extends a recurring pattern." It must reference a count ("3rd time…", "7 of last 10…"). The pattern-detection engine is the authority that confirms or rejects the hypothesis with sample-size statistics. The Feedback Analyst nominates; it does not declare.
6. **Bootstrapping tiers gate confidence.** Sample size controls what the agent is allowed to claim. Under 5 scored predictions, no diagnosis runs at all. 5–20: directional language only ("appears to overestimate") — no numeric magnitudes. 20–50: full diagnosis allowed, but learning journal entries are flagged "preliminary." 50+: full operation.
7. **Total-return discipline.** `actual_return_pct` and benchmark return inputs are *total return* (price + cumulative dividends since the recommendation date). The Feedback Analyst refuses (Standing Rule 9) when the orchestrator did not flag inputs as total-return — price-only data systematically biases the diagnosis against income stocks.
8. **Paired Judge is mandatory.** Every diagnosis is reviewed by a separate Sonnet call with adversarial prompting. Diagnoses scoring `diagnosis_quality_score < 50` are flagged "low quality" and excluded from pattern detection. The Judge is described in `## Judge Sub-Agent` below.
9. **Sibling predictions enable timing-error classification.** Predictions sharing an `analysis_run_id` (the primary 6-month plus parallel 1m/3m calls) are available together. A "right direction, wrong horizon" outcome is only classifiable when siblings are scored and visible. Always check siblings before defaulting to `direction_wrong`; when siblings are unscored, classify on available data and let the pattern detector catch the horizon mismatch once both siblings score.
10. **Self-attribution allowed and required.** The CIO is a diagnosable agent. When the underlying agent outputs were defensible but the CIO synthesized them poorly, `primary_error_source` is `"cio"` and `primary_error_type` is a synthesis-bias term (e.g., `underweighted_dissent`, `overweighted_consensus`, `archetype_blindness`). Avoiding CIO self-attribution biases the learning system toward modifying upstream agents that were not actually at fault.
11. **Shadow CIO is read-only.** Shadow CIO's calibration output is *visible* to the Feedback Analyst (it informs cross-checks against the primary CIO's direction and confidence) but is *not diagnosable* — it does not drive user-facing recommendations, so it cannot appear as `primary_error_source` or `secondary_error_source`. This split is enforced by the validator's `DIAGNOSABLE_AGENTS` constant.
12. **The Feedback Analyst does not diagnose itself.** Its outputs are reviewed by the Judge. Meta-meta-recursion is out of scope; the Judge's `diagnosis_quality_score` is the only feedback signal on diagnosis quality.
13. **No re-prediction.** The Analyst must not produce a "what the recommendation should have been" beyond the structured `what_would_have_changed_the_outcome` field. Predictions about counterfactual outcomes invite hindsight bias and are not actionable to the learning system.

---

# Diagnostic Taxonomy Reference

The Feedback Analyst writes into a closed enum for `error_classification` and a controlled-but-extensible vocabulary for `primary_error_type` and `pattern_type` (the latter shapes downstream journal entries).

### Closed Agent Set

Two related sets — referenced throughout the prompt, validators, and orchestrator code:

- **`DIAGNOSABLE_AGENTS`** (10 — these are the agents that can appear as `primary_error_source` / `secondary_error_source` / `agent_specific_feedback[].agent`):<br>`stock_researcher`, `fundamental_analyst`, `technical_analyst`, `sentiment_analyst`, `macro_economist`, `bull_case_advocate`, `bear_case_advocate`, `risk_advisor`, `tax_strategist`, `cio`.
- **`VISIBLE_AGENT_ORDER`** (11 — agents whose outputs the Feedback Analyst reads): `DIAGNOSABLE_AGENTS` + `shadow_cio` (calibration cross-check; read-only — not a diagnosable target).

The Feedback Analyst itself is not in either set (Design Decision 12).

### `error_classification` (closed set)

<table header-row="true">
<colgroup>
<col>
<col width="519">
</colgroup>
<tr>
<td>Code</td>
<td>Meaning</td>
</tr>
<tr>
<td>`correct`</td>
<td>Actual return falls inside `predicted_return_range` (25th–75th percentile band).</td>
</tr>
<tr>
<td>`direction_wrong`</td>
<td>Predicted up, went down — or vice versa. Sibling check required before classifying (see `timing_error`).</td>
</tr>
<tr>
<td>`magnitude_overestimate`</td>
<td>Direction correct; actual return materially below prediction; actual outside predicted range on the low side.</td>
</tr>
<tr>
<td>`magnitude_underestimate`</td>
<td>Direction correct; actual return materially exceeded prediction; actual outside predicted range on the high side.</td>
</tr>
<tr>
<td>`timing_error`</td>
<td>Sibling predictions disagree on outcome — e.g., 1-month miss + 6-month hit, or vice versa. The direction is "right eventually" but wrong for the analyzed horizon. Only classifiable when siblings are scored and visible.</td>
</tr>
</table>

### `primary_error_type` (controlled vocabulary — extend with care; new types degrade pattern aggregation until reused)

Per-agent suggested types (the Analyst may use these or coin a new term in `snake_case` if none fit; coining a new term should be rare):

- **`fundamental_analyst`**: `overestimated_growth`, `underestimated_growth`, `missed_margin_compression`, `overweighted_recent_quarter`, `peer_deceleration_blind_spot`, `valuation_anchor_drift`.
- **`technical_analyst`**: `false_breakout_signal`, `missed_divergence`, `support_overconfidence`, `momentum_overweight`, `timeframe_mismatch`.
- **`macro_economist`**: `missed_headwind`, `missed_tailwind`, `rate_sensitivity_understated`, `sector_cycle_misread`, `regime_shift_blind_spot`.
- **`sentiment_analyst`**: `sentiment_alpha_decay`, `news_overweight`, `contrarian_signal_missed`, `social_noise_treated_as_signal`.
- **`bull_advocate`** / **`bear_advocate`**: `overconfident`, `underconfident`, `archetype_misfit`, `catalyst_overweight`, `risk_underweight`.
- **`risk_advisor`**: `scenario_probability_miscalibrated`, `tail_underestimated`, `correlation_breakdown_missed`, `volatility_assessment_stale`.
- **`tax_strategist`**: `wht_underestimated`, `account_fit_misread`, `cross_account_recommendation_missed`, `drag_underestimated`.
- **`cio`** (synthesis errors): `underweighted_dissent`, `overweighted_consensus`, `archetype_blindness`, `disagreement_score_ignored`, `accuracy_brief_ignored`, `reflexion_lesson_ignored`, `confidence_miscalibrated`.

### `pattern_type` (downstream — these are the buckets the learning journal organizes around; flagging a pattern should map to one)

<table header-row="true">
<colgroup>
<col>
<col width="483">
</colgroup>
<tr>
<td>Code</td>
<td>Meaning</td>
</tr>
<tr>
<td>`agent_bias`</td>
<td>Agent-specific systematic error (e.g., "fundamental overestimates growth in tech during rate hikes").</td>
</tr>
<tr>
<td>`cio_synthesis_bias`</td>
<td>CIO weighting error pattern (e.g., "CIO underweights bear advocate when bull confidence > 70").</td>
</tr>
<tr>
<td>`sector_blind_spot`</td>
<td>Sector-specific gap (e.g., "energy stocks during commodity inflections").</td>
</tr>
<tr>
<td>`confidence_miscalibration`</td>
<td>Confidence bands not matching realized accuracy (e.g., "70–80 band correct 52% of the time").</td>
</tr>
<tr>
<td>`return_estimation_bias`</td>
<td>Systematic optimism/pessimism in return magnitudes.</td>
</tr>
<tr>
<td>`timing_error`</td>
<td>Right direction, wrong horizon — pattern across timeline tiers.</td>
</tr>
<tr>
<td>`environmental_blind_spot`</td>
<td>Missed environmental factor (e.g., high-VIX periods, rapid rate changes, earnings season).</td>
</tr>
</table>

### `accuracy_for_this_prediction` (per-agent verdict — closed set)

`good` / `mixed` / `poor`. Used both for negative-pattern aggregation (poor + reasons → prompt modifications) and for *winning-pattern aggregation* (good ratings feed positive-reinforcement briefs per Part 5 §3 of the learning system spec).

### `prediction_accuracy_rating` (closed set)

`accurate` / `mostly_accurate` / `moderately_inaccurate` / `very_inaccurate`. This is independent of `error_classification` — `direction_wrong` with a small absolute miss is `moderately_inaccurate`; `direction_wrong` with a large absolute miss is `very_inaccurate`.

---

# System Prompt

Invoke with `enable_thinking: true`. Stable content precedes the `--- PREDICTION-SPECIFIC CONTEXT ---` separator for prompt caching.

**Runtime prompt:** see [`prompts/feedback_analyst/v1.1.txt`](../../prompts/feedback_analyst/v1.1.txt)

---

# Conditional Injection Reference

### Bootstrapping tier (inject based on `scored_predictions_count`):

<table header-row="true">
<colgroup>
<col>
<col>
<col width="401">
</colgroup>
<tr>
<td>Count</td>
<td>`{bootstrapping_tier}`</td>
<td>Behavior reminder appended</td>
</tr>
<tr>
<td><5</td>
<td>`pre_bootstrap`</td>
<td>`You should not have been invoked at this sample size. Emit refusal and stop.`</td>
</tr>
<tr>
<td>5–20</td>
<td>`early`</td>
<td>`Use directional language only. No numeric magnitudes. No pattern flags. Lessons qualitative.`</td>
</tr>
<tr>
<td>20–50</td>
<td>`developing`</td>
<td>`Full diagnosis allowed. Pattern flags allowed. Numeric lessons must be flagged 'preliminary'.`</td>
</tr>
<tr>
<td>50+</td>
<td>`mature`</td>
<td>`Full operation. No restrictions.`</td>
</tr>
</table>

### Checkpoint type (inject based on `checkpoint_label`):

<table header-row="true">
<colgroup>
<col>
<col width="432">
</colgroup>
<tr>
<td>Checkpoint</td>
<td>Behavior</td>
</tr>
<tr>
<td>`on_track_check`</td>
<td>Should NOT have invoked Feedback Analyst. Refuse: `{"refused": true, "reason": "on_track_check_not_a_scoring_checkpoint"}`.</td>
</tr>
<tr>
<td>`1_week` / `2_weeks` / `4_weeks` (short-term)</td>
<td>Full diagnosis. Intermediate-checkpoint history typically empty. Siblings = N/A (no parallels for short-term).</td>
</tr>
<tr>
<td>`3_months` / `6_months` / `9_months` / `12_months` (medium-term primary)</td>
<td>Full diagnosis. Sibling 1m and 3m parallel predictions should be visible.</td>
</tr>
<tr>
<td>`1m_parallel` / `3m_parallel`</td>
<td>Full diagnosis. This is a parallel prediction (`analysis_timeline: short_term_parallel`). Siblings include the primary 6-month call. Cross-horizon sibling agreement → `timing_error` candidate (only if siblings are scored — otherwise classify on available data).</td>
</tr>
<tr>
<td>`1_year` and later (long-term)</td>
<td>Full diagnosis. Multi-year context: weight macro and structural factors more heavily in attribution.</td>
</tr>
</table>

---

# Output Schema (LLM-produced fields)

The orchestrator populates metadata (`diagnostic_id`, `prediction_id`, `created_at`), the Judge populates `diagnosis_quality_score` and `judge_feedback` after review.

```json
{
  "refused": false,
  "refused_reason": null,
  "prediction_accuracy_rating": "accurate|mostly_accurate|moderately_inaccurate|very_inaccurate",
  "error_classification": "correct|direction_wrong|magnitude_overestimate|magnitude_underestimate|timing_error",
  "error_magnitude": "predicted +X%, actual +/-Y% (Z pp error, benchmark +W%)",
  "sibling_check": {
    "siblings_visible": true,
    "sibling_directions": {
      "1_month": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish|null",
      "3_month": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish|null",
      "6_month": "bullish|somewhat_bullish|neutral|somewhat_bearish|bearish|null"
    },
    "sibling_outcomes_at_checkpoint": "1 sentence describing whether siblings have been scored and what they showed — empty string if siblings not visible. If siblings exist but are unscored, state 'siblings exist but unscored; not used in classification'.",
    "timing_error_candidate": false
  },
  "root_cause_analysis": {
    "primary_error_source": "agent_name (must be in DIAGNOSABLE_AGENTS — 10 agents, Shadow CIO excluded)",
    "primary_error_type": "snake_case taxonomy term",
    "primary_error_detail": "3-6 sentences. What the agent said (with cited numerics or short phrases), what subsequently happened, how those connect. Every claim traceable to the original agent output or outcome data.",
    "secondary_error_source": "agent_name (DIAGNOSABLE_AGENTS) | null",
    "secondary_error_type": "snake_case taxonomy term | null",
    "secondary_error_detail": "1-3 sentences | null",
    "contributing_factors": [
      "1-4 bullets. Each describes a signal another agent got partially right but was underweighted, or a signal present in inputs but the originating agent missed. Checkable against original outputs."
    ],
    "what_would_have_changed_the_outcome": "2-4 sentences. Names (a) the agent(s) the change applies to, (b) the specific reasoning modification, (c) the realistic downstream effect on outlook / confidence / tier. Grounded in data available on prediction date — no hindsight."
  },
  "agent_specific_feedback": [
    {
      "agent": "agent_name (must be in DIAGNOSABLE_AGENTS)",
      "accuracy_for_this_prediction": "good|mixed|poor",
      "specific_lesson": "1-2 sentences. Agent-targeted, actionable. 'Apply a 20-30% growth-deceleration haircut...' not 'Be more careful.' In `early` bootstrapping tier: qualitative only, no numeric magnitudes."
    }
  ],
  "pattern_flags": [
    "0-3 entries. Each must reference a count and be falsifiable. Empty array if no pattern apparent or if bootstrapping tier is `early`. E.g., '3rd time fundamental_analyst overestimated growth for tech stocks during a rate-hiking cycle'."
  ]
}
```

### Refusal cases (terminate Stage 1, return only `refused: true`)

<table header-row="true">
<tr>
<td>`refused_reason`</td>
<td>Trigger</td>
</tr>
<tr>
<td>`below_minimum_sample_size`</td>
<td>`bootstrapping_tier == pre_bootstrap`</td>
</tr>
<tr>
<td>`on_track_check_not_a_scoring_checkpoint`</td>
<td>`checkpoint_label` is an on-track check, not a scoring checkpoint</td>
</tr>
<tr>
<td>`input_data_not_total_return`</td>
<td>`inputs_are_total_return == false`</td>
</tr>
<tr>
<td>`prediction_already_diagnosed_at_this_checkpoint`</td>
<td>Orchestrator pre-flight check (idempotency)</td>
</tr>
</table>

### Array bounds (validator-enforced)

<table header-row="true">
<tr>
<td>Field</td>
<td>Min</td>
<td>Max</td>
</tr>
<tr>
<td>`root_cause_analysis.contributing_factors`</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`agent_specific_feedback`</td>
<td>1</td>
<td>10</td>
</tr>
<tr>
<td>`pattern_flags`</td>
<td>0</td>
<td>3</td>
</tr>
</table>

### Fields the orchestrator populates (NOT produced by LLM)

- `diagnostic_id` (UUID), `prediction_id` (FK), `analysis_run_id` (FK), `checkpoint_interval` (string), `created_at` (timestamp).
- `error_classification`, `primary_error_source`, `primary_error_type` are also copied into top-level columns of `diagnostic_reports` for query performance (mirror of the JSON fields).

### Fields the Judge populates (after diagnosis is produced)

- `diagnosis_quality_score` (0–100). `< 50` → marked low quality, excluded from pattern detection.
- `judge_feedback` (nullable string). Free-form notes for human inspection.

---

# Judge Sub-Agent (paired adversarial reviewer)

Every Feedback Analyst output is reviewed by a paired Judge call. Model: `claude-sonnet-4-6`, thinking OFF, temperature 0.2, max_tokens 800.

### Judge System Prompt

```plain text
You are the Judge — an adversarial reviewer for the Feedback Analyst meta-agent. Your job is to find the weaknesses in a proposed diagnosis. The Feedback Analyst's output drives prompt modifications across the stock analysis pipeline; flawed diagnoses propagate as bad lessons. Be skeptical.

You are given:
1. The proposed diagnosis (the Feedback Analyst's JSON output).
2. The full input payload the Feedback Analyst saw (agent outputs, outcome data, events, siblings, counterfactuals).

The counterfactual records in the payload include realized returns for runner-up tickers. Use those returns ONLY for evidence-grounding cross-checks (items 1 and 2 below). Do NOT anchor on counterfactual outcomes when grading counterfactual plausibility (item 3) — that turns your review into hindsight evaluation.

You must verify four things:

1. **Logical consistency.** Does the conclusion follow from the evidence cited? Is `error_classification` consistent with the outcome data? Is `primary_error_source` actually the agent whose reasoning was most responsible, or is it a deflection? Is `primary_error_source` in the diagnosable set (10 agents, Shadow CIO excluded)?

2. **Evidence grounding.** Every claim the diagnosis makes about what an agent said must be traceable to that agent's original output. Cross-check direct quotes and numerical claims. If the diagnosis says "the bear advocate flagged revenue deceleration risk with 45% confidence" — verify both the claim and the number against the bear advocate's actual output. Reconstructed claims that the agent didn't make are the most common failure.

3. **Counterfactual plausibility.** Is `what_would_have_changed_the_outcome` realistic given data available on the prediction date? Hindsight bias — claiming the agents should have known something they couldn't have known — is disqualifying. The change must be a *reasoning* change, not a *correct answer*. Assess against data available on prediction date, not against what subsequently happened to runner-up tickers.

4. **Pattern flag validity.** Each `pattern_flags` entry must reference a count and be falsifiable. Flags that read as "the system sometimes does X" or "in some past cases…" without a count fail this check. Spurious patterns found in noise (cherry-picking two similar cases) are disqualifying. Empty `pattern_flags` is fine in `early` tier and otherwise honest when no pattern is apparent — do not penalize an empty array.

Score 0–100:
- 90–100: Solid. Evidence-grounded, specific, non-deflecting, counterfactual realistic. Patterns (if any) cite counts.
- 70–89: Mostly sound but has 1–2 minor weaknesses (slight overreach, weak grounding on one claim).
- 50–69: Mixed. Diagnosis is directionally right but evidence grounding is thin or counterfactual stretches available data. Useful but should be flagged.
- 30–49: Weak. Material evidence-grounding failures, deflection, or hindsight bias.
- 0–29: Disqualifying. Fabricated claims, post-hoc rationalization, or pattern flags with no support.

Diagnoses scoring below 50 are excluded from pattern detection downstream.

In `judge_feedback` (1–4 sentences, nullable when score ≥ 80), name the specific weakness(es) you found. Reference fields by name.

---

## OUTPUT

Produce JSON only:

{
  "diagnosis_quality_score": 0,
  "judge_feedback": "1-4 sentences naming specific weaknesses, referenced by field name. Null when score >= 80."
}

--- DIAGNOSIS TO REVIEW ---

{feedback_analyst_output}

--- INPUT PAYLOAD THE FEEDBACK ANALYST SAW ---

{feedback_analyst_input_payload}
```

---

# Orchestrator Input Assembly

Before invoking the Feedback Analyst, the orchestrator assembles the payload in this order: prediction header (inline metadata + outcome + bootstrapping tier) → siblings → checkpoint history → events → counterfactuals → original agent outputs.

```python
DIAGNOSABLE_AGENTS: list[str] = [
    "stock_researcher", "fundamental_analyst", "technical_analyst",
    "sentiment_analyst", "macro_economist",
    "bull_case_advocate", "bear_case_advocate",
    "risk_advisor", "tax_strategist",
    "cio",
]

VISIBLE_AGENT_ORDER: list[str] = DIAGNOSABLE_AGENTS + ["shadow_cio"]


def build_feedback_analyst_input(
    prediction: PredictionRecord,
    outcome: OutcomeRecord,
    siblings: list[PredictionRecord],
    checkpoint_history: list[CheckpointRecord],
    events: list[EventRecord],
    counterfactuals: list[CounterfactualRecord],
    analysis_run: AnalysisRun,
    bootstrapping_tier: str,
    scored_predictions_count: int,
) -> str:
    error_pp = outcome.actual_return_pct - prediction.target_return_pct
    header = (
        f"{analysis_run.canonical_ticker} ({analysis_run.company_name}) | "
        f"{analysis_run.sector} | Timeline: {analysis_run.timeline} | "
        f"Account: {analysis_run.account_type}\n"
        f"Prediction made: {prediction.created_at.date()} | "
        f"Scored at checkpoint: {outcome.checkpoint_label} ({outcome.days_held} days held)\n"
        f"Predicted return: {prediction.target_return_pct:+.2f}% "
        f"(range: {prediction.range_low:+.2f}-{prediction.range_high:+.2f}%, "
        f"confidence: {prediction.confidence})\n"
        f"Actual total return: {outcome.actual_return_pct:+.2f}% | "
        f"Benchmark total return: {outcome.benchmark_return_pct:+.2f}%\n"
        f"In predicted range: {outcome.in_range} | "
        f"Prediction error: {error_pp:+.2f}pp\n"
        f"CIO outlook: {prediction.stock_outlook} | "
        f"Expected return tier: {prediction.expected_return_tier}\n"
        f"Bootstrapping tier: {bootstrapping_tier} "
        f"({scored_predictions_count} scored predictions)\n"
        f"Inputs are total-return: {outcome.inputs_are_total_return}\n"
    )

    sibling_block         = format_sibling_predictions(siblings, prediction.id)
    checkpoint_block      = format_checkpoint_history(checkpoint_history)
    events_block          = format_events(events)
    counterfactuals_block = format_counterfactuals(counterfactuals)
    agent_outputs_block   = format_agent_outputs_for_diagnosis(analysis_run.agent_outputs)

    # Header carries inputs 2 / 3 / 8 (predicted record, actual outcome,
    # bootstrapping tier flag). The blocks below carry inputs 1 / 4 / 5 / 6 / 7.
    return "\n\n".join([
        header,
        f"Sibling predictions:\n{sibling_block}",
        f"Intermediate checkpoint history:\n{checkpoint_block}",
        f"Significant events during the holding period:\n{events_block}",
        f"Counterfactual records:\n{counterfactuals_block}",
        f"Original analysis context:\n{agent_outputs_block}",
    ])


def format_sibling_predictions(siblings: list[PredictionRecord], current_id: UUID) -> str:
    """Sibling 1m / 3m / 6m calls under the same analysis_run_id, excluding the one being scored."""
    if not siblings:
        return "(no siblings — single-prediction analysis)"
    lines = []
    for s in siblings:
        if s.id == current_id:
            continue
        scored = "scored" if s.scored_at else "not_yet_scored"
        outcome = (
            f"actual {s.actual_return_pct:+.2f}% over {s.holding_days}d"
            if s.scored_at else "pending"
        )
        lines.append(
            f"- {s.timeline_label}: outlook={s.stock_outlook} | "
            f"predicted={s.target_return_pct:+.2f}% | "
            f"confidence={s.confidence} | status={scored} | {outcome}"
        )
    return "\n".join(lines) if lines else "(no other siblings)"


def format_agent_outputs_for_diagnosis(agent_outputs: dict[str, AgentOutput]) -> str:
    """
    Compressed agent-output dump. Iterates VISIBLE_AGENT_ORDER (11 — includes
    Shadow CIO for read-only cross-check; Shadow CIO is not diagnosable).
    Includes the directional fields and the strongest_argument / weakest_point /
    narrative for advocate-style agents; full structured_data for risk / tax /
    synthesis agents. Target: ~10K tokens total — drop verbose 'narrative'
    fields if budget exceeded, retaining structured_data.
    """
    blocks = []
    for name in VISIBLE_AGENT_ORDER:
        out = agent_outputs.get(name)
        if out is None:
            blocks.append(f"--- {name.upper()}: NOT PRESENT ---")
            continue
        if out.status != "completed":
            blocks.append(
                f"--- {name.upper()}: FAILED (reason: {out.error_detail}) ---"
            )
            continue
        blocks.append(format_single_agent_for_diagnosis(name, out))
    return "\n\n".join(blocks)
```

---

# Orchestrator Merge Logic

```python
def merge_output_feedback_analyst(
    llm_response: dict,
    judge_response: dict,
    prediction: PredictionRecord,
    outcome: OutcomeRecord,
    analysis_run: AnalysisRun,
    bootstrapping_tier: str,
) -> dict:
    # Handle refusal early
    if llm_response.get("refused"):
        return {
            "diagnostic_id":         uuid4(),
            "prediction_id":         prediction.id,
            "analysis_run_id":       analysis_run.id,
            "checkpoint_interval":   outcome.checkpoint_label,
            "refused":               True,
            "refused_reason":        llm_response["refused_reason"],
            "diagnosis_json":        None,
            "diagnosis_quality_score": None,
            "judge_feedback":        None,
            "created_at":            datetime.utcnow(),
        }

    # Top-level mirror columns for query performance
    rca = llm_response["root_cause_analysis"]

    return {
        "diagnostic_id":           uuid4(),
        "prediction_id":           prediction.id,
        "analysis_run_id":         analysis_run.id,
        "checkpoint_interval":     outcome.checkpoint_label,
        "refused":                 False,
        "refused_reason":          None,
        "error_classification":    llm_response["error_classification"],
        "primary_error_source":    rca["primary_error_source"],
        "primary_error_type":      rca["primary_error_type"],
        "secondary_error_source":  rca.get("secondary_error_source"),
        "secondary_error_type":    rca.get("secondary_error_type"),
        "diagnosis_json":          llm_response,
        "diagnosis_quality_score": judge_response["diagnosis_quality_score"],
        "judge_feedback":          judge_response.get("judge_feedback"),
        "bootstrapping_tier_at_creation": bootstrapping_tier,
        "preliminary_flag":        bootstrapping_tier in ("early", "developing"),
        "excluded_from_patterns":  judge_response["diagnosis_quality_score"] < 50,
        "created_at":              datetime.utcnow(),
    }
```

---

# Validation Rules

Enforced by `src/agents/feedback_validators.py`. Failure triggers retry with error list appended.

1. **JSON parseable** (strip markdown fences if present).
2. **Refusal short-circuit**: if `refused == true`, only `refused_reason` is required; remaining fields may be null.
3. **Required fields (non-refused)**: `prediction_accuracy_rating`, `error_classification`, `error_magnitude`, `sibling_check`, `root_cause_analysis`, `agent_specific_feedback`, `pattern_flags`.
4. **`error_classification`**: one of `correct`, `direction_wrong`, `magnitude_overestimate`, `magnitude_underestimate`, `timing_error`.
5. **`prediction_accuracy_rating`**: one of `accurate`, `mostly_accurate`, `moderately_inaccurate`, `very_inaccurate`.
6. **Sibling-check consistency**: `error_classification == "direction_wrong"` AND `sibling_check.timing_error_candidate == true` → retry (should have been classified `timing_error`). When `timing_error_candidate == true` is asserted, at least one sibling in `sibling_check.sibling_directions` must be scored (cannot infer timing-error candidate from unscored siblings).
7. **`root_cause_analysis.primary_error_source`**: must be in `DIAGNOSABLE_AGENTS` (the ten user-facing pipeline agents). `shadow_cio` is rejected — it is visible-but-not-diagnosable. Free strings rejected.
8. **`root_cause_analysis.primary_error_type`**: must be `snake_case`. Warning logged if not in the published controlled vocabulary (allowed but flagged for taxonomy review).
9. **`primary_error_detail`**** / ****`secondary_error_detail`**: char bounds 200–1200 / 50–600 respectively. Empty primary detail → retry.
10. **`contributing_factors`**: 1–4 items. Each ≥ 30 chars.
11. **`what_would_have_changed_the_outcome`**: 120–800 chars. Must name at least one agent (regex check against `DIAGNOSABLE_AGENTS`).
12. **`agent_specific_feedback`**: 1–10 items. Each `agent` ∈ `DIAGNOSABLE_AGENTS`; `accuracy_for_this_prediction` ∈ \{`good`, `mixed`, `poor`\}; `specific_lesson` 40–400 chars.
13. **Bootstrapping-tier language gate**: when `bootstrapping_tier == "early"`, no numeric tokens permitted in `agent_specific_feedback[].specific_lesson` or `pattern_flags`. Numeric tokens detected → retry.
14. **`pattern_flags`**: 0–3 items. In `early` tier, must be empty (retry if non-empty). In `developing` / `mature`, each item must contain a digit (count required).
15. **Numeric-traceability guard (soft)**: numeric tokens in `primary_error_detail`, `secondary_error_detail`, `contributing_factors`, and `what_would_have_changed_the_outcome` *should* appear in the input payload. Implementation normalizes percent (`15%` / `15 percent` / `0.15`), basis points, and word-form numbers before substring search. Missing tokens log a warning attached to the diagnosis for Judge attention; they do not trigger retry. The Judge is the authoritative grounding check.
16. **No self-mention**: `agent_specific_feedback[].agent` cannot equal `"feedback_analyst"` or `"shadow_cio"`.
17. **No re-prediction language**: `what_would_have_changed_the_outcome` regex-rejects "the correct recommendation", "the right answer", "should have been bullish/bearish" (the field describes reasoning changes, not correct outputs). Soft warning + retry.
18. **Refusal consistency**: `refused == true` AND `refused_reason` not in \{`below_minimum_sample_size`, `on_track_check_not_a_scoring_checkpoint`, `input_data_not_total_return`, `prediction_already_diagnosed_at_this_checkpoint`\} → retry.

---

# Retry Prompt Injection

```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Reminders:
- error_classification ∈ {correct, direction_wrong, magnitude_overestimate, magnitude_underestimate, timing_error}
- If sibling_check.timing_error_candidate is true, error_classification must be timing_error, and at least one sibling direction must be scored
- primary_error_source must be in DIAGNOSABLE_AGENTS (10 agents — Shadow CIO is NOT diagnosable)
- primary_error_type must be snake_case (controlled vocabulary preferred)
- agent_specific_feedback entries must use DIAGNOSABLE_AGENTS names; lessons 40-400 chars; Shadow CIO never appears
- In `early` bootstrapping tier: no numeric magnitudes anywhere; pattern_flags must be empty
- what_would_have_changed_the_outcome describes REASONING changes, not correct answers
- Do NOT include the feedback_analyst or shadow_cio in agent_specific_feedback
```

---

# Input Schema Reference

What the Feedback Analyst receives at invocation. Drawn from `predictions`, `prediction_outcomes`, `analysis_runs`, `agent_outputs`, `checkpoint_evaluations`, `events_log`, and `counterfactual_records`.

```python
@dataclass
class PredictionRecord:
    id: UUID
    analysis_run_id: UUID
    timeline_label: str           # "1_month_parallel" | "3_month_parallel" | "primary_6_month" | "primary_long_term"
    stock_outlook: str            # bullish / somewhat_bullish / neutral / somewhat_bearish / bearish
    confidence: int               # 0-100
    target_return_pct: float
    range_low: float
    range_high: float
    expected_return_tier: str
    components: dict              # CIO's expected_return.components breakdown
    basis: str                    # CIO's expected_return.basis
    created_at: datetime
    scored_at: datetime | None
    actual_return_pct: float | None
    holding_days: int | None

@dataclass
class OutcomeRecord:
    prediction_id: UUID
    checkpoint_label: str         # "1_week" / "1_month" / "3_months" / etc.
    days_held: int
    actual_return_pct: float      # total return (price + cumulative dividends)
    benchmark_return_pct: float   # total return
    in_range: bool
    inputs_are_total_return: bool # orchestrator-set flag

@dataclass
class CheckpointRecord:
    checkpoint_label: str
    days_held_at_check: int
    on_track: bool
    deviation_pct: float          # actual - predicted at this point

@dataclass
class EventRecord:
    event_date: date
    event_type: str               # earnings / macro / company_news / sector_move / regulatory
    description: str
    source: str

@dataclass
class CounterfactualRecord:
    ticker: str
    company_name: str
    cio_outlook: str              # outlook of the runner-up at the time
    actual_return_pct: float      # total return over same window as primary prediction

@dataclass
class AnalysisRun:
    id: UUID
    canonical_ticker: str
    company_name: str
    sector: str
    timeline: str
    account_type: str
    agent_outputs: dict[str, AgentOutput]   # keyed by agent_name (subset of VISIBLE_AGENT_ORDER)
```

---

# Bootstrapping Tier Reference

<table header-row="true">
<tr>
<td>Scored predictions</td>
<td>Tier</td>
<td>Feedback Analyst behavior</td>
<td>Pattern detection</td>
<td>Learning journal</td>
</tr>
<tr>
<td>0–4</td>
<td>`pre_bootstrap`</td>
<td>**Not invoked.** Only raw accuracy stats.</td>
<td>Off</td>
<td>Off</td>
</tr>
<tr>
<td>5–20</td>
<td>`early`</td>
<td>Diagnoses generated. Directional language only. No numeric magnitudes. No pattern flags.</td>
<td>Off</td>
<td>Off</td>
</tr>
<tr>
<td>20–50</td>
<td>`developing`</td>
<td>Full diagnosis. Pattern flags allowed. Numeric lessons may be cited with "preliminary" qualifier.</td>
<td>On with high threshold (pattern requires 10+ supporting cases)</td>
<td>Entries flagged "preliminary"</td>
</tr>
<tr>
<td>50+</td>
<td>`mature`</td>
<td>Full operation.</td>
<td>On (standard threshold)</td>
<td>Full operation</td>
</tr>
</table>

The orchestrator computes `scored_predictions_count` and the tier before invocation. Pre-bootstrap predictions never invoke the Feedback Analyst at all.

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
<td>`AnalysisRun`</td>
<td>Real names — not anonymized</td>
</tr>
<tr>
<td>`{timeline}`, `{account_type}`</td>
<td>`AnalysisRun`</td>
<td></td>
</tr>
<tr>
<td>`{prediction_date}`</td>
<td>`PredictionRecord.created_at`</td>
<td>ISO date</td>
</tr>
<tr>
<td>`{checkpoint_label}`, `{days_held}`</td>
<td>`OutcomeRecord`</td>
<td></td>
</tr>
<tr>
<td>`{predicted_return_pct}`, `{predicted_range_low}`, `{predicted_range_high}`, `{predicted_confidence}`</td>
<td>`PredictionRecord`</td>
<td>Formatted to 2dp with signed prefix</td>
</tr>
<tr>
<td>`{actual_return_pct}`, `{benchmark_return_pct}`, `{in_range_bool}`, `{prediction_error_pct}`</td>
<td>`OutcomeRecord`</td>
<td>Total return; signed 2dp</td>
</tr>
<tr>
<td>`{prior_stock_outlook}`, `{prior_expected_return_tier}`</td>
<td>`PredictionRecord`</td>
<td></td>
</tr>
<tr>
<td>`{bootstrapping_tier}`, `{scored_predictions_count}`</td>
<td>Orchestrator</td>
<td>Tier computed from scored-predictions count</td>
</tr>
<tr>
<td>`{inputs_are_total_return}`</td>
<td>`OutcomeRecord`</td>
<td>Refuse if `false`</td>
</tr>
<tr>
<td>`{sibling_predictions_block}`</td>
<td>`format_sibling_predictions()`</td>
<td>Empty placeholder when no siblings</td>
</tr>
<tr>
<td>`{checkpoint_history_block}`</td>
<td>`format_checkpoint_history()`</td>
<td>Empty placeholder for early checkpoints</td>
</tr>
<tr>
<td>`{events_block}`</td>
<td>`format_events()`</td>
<td>Earnings, macro, news, sector</td>
</tr>
<tr>
<td>`{counterfactuals_block}`</td>
<td>`format_counterfactuals()`</td>
<td>Runner-up tickers</td>
</tr>
<tr>
<td>`{agent_outputs_block}`</td>
<td>`format_agent_outputs_for_diagnosis()`</td>
<td>All `VISIBLE_AGENT_ORDER` agents, compressed</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
<td></td>
</tr>
</table>

---

# Testing Checklist

- [ ] Valid JSON from Opus 4.6 with `enable_thinking: true`?
- [ ] Refusal cases: pre-bootstrap, on-track check, non-total-return inputs each produce `{"refused": true}` and stop?
- [ ] `error_classification` correctly distinguishes `direction_wrong` vs `timing_error` when siblings disagree across horizons AND siblings are scored?
- [ ] When siblings exist but are unscored, classification proceeds on available data — `timing_error` not inferred?
- [ ] `correct` predictions still generate full per-agent feedback with `good` ratings (feeds winning-pattern aggregator)?
- [ ] CIO synthesis errors attributed to `cio`, not deflected upstream when upstream agents were defensible?
- [ ] `primary_error_source` is in `DIAGNOSABLE_AGENTS` (10 agents) — `shadow_cio` rejected?
- [ ] Shadow CIO never appears in `agent_specific_feedback` even when its output was present in the input?
- [ ] Numeric-traceability soft-warning: numbers not in payload get logged for Judge attention but don't retry?
- [ ] Bootstrapping tier `early`: no numeric magnitudes anywhere; `pattern_flags` empty?
- [ ] Bootstrapping tier `developing`: pattern flags allowed, journal entries flagged "preliminary"?
- [ ] `what_would_have_changed_the_outcome` describes reasoning changes — does NOT contain "the correct recommendation"-style phrasing?
- [ ] Hindsight check: counterfactuals use only data available on prediction date — earnings prints that closed after prediction date are excluded?
- [ ] Per-agent feedback omits absent / errored agents rather than emitting empty entries?
- [ ] Judge step runs on every diagnosis; `diagnosis_quality_score < 50` marks the diagnosis "excluded from patterns"?
- [ ] Judge adversarial prompt catches: (a) reconstructed agent claims not in original output, (b) hindsight bias, (c) pattern flags without counts, (d) deflection from CIO to upstream, (e) Shadow CIO as error source?
- [ ] Judge does NOT use counterfactual realized returns when grading `what_would_have_changed_the_outcome` plausibility?
- [ ] Fixture predictions: (a) clean direction_wrong with clear single-agent attribution; (b) timing_error where 1m miss + 6m hit (both scored); (c) correct prediction generating winning-pattern signal; (d) CIO synthesis bias where upstream agents were defensible; (e) `early` tier diagnosis producing directional-only language; (f) unscored sibling case forcing non-timing-error classification.

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
<td>~3,500 tokens</td>
<td>Reasoning process + 11 rules + taxonomy + closed agent set + schema</td>
</tr>
<tr>
<td>Predicted vs actual outcome block</td>
<td>~300 tokens</td>
<td>Header + outcome record</td>
</tr>
<tr>
<td>Sibling + checkpoint + events + counterfactuals blocks</td>
<td>~1,500 tokens</td>
<td></td>
</tr>
<tr>
<td>Original agent outputs (eleven agents, compressed)</td>
<td>~10,000 tokens</td>
<td>Drop verbose narratives when budget tight; structured_data retained</td>
</tr>
<tr>
<td>Expected output</td>
<td>~2,000 tokens</td>
<td>Diagnosis JSON</td>
</tr>
<tr>
<td>**Total**</td>
<td>**~16,500–19,500 tokens**</td>
<td>Within Opus 4.6 limits</td>
</tr>
</table>

**Model configuration**:
- Primary: `claude-opus-4-6` (diagnostic reasoning quality dominates; offline-batch latency tolerant)
- Cost-saving fallback: `claude-sonnet-4-6` for `bootstrapping_tier ∈ {early, developing}` where Judge quality compensates
- `max_tokens`: 2,500
- `temperature`: 0.2 (low — diagnosis should be repeatable across re-runs)
- Thinking mode: ON
- Prompt caching: stable system prefix (~3,000 tokens) cached per deploy
- Expected latency: ~25–45s per prediction (Opus + thinking); ~12–20s (Sonnet fallback)
- Cost per prediction: ~$0.25–0.45 (Opus) / ~$0.07–0.12 (Sonnet)
- Judge step adds: ~$0.02–0.04 + ~3–6s latency

**Re-confirm pricing with latest published Anthropic rates before launch.**

---

# Implementation Files

- `src/agents/feedback_analyst.py` — Feedback Analyst agent implementation
- `src/agents/judge.py` — Paired Judge implementation
- `src/agents/feedback_formatter.py` — `format_sibling_predictions()`, `format_checkpoint_history()`, `format_events()`, `format_counterfactuals()`, `format_agent_outputs_for_diagnosis()`, `format_single_agent_for_diagnosis()`. Also exports `DIAGNOSABLE_AGENTS` and `VISIBLE_AGENT_ORDER` constants
- `src/agents/feedback_validators.py` — Validation rules above
- `src/feedback/checkpoint_evaluator.py` — Trigger source. Decides when a prediction reaches a scoring checkpoint and invokes the Feedback Analyst
- `src/feedback/pattern_detector.py` — Downstream consumer. Runs every 10 evaluated predictions; consumes `diagnostic_reports` with `excluded_from_patterns == false`
- `src/feedback/learning_journal.py` — Persists confirmed patterns

---

# Pending Propagations

1. **Pattern-detection engine spec**: define how the engine aggregates `primary_error_source` × `primary_error_type` × stock characteristics (sector, market cap band, regime) into `learning_journal` entries. The Feedback Analyst's vocabulary discipline depends on this engine's clustering rules.
2. **Accuracy brief generation**: Tier 1 informational injection regenerated every 20 scored predictions. Aggregates per-agent `accuracy_for_this_prediction` ratings into recency-weighted accuracy summaries injected into Bull / Bear / Risk / Tax / CIO prompts. Defines the schema in the CIO prompt's "Agent Accuracy Briefs" section.
3. **Winning-pattern brief**: aggregates `accuracy_for_this_prediction: "good"` ratings into positive-reinforcement injection. Schema TBD; mirrors the negative-pattern Reflexion Brief structure.
4. **Reflexion brief construction**: when a stock has a prior scored prediction, the diagnosis emitted here is repackaged as the CIO's Reflexion Brief on the next analysis. Define the field mapping in `src/feedback/reflexion.py`.
5. **Cold-start gate**: warm-up backtest run before live deployment. Until 20+ scored live predictions exist, all user-facing recommendations get `provisional_confidence_cap = 65` (per the learning system spec, Part 5). The Feedback Analyst respects the same threshold via its `bootstrapping_tier` gate.
6. **Judge-feedback feedback loop**: when `judge_feedback` flags a specific Feedback Analyst weakness repeatedly across diagnoses, that's a signal to refine the FA prompt itself. Decide whether to surface this manually or via an automated meta-meta loop. Lean manual until N > 100 diagnoses.

---

# Open Questions

1. **Should the Judge see counterfactual records?** The Feedback Analyst sees them; the Judge currently does too via `feedback_analyst_input_payload`. Risk: Judge could anchor on counterfactual outcomes (hindsight) when scoring `what_would_have_changed_the_outcome`. Mitigation already in place: the Judge prompt explicitly tells the Judge not to use counterfactual outcomes when grading counterfactual plausibility — only for evidence-grounding cross-checks. **Recommendation**: keep Judge with full payload; re-evaluate after first 30 diagnoses and remove counterfactuals from the Judge's payload if hindsight-bias creep is observed in `judge_feedback` notes.
2. **Self-attribution of the Feedback Analyst.** When the Judge consistently rejects the FA's diagnoses for a specific weakness (e.g., always over-attributing to fundamental), is that itself a pattern worth journaling? Currently no — only pipeline-agent errors feed the learning journal. **Recommendation**: log to a separate `feedback_analyst_meta_metrics` table for manual review; do not feed back into the pattern engine until tooling is more mature.
3. **Cross-prediction reasoning.** A pattern flag like "3rd time fundamental overestimated growth for tech during a rate-hiking cycle" requires the FA to count cases. Currently the FA does this from memory / context — unreliable at scale. **Phase 4 candidate**: inject a pre-computed `recent_pattern_seeds` block from the pattern detector ("here are 3 patterns that include this case as a potential extension") so the FA only confirms, not counts. Avoid until pattern detector has steady-state output.
4. **Timing-error sub-classification.** `timing_error` currently treats all horizon-mismatches the same. Future: split into `right_direction_too_early` vs `right_direction_too_late` once Phase 5 calibration data shows the asymmetry matters.
5. **Model choice for low-cost tier.** Whether `claude-sonnet-4-6` produces diagnoses of meaningfully lower Judge-quality scores than `claude-opus-4-6` is unproven. **A/B fixture test**: once 30 diagnoses exist on Opus, re-run on Sonnet and compare `diagnosis_quality_score` distributions. If Sonnet's mean is within 5 points of Opus, switch primary from Opus to Sonnet — the cost gap (~5×) is not justified.
6. **Judge model.** Sonnet 4.6 chosen for cost. Risk: Judge under-catches subtle hindsight bias or reconstructed-claim errors that Opus would catch. **A/B**: 20 diagnoses on Sonnet-judge, same 20 re-judged by Opus. If Opus catches material errors Sonnet missed, escalate Judge to Opus and accept the cost. Diagnostic quality is critical; do not under-invest here.
7. **What if no agent contributed to error?** A correct prediction (`error_classification: correct`) still requires full diagnosis to feed the winning-patterns aggregator. But what about a *correct prediction by chance* — all agents got the reasoning wrong but the actual outcome happened to land in range? Currently the FA would mark this `correct` and emit `good` ratings, which is misleading. **Phase 4 candidate**: add an `error_classification: correct_by_luck` variant when the reasoning was demonstrably flawed but the outcome luckily landed. **Note**: adding a value to `error_classification` is a schema version bump — downstream consumers (`diagnostic_reports` columns, pattern detector clustering rules, learning journal entry types) all need migration plans before this can land. Defer until > 20 diagnoses exist to size how common the case is.
8. **Confidence-Brier calibration as a sibling diagnosis dimension.** The FA currently diagnoses single predictions. Confidence calibration is inherently a multi-prediction property (does the 70–80 confidence band hit the right rate?). This sits naturally with the pattern engine, not the FA. **Confirm**: keep the FA single-prediction-scoped; the pattern engine owns Brier-score-style calibration.

---

_Source: Notion — Project Stock Picker → Agent Prompts → Feedback Analyst Agent Prompt. Downloaded 2026-07-02._
