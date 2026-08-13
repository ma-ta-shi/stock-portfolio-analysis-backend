**Agent**: Tax Strategist<br>**Status:** Draft v1.5<br>**Phase:** Phase 2<br>**Pass**: Pass 2 (Perspective & Strategy)<br>**Role**: Assess the after-tax impact of \{ticker\} for the specified timeline and account, given Canadian tax rules during the user’s accumulation phase. The Tax Strategist quantifies after-tax yield, account-fit, and cross-account placement opportunities. Independent of the Bull/Bear advocates and the Risk Advisor: they argue direction or quantify outcome range; the Tax Strategist quantifies how much of the realized return the user actually keeps after Canadian tax.<br>**Target Model**: **Cloud-preferred: ****`claude-sonnet-4-6`** (per [CLAUDE.md](http://CLAUDE.md) — tax-rule complexity favors Sonnet over the local model). Local `qwen3.6:35b-a3b` fallback is dev-test only until a tax-rule accuracy gate is met.<br>**Token Budget**: \~1,400 system prompt (includes \~500-token injected reference) \| \~3,800 data input \| \~1,200 expected output \| \~6,400 total.

**User-context assumptions** (baked in by design — revisit if any change):
- Canadian tax resident, accumulation phase, no RRSP withdrawals.
- Total portfolio \<\$250K CAD. Annual realized capital gains will not approach the \$250K higher-inclusion-tier threshold; 50% inclusion rate is the only rate the agent needs.
- Default province: Ontario (override via `user_tax_context.province` if applicable).
- US estate tax exposure is a low-priority caveat at this portfolio size — surfaced only when US-situs concentration is genuinely material.
- Foreign-listed (non-US, non-Canadian) names are vanishingly rare in scope; treated as a one-line edge case.
- US MLPs are not held; flagged as inappropriate if encountered.
- Sophistication: marginal-rate-arbitrage uses a default heuristic (current \> retirement) rather than requiring an explicit `expected_retirement_marginal_rate_pct` parameter.
---
# Design Decisions & Rationale
Built following the [Agent Design Patterns](https://www.notion.so/Agent-Design-Patterns-33fea067982380e6a055d3ed74c8eada?pvs=21).
1. **Analyst, not advocate**. No `recommendation: "bullish|bearish"`. Output is `tax_efficiency_for_account: favorable | neutral | unfavorable` plus a structured `tax_profile`. **Honest framing**: `tax_efficiency_for_account` is an indirect directional signal — the CIO treats `unfavorable` as a constraint that can downgrade a bullish thesis when tax drag eats the alpha.
2. **Cloud-preferred**. Tax-rule mistakes have real consequences; local-model failure modes on tax content are plausible-sounding wrong answers. Cloud is the production path.
3. **Pre-computed tax metrics are first-class inputs**. The data pipeline computes deterministically: listing exchange, security structure, dividend yield, dividend type classification, per-account WHT rate, account state (TFSA/RRSP room, trading capital-gains YTD, superficial-loss-window status, dual-listed flag for Canadian cross-listed names). The LLM interprets — it does not compute.
4. **Account context is the first-class CoT axis** — Step 1 of the CoT. Account_type drives the analysis more than any other variable.
5. **9-step CoT**, conditional steps default-output the inapplicable case rather than skipping. Each step maps to at least one structured field.
6. **Citations required on every claim**. Pass 1 IDs (RSRCH/FUND/TECH/SENT/MACRO), pre-computed tokens (DIVID/LIST/DOM/WHT/CGAIN/ROOM/LOSS/ELIG), or REF (the injected Canadian Tax Rules Reference).
7. **Hardcoded constraint: no RRSP withdrawals**. Validator strips any action implying withdrawal.
8. **Contribution-room awareness**: TFSA sells must reference room recovery timing (next-year recovery); RRSP buys must reference permanent room consumption. Over-contribution is treated as a hard blocker — the agent does not propose actions that exceed room.
9. **Superficial loss rule (s. 54 ITA): 30-day cooldown**. **Canadian dual-listed names** (RY/RY.TO, BNS/BNS.TO, ENB/ENB.TO, etc.) are “identical” under s. 54 — the LOSS token carries a `dual_listed` flag and the agent must apply the cross-listed check.
10. **TFSA day-trading classification risk** is surfaced as a `key_tax_risks` entry when `account_type == "tfsa"` AND `timeline == "short_term"`, OR when the Researcher archetype implies near-term realization. CRA may classify frequent TFSA trading as carrying on a business — making gains taxable.
11. **Marginal-rate arbitrage uses a default heuristic, not an explicit field**. For accumulation-phase Canadian retail, current marginal rate \> expected retirement rate is the default assumption. RRSP placement therefore gets a deferral-arbitrage benefit on top of WHT exemption. If the user provides `user_tax_context.expected_retirement_marginal_rate_pct` and the spread is materially different, the agent uses it; otherwise it assumes a positive spread and flags the assumption in caveats.
12. **Architecture (b): Canadian Tax Rules Reference is the source of truth, injected at runtime**. The maintained [Canadian Tax Rules Reference](https://www.notion.so/Canadian-Tax-Rules-Reference-325ea06798238126a06df7650f38657e?pvs=21) is bundled as a local file at `prompts/tax_strategist/canadian_tax_rules_reference.md` and loaded by the orchestrator. Rule changes are made to the local file (kept in sync with the Notion reference page). Cache invalidates when the file's `last_verified` date header changes (\~once per year nominal). No live Notion fetch occurs at runtime.
13. **No autonomous tax advice — disclaimer enforced by orchestrator merge**. Top-level `disclaimer` field appended by orchestrator, not LLM-produced.
14. **Memory drops on rule change OR cross-account mismatch**. Reference-version drift (any change to the reference’s `last_verified` field) drops memory. Tax conclusions are situational on the rule set.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- Dividend history, type, yield — openbb-tmx .equity.fundamental.dividends (CA) / FMP /stock_dividend (US)
- Company profile (exchange, country) — openbb-tmx .equity.profile (CA) / FMP /profile (US)
- Withholding tax rules — static reference data, no API (CA & US)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- `precomputed_tax_metrics` block — DIVID (yield + frequency), ELIG (Canadian-eligible-dividend status), LIST (listing exchange/country), DOM (security structure + distribution decomposition), WHT (per-account withholding rate, recoverable flag), effective after-tax yield, CGAIN (50% inclusion + YTD realized gains in Trading)
- `account_state_block` — TFSA/RRSP contribution room remaining, superficial-loss-window status + dual_listed flag (LOSS), US-situs aggregate when relevant
- `tax_rules_reference` — the injected Canadian Tax Rules Reference, loaded from the local bundled file (source of truth; cited as REF)
- `TAX_RULE_SNAPSHOT` — snapshot date + reference last-verified date, for memory-drop detection
- `user_tax_context` — province (default Ontario), optional `expected_retirement_marginal_rate_pct`
- `memory_brief` — prior Tax Strategist run for this stock/account (dropped on rule-version drift or account-type mismatch)
- `pass1_reliability_warnings`, `accuracy_brief`, `winning_patterns_brief` — compressed Pass 1 outputs and calibration briefs (post-20/50 predictions)
- `researcher_thesis_archetype` — from `researcher_output.pass2_view.thesis_archetype`, used to detect short-term/catalyst realization risk
---
# System Prompt
Stable content (role framing, injected reference, rules, CoT, schema) precedes the `--- STOCK-SPECIFIC CONTEXT ---` separator so the cacheable prefix is real. Caching invalidates only on reference-file edits (last_verified date change) or prompt deploys.

**Runtime prompt:** see [`prompts/tax_strategist/v1.5.txt`](../../prompts/tax_strategist/v1.5.txt)
## Conditional Injection Reference
**Timeline**:
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
<td>`SHORT-TERM (1-4 wks). Capital-gains realization within horizon plausible. Loss-harvesting time-sensitive. Dividend drag small over this horizon. CRITICAL: short_term + tfsa triggers Rule 9 — surface CRA business-income reclassification risk in key_tax_risks.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Annual dividend drag is materially relevant. Capital-gains realization possible mid-horizon.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). WHT drag compounds for income holdings. Permanent-capital-loss cost is the binding TFSA risk. RRSP wins for US dividend names; TFSA wins for non-dividend growth (RRSP would tax withdrawals as ordinary income, losing the capital-gains rate). For Canadian-eligible-dividend names in RRSP, DTC is lost at withdrawal — Trading preferred at moderate brackets.`</td>
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
<td>`TFSA. Permanent capital shelter on Canadian-source returns AND on capital gains. Capital losses PERMANENTLY LOST. US ordinary dividends: 15% non-recoverable WHT. US REIT/MLP distributions: structure-specific WHT, often 30% on non-qualified portions. Loss-harvesting INADMISSIBLE. Contribution room recoverable Jan 1 of year following sell. Over-contribution is blocking. Frequent trading risks CRA business-income classification.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US ordinary dividends WHT-EXEMPT under treaty Article XVIII(2). NO WITHDRAWALS during portfolio operation. In-account swaps admissible. Capital gains/losses tax-deferred until withdrawal — at withdrawal, taxed as ORDINARY INCOME. Canadian-eligible-dividend stocks LOSE the DTC at withdrawal. Foreign dividends generally do NOT receive treaty exemption. RRSP contribution room is permanently consumed by buys. $2,000 lifetime buffer is inviolable.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading (non-registered, taxable). Canadian-eligible dividends: gross-up + DTC per REF/Taxable — most tax-efficient at moderate brackets. Capital gains: 50% inclusion at marginal. Capital losses: deductible. Foreign tax credit on US/foreign WHT. Superficial loss rule (s. 54 ITA): 30-day window before-and-after; loss disallowed if repurchased in any account including spouse's; "identical" includes Canadian dual-listed cross-exchange equivalents per REF/Superficial-loss. Loss-harvesting is the most powerful tax lever in this account.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General — no specific account. Compare all three per REF/Account-optimization-decision-matrix. account_fit_score reflects the BEST account; cross_account_recommendation surfaces the optimal placement (or null when accounts tie).`</td>
</tr>
</table>
---
# Output Schema (LLM-produced fields)
```javascript
{
  "groundedness_score": 0-100,
  "thesis_summary": "2-3 sentences. Tax profile in compressed form. Must reference timeline AND account, AND at least one of: DIVID, WHT, CGAIN, LIST.",
  "strongest_signal": "1-2 sentences. The single most important tax observation. Must cite a Pass 1 ID, metric token, or REF.",
  "caveats": ["data gaps, reliability flags, retirement-rate-unknown caveat, US-estate-tax soft caveat, memory-drop acknowledgments"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "negative|neutral|positive", "evidence": "PASS1_ID_OR_METRIC_OR_REF: compact citation"}
  ],
  "narrative": "200-350 word tax profile. Quantitative, grounded.",
  "tax_profile": {
    "account_axis_summary": "1-2 sentences.",
    "dividend_classification": "canadian_eligible|us|foreign|mixed|none",
    "dividend_yield_pct": <number passthrough, one-decimal>,
    "yield_interpretation": "1-2 sentences.",
    "listing_summary": "1 sentence including DOM implications.",
    "withholding_tax_rate_pct": <number passthrough>,
    "wht_interpretation": "1-2 sentences.",
    "effective_after_tax_yield_pct": <number passthrough, one-decimal>,
    "annual_tax_drag_pct": <number, one-decimal; positive = drag>,
    "capital_gains_treatment_summary": "1-2 sentences. Note RRSP DTC loss for Canadian dividend stocks held in RRSP.",
    "loss_harvesting_opportunity": {
      "available": true|false,
      "estimated_tax_savings_pct": <number, one-decimal, OR null>,
      "superficial_loss_window_safe": true|false|not_applicable,
      "dual_listed_check_applied": true|false|not_applicable,
      "remediation_suggestion": "wait 31 days OR similar-but-not-identical substitute OR null",
      "reasoning": "1-2 sentences citing LOSS, CGAIN tokens"
    } | null,
    "tfsa_contribution_room_impact": "string OR null",
    "rrsp_contribution_room_impact": "string OR null",
    "cross_account_recommendation": {
      "better_account": "tfsa|rrsp|trading",
      "reasoning": "1-2 sentences citing tax mechanics + Rule 6 placement principle + Rule 13 marginal-rate arbitrage when applicable",
      "drag_delta_pct": <number, one-decimal; negative = improvement>
    } | null,
    "key_tax_risks": [
      {"risk": "1-sentence specific risk", "severity": "high|medium|low", "evidence": "PASS1_ID_OR_METRIC_OR_REF: compact citation"}
    ],
    "tax_optimization_actions": [
      {
        "action": "1-sentence specific action",
        "applies_to": "tfsa|rrsp|trading|cross_account",
        "estimated_benefit_pct": <number, one-decimal>,
        "justification": "1 sentence with citation",
        "blocking_constraints": "string OR null"
      }
    ],
    "account_fit_score": "excellent|good|fair|poor",
    "tax_efficiency_for_account": "favorable|neutral|unfavorable",
    "context_aware_strongest_argument": "for this {timeline} + {account_type}, which tax mechanic dominates"
  }
}
```
### Array bounds
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
<td>`tax_profile.key_tax_risks`</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`tax_profile.tax_optimization_actions`</td>
<td>0</td>
<td>3</td>
</tr>
</table>
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name` (`"tax_strategist"`), `agent_pass` (`"pass2"`), `analysis_context`, `data_sources_used`, `data_quality_assessment`.
- **`disclaimer`**: hardcoded string appended by orchestrator.
- **`tax_rule_snapshot`**: `{capital_gains_inclusion_rate_pct, tfsa_annual_limit_cad, rrsp_annual_max_cad, reference_last_verified_date, reference_page_id}` (5 fields — simplified from v1.3’s 15).
- **`data_quality_assessment`**: derived from Pass 1 reliability + tax-block completeness + reference-fetch success.
- **`recommendation`**, **`reliability_factors`**, **`weakest_point`**: intentionally omitted.
---
# Orchestrator Pre-computation
```python
def load_tax_rules_reference(reference_path: str = "prompts/tax_strategist/canadian_tax_rules_reference.md") -> tuple[str, date]:
    """Load the Canadian Tax Rules Reference from the local bundled file.

    The reference is maintained as a static local file and updated manually
    when tax rules change (~annually). Cache invalidates when the file's
    last_verified date header changes.

    Raises TaxReferenceUnavailable if the file is missing or unreadable.
    No Notion fetch occurs — the local file is the source of truth at runtime.
    """
    try:
        cached = cache.get("tax_rules_reference:local")
        mtime = os.path.getmtime(reference_path)
        if cached and cached["mtime"] == mtime:
            return cached["rendered"], cached["last_verified_date"]
        with open(reference_path, "r", encoding="utf-8") as f:
            raw = f.read()
        last_verified = parse_last_verified_date(raw)  # extracts "Last verified: ..." header
        rendered = render_for_prompt_injection(raw)  # core sections only; strip maintenance notes
        cache.set("tax_rules_reference:local",
                  {"rendered": rendered, "last_verified_date": last_verified, "mtime": mtime})
        return rendered, last_verified
    except (FileNotFoundError, PermissionError) as e:
        raise TaxReferenceUnavailable(
            f"Canadian Tax Rules Reference file not found or unreadable at {reference_path} — "
            "Tax Strategist cannot run."
        ) from e

def build_precomputed_tax_metrics(stock_id, ctx, portfolio, user_tax, snapshot) -> str:
    """Assemble the pre-computed tax metrics block."""
    metrics = tax_pipeline.compute(stock_id, account=ctx.account_type, user=user_tax, snapshot=snapshot)
    state = portfolio_service.account_state(user_tax.user_id, ctx.account_type) if portfolio else None

    block = [
        f"DIVID:{metrics.dividend_yield_pct:.1f}% yield,{metrics.dividend_frequency} payments",
        f"ELIG:{'canadian_eligible' if metrics.is_canadian_eligible else 'not_canadian_eligible'}",
        f"LIST:{metrics.listing_exchange} ({metrics.listing_country})",
        f"DOM:{metrics.security_structure}"
        + (f" —{metrics.distribution_decomposition}" if metrics.distribution_decomposition else ""),
        f"WHT (this account,{ctx.account_type}):{metrics.wht_rate_pct:.1f}%, "
        f"{'recoverable as FTC' if metrics.wht_recoverable else 'non-recoverable'}",
        f"Effective after-tax yield:{metrics.effective_after_tax_yield_pct:.1f}%",
        f"CGAIN: 50% inclusion at marginal"
        + (f"; YTD realized in Trading: ${state.trading_ytd_realized_gains_cents/100:.0f}" if state else ""),
    ]
    if state and ctx.account_type == "tfsa":
        block.append(f"ROOM: TFSA remaining ${state.tfsa_room_remaining_cents/100:,.0f}")
    if state and ctx.account_type == "rrsp":
        block.append(f"ROOM: RRSP remaining ${state.rrsp_room_remaining_cents/100:,.0f}")
    if state and ctx.account_type == "trading":
        loss_status = "BLOCKED — repurchased within 30d" if state.superficial_loss_blocked else "available"
        dual_listed_flag = ", DUAL_LISTED" if metrics.is_canadian_dual_listed else ""
        block.append(
            f"LOSS: superficial-loss window:{loss_status}{dual_listed_flag}; "
            f"YTD realized losses: ${state.trading_ytd_realized_losses_cents/100:.0f}"
        )
    # US-situs aggregate only when LIST is US AND portfolio context exists
    if metrics.is_us_listed and state and state.us_situs_aggregate_usd:
        block.append(f"US_SITUS aggregate: ${state.us_situs_aggregate_usd:,.0f} USD")

    block.append(
        f"TAX_RULE_SNAPSHOT: snapshot_date={snapshot.snapshot_date.isoformat()}, "
        f"reference_last_verified={snapshot.reference_last_verified_date.isoformat()}"
    )
    return "\n".join(block)
```
---
# Orchestrator Merge Logic
```python
def merge_output_tax_strategist(llm_response, pass1_outputs, precomputed_metrics,
                                tax_rule_snapshot, context):
    scores = [o.reliability_score for o in pass1_outputs if o is not None]
    fund = next((o for o in pass1_outputs if o and o.agent_name == "fundamental_analyst"), None)
    fund_score = fund.reliability_score if fund else 0
    if not scores: dqa = "insufficient"
    elif sum(scores)/len(scores) >= 70 and fund_score >= 50: dqa = "high"
    elif sum(scores)/len(scores) >= 50: dqa = "medium"
    elif sum(scores)/len(scores) >= 30: dqa = "low"
    else: dqa = "insufficient"

    tp = llm_response["tax_profile"]
    tp["dividend_yield_pct"] = round(precomputed_metrics.dividend_yield_pct, 1)
    tp["withholding_tax_rate_pct"] = round(precomputed_metrics.wht_rate_pct, 1)
    tp["effective_after_tax_yield_pct"] = round(precomputed_metrics.effective_after_tax_yield_pct, 1)

    for forbidden in ("recommendation", "tax_archetype", "bull_archetype", "bear_archetype", "risk_archetype"):
        llm_response.pop(forbidden, None)

    # Strip RRSP-withdrawal actions
    if "tax_optimization_actions" in tp:
        before = len(tp["tax_optimization_actions"])
        tp["tax_optimization_actions"] = [
            a for a in tp["tax_optimization_actions"]
            if not _action_implies_rrsp_withdrawal(a)
        ]
        if len(tp["tax_optimization_actions"]) < before:
            llm_response.setdefault("caveats", []).append(
                "orchestrator: stripped action(s) implying RRSP withdrawal (hardcoded constraint)"
            )

    return {
        "agent_name": "tax_strategist",
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
        "tax_profile": tp,
        "data_quality_assessment": dqa,
        "data_sources_used": [p.agent_name for p in pass1_outputs if p]
                             + ["precomputed_tax_pipeline", "portfolio_service", "tax_rules_reference"],
        "tax_rule_snapshot": tax_rule_snapshot.to_dict(),
        "disclaimer": (
            "Tax analysis is decision-support only and not legal or tax advice. "
            "Consult a qualified Canadian tax professional for personal tax decisions."
        ),
    }
```
---
# Validation Rules
1. **JSON parseable**.
2. **Required fields**: `groundedness_score`, `thesis_summary`, `strongest_signal`, `caveats`, `key_factors`, `narrative`, `tax_profile` (all sub-fields per schema).
3. **No ****`recommendation`**** or archetype fields** — stripped silently.
4. **groundedness_score**: integer 0–100. Tax-profile groundedness, not directional conviction.
5. **Pre-computed metric passthrough**: tolerance ±0.1 on yields and effective_after_tax_yield_pct; ±0.1 (one-decimal source) or ±0 (integer source) on WHT. Mismatch triggers retry; orchestrator overwrites as defense-in-depth.
6. **thesis_summary**: 80–500 chars, references timeline OR account_type AND at least one of `DIVID`, `WHT`, `CGAIN`, `LIST`.
7. **strongest_signal**: 40–400 chars, ≥1 valid citation token.
8. **caveats**: 0–5. If reliability_warnings present, must contain ≥1 entry referencing a flagged agent (FUND specifically when FUND \<40). If memory was dropped, must reference the drop. If `expected_retirement_marginal_rate_pct` is unprovided AND `cross_account_recommendation` involves RRSP, must contain the Rule 13 retirement-rate-unknown caveat.
9. **key_factors**: 2–4. Sentiment ∈ `negative|neutral|positive`. Evidence starts with a valid citation token followed by `:`.
10. **narrative**: 300–500 words (stop at 500 words; \~1,800–3,200 characters). ≥2 distinct Pass 1 IDs (or 1 + FUND specifically) AND ≥3 distinct pre-computed metric tokens AND ≥1 REF citation. Empirical 3,200-char ceiling (raised from 2,000 after 2026-05-15 simulation — multi-agent + multi-token citation requirements naturally produced 2,800–3,200-char narratives). Length is validator-enforced; `base.py` auto-trims to the last sentence boundary when overshoot is small.
11. **account_fit_score**: if `poor` AND `cross_account_recommendation` is null, retry. If `excellent` AND `cross_account_recommendation` is non-null with `drag_delta_pct < -0.10`, log warning. If `account_type == "rrsp"` AND `dividend_classification == "canadian_eligible"` AND `account_fit_score == "excellent"`, log warning (DTC loss makes excellent fit unlikely).
12. **key_tax_risks**: 1–4. Per risk: `risk`, `severity`, `evidence` required; evidence must contain at least one valid citation token. **Soft enforcement of risk surfaces**:
	- TFSA + (short_term OR cyclical_recovery): should contain a CRA business-income classification entry citing REF/TFSA. Missing logs warning (no retry). # v1.5c: event_driven/special_situation were invalid enum values; cyclical_recovery is the correct proxy.
		- DOM == “MLP” AND any Canadian account: should contain an entry flagging MLP inappropriateness. Missing logs warning.
13. **tax_optimization_actions**: 0–3. Per action: required fields present; justification cites at least one valid token. Actions implying RRSP withdrawal stripped.
14. **Loss-harvesting account constraint**: TFSA/RRSP → null; Trading → object or null.
15. **Superficial-loss-window**: when `loss_harvesting_opportunity.available == true`:
	- `superficial_loss_window_safe` ∈ `true|false|not_applicable`; consistent with LOSS token.
		- When `LOSS.dual_listed == true`, `dual_listed_check_applied` MUST be `true`. Failure triggers retry.
		- When `superficial_loss_window_safe == false`, `remediation_suggestion` MUST be non-null.
16. **Cross-account recommendation**: when non-null:
	- `better_account` ≠ current `account_type`.
		- `drag_delta_pct ≤ 0`. Positive triggers retry.
		- When Rule 13 marginal-rate arbitrage applies, `reasoning` must reference the spread (or its absence as a caveat).
		- When `better_account == "rrsp"` AND user has zero RRSP room AND no implied new capital, `blocking_constraints` MUST surface the limitation.
		- For `account_type == "general"`: no restriction.
17. **TFSA contribution-room consistency**: TFSA actions with sell implication require `tfsa_contribution_room_impact` referencing room recovery. Buys exceeding room must be marked blocking.
18. **RRSP contribution-room consistency**: RRSP buys require `rrsp_contribution_room_impact` referencing permanent room consumption. In-RRSP swaps must reference that sells do NOT free room. Buys exceeding room are blocking.
19. **drag_delta_pct sanity range**: `[-5.0, 0.0]`. Out of range triggers retry.
20. **Memory-drop conditions**: tax-rule snapshot drift (`reference_last_verified_date` change) OR cross-account mismatch. If dropped, caveats must reference the drop.
21. **TAX_RULE_SNAPSHOT and reference completeness**: pre-computed block must include TAX_RULE_SNAPSHOT line; orchestrator must have successfully fetched and rendered the reference. Hard pipeline error on missing.
22. **Numeric plausibility check**: numeric tokens in narrative + signals + interpretations must appear in input payload (Pass 1 + pre-computed block + injected REF). Missing triggers warning + single retry.
---
# Retry Prompt Injection
```javascript
VALIDATION ERRORS:
{validation_error_list}

Fix these. Respond with corrected JSON only. Remember:
- No directional recommendation, no archetype field
- pre-computed numerics must passthrough at SAME PRECISION
- every key_tax_risks.evidence and tax_optimization_actions.justification cites a Pass 1 ID, metric token, or REF
- NEVER propose RRSP withdrawal — hardcoded
- loss_harvesting_opportunity is null in TFSA/RRSP
- when LOSS.dual_listed is true, dual_listed_check_applied must be true
- when cross_account_recommendation is non-null, drag_delta_pct ≤ 0
- TFSA + short_term or TFSA + cyclical_recovery should produce a CRA-business-income key_tax_risks entry  # v1.5c: event_driven/special_situation replaced with cyclical_recovery
- account_fit_score "poor" requires a non-null cross_account_recommendation
- tax-rule numerics come from the injected REF; do NOT recall from training data
- generic phrasing without ticker-specific mechanics is inadmissible
```
---
# Memory Brief Schema
Populated when: prior Tax Strategist run exists AND tax-rule snapshot has not drifted (reference’s `last_verified_date` matches) AND prior `account_type` matches current `account_type`.
```javascript
Prior Tax Profile ({days_since} days ago, timeline={prior_timeline}, account={prior_account}):
  prior_dividend_classification: {classification}
  prior_dividend_yield_pct: {yield}%
  prior_account_fit_score: {fit}
  prior_tax_efficiency_for_account: {efficiency}
  prior_cross_account_recommendation: {rec}
  prior_tax_optimization_actions:
    - {action}; estimated_benefit: {b}%; acted_on: yes|no|partial
  prior_tax_rule_snapshot: reference_last_verified={prior_ref_verified}
  outcome_observed: {actual_drag_pct}% over {holding_period_days} days
  if_scored_prior_was: correct|incorrect|partially_correct
```
**`acted_on`**** enum**: `yes` (state change consistent with action within 90 days); `partial`; `no`; `not_yet`.
**Drop conditions**: reference-page version drift, OR cross-account mismatch.
**How the analyst uses memory**: Reads but does not defer. The most useful signal is `acted_on`: if not acted on, consider re-surfacing OR justify why current state changes the recommendation.
---
# Accuracy & Winning Patterns Briefs (Post-20 Predictions)
**Accuracy brief** (\~80 tokens):
```javascript
YOUR ACCURACY: {N} scored tax profiles. Account-fit accuracy: {X}% (within ±0.30pct annual). Cross-account recommendation accuracy: {Y}% (drag_delta_pct realized within ±0.30pct of estimate when user acted). TFSA day-trading risk-surface rate: {Z}%. Known bias: {e.g., "you under-estimate WHT drag on US-listed REITs in TFSA by 0.20-0.40pct"}. Adjustment: {e.g., "for US-listed REITs in TFSA, increase annual_tax_drag_pct by 0.30pct"}.
```
**Winning patterns brief** (\~120 tokens):
```javascript
YOUR WINNING PATTERNS:
Most accurate recent tax call: {ticker} on {date} — recommended {recommendation}, drag_delta realized {realized}% vs estimated {estimated}%.
- Pattern that works: {e.g., "for US dividend aristocrats with yield >2.5% in TFSA, RRSP-move recommendation produces drag_delta within ±0.15pct of estimate 85% of the time"}
- Action patterns: {e.g., "tax-loss selling proposals with explicit dual_listed_check_applied act on 70% of the time vs 35% baseline"}
Double down on these patterns when you see them.
```
---
# Minimum Data Threshold
Per Gate 1: ≥3 of 5 Pass 1 agents with `reliability_score >= 30`, AND pre-computed tax metrics block contains valid `DIVID` (or no-dividend marker), `LIST`, `DOM`, `WHT`, `TAX_RULE_SNAPSHOT`, AND the Canadian Tax Rules Reference local file (`prompts/tax_strategist/canadian_tax_rules_reference.md`) exists and is readable. Gate 1 failure: stub output with `groundedness_score: 0`, `completed_with_warnings`. Reference-file missing or unreadable: orchestrator does NOT invoke the agent.
**Pre-computed-metric independence**: The agent can produce useful output with weak Pass 1 IF tax metrics + reference are strong. Cap confidence at 60, list Pass 1 gaps in caveats, rely on metric-grounded + REF-grounded risks.
**FUND-specific dependence (edge case)**: When dividend classification cannot be determined from `LIST + DOM` alone (e.g., TSX-listed stock with mixed US-source dividend treaty treatment), FUND is binding. Cap confidence at 50 if FUND \<40 in this regime.
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
<td>`AnalysisContext` + `DataBundle`</td>
<td></td>
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
<td>`{researcher_thesis_archetype}`</td>
<td>`researcher_output.pass2_view.thesis_archetype`</td>
<td>Falls back to `"unclassified"`. **v1.5c: corrected from structured_data — always use pass2_view (stable contract).**</td>
</tr>
<tr>
<td>`{user_tax_context}`</td>
<td>`UserTaxContext` (province default Ontario; `expected_retirement_marginal_rate_pct` optional)</td>
<td>Sparse for retail user</td>
</tr>
<tr>
<td>`{account_state_block}`</td>
<td>`portfolio_service.account_state(user_id, account_type)`</td>
<td>Empty when account_type == “general”</td>
</tr>
<tr>
<td>`{precomputed_tax_metrics}`</td>
<td>`build_precomputed_tax_metrics()`</td>
<td>Required by Gate 1</td>
</tr>
<tr>
<td>`{tax_rules_reference}`</td>
<td>`load_tax_rules_reference()` — local file `prompts/tax_strategist/canadian_tax_rules_reference.md`</td>
<td>Required by Gate 1; loaded from local file; cached per file mtime</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory`; dropped per Rule 20</td>
<td>May be empty</td>
</tr>
<tr>
<td>`{pass1_reliability_warnings}`, `{accuracy_brief}`, `{winning_patterns_brief}`</td>
<td>Synthesized from Pass 1 / Feedback Analyst</td>
<td>May be empty; latter two empty when N\<20</td>
</tr>
</table>
---
# Token Budget
**Cloud (Sonnet)**: \~6,400 tokens comfortable. `max_tokens`: **4000** (empirical min safe value from 2026-05-15 simulation — lower values caused JSON truncation given the expanded narrative ceiling). Set to min safe + 20% headroom per the Orchestration Engine LLM Call Parameters section. JSON output prefilled with `{` per the same section. Sonnet input cache covers the \~2,800-token system-prompt prefix (system prompt + injected REF + rules + CoT + schema). Cache invalidates only on reference-file edits (last_verified date change) or prompt deploys. Per-analysis cost: \~\$0.025/run. 50-stock weekly batch: \~\$5/month steady state.
**Local (qwen3.6:35b-a3b)**: comfortable headroom; \~6,400 in `num_ctx: 8192` leaves room for 1–2 retries. Local promotion gated on tax-rule accuracy (Sonnet vs local-model agreement on the basic test cases listed below).
---
# Testing Checklist
**Functional tests**:
- TSX-listed Canadian eligible-dividend stock: TFSA → ≥`good`, WHT 0%. RRSP → ≤`fair` at moderate brackets (DTC loss at withdrawal). Trading → `excellent`.
- US dividend aristocrat: TFSA → `fair`/`poor`, recommendation `rrsp`, `drag_delta_pct ≈ -(yield × 0.15)` (or larger with marginal-rate arbitrage). RRSP → `excellent`. Trading → recommendation `rrsp` if room available.
- US REIT in TFSA: `key_tax_risks` includes a 30% non-qualified portion entry; very poor fit.
- US MLP in any account: `key_tax_risks` flags inappropriateness; recommendation typically null.
- Growth-only no-dividend: TFSA → `excellent`. RRSP → `good`; recommendation TFSA only when room exists AND timeline is long_term.
- Trading + YTD gains + hypothetical loss: loss-harvesting surfaces with explicit superficial-loss-window check.
- Trading + Canadian dual-listed (RY/RY.TO etc.): `dual_listed_check_applied = true`.
- TFSA + short_term: `key_tax_risks` includes CRA business-income classification entry.
- TFSA buy exceeding room: action marked blocking; over-contribution penalty referenced.
- RRSP buy exceeding room (above \$2K buffer): action marked blocking.
- US-listed stock when portfolio us_situs aggregate ≥30% of total portfolio: `caveats` (NOT key_tax_risks) includes US estate tax soft note.
**Adversarial tests** (must NOT produce):
- Any action implying RRSP withdrawal.
- `loss_harvesting_opportunity` non-null in TFSA/RRSP.
- `cross_account_recommendation.drag_delta_pct > 0`.
- `account_fit_score = "poor"` AND `cross_account_recommendation = null`.
- Generic tax phrasing without ticker-specific mechanics.
- Tax rule numerics that don’t match the injected REF.
- Loss-harvesting on a `LOSS.dual_listed == true` name without `dual_listed_check_applied = true`.
- US estate tax surfaced as a `key_tax_risks` entry (should be a soft caveat, not a risk, at this portfolio size).
---
# Open Questions
1. **CIO groundedness_score semantics across Pass 2 agents**: Pending Propagation 1 — extend to include Tax Strategist (`groundedness_score` = tax-profile groundedness, not directional conviction). Renamed from `confidence` in v1.5 to prevent semantics collision with Bull/Bear advocates.
2. **Cross-account recommendation routing to Portfolio Optimizer**: should the optimizer consume `cross_account_recommendation` directly, bypassing CIO summarization? Define interface in Milestone 2+.
3. **Tax pipeline implementation** (`src/data_pipeline/tax_pipeline.py`): confirm full token set emission (DIVID, ELIG, LIST, DOM, WHT, CGAIN, ROOM, LOSS, dual_listed flag) before v1 testing.
4. **Reference page rendering format**: `render_for_prompt_injection()` should include core mechanics sections (TFSA, RRSP, Taxable, Superficial Loss, Cross-border, Decision Matrix) verbatim; strip maintenance notes and inline “for the Tax Strategist agent” comments. Validate token cost in v1 testing.
5. **~~Reference-page fetch fallback~~**: Resolved. The reference is now the local file at `prompts/tax_strategist/canadian_tax_rules_reference.md` in all environments — including dev/test. No Notion fetch occurs at runtime.
6. **`expected_retirement_marginal_rate_pct`**** data source**: optional field; if the user later provides it, surface in `user_tax_context`. Default heuristic per Rule 13 holds in the meantime.
7. **Aggregate US-situs tracking**: `portfolio_service.account_state` should aggregate US-situs holdings across ALL accounts (not just the analysis account) for the Rule 11 caveat. Confirm or scope.
8. **Revisit list if portfolio context changes** (above \$250K, MLPs introduced, foreign-listed coverage, etc.): the cuts documented in Design Decisions are explicit on what would need to come back.

_Source: Notion — Project Stock Picker → Agent Prompts → Tax Strategist Agent Prompt. Downloaded 2026-07-02._
