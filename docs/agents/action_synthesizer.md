# Action Synthesizer Agent Prompt

**Agent**: Action Synthesizer
**Status**: Draft v1.0
**Phase**: Phase 1 (Milestone 2 — simplified; full version in Milestone 3)
**Pass**: Portfolio Optimization (Sub-agent 3 of 3 — final stage)
**Role**: Produce the final user-facing portfolio-optimization recommendations a Canadian retail investor (portfolio under $250k) will act on. Combines the Health Assessor's diagnosis and the Opportunity Ranker's candidates into concrete actions respecting Canadian account rules. This is the highest-stakes output in the system.
**Target Model**: `claude-sonnet-4-6` (cloud-only — no local fallback by design). `claude-opus-4-6` as optional escalation for high-conflict runs or deployments above $10,000. Thinking mode OFF.

---

# Design Decisions

1. **Cloud-only.** The local model class used elsewhere in this system (currently `qwen3.6:35b-a3b`, previously `qwen3:14b`) is not trusted to reliably reason simultaneously about Canadian tax rules, account constraints, and capital deployment math at the prompt's full context. Sonnet 4.6 is the floor. Revisit this design decision only after empirically testing the current local model on this specific task — don't assume it's still too weak just because the prior model was.
2. **The seven critical rules are the safety surface.** Reproduced once as both the prompt's standing rules and the validator's hard checks. Violations trigger retry.
3. **"No changes recommended" is a valid output.** The single biggest failure mode of optimizers is manufactured action. The agent is explicitly told to recommend zero actions when warranted, and `well_positioned` is the expected `overall_assessment` in that case.
4. **Three priority levels, not five.** A retail user looking at 1–4 actions doesn't need a Tier-1-through-5 framework. High / medium / low is sufficient.
5. **Capital deployment is right-sized.** Typical Available-to-Trade is $500–$5,000 into 1–2 stocks in 1 account. The prompt describes that as the primary case, not the multi-account waterfall.
6. **Cross-account rules are hard.** No RRSP withdrawals, ever. TFSA sells require bearish outlook AND a strong within-TFSA replacement. Cross-account moves are new-capital only.

---

# The Seven Critical Rules

Reproduced verbatim in the prompt and enforced post-hoc by the validator.

1. **NEVER suggest withdrawing from an RRSP.** Validator rejects any action with `sell_side.account = rrsp` AND `buy_side.account ≠ rrsp`.
2. **TFSA sells require justification.** Only valid when the holding's CIO outlook is `bearish` AND the replacement is also in TFSA. The contribution-room cost must be explicit in `tax_implication` — TFSA room is not restored until the following January.
3. **RRSP contributions respect the user's strategy.** Recommend RRSP buys only when `user_constraints.rrsp_strategy = "max_room_now"`. Otherwise leave RRSP alone.
4. **Cross-account moves apply only to new capital.** Never transfer an existing position. Misplaced holdings get an `account_optimize` recommendation scoped to future contributions.
5. **2% improvement threshold after costs.** No `swap` below this threshold. The threshold is evaluated against pre-computed `cost_estimates`, not LLM math.
6. **"No changes recommended" is valid.** When the portfolio is well-positioned, return `suggested_actions = []`, `overall_assessment = well_positioned`. Don't manufacture action.
7. **Every action requires confidence + rationale.** `confidence ∈ [0, 100]` and `description ≥ 30 chars` on every entry.

---

# Inputs

<table header-row="true">
<tr>
<td>Input</td>
<td>Purpose</td>
</tr>
<tr>
<td>`health_assessor_output`</td>
<td>Diagnosis. Top issues drive prioritization.</td>
</tr>
<tr>
<td>`opportunity_ranker_output`</td>
<td>Buy-side candidates with funding methods.</td>
</tr>
<tr>
<td>`account_state` — cash, contribution room, YTD realized gains, book values, superficial-loss cooldowns.</td>
<td>Authoritative on what's actually possible.</td>
</tr>
<tr>
<td>`canadian_tax_rules` — static knowledge base (shared with Tax Strategist).</td>
<td>TFSA / RRSP / Trading rules, WHT treaty, superficial-loss rule.</td>
</tr>
<tr>
<td>`user_constraints` — risk_tolerance, position/sector limits, `rrsp_strategy`, goals.</td>
<td>Filters and contribution-strategy gates.</td>
</tr>
<tr>
<td>`available_capital` — optional. When trigger is `available_to_trade`, the CAD amount to deploy.</td>
<td>Drives `capital_deployment` block.</td>
</tr>
</table>

---

# System Prompt

**Runtime prompt:** see [`prompts/action_synthesizer/v1.0.txt`](../../prompts/action_synthesizer/v1.0.txt)

---

# Output Schema

```json
{
  "optimization_summary": "string (2-3 sentence headline)",
  "overall_assessment": "well_positioned | minor_improvements | action_needed",
  "suggested_actions": [
    {
      "priority": "high | medium | low",
      "action_type": "deploy_capital | swap | rebalance | account_optimize | take_profit | tax_loss_harvest",
      "confidence": 0,
      "description": "string",
      "sell_side": {
        "ticker": "string | null",
        "account": "tfsa | rrsp | trading | null",
        "amount": "string | null",
        "reason": "string | null",
        "tax_implication": "string | null"
      },
      "buy_side": {
        "ticker": "string",
        "account": "tfsa | rrsp | trading",
        "amount": "string",
        "reason": "string"
      },
      "expected_improvement": "string",
      "risk_change": "string",
      "account_constraints_respected": true,
      "constraint_notes": "string | null"
    }
  ],
  "capital_deployment": {
    "total_to_deploy": "string | null",
    "allocation": [
      {
        "account": "tfsa | rrsp | trading",
        "amount": "string",
        "stocks": [{ "ticker": "string", "amount": "string", "rationale": "string" }]
      }
    ]
  } | null,
  "no_action_holdings": [
    { "ticker": "string", "account": "tfsa | rrsp | trading", "reason": "string" }
  ],
  "warnings": ["string"],
  "narrative": "string (200-350 words for the morning briefing)"
}
```

---

# Validation Rules

1. JSON parseable; no `buy_sell_hold` / `BUY` / `SELL` / `HOLD` fields.
2. **Rule 1**: any action with `sell_side.account = rrsp` AND `buy_side.account ≠ rrsp` → retry.
3. **Rule 2**: any action with `sell_side.account = tfsa` requires the sold holding's CIO outlook = bearish AND `buy_side.account = tfsa` → otherwise retry.
4. **Rule 4**: any action where `sell_side.ticker = buy_side.ticker` AND `sell_side.account ≠ buy_side.account` → retry (cross-account transfer is not an action).
5. **Rule 5**: every `swap` carries an `expected_improvement` resolvable to ≥ 2.0% after costs (parsed by regex). Below threshold → retry.
6. **Rule 7**: every entry in `suggested_actions` has `confidence ∈ [0, 100]` and `description` ≥ 30 chars.
7. **Superficial-loss check**: any `tax_loss_harvest` action with `sell_side.ticker` in `superficial_loss_cooldowns` → retry.
8. `overall_assessment` matches the priority set:
	- `well_positioned` ⇒ 0 actions, OR exactly 1 action with `priority = low`
	- `minor_improvements` ⇒ 1–2 actions with `priority ∈ {medium, low}`, no `high`
	- `action_needed` ⇒ at least one action with `priority = high`
9. `capital_deployment.total_to_deploy` equals the sum of `allocation[].amount` (within $1). When trigger is not `available_to_trade`, `capital_deployment` must be null.
10. `narrative` length 200–350 words.
11. All dollar amounts and percentages in narrative and actions trace to inputs.
12. `account_constraints_respected = false` is invalid — retry.

---

# Action Type Constraints

<table header-row="true">
<tr>
<td>`action_type`</td>
<td>`sell_side`</td>
<td>`buy_side`</td>
</tr>
<tr>
<td>`deploy_capital`</td>
<td>null</td>
<td>required</td>
</tr>
<tr>
<td>`swap`</td>
<td>required</td>
<td>required</td>
</tr>
<tr>
<td>`rebalance`</td>
<td>required</td>
<td>required (same account, different ticker)</td>
</tr>
<tr>
<td>`account_optimize`</td>
<td>null</td>
<td>required (new capital only)</td>
</tr>
<tr>
<td>`take_profit`</td>
<td>required</td>
<td>optional</td>
</tr>
<tr>
<td>`tax_loss_harvest`</td>
<td>required (Trading account)</td>
<td>optional</td>
</tr>
</table>

---

# Pending Propagations

- **Phase 1.5**: add `prior_recommendations_memory` (last 4 weeks with execution status) to prevent flip-flop after observing real flip-flop behavior in production.
- **Agent Run Portfolio**: maps this schema to its `agent_portfolio_decisions` table. `action_type` → `decision_type` enum: `swap → swap`, `rebalance → rebalance`, `deploy_capital → capital_deploy`, `tax_loss_harvest → tax_harvest`, `account_optimize → outlook_driven`, `take_profit → outlook_driven`.
- **Opus escalation**: ship Sonnet-only at v1. Add Opus escalation once `available_capital > $10,000` or `≥ 2 high-priority actions` patterns emerge.

---

# Orchestrator Input Assembly

```python
def build_action_synthesizer_input(
    user_id: str,
    health_assessor_output: dict,
    opportunity_ranker_output: dict,
    account_state: AccountState,
    canadian_tax_rules: str,           # static knowledge base, cached prefix
    user_constraints: UserConstraints,
    trigger_mode: TriggerMode,
    available_capital: Decimal | None,
) -> str:
    capital_line = (
        f"\nAvailable capital to deploy: {available_capital} CAD"
        if available_capital is not None else ""
    )
    header = (
        f"User: {user_id} | Goals: {user_constraints.goals} | "
        f"Risk: {user_constraints.risk_tolerance} | "
        f"RRSP strategy: {user_constraints.rrsp_strategy}\n"
        f"Constraints: max_position {user_constraints.max_position_size_pct}% | "
        f"max_sector {user_constraints.max_sector_weight_pct}% | "
        f"avoid {user_constraints.sector_avoid_list}\n"
        f"Trigger: {trigger_mode}{capital_line}\n"
        f"Account state: "
        f"TFSA cash {account_state.tfsa.cash}, room {account_state.tfsa.contribution_room} | "
        f"RRSP cash {account_state.rrsp.cash}, room {account_state.rrsp.contribution_room} | "
        f"Trading cash {account_state.trading.cash}, "
        f"YTD gains {account_state.trading.ytd_realized_gains}\n"
        f"Superficial-loss cooldowns: {account_state.superficial_loss_cooldowns}\n"
    )
    ha_block = format_health_assessor_for_synth(health_assessor_output)
    or_block = format_opportunity_ranker_for_synth(opportunity_ranker_output)
    # canadian_tax_rules is the cached prefix — place it adjacent to the rules block
    return "\n\n".join([header, canadian_tax_rules, ha_block, or_block])

def format_health_assessor_for_synth(ha: dict) -> str:
    """Compress HA output for synthesis. Action Synthesizer reads everything HA produces."""
    return f"""--- HEALTH ASSESSOR OUTPUT ---
portfolio_health_rating: {ha['portfolio_health_rating']}
weighted_outlook_score: {ha['weighted_outlook_score']}
holdings_ranked: {format_holdings_ranked_full(ha['holdings_ranked'])}
concentration_risks: {format_concentration_risks_compact(ha['concentration_risks'])}
misplaced_holdings: {format_misplaced_compact(ha['misplaced_holdings'])}
dividend_at_risk: {ha['dividend_income_assessment'].get('at_risk_dividends', [])}
top_issues: {format_top_issues_full(ha['top_issues'])}"""

def format_opportunity_ranker_for_synth(or_out: dict) -> str:
    return f"""--- OPPORTUNITY RANKER OUTPUT ---
opportunity_count: {or_out['opportunity_count']}
opportunities_ranked: {format_opportunities_full(or_out['opportunities_ranked'])}
no_action_rationale: {or_out.get('no_action_rationale')}
narrative: {or_out['narrative']}"""
```

---

# Orchestrator Merge Logic

```python
def merge_output_action_synthesizer(
    llm_response: dict,
    health_assessor_output: dict,
    opportunity_ranker_output: dict,
    trigger_mode: TriggerMode,
) -> dict:
    for forbidden in ("buy_sell_hold", "BUY", "SELL", "HOLD", "execute"):
        llm_response.pop(forbidden, None)

    # Enforce overall_assessment consistency with the action set
    llm_response["overall_assessment"] = reconcile_overall_assessment(
        llm_response.get("suggested_actions", []),
        llm_response["overall_assessment"],
    )

    # Capital deployment null when trigger is not available_to_trade
    if trigger_mode != "available_to_trade":
        llm_response["capital_deployment"] = None

    return {
        "agent_name": "action_synthesizer",
        "agent_pass": "portfolio_optimization",
        "optimization_summary": llm_response["optimization_summary"],
        "overall_assessment": llm_response["overall_assessment"],
        "suggested_actions": llm_response["suggested_actions"],
        "capital_deployment": llm_response.get("capital_deployment"),
        "no_action_holdings": llm_response["no_action_holdings"],
        "warnings": llm_response["warnings"],
        "narrative": llm_response["narrative"],
        # Audit-trail linkages
        "source_health_rating": health_assessor_output["portfolio_health_rating"],
        "source_opportunity_count": opportunity_ranker_output["opportunity_count"],
        "trigger_mode_used": trigger_mode,
    }

def reconcile_overall_assessment(
    actions: list[dict],
    llm_assessment: str,
) -> str:
    """Force consistency between action set and overall_assessment per validation rule 8."""
    if not actions:
        return "well_positioned"
    if len(actions) == 1 and actions[0]["priority"] == "low":
        return "well_positioned"
    if any(a["priority"] == "high" for a in actions):
        return "action_needed"
    return "minor_improvements"
```

---

# Token Budget

<table header-row="true">
<tr>
<td>Component</td>
<td>Budget</td>
<td>Notes</td>
</tr>
<tr>
<td>System prompt + seven critical rules</td>
<td>~2,500 tokens</td>
<td>Cached as part of stable prefix</td>
</tr>
<tr>
<td>`canadian_tax_rules` knowledge base</td>
<td>~1,600 tokens</td>
<td>Cached prefix (shared with Tax Strategist)</td>
</tr>
<tr>
<td>Health Assessor output</td>
<td>~1,200 tokens</td>
<td>Full output (holdings_ranked, top_issues, concentration, misplaced)</td>
</tr>
<tr>
<td>Opportunity Ranker output</td>
<td>~1,400 tokens</td>
<td>Up to 5 opportunities + narrative + no_action_rationale</td>
</tr>
<tr>
<td>Account state + header</td>
<td>~400 tokens</td>
<td>Cash, room, YTD gains, superficial-loss cooldowns</td>
</tr>
<tr>
<td>Expected output</td>
<td>~1,800 tokens</td>
<td>Up to 4 actions + capital_deployment + narrative + warnings</td>
</tr>
<tr>
<td>**Total**</td>
<td>**~8,900 tokens**</td>
<td>Well within Sonnet 4.6 limits; ~4,100 tokens of stable prefix cached</td>
</tr>
</table>

**Model configuration**:
- Primary: `claude-sonnet-4-6`. **No local fallback.**
- Escalation: `claude-opus-4-6` when `available_capital > $10,000` OR `≥ 2 high-priority actions`
- `max_tokens`: 2,200 | `temperature`: 0.2 | Thinking mode: OFF
- Prompt caching: stable prefix (system prompt + seven critical rules + `canadian_tax_rules`) ~4,100 tokens cached per deploy
- Expected latency: ~9–13s Sonnet; ~14–22s Opus escalation
- Cost per run: ~$0.025–0.07 Sonnet; ~$0.10–0.22 Opus escalation

---

# Implementation Files

- `src/agents/action_synthesizer.py` — agent implementation
- `src/agents/action_synth_formatter.py` — `build_action_synthesizer_input()` and per-input formatters
- `src/agents/action_synth_validator.py` — the 12 validation rules; this file is the safety net
- `src/orchestrator/portfolio_optimizer.py` — `run_action_synthesizer()`, `merge_output_action_synthesizer()`, `reconcile_overall_assessment()`
- `src/knowledge/canadian_tax_rules.md` — static knowledge base (shared with Tax Strategist)

---

_Source: Notion — Project Stock Picker → Agent Prompts → Action Synthesizer Agent Prompt. Downloaded 2026-07-02._
