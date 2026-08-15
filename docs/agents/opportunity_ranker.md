# Opportunity Ranker Agent Prompt

**Agent**: Opportunity Ranker
**Status**: Draft v1.0
**Phase**: Phase 1 (Milestone 2 — single-agent; promoted to 3-agent chain in Milestone 3)
**Pass**: Portfolio Optimization (Sub-agent 2 of 3 — runs after Health Assessor)
**Role**: Evaluate watchlist stocks as potential portfolio additions. Rank them by outlook + portfolio fit + tax efficiency + displacement value, and identify how each would be funded. Internal-only — output is consumed by the Action Synthesizer.
**Target Model**: `qwen3.6:35b-a3b` via Ollama (local primary). `claude-sonnet-4-6` cloud fallback. Thinking mode OFF (verify `enable_thinking:false` fully suppresses reasoning on this model — see Agent Design Patterns Pattern 10 caveat).

---

# Design Decisions

1. **Consumes Health Assessor output directly.** Displacement candidates (`position_assessment ∈ {underperforming, consider_replacing}`) and top issues come from the prior sub-agent — the Ranker doesn't re-derive them.
2. **Watchlist outlook is inherited from the CIO.** Same rule as the Health Assessor: read `stock_outlook`, `confidence`, `expected_return_tier` verbatim.
3. **Pre-computed cost estimates are inputs, not LLM math.** Transaction costs (FX, commissions, bid/ask spreads) and tax impact (capital gains on Trading-account sells) are computed before the agent runs and passed in. The LLM evaluates the 2% improvement threshold but doesn't compute the underlying numbers.
4. **"No watchlist stocks materially improve the portfolio" is a valid output.** Top-5 cap; if fewer materially help, return fewer (or zero).
5. **Local-primary by design.** Ranking at this scale fits `qwen3.6:35b-a3b`.

---

# Inputs

<table header-row="true">
<tr>
<td>Input</td>
<td>Purpose</td>
</tr>
<tr>
<td>`watchlist_summary_matrix` — same shape as the holdings matrix, applied to watchlist stocks.</td>
<td>Canonical view of each candidate.</td>
</tr>
<tr>
<td>`health_assessor_output` — compressed to: `holdings_ranked` (displacement candidates only), `top_issues`, `concentration_risks`, `misplaced_holdings`.</td>
<td>Diagnostic context. Where are the gaps?</td>
</tr>
<tr>
<td>`correlation_matrix_with_watchlist` — watchlist × holdings pairwise correlations.</td>
<td>Portfolio-fit scoring.</td>
</tr>
<tr>
<td>`account_state` — per-account cash, contribution room, YTD realized gains.</td>
<td>Funding feasibility.</td>
</tr>
<tr>
<td>`cost_estimates` — pre-computed per candidate: `transaction_cost_pct`, `tax_impact_pct_if_displacing` for each potential displacement target.</td>
<td>Inputs to the 2% threshold check. LLM doesn't compute these.</td>
</tr>
<tr>
<td>`user_constraints` — risk_tolerance, max_position_size_pct, max_sector_weight_pct, sector_avoid_list, goals.</td>
<td>Filter candidates that would breach limits.</td>
</tr>
</table>

---

# System Prompt

**Runtime prompt:** see [`prompts/opportunity_ranker/v1.0.txt`](../../prompts/opportunity_ranker/v1.0.txt)

---

# Output Schema

```json
{
  "opportunities_ranked": [
    {
      "rank": 1,
      "ticker": "string",
      "sector": "string",
      "outlook": "bullish | somewhat_bullish | neutral",
      "confidence": 0,
      "expected_return_tier": "strong_outperform | outperform | market_perform | underperform | strong_underperform",
      "opportunity_score": 0,
      "portfolio_fit": "strong | good | neutral | poor",
      "portfolio_fit_rationale": "string (1 sentence)",
      "optimal_account": "tfsa | rrsp | trading",
      "account_room_available": true,
      "funding_method": "available_cash | displacement | rebalance | new_capital",
      "displacement_target": {
        "ticker": "string",
        "account": "tfsa | rrsp | trading",
        "expected_improvement_pct_after_costs": 0.0
      },
      "transaction_cost_impact": "string (e.g., 'TFSA sell removes $4,200 contribution room until Jan 2027')",
      "recommended_amount": "string (e.g., '3-5% of portfolio, ~$3,000-5,000 in RRSP')",
      "rationale": "string (2 sentences for the Action Synthesizer)"
    }
  ],
  "no_action_rationale": "string | null (only when opportunities_ranked is empty)",
  "narrative": "string (150-250 words for the Action Synthesizer)"
}
```

---

# Validation Rules

1. JSON parseable; no `recommendation` / `buy_sell_hold` fields.
2. `opportunities_ranked` length ≤ 5; ranked descending by `opportunity_score`.
3. `outlook` and `confidence` per opportunity match the CIO output exactly.
4. `funding_method = displacement` requires `displacement_target` populated AND `expected_improvement_pct_after_costs ≥ 2.0`.
5. `funding_method = new_capital` valid only when trigger is `available_to_trade`.
6. No opportunity in `opportunities_ranked` has outlook ∈ \{somewhat_bearish, bearish\}.
7. If `opportunities_ranked = []`, `no_action_rationale` is non-empty (≥ 40 chars).
8. No "withdraw from RRSP" / "RRSP withdrawal" string in any field.
9. Numeric tokens in narrative + rationale fields trace to inputs (especially `cost_estimates`).
10. `narrative` length 150–250 words.

---

# Funding Method → Action Type Mapping

The Action Synthesizer reads these directly:

<table header-row="true">
<tr>
<td>Opportunity Ranker `funding_method`</td>
<td>Action Synthesizer `action_type`</td>
</tr>
<tr>
<td>`available_cash`</td>
<td>`deploy_capital`</td>
</tr>
<tr>
<td>`displacement`</td>
<td>`swap`</td>
</tr>
<tr>
<td>`rebalance`</td>
<td>`rebalance`</td>
</tr>
<tr>
<td>`new_capital`</td>
<td>`deploy_capital` (only when trigger is `available_to_trade`)</td>
</tr>
</table>

---

# Pending Propagations

- When Phase 2 ships FUND-anchored price scenarios, `expected_improvement_pct_after_costs` becomes a scenario-weighted expectation rather than a tier-difference heuristic.
- At >40 combined watchlist+holdings stocks, plan a pre-filter step before LLM invocation.

---

# Orchestrator Input Assembly

```python
def build_opportunity_ranker_input(
    user_id: str,
    watchlist: list[WatchlistEntry],
    cio_outputs: dict[str, CIOOutput],
    health_assessor_output: dict,
    correlation_matrix: CorrelationMatrix,
    account_state: AccountState,
    cost_estimates: CostEstimates,
    user_constraints: UserConstraints,
    trigger_mode: TriggerMode,
    available_capital: Decimal | None = None,
) -> str:
    capital_line = (
        f"\nAvailable capital to deploy: {available_capital} CAD"
        if trigger_mode == "available_to_trade" else ""
    )
    header = (
        f"User: {user_id} | Goals: {user_constraints.goals} | "
        f"Risk: {user_constraints.risk_tolerance} | "
        f"Dividend focus: {user_constraints.dividend_focus_level}\n"
        f"Constraints: max_position {user_constraints.max_position_size_pct}% | "
        f"max_sector {user_constraints.max_sector_weight_pct}% | "
        f"avoid {user_constraints.sector_avoid_list}\n"
        f"Trigger: {trigger_mode}{capital_line}\n"
    )
    matrix_block = format_watchlist_summary_matrix(watchlist, cio_outputs)
    ha_block = format_health_assessor_output(health_assessor_output)
    correlation_block = format_correlation_matrix_with_watchlist(correlation_matrix)
    account_block = format_account_state(account_state)
    cost_block = format_cost_estimates(cost_estimates)
    constraints_block = format_user_constraints(user_constraints)
    return "\n\n".join([
        header, matrix_block, ha_block,
        correlation_block, account_block, cost_block, constraints_block,
    ])

def format_watchlist_summary_matrix(
    watchlist: list[WatchlistEntry],
    cio_outputs: dict[str, CIOOutput],
) -> str:
    lines = ["--- WATCHLIST SUMMARY MATRIX (CIO outputs, most recent per stock) ---"]
    for w in watchlist:
        cio = cio_outputs.get(w.ticker)
        if cio is None:
            lines.append(f"{w.ticker}: NO CIO OUTPUT — exclude from ranking")
            continue
        cross = cio.tax_summary.get("cross_account_note", "")
        lines.append(
            f"{w.ticker} ({w.sector}) | outlook {cio.stock_outlook} | "
            f"confidence {cio.confidence} | tier {cio.expected_return_tier}\n"
            f"  thesis: {cio.thesis_summary}\n"
            f"  tax: {cio.tax_summary['tax_efficiency_consumed_as']} | cross: {cross}"
        )
    return "\n".join(lines)

def format_health_assessor_output(ha: dict) -> str:
    """Compress HA output to the fields the Ranker actually uses."""
    return f"""--- HEALTH ASSESSOR OUTPUT (Sub-agent 1) ---
portfolio_health_rating: {ha['portfolio_health_rating']}
weighted_outlook_score: {ha['weighted_outlook_score']}
displacement candidates (position_assessment ∈ underperforming/consider_replacing):
{format_displacement_candidates(ha['holdings_ranked'])}
top_issues: {format_top_issues_compact(ha['top_issues'])}
concentration_risks: {format_concentration_risks_compact(ha['concentration_risks'])}
diversification verdict: {ha['diversification_notes']['verdict']}
misplaced_holdings: {format_misplaced_compact(ha['misplaced_holdings'])}"""

def format_cost_estimates(cost_estimates: CostEstimates) -> str:
    """Per-candidate transaction costs and tax-impact estimates."""
    lines = ["--- COST ESTIMATES (pre-computed, use as given) ---"]
    for ce in cost_estimates.entries:
        lines.append(
            f"{ce.candidate_ticker} → {ce.displacement_target or 'no displacement'}: "
            f"transaction_cost_pct {ce.transaction_cost_pct:.2f}% | "
            f"tax_impact_pct {ce.tax_impact_pct_if_displacing:.2f}%"
        )
    return "\n".join(lines)
```

---

# Orchestrator Merge Logic

```python
def merge_output_opportunity_ranker(
    llm_response: dict,
    watchlist: list[WatchlistEntry],
) -> dict:
    for forbidden in ("recommendation", "buy_sell_hold", "execute"):
        llm_response.pop(forbidden, None)

    # Enforce top-5 cap
    llm_response["opportunities_ranked"] = llm_response["opportunities_ranked"][:5]

    return {
        "agent_name": "opportunity_ranker",
        "agent_pass": "portfolio_optimization",
        "opportunities_ranked": llm_response["opportunities_ranked"],
        "no_action_rationale": llm_response["no_action_rationale"],
        "narrative": llm_response["narrative"],
        "watchlist_size": len(watchlist),
        "opportunity_count": len(llm_response["opportunities_ranked"]),
    }
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
<td>System prompt</td>
<td>~1,800 tokens</td>
<td>CoT + 7 rules + schema + funding mapping</td>
</tr>
<tr>
<td>Watchlist summary matrix</td>
<td>~2,000 tokens</td>
<td>~150 × ~13 watchlist stocks (steady state)</td>
</tr>
<tr>
<td>Health Assessor output (compressed)</td>
<td>~1,000 tokens</td>
<td>Displacement candidates + top_issues + concentration + misplaced</td>
</tr>
<tr>
<td>Correlation + accounts + cost_estimates</td>
<td>~1,400 tokens</td>
<td>Cost estimates ~250 tokens; correlation pairs O(N×M)</td>
</tr>
<tr>
<td>User constraints + header</td>
<td>~200 tokens</td>
<td></td>
</tr>
<tr>
<td>Expected output</td>
<td>~1,200 tokens</td>
<td>Up to 5 opportunities + narrative</td>
</tr>
<tr>
<td>**Total**</td>
<td>**~7,600 tokens**</td>
<td>Within `qwen3.6:35b-a3b`'s reliable budget; Sonnet fallback well within limits</td>
</tr>
</table>

**Model configuration**:
- Primary: `qwen3.6:35b-a3b` via Ollama
- Fallback: `claude-sonnet-4-6` (cloud) when local unavailable OR combined watchlist+holdings > 40
- `max_tokens`: 1,800 | `temperature`: 0.2 | Thinking mode: OFF
- Expected latency: ~36–48s local (unverified estimate, re-measure after deployment); ~6–9s cloud
- Cost per run: $0 local; ~$0.025 cloud fallback

---

# Implementation Files

- `src/agents/opportunity_ranker.py` — agent implementation
- `src/agents/opportunity_ranker_formatter.py` — input assembly + per-input formatters
- `src/orchestrator/portfolio_optimizer.py` — `run_opportunity_ranker()`, `merge_output_opportunity_ranker()`
- `src/precompute/cost_estimates.py` — produces `cost_estimates` input (transaction costs + tax-impact estimates per candidate/displacement pair)

---

_Source: Notion — Project Stock Picker → Agent Prompts → Opportunity Ranker Agent Prompt. Downloaded 2026-07-02._
