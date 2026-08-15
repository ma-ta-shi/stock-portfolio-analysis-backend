# Portfolio Health Assessor Agent Prompt

**Agent**: Portfolio Health Assessor
**Status**: Draft v1.0
**Phase**: Phase 1 (Milestone 2 — single-agent; promoted to 3-agent chain in Milestone 3)
**Pass**: Portfolio Optimization (Sub-agent 1 of 3)
**Role**: Diagnose the current portfolio. Rank holdings, identify concentration and placement issues, and surface up to 5 top issues to address. Internal-only — its output is consumed by the Opportunity Ranker and Action Synthesizer.
**Target Model**: `qwen3.6:35b-a3b` via Ollama (local primary). `claude-sonnet-4-6` cloud fallback when local is unavailable. Thinking mode OFF for JSON stability (verify `enable_thinking:false` fully suppresses reasoning on this model — see Agent Design Patterns Pattern 10 caveat).

---

# Design Decisions

1. **Diagnosis only, no actions.** The Health Assessor labels holdings (`core_holding | solid | monitor | underperforming | consider_replacing`) and lists top issues. The Action Synthesizer turns these into actions.
2. **No original stock research.** Each holding's outlook is read verbatim from the most recent CIO output. Never re-rated.
3. **Pre-computation does the math.** Sector weights, position weights, correlations, and account totals are computed before the LLM runs and passed in as facts. The LLM reasons; it does not arithmetic.
4. **Misplaced ≠ weak.** A bullish holding in the wrong account is `misplaced` (action: place future contributions in the optimal account), not `underperforming`. The two go in separate output blocks.
5. **Local-primary by design.** Ranking and structured diagnosis at this scale fits `qwen3.6:35b-a3b` reliably. Cloud fallback exists for portfolios that exceed local context.

---

# Inputs

<table header-row="true">
<tr>
<td>Input</td>
<td>Purpose</td>
</tr>
<tr>
<td>`per_stock_summary_matrix` — one row per holding from the most recent CIO output. Fields: `ticker, account, stock_outlook, confidence, expected_return_tier, position_sizing_recommendation, thesis_summary, tax_efficiency_for_account, cross_account_recommendation, dividend_yield_pct`.</td>
<td>Canonical view of each holding. Use verbatim.</td>
</tr>
<tr>
<td>`portfolio_metrics` — pre-computed: `weighted_outlook_score, sector_weights{}, position_weights{}, account_value{tfsa,rrsp,trading}, total_value`.</td>
<td>Concentration facts.</td>
</tr>
<tr>
<td>`correlation_matrix` — pairwise 1Y daily-return correlations between holdings.</td>
<td>Diversification check; flag pairs with `|r| > 0.7`.</td>
</tr>
<tr>
<td>`account_state` — per-account cash, contribution room (TFSA/RRSP), YTD realized gains (Trading), book value per position.</td>
<td>Tax context.</td>
</tr>
<tr>
<td>`user_constraints` — risk_tolerance, max_position_size_pct, max_sector_weight_pct, sector_avoid_list, goals, dividend_focus_level ∈ \{primary, secondary, none\}.</td>
<td>Thresholds to compare against.</td>
</tr>
</table>

All numbers in the output trace to these inputs. No fabrication.

Outlook scale: `bullish | somewhat_bullish | neutral | somewhat_bearish | bearish`. Account types: `tfsa | rrsp | trading`.

---

# System Prompt

**Runtime prompt:** see [`prompts/portfolio_health_assessor/v1.0.txt`](../../prompts/portfolio_health_assessor/v1.0.txt)

---

# Output Schema

```json
{
  "portfolio_health_rating": "strong | adequate | needs_attention | weak",
  "holdings_ranked": [
    {
      "ticker": "string",
      "account": "tfsa | rrsp | trading",
      "outlook": "bullish | somewhat_bullish | neutral | somewhat_bearish | bearish",
      "confidence": 0,
      "position_weight_pct": 0.0,
      "position_assessment": "core_holding | solid | monitor | underperforming | consider_replacing",
      "issues": ["string"]
    }
  ],
  "concentration_risks": [
    {
      "type": "sector | single_position | account",
      "detail": "string (must cite specific numbers from portfolio_metrics)",
      "severity": "high | medium | low"
    }
  ],
  "diversification_notes": {
    "correlation_concerns": [
      { "ticker_a": "string", "ticker_b": "string", "correlation": 0.0 }
    ],
    "geographic_split": { "canada_pct": 0.0, "us_pct": 0.0, "other_pct": 0.0 },
    "verdict": "well_diversified | acceptable | concentrated"
  },
  "misplaced_holdings": [
    {
      "ticker": "string",
      "current_account": "tfsa | rrsp | trading",
      "optimal_account": "tfsa | rrsp | trading",
      "annual_tax_cost": "string",
      "action_scope": "new_capital_only"
    }
  ],
  "dividend_income_assessment": {
    "applicable": true,
    "total_annual_income_estimate": "string",
    "at_risk_dividends": [{ "ticker": "string", "risk": "string" }],
    "withholding_tax_drag": "string"
  },
  "top_issues": [
    {
      "rank": 1,
      "issue": "string",
      "impact": "string (concrete)",
      "suggested_action_type": "swap | rebalance | account_optimize | reduce | none",
      "affected_tickers": ["string"]
    }
  ],
  "narrative": "string (150-250 words for the downstream Action Synthesizer)"
}
```

---

# Validation Rules

1. JSON parseable; no `recommendation` / `buy_sell_hold` fields.
2. `holdings_ranked` length equals input holding count; tickers match.
3. `outlook` and `confidence` per holding match the CIO output exactly (no re-rating).
4. `position_assessment` ∈ enum; `suggested_action_type` ∈ \{swap, rebalance, account_optimize, reduce, none\}; `severity` ∈ enum.
5. `top_issues` length ≤ 5.
6. `misplaced_holdings[].action_scope = "new_capital_only"` (hardcoded).
7. No "withdraw from RRSP" / "RRSP withdrawal" string in any field.
8. Numeric tokens in `narrative` and `concentration_risks[].detail` appear in the input data.
9. `narrative` length 150–250 words.
10. If `dividend_focus_level = none`, `dividend_income_assessment.applicable = false`.

---

# Pending Propagations

- Action enum changes (`position_assessment`, `suggested_action_type`) propagate to the Opportunity Ranker and Action Synthesizer.
- When Phase 2 ships Black-Litterman pre-computation, add a `theoretical_weights` reference to `portfolio_metrics` and surface deviation as a new `concentration_risks.type`.

---

# Orchestrator Input Assembly

```python
def build_health_assessor_input(
    user_id: str,
    holdings: list[Holding],
    cio_outputs: dict[str, CIOOutput],
    portfolio_metrics: PortfolioMetrics,
    correlation_matrix: CorrelationMatrix,
    account_state: AccountState,
    user_constraints: UserConstraints,
) -> str:
    header = (
        f"User: {user_id} | Goals: {user_constraints.goals} | "
        f"Risk: {user_constraints.risk_tolerance} | "
        f"Dividend focus: {user_constraints.dividend_focus_level}\n"
        f"Constraints: max_position {user_constraints.max_position_size_pct}% | "
        f"max_sector {user_constraints.max_sector_weight_pct}% | "
        f"avoid {user_constraints.sector_avoid_list}\n"
        f"Total value: {portfolio_metrics.total_value} | "
        f"TFSA: {portfolio_metrics.account_value.tfsa} | "
        f"RRSP: {portfolio_metrics.account_value.rrsp} | "
        f"Trading: {portfolio_metrics.account_value.trading}\n"
    )
    matrix_block = format_per_stock_summary_matrix(holdings, cio_outputs)
    metrics_block = format_portfolio_metrics(portfolio_metrics)
    correlation_block = format_correlation_matrix(correlation_matrix)
    account_block = format_account_state(account_state)
    constraints_block = format_user_constraints(user_constraints)
    return "\n\n".join([
        header, matrix_block, metrics_block,
        correlation_block, account_block, constraints_block,
    ])

def format_per_stock_summary_matrix(
    holdings: list[Holding],
    cio_outputs: dict[str, CIOOutput],
) -> str:
    """One row per holding. ~150 tokens per stock."""
    lines = ["--- PER-STOCK SUMMARY MATRIX (CIO outputs, most recent per holding) ---"]
    for h in holdings:
        cio = cio_outputs.get(h.ticker)
        if cio is None:
            lines.append(f"{h.ticker} ({h.account}): NO CIO OUTPUT — skip")
            continue
        cross = cio.tax_summary.get("cross_account_note", "")
        lines.append(
            f"{h.ticker} ({h.account}) | weight {h.position_weight_pct}% | "
            f"outlook {cio.stock_outlook} | confidence {cio.confidence} | "
            f"tier {cio.expected_return_tier}\n"
            f"  thesis: {cio.thesis_summary}\n"
            f"  tax: {cio.tax_summary['tax_efficiency_consumed_as']} | cross: {cross}\n"
            f"  dividend_yield_pct: {cio.dividend_yield_pct or 'n/a'}"
        )
    return "\n".join(lines)
```

---

# Orchestrator Merge Logic

```python
def merge_output_health_assessor(
    llm_response: dict,
    portfolio_metrics: PortfolioMetrics,
    holdings: list[Holding],
) -> dict:
    for forbidden in ("recommendation", "buy_sell_hold", "action_plan"):
        llm_response.pop(forbidden, None)

    return {
        "agent_name": "portfolio_health_assessor",
        "agent_pass": "portfolio_optimization",
        "portfolio_health_rating": llm_response["portfolio_health_rating"],
        "holdings_ranked": llm_response["holdings_ranked"],
        "concentration_risks": llm_response["concentration_risks"],
        "diversification_notes": llm_response["diversification_notes"],
        "misplaced_holdings": llm_response["misplaced_holdings"],
        "dividend_income_assessment": llm_response["dividend_income_assessment"],
        "top_issues": llm_response["top_issues"],
        "narrative": llm_response["narrative"],
        "weighted_outlook_score": portfolio_metrics.weighted_outlook_score,
        "holding_count": len(holdings),
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
<td>CoT + 7 rules + schema</td>
</tr>
<tr>
<td>Per-stock summary matrix</td>
<td>~2,000 tokens</td>
<td>~150 × ~13 holdings (steady state)</td>
</tr>
<tr>
<td>Portfolio metrics + correlation + accounts</td>
<td>~1,000 tokens</td>
<td>Pre-computed structured data</td>
</tr>
<tr>
<td>User constraints + header</td>
<td>~200 tokens</td>
<td></td>
</tr>
<tr>
<td>Expected output</td>
<td>~1,000 tokens</td>
<td>Holdings_ranked + concentration + dividend + top_issues + narrative</td>
</tr>
<tr>
<td>**Total**</td>
<td>**~6,000 tokens**</td>
<td>Comfortably within `qwen3.6:35b-a3b`'s reliable context</td>
</tr>
</table>

**Model configuration**:
- Primary: `qwen3.6:35b-a3b` via Ollama
- Fallback: `claude-sonnet-4-6` (cloud) when local unavailable
- `max_tokens`: 1,500 | `temperature`: 0.2 | Thinking mode: OFF
- Expected latency: ~30–42s local (unverified estimate, re-measure after deployment); ~5–8s cloud
- Cost per run: $0 local; ~$0.015 cloud fallback

---

# Implementation Files

- `src/agents/portfolio_health_assessor.py` — agent implementation
- `src/agents/health_assessor_formatter.py` — `build_health_assessor_input()` and per-input formatters
- `src/orchestrator/portfolio_optimizer.py` — `run_health_assessor()`, `merge_output_health_assessor()`
- `src/precompute/portfolio.py` — produces `portfolio_metrics` and `correlation_matrix`

---

_Source: Notion — Project Stock Picker → Agent Prompts → Portfolio Health Assessor Agent Prompt. Downloaded 2026-07-02._
