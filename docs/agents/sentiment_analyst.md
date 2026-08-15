# Sentiment Analyst Agent Prompt
**Agent**: Sentiment Analyst
**Status:** Draft v1.2 — retail-Canadian simplification pass
**Phase:** Phase 1
**Pass**: Pass 1 (Research & Analysis)
**Role**: Gauge market mood, narrative momentum, and crowded positioning — what the market “feels” about this stock beyond the numbers.
**Prompt Version**: v1.2
**Target Model**: `qwen3.6:35b-a3b` via Ollama (non-thinking mode — verify `enable_thinking:false` fully suppresses reasoning on this model; see Agent Design Patterns Pattern 10 caveat)
**Audience**: Canadian retail investors with total portfolio values under \$250k CAD, holding stocks across TFSA / RRSP / Trading / General accounts. Institutional-grade complexity that doesn’t pay rent at this scale has been trimmed.
**Token Budget**: \~430 system prompt \| \~1,630 data input \| \~580 expected output \| \~2,640 total
---
# Design Decisions & Rationale
Built from the Sentiment Analyst design notes (previously on this page, now superseded). Follows validated [Agent Design Patterns](https://www.notion.so/Agent-Design-Patterns-33fea067982380e6a055d3ed74c8eada?pvs=21) 1, 2, 3, 4, 7, 9, 10, and 12 directly. Reference implementations: [Fundamental Analyst v4.1](https://www.notion.so/Fundamental-Analyst-Agent-Prompt-333ea06798238063a42bed3e2c132ea7?pvs=21) (hybrid orchestrator/LLM split for quantitative inputs) and [Stock Researcher v1.2](https://www.notion.so/Stock-Researcher-Agent-Prompt-333ea06798238081bb20ff087cc7e76a?pvs=21) (verifiable grounding for qualitative inputs). The decisions below are the ones where Sentiment *adapts* or *departs* from the reference implementations; all other patterns apply as-is.
## 1. 3-phase reasoning frame (departure from Pattern 6 step-by-step CoT)
The agent does not narrate a step-by-step trace. The prompt asks for three cognitively distinct phases: (a) **Per-signal interpretation** — interpret news, analyst, insider, peer sentiment, short interest individually, filling the structured schema fields; (b) **Cross-signal pattern detection** — look for contrarian archetypes across the just-interpreted signals; (c) **Synthesis** — produce a 120–180 word narrative about the resulting story.
**Why depart from Fundamental/Researcher’s step-by-step CoT:** A 7- or 9-step trace forces the narrative to do double duty (reasoning trace + synthesis) within a fixed token budget. With 9 analytical dimensions and a 150-word budget, that’s \~17 words per step — sentence fragments, not analysis. The structured schema already enforces coverage (every dimension has a required field); the step-by-step prompt instruction is redundant overhead. The 3-phase frame matches what the analytical work actually is — three distinct cognitive operations — and the narrative becomes about findings, not process.
**Audit trail consequence:** To debug “why did the model rate sentiment as positive when news was mixed,” read the structured fields (`news_sentiment.overall`, `dominant_themes`, `contrarian_signals`), not a prose trace. The audit trail moves from prose to schema. This is a deliberate trade.
## 2. Social sentiment is “unknown” at MVP — orchestrator-owned, no validation enforcement
Orchestrator hard-codes `social_sentiment.buzz_level = "unknown"` and `social_sentiment.sentiment = "unknown"`. The system prompt simply says "the orchestrator handles social sentiment; ignore it." If the LLM produces a `social_sentiment` block anyway, the orchestrator silently overwrites it during merge — no retry, no validation rule, no token tax.
## 3. Canadian sentiment-inference cap matches Researcher (70)
When `canadian_sentiment_inferred = true`, the orchestrator caps `base_reliability_score` at 70 — same as Researcher's Canadian cap. The underlying data source has changed: Canadian stocks now receive **LLM-scored** per-article sentiment (qwen3.6:35b-a3b, non-thinking mode) over FMP + GlobeNewswire/[Newswire.ca](http://Newswire.ca) RSS articles, rather than raw FMP headline inference. The cap is retained pending calibration data from 20+ scored predictions. The system prompt requires the templated caveat *“Canadian AI sentiment: scored by local LLM (qwen3.6:35b-a3b) rather than Finnhub — confidence scores are available but not independently validated against a ground-truth dataset.”*
## 4. No anonymization for v1.0–v1.2 — flagged for v1.3+ A/B test
Stock Researcher anonymizes to combat training-data leakage on filings/transcripts. Sentiment receives current news + transactions the LLM could not have memorized, so that specific leakage path is closed. However, the LLM has strong company-level priors that may color sentiment interpretation independent of input data. v1.0–v1.2 ship without anonymization on the assumption that current data dominates priors; v1.3+ will run an A/B test.
The Agent Design Patterns anonymization principle formalizes this: anonymize when agent inputs include raw or lightly-processed text from sources the model may have memorized (filings, transcripts). Sentiment’s inputs are current news headlines and structured analyst/insider/short-interest data — none qualify.
## 5. Free-form contrarian signals with two seeded archetypes (v1.2 — reduced from four)
The `contrarian_signals.signal` field is a free-form string. v1.2 seeds two archetypes as patterns to look for (`washed_out`, `insider_news_divergence`) but does not constrain to them. v1.0/v1.1 also seeded `crowded_long` and `analyst_short_divergence`; both are dropped in v1.2:
- **`crowded_long`**** dropped**: requires the social sentiment overlay that is hard-coded `unknown` at MVP. Without it, the archetype reduces to “all of analyst + news + insider are positive at the same time” — a positive-signal description, not a contrarian flag.
- **`analyst_short_divergence`**** dropped**: requires reliable short-interest data, which is “limited availability for TSX names” per the Canadian asymmetry table. The signal triggers almost exclusively on US large-caps where the divergence is well-known and already priced in — low marginal value for the target user.
A strict 2-value enum forces the model to either fabricate a fit or omit a real signal because it doesn’t have a name. Free-form preserves the option to flag other patterns when the model can name and evidence them.
## 6. Single news_id per dominant theme
`dominant_themes` is an array of `{theme, sentiment, primary_news_id}` — one anchor news_id per theme, not an array of ids per theme. Nested array-of-objects-with-array-of-strings is exactly the structure local 14B models break on. Single anchor matches Researcher’s `recent_developments.news_id` pattern, which is validated and works.
## 7. Insider transaction filtering scales with market cap and filters by transaction type
Threshold is `max(0.001% of market cap, $25K)`, applied only to Form 4 P-purchase open-market and S-sale open-market transactions. Option exercises (M), gifts (G), and other non-discretionary codes are excluded entirely from `notable_transactions` — they dominate small-dollar filings and aren’t real signal regardless of size.
## 8. Minimum data threshold: 5 articles OR 5 analysts with recent rating change
≥5 news articles in the timeline window OR ≥5 analysts with at least one rating change in the last 90 days. Either input alone supports a baseline sentiment read; both missing means there is no sentiment to analyze.
## 9. Memory decay matches Researcher’s thresholds
Memory included if \<30 days old at short_term, \<90 days at medium_term, \<180 days at long_term. Thresholds updated to match Researcher v1.3, which changed from 14/60/180 → 30/90/180 days. Match the validated Researcher logic until the feedback engine produces evidence for divergent thresholds.
## 10. Local-primary model, fixed assignment (updated 2026-07-01)
Model updated from `qwen3:14b` to `qwen3.6:35b-a3b` — outperforms it on MMLU-Pro/GPQA, the closest available proxies for this agent's interpretive-judgment task. Note: the automatic quality-triggered promotion-to-cloud described in the prior version of this decision is no longer planned; model assignment per agent is a fixed upfront choice. If sentiment-inference quality is poor on \>20% of Canadian stocks (measured by Pass 2 disagreement rate, Shadow CIO flag rate, or post-hoc accuracy comparison vs US stocks), that should prompt a deliberate re-evaluation of this agent's model assignment, not an automatic runtime escalation.
## 11. Globally-stable news IDs across agents (v1.1)
News IDs (`N{num}`) are assigned globally per stock per run by the Data Pipeline, not per-agent. The same news item carries the same `N{num}` identifier in Sentiment’s payload, Researcher’s payload, and any future Pass 1 agent that references news. IDs are stable across agent retries within the same run.
Consequence: if a news item is in Researcher’s 90-day medium-term window but outside Sentiment’s 14-day medium-term window, it has a `N{num}` in Researcher but does not appear in Sentiment at all — the ID is not reassigned. This means Sentiment’s payload may have non-contiguous IDs (e.g., `N7, N8, N11, N15`) when the global assignment order (oldest first) puts items outside Sentiment’s tighter window between them. This is intentional and does not affect Sentiment’s analysis.
Enables: the Feedback Analyst can cross-reference “Researcher and Sentiment both flagged N5 as material” without ID reconciliation.
## 12. Single-peer contrastive analysis (v1.2 — reduced from two peers)
The user message includes only PEER_1 in v1.2; v1.0/v1.1 included PEER_1 + PEER_2 with an optional PEER_3 block. For the target user (Canadian retail, \<\$250k portfolios), peer comparison answers a binary question — “is this stock-specific or sector-wide?” — that one peer answers adequately. The second peer added cross-validation value for institutional analysis but cost \~80 tokens for marginal decision value at this scale. Trade-off accepted: if PEER_1 happens to have idiosyncratic news in the same window, peer comparison is misleading; the orchestrator mitigates this by selecting the peer with the best news-coverage match for the window.
## 13. Short interest interpretation collapsed to three values (v1.2)
v1.0/v1.1 enum: `squeeze_setup | bearish_conviction | crowded_short | not_significant | insufficient_data`. v1.2 enum: `elevated_volatility_risk | normal | insufficient_data`.
Rationale: a retail investor with \$5k–\$25k positions has effectively zero exposure to short-squeeze dynamics. The relevant signal is “elevated volatility ahead” — fold all the high-short-interest variants into one. The Canadian asymmetry compounds it: short data is thin for \~half the TSX watchlist, so the high-resolution enum is unavailable in many cases anyway. The “squeeze setups get full weight” line from the trading-account instruction is removed in v1.2 for consistency.
## 14. `informed_vs_noise_summary` field removed (v1.2)
v1.0/v1.1 required the LLM to output a 1-sentence summary classifying signals as informed (analyst/insider/peer) vs noisy (headlines) and whether they agreed. v1.2 drops the field. The concept is sound but the value is fuzzy for retail (no direct action), it duplicates content already implicit in `contrarian_signals` (the divergence cases) and `narrative` (which synthesizes across signal types), and the per-field schema cost wasn’t justified. Insight folds into narrative naturally if material.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- News with sentiment — openbb-tmx `.news.company` (CA, no scores — see Gap 2) / Finnhub `/company-news` (US, scored)
- Analyst ratings — yfinance `recommendations_summary` (CA) / FMP `/rating` + Finnhub `/stock/recommendation` (US)
- Analyst price targets — yfinance `analyst_price_targets` (CA) / FMP `/analyst-estimates` (US)
- Insider trading — openbb-tmx `.equity.ownership.insider_trading` (CA, aggregates only — see Gap 3) / edgartools Form 4 `get_filings(form="4").obj().transactions` (US, transactional level)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- `base_reliability_score` — computed from news article count/source concentration, analyst coverage, insider data age, short-interest data age, and peer-block presence (capped at 70 if `canadian_sentiment_inferred` is true)
- Numeric passthrough fields — `article_count`, analyst distribution counts, `average_price_target`, `price_target_vs_current_pct`, recent upgrade/downgrade counts, `short_interest_pct`, `days_to_cover` — the LLM must not recompute or restate these
- Globally-stable per-run news IDs (`N{num}`), assigned once across all Pass 1 agents so the same article carries the same ID in Sentiment's and Researcher's payloads
- Hard-coded `social_sentiment` block (`buzz_level`/`sentiment` = `"unknown"` at MVP) — silently overwritten at merge if the LLM produces one
- Filtered `notable_transactions` — insider Form 4 purchases/sales scaled to market cap (`max(0.001% of market cap, $25K)`), with option exercises/gifts excluded
- Single best-match peer block (`PEER_1`) — orchestrator selects the peer with the best news-coverage overlap for the analysis window
- `canadian_sentiment_inferred` flag — drives the mandatory templated caveat about LLM-scored (qwen3.6:35b-a3b) vs. Finnhub-scored Canadian sentiment
- Data availability flags (`short_interest_available`, `missing_sources_list`) and `data_pipeline_warnings`
---
# System Prompt (v1.2)
Invoke with `enable_thinking: false`. The `build_messages()` method renders this template.

**Runtime prompt:** see [`prompts/sentiment_analyst/v1.2.txt`](../../prompts/sentiment_analyst/v1.2.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`). The data pipeline scopes the news window by timeline so payload contents and instructions stay aligned.
<table header-row="true">
<colgroup>
<col>
<col width="450">
<col>
</colgroup>
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
<td>News window</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1–4 wks). News sentiment and recent analyst rating changes drive the signal. Insider activity is secondary at this horizon.`</td>
<td>7 days</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1–12 mo). Weight analyst consensus trend and sustained news themes over individual headlines. Insider activity becomes meaningful.`</td>
<td>14 days</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1–5+ yrs). Insider buying is the strongest signal — insiders see the long horizon. Daily news is mostly noise; flag only structural narrative shifts.`</td>
<td>30 days</td>
</tr>
</table>
**Account** (inject based on `AnalysisContext.account_type`):
<table header-row="true">
<colgroup>
<col>
<col width="577">
</colgroup>
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Gains tax-free, no loss harvesting available. Tolerate sentiment-driven volatility for a strong long thesis.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. Long-horizon, no withdrawals. De-emphasize headline noise; weight insider activity and consensus stability.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading account. Capital gains 50% inclusion; superficial loss rule applies on rebuys within 30 days. Near-term sentiment shifts and contrarian setups are actionable.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General. Note which sentiment signals matter most for each account type.`</td>
</tr>
</table>
---
# Output Schema (v1.2 — LLM-produced fields only)
The orchestrator populates metadata, derived enum fields, all numeric passthrough fields, the hard-coded social_sentiment block, and orchestrator-computed insider transaction details.
```javascript
{
  "assessment_summary": "≤80 words. Most important sentiment finding first.",
  "reliability_score": 0-100,
  "analysis_confidence": "high|medium|low",
  "caveats": ["specific data gaps, anomalies, source-mix concerns, or data-asymmetry flags"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "positive|negative|neutral", "evidence": "citation + brief note"}
  ],
  "risks": [
    {"risk": "name", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "citation + brief note"}
  ],
  "narrative": "120-180 word synthesis. Reference specific citations.",
  "structured_data": {
    "news_sentiment": {
      "overall": "very_positive|positive|neutral|negative|very_negative",
      "dominant_themes": [
        {"theme": "short label", "sentiment": "positive|negative|neutral|mixed", "primary_news_id": "N1"}
      ],
      "sentiment_trend": "improving|stable|deteriorating"
    },
    "analyst_sentiment": {
      "consensus_trend": "improving|stable|deteriorating",
      "consensus_interpretation": "1 sentence on what the consensus + recent changes signal"
    },
    "insider_activity_interpretation": "1 sentence on net direction (buying/selling/neutral) and what it signals at this timeline. Reference a specific transaction only if materially outsized. 'insufficient_data' if no insider data.",
    "short_interest_interpretation": {
      "trend": "increasing|stable|decreasing|unknown",
      "interpretation": "elevated_volatility_risk|normal|insufficient_data"
    },
    "peer_sentiment_comparison": "1-2 sentences comparing this stock's sentiment to PEER_1. MUST reference PEER_1.",
    "contrarian_signals": [
      {"signal": "free-form name (see seeded archetypes in rule 5)", "evidence": "citation + brief note"}
    ]
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
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`key_factors`</td>
<td>2</td>
<td>3</td>
</tr>
<tr>
<td>`risks`</td>
<td>1</td>
<td>2</td>
</tr>
<tr>
<td>`dominant_themes`</td>
<td>2</td>
<td>3</td>
</tr>
<tr>
<td>`contrarian_signals`</td>
<td>0</td>
<td>2</td>
</tr>
</table>
`contrarian_signals` is allowed empty — most stocks have no contrarian signal worth flagging.
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name`, `agent_pass`, `analysis_context`, `data_sources_used`
- **Numeric passthrough** (under `structured_data`): `news_sentiment.article_count`, `analyst_sentiment.consensus_rating`, `analyst_sentiment.buy_hold_sell_distribution`, `analyst_sentiment.average_price_target`, `analyst_sentiment.price_target_vs_current_pct`, `analyst_sentiment.recent_upgrades`, `analyst_sentiment.recent_downgrades`, `insider_activity.net_insider_activity_90d`, `insider_activity.notable_transactions`, `short_interest.short_interest_pct`, `short_interest.days_to_cover`
- **Hard-coded at MVP**: `social_sentiment.buzz_level = "unknown"`, `social_sentiment.sentiment = "unknown"`. If LLM produces a `social_sentiment` block, the orchestrator silently overwrites it during merge. No retry.
- **Derived enums**: `data_quality_assessment` (from `reliability_score`), `reliability_factors.data_completeness`, `reliability_factors.data_freshness`
---
# Orchestrator Pre-computation: `base_reliability_score`
The orchestrator owns all mechanical reliability adjustments. The LLM’s ±15pt adjustment is reserved for qualitative concerns (Rule 8).
```python
def compute_base_reliability_sentiment(data_bundle: DataBundle) -> tuple[int, str]:
    # Real gap fixed 2026-08-07 (contract-vs-consumer audit): this function
    # previously aliased `src = data_bundle.sentiment_sources`, a field that
    # doesn't exist anywhere on DataBundle. The real data is spread across
    # several top-level DataBundle fields (news_with_sentiment,
    # canadian_data_flags, analyst_consensus, insider_activity,
    # short_interest, peer_sentiment) — no single alias replaces `src`.
    score = 100
    reasons = []
    news = data_bundle.news_with_sentiment or []
    canadian_flags = data_bundle.canadian_data_flags

    n = len(news)
    if n == 0:
        score -= 35; reasons.append("no news articles in window")
    elif n < 3:
        score -= 20; reasons.append(f"only{n} news articles")
    elif n < 5:
        score -= 10; reasons.append(f"only{n} news articles")

    if n > 0:
        # top_source_share / primary_tier_share: no stored field anywhere —
        # computed from news_with_sentiment's per-article source/tier data.
        # Exact derivation is sentiment.py's decision when it's built; not
        # asserting settled key names here (news_with_sentiment is
        # `list[dict]`, not a strictly-typed model like ResearchSourcesBundle's
        # `news_items: list[NewsItem]`, so per-article keys aren't contract-
        # guaranteed).
        top_source_share = compute_top_source_share(news)
        primary_tier_share = compute_primary_tier_share(news)
        if top_source_share > 0.6:
            score -= 8; reasons.append(f"{int(top_source_share*100)}% from single source")
        if primary_tier_share < 0.3:
            score -= 5; reasons.append(f"only{int(primary_tier_share*100)}% primary-tier news")

    # analyst_count: CanadianDataFlags.analyst_count is a real, typed field
    # for CA stocks (canadian_flags is None for US stocks). No confirmed
    # source for the US-side count — analyst_consensus is an unshaped dict;
    # flagging as an open question for sentiment.py rather than guessing a
    # dict key.
    a = canadian_flags.analyst_count if canadian_flags is not None else get_us_analyst_count(data_bundle.analyst_consensus)
    if a == 0:
        score -= 20; reasons.append("no analyst coverage")
    elif a < 3:
        score -= 10; reasons.append(f"only{a} analysts")

    # insider_data_age_days: a key inside insider_activity (dict) — exact
    # key not pinned by the contract, sentiment.py's decision when built.
    insider_data_age_days = get_insider_data_age_days(data_bundle.insider_activity)
    if insider_data_age_days is None:
        score -= 10; reasons.append("no insider data")
    elif insider_data_age_days > 120:
        score -= 5; reasons.append(f"insider data{insider_data_age_days}d old")

    short_interest_available = data_bundle.short_interest is not None
    if short_interest_available:
        # short_interest_age_days: a key inside short_interest (dict) —
        # same caveat as insider_data_age_days above.
        short_interest_age_days = get_short_interest_age_days(data_bundle.short_interest)
        if short_interest_age_days > 30:
            score -= 3; reasons.append(f"short interest{short_interest_age_days}d old")

    # v1.2: single-peer payload — penalize only if no peer block at all
    if len(data_bundle.peer_sentiment) < 1:
        score -= 8; reasons.append("no peer sentiment block")

    canadian_sentiment_inferred = (
        canadian_flags is not None and canadian_flags.sentiment_source == "local_llm"
    )
    if canadian_sentiment_inferred:
        score = min(score, 70)
        reasons.append("Canadian sentiment LLM-scored (qwen3.6:35b-a3b); cap at 70 pending calibration data")

    reason_str = "; ".join(reasons) if reasons else "all sources present and current"
    return (max(0, score), reason_str)
```
---
# User Message (Data Payload)
```plain text
{canonical_ticker} ({company_name}) | {sector} | {primary_exchange} | {currency}
As of: {data_timestamp} | Timeline: {timeline} | News window: {news_window_days} days

NEWS (last {news_window_days} days, {article_count} articles):
  N1 | {date} | {headline} | source: {source} | tier: {tier} | sentiment: {sentiment_score_or_none}
  N2 | {date} | {headline} | source: {source} | tier: {tier} | sentiment: {sentiment_score_or_none}
  ...

ANALYST:
  Consensus rating: {consensus_rating} | Total analysts: {num_analysts}
  Distribution: buy={buy_count} hold={hold_count} sell={sell_count}
  Average price target: {avg_price_target} ({price_target_vs_current_pct}% vs current price {current_price})
  Last 30 days: upgrades={recent_upgrades} downgrades={recent_downgrades}
  Recent rating changes: {recent_rating_change_list}

INSIDER (last 90 days):
  Net direction: {net_insider_activity_90d}
  Buy value: {total_buy_value} | Sell value: {total_sell_value}
  Notable transactions:
    {date} | {insider_name} ({role}) | {action} | {amount}
    ...

SHORT INTEREST (as of {short_interest_date}):
  Short interest: {short_interest_pct}% of float | Days to cover: {days_to_cover}
  30-day trend: {short_interest_30d_change} ({short_interest_30d_change_pct}%)

PEER SENTIMENT (contrastive analysis):
  PEER_1 = {peer_1_ticker} ({peer_1_name}):
    News sentiment summary: {peer_1_news_sentiment_summary}
    Article count (same window): {peer_1_article_count}
    Analyst consensus: {peer_1_consensus_rating} | Recent: U{peer_1_upgrades}/D{peer_1_downgrades}

DATA AVAILABILITY FLAGS:
  canadian_sentiment_inferred: {canadian_sentiment_inferred}
  short_interest_available: {short_interest_available}
  missing_sources: {missing_sources_list}

{data_pipeline_warnings}
```
**Source ID stability** (from v1.1): News IDs (`N{num}`) are **assigned globally per stock per run by the Data Pipeline**, not per-agent. The same news item carries the same `N{num}` identifier across every Pass 1 agent’s payload within a single run. IDs are assigned in date order (oldest first) over the widest window any agent uses (typically Researcher’s 180-day long-term window); Sentiment’s narrower window (7/14/30 days by timeline) sees a contiguous **suffix** of the globally-assigned IDs but the IDs themselves are not renumbered. So Sentiment may legitimately receive `N12, N13, N14, N15, N16` rather than `N1, N2, N3, N4, N5` — the missing earlier IDs are news items outside Sentiment’s window. This is intentional. Enables the Feedback Analyst and CIO to cross-reference findings without a mapping table.
**Single-peer payload (v1.2)**: PEER_1 only. The orchestrator selects the peer with the best news-coverage match for the analysis window. PEER_2 and the optional PEER_3 block are removed.
---
# Orchestrator Merge Logic
```python
def merge_output_sentiment(llm_response: dict, data_bundle: DataBundle,
                           context: AnalysisContext, base_score: int) -> dict:
    derived = derive_quality_enums_sentiment(base_score, data_bundle)
    news = data_bundle.news_with_sentiment or []

    # news_with_sentiment is list[dict], not a strictly-typed model — id
    # lookup is dict-key access, not attribute access.
    valid_news_ids = {item["id"] for item in news}
    for theme in llm_response["structured_data"]["news_sentiment"]["dominant_themes"]:
        nid = theme["primary_news_id"]
        if nid not in valid_news_ids:
            raise ValidationError(f"invalid primary_news_id in dominant_themes:{nid}")

    sd = llm_response["structured_data"]
    analyst = data_bundle.analyst_consensus  # dict — exact keys are sentiment.py's decision when built
    insider = data_bundle.insider_activity   # dict — same caveat
    short_interest = data_bundle.short_interest  # dict | None — same caveat
    merged_structured = {
        "news_sentiment": {**sd["news_sentiment"], "article_count": len(news)},
        "analyst_sentiment": {
            **sd["analyst_sentiment"],
            "consensus_rating": map_consensus_rating(
                analyst["buy_count"], analyst["hold_count"], analyst["sell_count"]
            ),
            "buy_hold_sell_distribution": {
                "buy": analyst["buy_count"], "hold": analyst["hold_count"], "sell": analyst["sell_count"]
            },
            "average_price_target": analyst["avg_price_target"],
            "price_target_vs_current_pct": analyst["price_target_vs_current_pct"],
            "recent_upgrades": analyst["recent_upgrades"],
            "recent_downgrades": analyst["recent_downgrades"],
        },
        "insider_activity": {
            "net_insider_activity_90d": compute_net_insider_direction(insider),
            "notable_transactions": filter_notable_transactions(
                insider, data_bundle.company_info["market_cap"]
            ),
            "interpretation": sd["insider_activity_interpretation"],
        },
        "short_interest": {
            "short_interest_pct": short_interest["short_interest_pct"] if short_interest else None,
            "days_to_cover": short_interest["days_to_cover"] if short_interest else None,
            "trend": sd["short_interest_interpretation"]["trend"],
            "interpretation": sd["short_interest_interpretation"]["interpretation"],
        },
        "social_sentiment": {"buzz_level": "unknown", "sentiment": "unknown"},
        "peer_sentiment_comparison": sd["peer_sentiment_comparison"],
        "contrarian_signals": sd["contrarian_signals"],
        # v1.2: informed_vs_noise_summary removed
    }

    return {
        "agent_name": "sentiment_analyst",
        "agent_pass": "pass1",
        "analysis_context": {
            "account_type": context.account_type,
            "timeline": context.timeline,
            "stock_id": str(context.stock_id),
            "canonical_ticker": context.canonical_ticker,
        },
        "assessment_summary": llm_response["assessment_summary"],
        "reliability_score": llm_response["reliability_score"],
        "reliability_factors": {
            "data_completeness": derived["data_completeness"],
            "data_freshness": derived["data_freshness"],
            "analysis_confidence": llm_response["analysis_confidence"],
            "caveats": llm_response["caveats"],
        },
        "key_factors": llm_response["key_factors"],
        "risks": llm_response["risks"],
        "data_quality_assessment": derived["data_quality_assessment"],
        "narrative": llm_response["narrative"],
        "structured_data": merged_structured,
        "data_sources_used": derive_sources_sentiment(data_bundle),
    }
```
---
# Pass 2 Consumption View
Pass 2 agents (Bull, Bear, Risk, Tax) receive a flat `pass2_view` dict produced by the orchestrator’s `compress_pass1_for_advocate()`, NOT the full nested `structured_data`. The view is the stable contract between Pass 1 and Pass 2 — Pass 1 schema changes under `structured_data` do not break Pass 2 as long as this view is preserved.
```python
pass2_view = {
    "news_sentiment_overall": str,             # structured_data.news_sentiment.overall
    "sentiment_trend": str,                    # structured_data.news_sentiment.sentiment_trend
    "article_count": int,
    "dominant_themes": [
        {"theme": str, "sentiment": str}       # primary_news_id dropped at pass2 level
    ],
    "consensus_rating": str|None,
    "consensus_trend": str,
    "consensus_interpretation": str,
    "average_price_target": float|None,
    "price_target_vs_current_pct": float|None,
    "recent_upgrades": int,
    "recent_downgrades": int,
    "net_insider_activity_90d": str,
    "insider_activity_interpretation": str,
    "short_interest_pct": float|None,
    "days_to_cover": float|None,
    "short_interest_trend": str,
    "short_interest_interpretation": str,      # v1.2 enum: elevated_volatility_risk|normal|insufficient_data
    "peer_sentiment_comparison": str,
    "contrarian_signals": [
        {"signal": str, "evidence": str}
    ],
    # v1.2: informed_vs_noise_summary removed from pass2_view
}
```
The envelope fields Pass 2 sees alongside `pass2_view` — `reliability_score`, `data_quality_assessment`, `caveats`, `assessment_summary`, truncated `narrative`, top-2 `key_factors`, top-2 `risks` — are defined in the Orchestration Engine’s Pass 1 compression spec.
`social_sentiment` is intentionally excluded from `pass2_view`. At MVP it’s hard-coded to `unknown` and provides no signal. When social sentiment becomes available, it can be added to the view.
`informed_vs_noise_summary` is removed from `pass2_view` in v1.2 — Bull/Bear/Risk/Tax should not depend on it. The equivalent insight is implicit in `contrarian_signals` (divergence cases) and the truncated `narrative`.
Note on archetype composition: Sentiment does not currently produce a `sentiment_archetype` enum (deferred to v1.3+). When introduced, it will be an **orthogonal classification dimension**, not a competing label (see Agent Design Patterns Pattern 13 for the composition rule).
---
# Validation Rules
Enforced by `src/agents/validators.py`.
**Output-only validators:**
1. **JSON parseable** (strip markdown fences/preamble if present).
2. **Required fields**: `assessment_summary`, `reliability_score`, `analysis_confidence`, `caveats`, `key_factors`, `risks`, `narrative`, `structured_data`.
3. **reliability_score**: integer 0–100, within ±15 of `base_reliability_score`.
4. **analysis_confidence**: one of `[high, medium, low]`.
5. **caveats**: 1–4 non-empty string items.
6. **key_factors**: 2–3 items, each with `factor`, `importance`, `sentiment`, `evidence`. Evidence must start with a valid citation token (`N\d+`, `ANALYST`, `INSIDER`, `SHORT`, or `PEER_1`).
7. **risks**: 1–2 items, each with `risk`, `severity`, `likelihood`, `evidence`. Same citation rule as key_factors.
8. **narrative**: 720–1,080 characters (\~120–180 words).
9. **news_sentiment.overall**: one of `[very_positive, positive, neutral, negative, very_negative]`.
10. **news_sentiment.dominant_themes**: 2–3 items, each with `theme`, `sentiment` (one of `[positive, negative, neutral, mixed]`), and `primary_news_id`.
11. **news_sentiment.sentiment_trend**: one of `[improving, stable, deteriorating]`.
12. **analyst_sentiment.consensus_trend**: one of `[improving, stable, deteriorating]`.
13. **analyst_sentiment.consensus_interpretation**: non-empty string.
14. **short_interest_interpretation.trend**: one of `[increasing, stable, decreasing, unknown]`.
15. **short_interest_interpretation.interpretation**: one of `[elevated_volatility_risk, normal, insufficient_data]`. *(v1.2 — collapsed from 5-value enum)*
16. **contrarian_signals**: 0–2 items, each with `signal` (non-empty string) and `evidence` (must contain a valid citation token).
17. **peer_sentiment_comparison**: non-empty string.
*(v1.1 validator 17 — **`informed_vs_noise_summary`** non-empty — removed in v1.2.)*
**Cross-validators:**
1. **`primary_news_id`**** exists in payload**: Each `dominant_themes[].primary_news_id` must match a news ID in the payload. (Note: the payload may contain non-contiguous IDs like `N12, N13, N15` per the global-ID spec — the validator checks set membership, not contiguity.)
2. **`peer_sentiment_comparison`**** cites PEER_1**: Must contain the literal token `PEER_1` (v1.2 — was `PEER_\d+`).
3. **`insider_activity_interpretation`**** matches data presence**: If payload has no qualifying insider transactions, the field must be `"insufficient_data"`.
4. **`short_interest_interpretation`**** matches data availability**: If payload’s `short_interest_available=false`, both `trend` must be `"unknown"` and `interpretation` must be `"insufficient_data"`.
5. **Canadian inference caveat present**: If payload's `canadian_sentiment_inferred=true`, `caveats` must contain the exact string `"Canadian AI sentiment: scored by local LLM (qwen3.6:35b-a3b) rather than Finnhub — confidence scores are available but not independently validated against a ground-truth dataset."`
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember: cite sources by ID only (N{num}, ANALYST, INSIDER, SHORT, PEER_1), do not recompute numeric fields the orchestrator provides.
```
---
# Memory Brief Schema
Schema (\~120 tokens):
```plain text
Prior sentiment analysis ({days_since} days ago, timeline={prior_timeline}, score={composite_score}/100):
  news_sentiment_then: {prior_overall} (trend: {prior_sentiment_trend})
  consensus_rating_then: {prior_consensus_rating} (trend: {prior_consensus_trend})
  insider_direction_then: {prior_net_insider_activity_90d}
  contrarian_signals_then: [{prior_contrarian_signal_names}]
  outcome: {return_pct if scored else "not yet scored"}
  what_changed_since: {short_list_of_material_news_since_prior}
```
**Timeline-scaled decay**: memory included if \<30d (short), \<90d (medium), \<180d (long). (Updated to match Researcher v1.3 — was 14/60/180.) Reconciliation logic is in the system prompt’s PRIOR ANALYSIS MEMORY block.
---
# Accuracy Brief (Post-20 Predictions)
Injected into the system prompt as `{accuracy_brief}`. \~80 tokens. Empty string until ≥20 scored predictions.
```plain text
YOUR ACCURACY: {N} sentiment calls scored. Bias: {bias_description}. Adjust: {calibration_instruction}. Strongest signal type: {best_signal_type}. Weakest signal type: {worst_signal_type}.
```
---
# Minimum Data Threshold
Per [Agents Overview](https://www.notion.so/Agents-Overview-325ea06798238004a793fc1720637698?pvs=21): **at least 5 news articles in the timeline window OR ≥5 analysts with at least one rating change in the last 90 days**.
Below threshold: `base_reliability_score` will be \<20. Orchestrator sets `data_quality_assessment` to `"insufficient"` automatically.
---
# Canadian vs US Data Asymmetry
<table header-row="true">
<tr>
<td>Source</td>
<td>US stocks</td>
<td>Canadian stocks</td>
</tr>
<tr>
<td>News</td>
<td>FMP + Finnhub with per-article AI sentiment scores</td>
<td>FMP + GlobeNewswire/[Newswire.ca](http://Newswire.ca) RSS (when FMP count \< 5); LLM-scored per-article sentiment via qwen3.6:35b-a3b</td>
</tr>
<tr>
<td>Analyst data</td>
<td>Full FMP coverage + Finnhub recommendation trends</td>
<td>Thinner FMP coverage; small-cap TSX often \<3 analysts</td>
</tr>
<tr>
<td>Insider data</td>
<td>FMP Form 4 data</td>
<td>FMP SEDI-sourced, less granular</td>
</tr>
<tr>
<td>Short interest</td>
<td>Available via FMP for major tickers</td>
<td>Limited availability for TSX names</td>
</tr>
</table>
**Orchestrator behavior when ****`canadian_sentiment_inferred`**** is true**: `base_reliability_score` capped at 70 (matches Researcher); `canadian_sentiment_inferred` injected into system prompt; cross-validator 5 forces the exact templated caveat.
---
# How the System Prompt and User Message Interact
- **Citation token consistency**: System prompt defines exact tokens (`N\d+`, `ANALYST`, `INSIDER`, `SHORT`, `PEER_1`) and the user message uses them as block headers.
- **3-phase reasoning, schema-enforced coverage**: Phases in the prompt do not require step-by-step narration. Coverage of all analytical dimensions is enforced by the structured schema’s required fields. The narrative is reserved for synthesis.
- **Numeric passthrough discipline**: Rule 3 tells the LLM not to restate numeric fields the orchestrator owns.
- **Mechanical-vs-qualitative reliability split**: Orchestrator owns mechanical reliability (`base_reliability_score`); LLM’s ±15pt window is for qualitative concerns (Rule 8). No double-counting.
- **Non-thinking mode**: `enable_thinking: false`.
- **Single exchange**: `[{role: "system", content: ...}, {role: "user", content: ...}]`. Retries append to user message.
---
# Prompt Variable Injection Reference
<table header-row="true">
<tr>
<td>Placeholder</td>
<td>Source</td>
<td>Notes</td>
</tr>
<tr>
<td>`{canonical_ticker}`</td>
<td>`DataBundle.stock.canonical_ticker`</td>
<td>e.g., “AAPL” or “RY.TO”</td>
</tr>
<tr>
<td>`{company_name}`</td>
<td>`DataBundle.company_info["name"]`</td>
<td>Full company name</td>
</tr>
<tr>
<td>`{sector}`</td>
<td>`DataBundle.company_info["sector"]`</td>
<td>No anonymization at v1.0–v1.2</td>
</tr>
<tr>
<td>`{timeline}`</td>
<td>`AnalysisContext.timeline`</td>
<td>short_term / medium_term / long_term</td>
</tr>
<tr>
<td>`{news_window_days}`</td>
<td>Conditional by timeline</td>
<td>7 / 14 / 30</td>
</tr>
<tr>
<td>`{account_type}`</td>
<td>`AnalysisContext.account_type`</td>
<td>“tfsa”, “rrsp”, “trading”, or “general”</td>
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
<td>`{base_reliability_score}`</td>
<td>`compute_base_reliability_sentiment()`</td>
<td>Orchestrator pre-computed</td>
</tr>
<tr>
<td>`{reliability_reason}`</td>
<td>`compute_base_reliability_sentiment()`</td>
<td>Human-readable reason string</td>
</tr>
<tr>
<td>`{canadian_sentiment_inferred}`</td>
<td>Derived: `data_bundle.canadian_data_flags is not None and data_bundle.canadian_data_flags.sentiment_source == "local_llm"`</td>
<td>“true” or “false”; drives rule 7 — computed by the orchestrator, not a raw `DataBundle` field</td>
</tr>
<tr>
<td>`{accuracy_brief}`</td>
<td>Feedback engine</td>
<td>Empty string until ≥20 scored predictions</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory` query</td>
<td>Empty string if none or below decay threshold</td>
</tr>
<tr>
<td>`{data_warnings}`</td>
<td>Pre-flight validation</td>
<td>Empty string if clean</td>
</tr>
<tr>
<td>`{output_schema}`</td>
<td>Hardcoded JSON template</td>
<td>The schema above</td>
</tr>
</table>
---
# Testing Checklist
- [ ] Does `qwen3.6:35b-a3b` in non-thinking mode produce valid JSON matching the v1.2 schema, with no leaked `<think>` reasoning tokens?
- [ ] Does thinking mode accidentally activate? (Check for `<think>` tags)
- [ ] Does `reliability_score` stay within ±15 of `base_reliability_score`?
- [ ] Does the narrative stay within 120–180 words (720–1,080 chars) and read as synthesis (not a phase-by-phase recap)?
- [ ] Does every `key_factors` / `risks` / `contrarian_signals.evidence` field use a valid citation token?
- [ ] Does `dominant_themes` use single `primary_news_id` per theme (not arrays)?
- [ ] Does `peer_sentiment_comparison` contain `PEER_1`?
- [ ] Does the model refuse to recompute or restate numeric passthrough fields?
- [ ] Does the orchestrator correctly silently overwrite any LLM-produced `social_sentiment` block (no retry triggered)?
- [ ] Does the model produce 2–3 dominant_themes, 2–3 key_factors, 1–2 risks, 0–2 contrarian_signals?
- [ ] Does the model refuse to recommend buy/sell?
- [ ] Does the model adapt focus based on timeline (short_term weighting news, long_term weighting insiders)?
- [ ] **Canadian asymmetry test**: Pass a Royal Bank of Canada ([RY.TO](http://RY.TO)) payload with `canadian_sentiment_inferred=true`. Does the exact templated caveat appear (cross-validator 5)? Does `base_reliability_score` cap at 70? Verify caveat text matches updated Rule 7 (LLM-scored language — not the old Finnhub-unavailable language).
- [ ] **Quiet stock test**: Pass a stock with 4 news articles in window (below 5-threshold). Does the agent appropriately fail with “insufficient” data quality?
- [ ] **No-news edge case**: Pass a stock with 0 news articles but 8+ analysts with recent rating changes. Does the model produce a coherent analysis from analyst + insider only?
- [ ] **Contrarian test (insider divergence)**: Pass a stock with positive news + insider selling at material size. Does an `insider_news_divergence`themed signal appear in `contrarian_signals`?
- [ ] **Insider data missing test**: Pass a stock with no qualifying discretionary open-market insider transactions. Does `insider_activity_interpretation` return `"insufficient_data"`?
- [ ] **Short interest unavailable test**: Pass a Canadian small-cap with `short_interest_available=false`. Does `short_interest_interpretation.trend` return `"unknown"` and `interpretation` return `"insufficient_data"`?
- [ ] **Short interest collapse test (new in v1.2)**: Pass a stock with high short interest. Does `short_interest_interpretation.interpretation` return `"elevated_volatility_risk"` and NOT a v1.0/v1.1 value (`squeeze_setup`, `bearish_conviction`, `crowded_short`)?
- [ ] **Single-peer test (new in v1.2)**: Verify the user message contains only `PEER_1` block; no `PEER_2` or `PEER_3` references. Verify `peer_sentiment_comparison` cites `PEER_1` literally.
- [ ] **No informed_vs_noise_summary test (new in v1.2)**: Verify the LLM does not produce an `informed_vs_noise_summary` field. If it does (training inertia from prior versions), the field should be silently dropped at merge — not a retry trigger. Validate against v1.2 schema.
- [ ] **Reduced array bounds test (new in v1.2)**: Verify validator rejects 5+ caveats, 3+ risks, 4+ key_factors, 4+ dominant_themes, 3+ contrarian_signals.
- [ ] **Phase-narration leakage test**: Verify the narrative does not contain phrases like “Phase 1:” or “First, looking at news…” or “Step 1:” — synthesis should read as a market-mood story, not a process recap.
- [ ] **Cross-agent news ID consistency test**: Run Researcher and Sentiment on the same stock. Confirm that a news item that appears in both payloads has the same `N{num}` identifier in both. Confirm Sentiment can correctly handle non-contiguous N-IDs (e.g., payload contains `N12, N13, N15`).
- [ ] **pass2_view shape test**: the orchestrator’s compression produces every key listed in the Pass 2 Consumption View section with correct types. Confirm `social_sentiment` and `informed_vs_noise_summary` are NOT in the pass2_view (intentionally excluded).
- [ ] **Fixture stocks**: AAPL (US large-cap, rich data), RY.TO (Canadian financial, sentiment-inference path), a US small-cap with thin coverage (sparse-data edge case), a meme stock candidate (high short interest test for the elevated_volatility_risk path)
---
# Token Budget
<table header-row="true">
<colgroup>
<col>
<col>
<col width="465">
</colgroup>
<tr>
<td>Component</td>
<td>Budget</td>
<td>Notes</td>
</tr>
<tr>
<td>System prompt</td>
<td>\~430 tokens</td>
<td>8 rules (was 9) + 3-phase reasoning frame + tightened injection blocks + schema + accuracy brief slot + memory reconciliation</td>
</tr>
<tr>
<td>User message</td>
<td>\~1,630 tokens</td>
<td>News block (up to 30 lines × \~20 tokens) + analyst (\~110) + insider (\~140) + short interest (\~80) + single peer block (\~80) + flags + headers</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~580 tokens</td>
<td>Single `primary_news_id` per theme, `informed_vs_noise_summary` removed, reduced array maxes</td>
</tr>
<tr>
<td>**Total**</td>
<td>**\~2,640 tokens**</td>
<td>\~340 tokens (\~11%) under v1.1’s \~2,980.</td>
</tr>
</table>
**Model configuration**: `num_ctx`: 8192; `enable_thinking`: false (verify suppression is clean on `qwen3.6:35b-a3b` — see Pattern 10 caveat); VRAM: \~18-22GB at Q4_K_M (MoE, partial CPU offload expected); Expected latency: unverified estimate, re-measure after deployment (was \~16–22s per stock on the prior `qwen3:14b`).
---
# Open Questions / Future Improvements
- **News pre-aggregation (v1.3+)**: If the LLM’s `news_sentiment.overall` enum reliably agrees with a simple weighted average of per-article scores (correlation \>0.85 across 50+ stocks), move the `overall` calculation into the orchestrator and remove it from the LLM’s responsibility.
- **Social sentiment when available**: When a Reddit/StockTwits API becomes accessible, unlock the `social_sentiment` block. The LLM gains new instructions, and `crowded_long` returns to the contrarian-archetype seed list. Add to `pass2_view` at that point.
- **Short interest re-expansion**: If user base ever broadens to active traders, restore the v1.0/v1.1 5-value short interest enum (`squeeze_setup | bearish_conviction | crowded_short | not_significant | insufficient_data`) for the trading-account path.
- **Headline pre-filtering**: Some news items are pure non-events (broker reiterations, share repurchase trivia). A small classifier could filter these from the payload.
- **Insider clustering**: Aggregating insider transactions by role (CEO, CFO, board, 10%-owners) and surfacing role-level direction would be more informative than raw value sums.
- **Few-shot example**: Start without one. If JSON retry rate \>10% after first batch, add a partial example (\~150 tokens).
- **`sentiment_archetype`**** enum (v1.3+)**: Researcher has `thesis_archetype`. Candidates for Sentiment: `momentum_following`, `momentum_reversal_setup`, `consensus_diverging`, `quiet_consensus`, `polarized`. Defer until first 50 runs. When introduced, must be specified as **orthogonal** to Researcher’s `thesis_archetype` per Agent Design Patterns Pattern 13.
- **Anonymization A/B test (v1.3+)**: Test whether removing company name improves sentiment accuracy by forcing the model to reason from current data rather than training priors.
- **`informed_vs_noise_summary`**** reintroduction**: If post-v1.2 monitoring shows CIO synthesis quality degrades (disagreement rate spikes after first 10–20 runs), reintroduce as a narrative-only requirement (no separate field) rather than a structured field.
---
# Related Documents
- [Agent Design Patterns](https://www.notion.so/Agent-Design-Patterns-33fea067982380e6a055d3ed74c8eada?pvs=21) — the 13 validated patterns this prompt applies, with one deliberate departure from Pattern 6 (see Decision #1)
- [Fundamental Analyst Agent Prompt v4.1](https://www.notion.so/Fundamental-Analyst-Agent-Prompt-333ea06798238063a42bed3e2c132ea7?pvs=21) — reference implementation: hybrid orchestrator/LLM split for quantitative inputs
- [Stock Researcher Agent Prompt v1.2](https://www.notion.so/Stock-Researcher-Agent-Prompt-333ea06798238081bb20ff087cc7e76a?pvs=21) — reference implementation: verifiable grounding for qualitative inputs
- [Agents Overview](https://www.notion.so/Agents-Overview-325ea06798238004a793fc1720637698?pvs=21) — AgentOutput schema, pass structure, token budgets, minimum data thresholds
- [🔄 Data Pipeline](https://www.notion.so/Data-Pipeline-341ea06798238191a875f3f4f63082f7?pvs=21) — globally-assigned news IDs (`N{num}`) are specified here; Sentiment’s Source ID stability note references this contract
- [Financial Data API Research](https://www.notion.so/Financial-Data-API-Research-325ea0679823810da733ce5809563e73?pvs=21) — Finnhub vs FMP coverage details that drive the US/Canadian asymmetry handling
---

_Source: Notion — Project Stock Picker → Agent Prompts → Sentiment Analyst Agent Prompt. Downloaded 2026-07-02._
