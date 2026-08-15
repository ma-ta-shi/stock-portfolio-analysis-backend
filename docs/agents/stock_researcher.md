**Agent**: Stock Researcher
**Status:** Draft — v1.3 
**Phase:** Phase 1
**Pass**: Pass 1 (Research & Analysis)
**Role**: Deep-dive qualitative research on the company — business model, competitive position, moat, management quality, and recent developments. Provides the “story” of the company that other agents build on.
**Prompt Version**: v1.3
**Target Model**: `qwen3.6:35b-a3b` via Ollama (non-thinking mode — verify `enable_thinking:false` fully suppresses reasoning on this model; see Agent Design Patterns Pattern 10 caveat)
**Token Budget**: \~700 system prompt \| \~2,750 data input \| \~1,340 expected output \| \~4,790 total (\~430 tokens lighter than v1.2)
**Audience**: Canadian retail investors with total portfolio values under \$250k CAD, holding stocks across TFSA / RRSP / Trading / General accounts. Institutional-grade complexity that doesn’t pay rent at this scale has been trimmed.
---
# Design Decisions & Rationale
Built following the [Agent Design Patterns](https://www.notion.so/Agent-Design-Patterns-33fea067982380e6a055d3ed74c8eada?pvs=21) validated through the Fundamental Analyst v1→v4 iteration. Reference implementation: [📊 Fundamental Analyst Agent Prompt](https://www.notion.so/Fundamental-Analyst-Agent-Prompt-333ea06798238063a42bed3e2c132ea7?pvs=21).
1. **Qualitative agent, not quantitative.** The DataBundle for this agent contains pre-summarized filings, transcript excerpts, news items, peer summaries, and a structured `management_signals` block — not pre-computed financial metrics. The LLM’s job is to interpret text and structured qualitative signals.
2. **Pattern 1 still applies — orchestrator owns everything it can compute.** Even though there are no financial-metric passthroughs, the orchestrator owns: quality enums (`data_quality_assessment`, `data_completeness`, `data_freshness`); `recent_developments` entries (hydrated from news IDs the LLM cites — LLM never types dates or headlines); `management_signals` (C-suite changes, insider direction, buyback activity, computed in pre-flight); peer block contents. The LLM only produces interpretive judgment on top of these inputs.
3. **Hybrid reliability scoring** (Pattern 2). Orchestrator pre-computes `base_reliability_score` from filing recency, transcript availability, news coverage, peer data presence, and the US/Canadian data-asymmetry flag. LLM adjusts ±15pts with justification in `caveats`.
4. **Contrastive peer analysis.** 2 top sector peers are included as structured blocks with brief summaries and recent news. CoT step 2 requires explicit peer comparison. Peer citation is enforced on `peer_comparison_summary` (not on the broader `key_factors`/`risks` arrays — see validation rules).
5. **7-step CoT** (Pattern 6). Each step maps to at least one structured output field. No analytical conclusion exists only in narrative.
6. **Bounded output arrays** (Pattern 7). `moats` 1–3, `recent_developments` 0–5, `competitive_threats` 1–3, `growth_drivers` 1–3, `key_factors` 2–4, `risks` 1–3, `caveats` 1–4. `recent_developments` has a zero floor — a quiet stock should not fabricate stub events. Maxes lowered in v1.3 to match retail attention span and reduce padding.
7. **Verifiable grounding directive** (Pattern 3). Every claim must be traceable to a source ID in the payload (`N1`, `FILING:Business`, `TRANSCRIPT:Q1`, `PEER_1`). If it cannot be cited, it is inadmissible. This is verifiable behavior, not self-censorship of training knowledge.
8. **Conditional injection** (Pattern 4). Timeline block, account block, and **news window length** are all conditional. Data payload and instructions both vary by timeline.
9. **Sector injection with moat-type hints** (Pattern 9). Sector appears in the header, and the orchestrator injects a sector-appropriate moat-type list so the LLM defaults to plausible moats for that industry. Table collapsed from 12 sectors to 8 in v1.3.
10. **Non-thinking mode** (Pattern 10). `enable_thinking: false`.
11. **Canadian data asymmetry handling.** When `ResearchSourcesBundle.sedar_filing_available` is false, the system prompt requires a specific templated caveat phrase (not substring heuristics) — SEDAR filing depth is disclosed directly rather than via a blanket reliability cap; the existing per-source penalties (filing/transcript/news/peer counts) already reflect Canadian coverage gaps.
12. **Training-data leakage mitigation — entity anonymization.** Filing and transcript digests reach the LLM with the company name and ticker replaced by `COMPANY_X` / `TICKER_X` tokens. The LLM sees the sector and the structured payload but not the named entity. Narrative is produced against `COMPANY_X`, and the orchestrator de-anonymizes on merge. This closes the biggest quality hole for well-known names like AAPL or MSFT where the model’s training knowledge overwhelms the payload. See also the Agent Design Patterns anonymization principle — Researcher is the only Pass 1 agent that anonymizes, because it is the only Pass 1 agent that ingests raw-ish text (filing/transcript digests) the model may have memorized.
13. **Controlled-vocabulary thesis archetype.** The `thesis_archetype` field is a fixed enum (secular grower, dividend compounder, cyclical recovery, quality compounder, value trap candidate). v1.3 trimmed the enum from 8 to 5: dropped `turnaround` (subset of cyclical_recovery for retail-held names), `disrupted_incumbent` (subset of value_trap_candidate), and `special_situation` (M&A arb / spinoffs are rarely held in sub-\$250k retail portfolios). Five archetypes also gives the feedback engine more samples per bucket. Pass 2 agents consume this field via the `pass2_view`.
14. **Single prompt, re-optimize on model change** (Pattern 12).
15. **Narrative bound normalized** (v1.2 change). Narrative is 120–180 words / 720–1080 chars — matches Fundamental v4.1, Technical v2.1, Sentiment v1.0, Macro v1.0. Converging all Pass 1 narratives gives the feedback engine stable training signal.
16. **Retail-scope simplification (v1.3 change).** Institutional-grade complexity that doesn’t pay rent for sub-\$250k Canadian retail portfolios was removed: dual-class voting analysis is data-pipeline-only (orchestrator injects caveat when the flag fires — LLM no longer told to consider voting structure); CIK verification stays in the data pipeline (not surfaced to LLM); news quality_tier filtering happens at ingest (LLM no longer sees per-headline tier labels); the optional third peer block is dropped; thesis archetype enum trimmed from 8 to 5; sector moat hint table from 12 sectors to 8; bounded-array maxes lowered (caveats 1–4, competitive_threats 1–3, growth_drivers 1–3); RiskFactors filing digest dropped from the payload (Business + MDA carry the analytic weight; RiskFactors is boilerplate for \~80% of retail-relevant names); rule 4 anomaly list simplified; rule 7 management focus tightened to capital-allocation signals; memory decay loosened to 30/90/180 days to match retail cadence. Anonymization, citation system, Canadian data asymmetry handling, account-type differentiation (TFSA/RRSP/Trading/General), 7-step CoT, and hybrid reliability scoring all retained — those earn their keep at any portfolio scale.
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- Company profile — openbb-tmx `.equity.profile` (CA) / FMP `/profile` (US)
- News — openbb-tmx `.news.company` (CA) / FMP `/stock_news` + Finnhub `/company-news` (US)
- Filings (dates, 8-K material events) — openbb-tmx `.equity.fundamental.filings` (CA) / FMP profile (US)
- Peer companies — static `peers.json` (CA, see Gap 1) / Finnhub `/stock/peers` (US)
- ETF info, where applicable — openbb-tmx `.etf.info`
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- Anonymized Business Model & MD&A digests (`FILING:Business`/`FILING:MDA`, ≤250 tokens each; RiskFactors digest dropped in v1.3)
- Earnings transcript excerpts, current + prior quarter (`TRANSCRIPT:Q1`/`Q0`) — US only; unavailable for Canadian tickers (Gap 4), flagged rather than left as a silent gap
- Recent news with stable per-run IDs (`N1`, `N2`...), window scoped to timeline (30/90/180 days)
- Management Signals block (`SIGNALS`) — C-suite changes, insider trading direction, buyback/dividend activity
- Analyst context — consensus rating, price target, recent rating changes
- Two peer blocks (`PEER_1`/`PEER_2`) — business summary + recent news
- `base_reliability_score` — computed from filing recency, transcript availability, news coverage, and peer presence
- Data availability flags (`sedar_filing_available`, `missing_sources_list`) and `data_pipeline_warnings`
---
# System Prompt
Invoke with `enable_thinking: false`. The `build_messages()` method renders this template. Note: `{COMPANY_X}` and `{TICKER_X}` are anonymized tokens — the actual company name and ticker never reach the LLM inside filing/transcript content.

**Runtime prompt:** see [`prompts/stock_researcher/v1.3.txt`](../../prompts/stock_researcher/v1.3.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`). Note: the data pipeline also scopes the news window by timeline — short_term=30d, medium_term=90d, long_term=180d — so the instructions here and the payload contents are aligned.
<table header-row="true">
<colgroup>
<col>
<col width="452">
<col>
</colgroup>
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
<td>News window</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks). Weight recent news, upcoming catalysts (earnings date, product launches, regulatory decisions), management near-term guidance. De-emphasize long-run moat durability.`</td>
<td>30 days</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Balance near-term catalysts with business quality. Management execution and sector positioning matter most. Recent developments shape the 3-12 month narrative.`</td>
<td>90 days</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). Emphasize moat durability, secular trends, management track record over multiple years, business model resilience. De-emphasize single-quarter news unless it changes the structural thesis.`</td>
<td>180 days</td>
</tr>
</table>
**Account** (inject based on `AnalysisContext.account_type`):
<table header-row="true">
<colgroup>
<col>
<col width="584">
</colgroup>
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Gains tax-free. Prefer durable businesses with compound-growth potential. For foreign holdings note 15% US/foreign dividend WHT not recoverable.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. US dividends WHT-exempt (treaty). Emphasize dividend sustainability and management's capital return track record.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading account. Capital gains 50% inclusion. Narrative can emphasize nearer-term catalysts and sentiment shifts.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General. Highlight which qualitative factors matter most for each account type.`</td>
</tr>
</table>
**Sector moat hint** (inject based on `DataBundle.company_info.sector`). Collapsed to 8 sectors with 3 moat types each in v1.3:
<table header-row="true">
<colgroup>
<col>
<col width="461">
</colgroup>
<tr>
<td>Sector</td>
<td>`{sector_moat_hint}`</td>
</tr>
<tr>
<td>Financials / Real Estate</td>
<td>`regulatory, scale, switching_costs`</td>
</tr>
<tr>
<td>Technology / Communication Services</td>
<td>`network_effects, switching_costs, data_advantage`</td>
</tr>
<tr>
<td>Healthcare / Pharma</td>
<td>`intellectual_property, regulatory, brand`</td>
</tr>
<tr>
<td>Consumer (Staples + Discretionary)</td>
<td>`brand, scale, switching_costs`</td>
</tr>
<tr>
<td>Industrials / Materials</td>
<td>`scale, cost_advantage, switching_costs`</td>
</tr>
<tr>
<td>Energy</td>
<td>`cost_advantage, scale, regulatory`</td>
</tr>
<tr>
<td>Utilities</td>
<td>`regulatory, cost_advantage, scale`</td>
</tr>
<tr>
<td>Default / Unknown</td>
<td>`scale, brand, switching_costs`</td>
</tr>
</table>
---
# Output Schema (LLM-produced fields)
The orchestrator populates metadata, derived enum fields, and hydrates `recent_developments` entries from news IDs. The LLM produces the interpretive fields below.
```javascript
{
  "assessment_summary": "≤80 words. Most important qualitative finding first.",
  "reliability_score": 0-100,
  "analysis_confidence": "high|medium|low",
  "caveats": ["specific data gaps, contradictions, stale sources, or data-asymmetry flags"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "positive|negative|neutral", "evidence": "source ID + brief note"}
  ],
  "risks": [
    {"risk": "name", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "source ID + brief note"}
  ],
  "narrative": "120-180 word analysis following steps 1-7. Reference specific source IDs.",
  "structured_data": {
    "business_model_summary": "2-3 sentences. How the company makes money. Cite FILING or TRANSCRIPT.",
    "revenue_mix_notes": "Primary segments and approximate weightings if stated. 'not disclosed' if absent.",
    "competitive_position": "dominant|strong|average|weak|deteriorating",
    "peer_comparison_summary": "1-2 sentences comparing to named peers. MUST reference at least one peer by PEER_{num}.",
    "moat_assessment": {
      "moats": [
        {
          "type": "network_effects|switching_costs|brand|cost_advantage|regulatory|scale|intellectual_property|data_advantage|none",
          "strength": "strong|moderate|weak",
          "evidence": "source ID + brief note"
        }
      ],
      "overall_moat_durability": "strong|moderate|weak|none",
      "moat_trend": "strengthening|stable|eroding"
    },
    "management_assessment": "excellent|competent|concerning|poor|insufficient_data",
    "management_notes": "1-2 sentences drawing on SIGNALS and transcript tone. 'insufficient data' if both are thin.",
    "recent_developments": [
      {
        "news_id": "N{num}",
        "significance": "high|medium|low",
        "sentiment": "positive|negative|neutral|mixed"
      }
    ],
    "competitive_threats": ["threat description with source ID"],
    "growth_drivers": ["driver description with source ID"],
    "thesis_archetype": "secular_grower|dividend_compounder|cyclical_recovery|quality_compounder|value_trap_candidate"
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
<td>4</td>
</tr>
<tr>
<td>`risks`</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>`moat_assessment.moats`</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>`recent_developments`</td>
<td>0</td>
<td>5</td>
</tr>
<tr>
<td>`competitive_threats`</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>`growth_drivers`</td>
<td>1</td>
<td>3</td>
</tr>
</table>
`recent_developments` is allowed to be empty — a quiet stock should not fabricate stub events.
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name`, `agent_pass`, `analysis_context`, `data_sources_used`
- **Derived enums**: `data_quality_assessment` (from `reliability_score`), `reliability_factors.data_completeness` (from source counts), `reliability_factors.data_freshness` (from max source age)
- **Hydrated recent developments**: For each `{news_id, significance, sentiment}` entry, the orchestrator looks up the news item in the payload and merges in `event` (headline), `date`, `source`. Invalid news IDs fail validation.
- **Dual-class caveat injection** (new in v1.3): When `dual_class_flag` is true in the data pipeline, the orchestrator appends a standard caveat to `caveats` post-LLM (not surfaced to the LLM during analysis). This keeps the prompt simpler without losing the signal for the rare case where it matters.
- **De-anonymization**: Before output is returned to the pipeline, the orchestrator replaces all `COMPANY_X` / `TICKER_X` tokens in LLM text fields with the real company name and ticker.
---
# Orchestrator Pre-computation: base_reliability_score
```python
def compute_base_reliability_researcher(data_bundle: DataBundle) -> tuple[int, str]:
    score = 100
    reasons = []
    sources = data_bundle.research_sources

    latest_filing_age_days = sources.latest_filing_age_days
    if latest_filing_age_days is None:
        score -= 30; reasons.append("no filings available")
    elif latest_filing_age_days > 180:
        score -= 20; reasons.append(f"latest filing{latest_filing_age_days}d old")
    elif latest_filing_age_days > 120:
        score -= 10; reasons.append(f"latest filing{latest_filing_age_days}d old")

    transcript_count = sources.transcript_count
    if transcript_count == 0:
        score -= 15; reasons.append("no earnings transcripts")
    elif transcript_count == 1:
        score -= 5; reasons.append("only 1 transcript (expected 2)")

    news_count = sources.news_item_count
    if news_count == 0:
        score -= 15; reasons.append("no news items")
    elif news_count < 3:
        score -= 8; reasons.append(f"only{news_count} news items")

    peer_count = len(sources.peer_blocks)
    if peer_count < 2:
        score -= 15; reasons.append(f"only{peer_count} peer blocks")

    reason_str = "; ".join(reasons) if reasons else "all sources present and current"
    return (max(0, score), reason_str)
```
**Derived quality enums** (same pattern as Fundamental Analyst):
```python
def derive_quality_enums_researcher(base_score: int, data_bundle: DataBundle) -> dict:
    if base_score >= 70: dqa = "high"
    elif base_score >= 40: dqa = "medium"
    elif base_score >= 20: dqa = "low"
    else: dqa = "insufficient"

    sources = data_bundle.research_sources
    source_checklist = [
        sources.latest_filing_age_days is not None,
        sources.transcript_count >= 2,
        sources.news_item_count >= 3,
        len(sources.peer_blocks) >= 2,
    ]
    present = sum(source_checklist)
    if present == 4: dc = "complete"
    elif present == 3: dc = "mostly_complete"
    elif present == 2: dc = "partial"
    else: dc = "minimal"

    freshest_days = min(
        sources.latest_filing_age_days or 9999,
        sources.latest_transcript_age_days or 9999,
        sources.latest_news_age_days or 9999,
    )
    if freshest_days <= 30: df = "current"
    elif freshest_days <= 90: df = "slightly_stale"
    elif freshest_days <= 180: df = "stale"
    else: df = "very_stale"

    return {"data_quality_assessment": dqa, "data_completeness": dc, "data_freshness": df}
```
---
# User Message (Data Payload)
The data pipeline pre-summarizes long sources and anonymizes company identity before the payload reaches the LLM. Raw 10-K text never reaches the agent — only compressed section digests with the company name and ticker replaced by `COMPANY_X` and `TICKER_X`. RiskFactors digest dropped in v1.3 — boilerplate for most retail-relevant names; Business + MDA carry the analytic weight.
```plain text
TICKER_X (COMPANY_X) | {sector} | {primary_exchange}
As of: {data_timestamp} | Research payload version: {payload_version}
Timeline: {timeline} | News window: {news_window_days} days

BUSINESS MODEL DIGEST (FILING:Business, pre-summarized, anonymized):
  {business_model_digest}

MD&A DIGEST (FILING:MDA, pre-summarized):
  {mda_digest}

REVENUE MIX (if disclosed):
  {revenue_mix_lines}

LATEST EARNINGS TRANSCRIPT EXCERPTS (TRANSCRIPT:Q1, {latest_quarter_label}):
  Management commentary: {transcript_mgmt_excerpt}
  Guidance: {transcript_guidance_excerpt}
  Q&A highlights: {transcript_qa_excerpt}

PRIOR QUARTER TRANSCRIPT EXCERPTS (TRANSCRIPT:Q0, {prior_quarter_label}):
  {prior_transcript_summary}

RECENT NEWS (last {news_window_days} days):
  N1 | {date} | {headline} | {source}
  N2 | {date} | {headline} | {source}
  ...

MANAGEMENT SIGNALS (SIGNALS — orchestrator pre-computed, factual):
  c_suite_changes_12mo: {count}
  changes_detail: {brief list or "none"}
  insider_net_direction_90d: {net_buying|net_selling|neutral|no_data}
  buyback_activity: {active|none|suspended|no_data}
  dividend_activity: {increased|held|cut|suspended|none|no_data}

ANALYST CONTEXT:
  Consensus rating: {consensus_rating} | Avg target: {avg_price_target}
  Recent rating changes: {recent_rating_changes}

PEER BLOCKS (contrastive analysis):
  PEER_1 = {peer_1_token} ({peer_1_anonymized_name}):
    Business: {peer_1_business_summary}
    Recent news: {peer_1_recent_news}
  PEER_2 = {peer_2_token} ({peer_2_anonymized_name}):
    Business: {peer_2_business_summary}
    Recent news: {peer_2_recent_news}

DATA AVAILABILITY FLAGS:
  sedar_filing_available: {sedar_filing_available}
  missing_sources: {missing_sources_list}

{data_pipeline_warnings}
```
**News quality filtering**: Low-tier sources (blogs, aggregators, contributor pieces) are filtered out by the data pipeline at ingest. The LLM no longer sees per-headline `quality_tier` labels (removed in v1.3) — every headline that reaches the agent is primary or secondary tier.
**Source ID stability**: News IDs (`N1`, `N2`, …) are **assigned globally per stock per run by the Data Pipeline** — the same news item carries the same `N{num}` identifier across every Pass 1 agent’s payload within a single run. IDs are assigned in date order (oldest first) over the widest window any agent uses (typically Researcher’s 180-day long-term window); agents with narrower windows see a contiguous subset of the globally-assigned IDs. This enables cross-agent cross-reference: Researcher’s `N3` and Sentiment’s `N3` refer to the same item. IDs are stable across agent retries within the same run.
**Pre-summarization budgets**: Each filing section ≤250 tokens (Business + MDA only — RiskFactors dropped in v1.3). Each transcript excerpt set ≤400 tokens. Each news item = one headline line. Each peer block ≤200 tokens. Total payload target \~2,750 tokens before data-pipeline warnings.
---
# Orchestrator Merge Logic
```python
def merge_output_researcher(llm_response: dict, data_bundle: DataBundle,
                            context: AnalysisContext, base_score: int) -> dict:
    derived = derive_quality_enums_researcher(base_score, data_bundle)

    news_by_id = {item.id: item for item in data_bundle.research_sources.news_items}
    hydrated_developments = []
    for dev in llm_response["structured_data"]["recent_developments"]:
        news = news_by_id.get(dev["news_id"])
        if news is None:
            raise ValidationError(f"invalid news_id:{dev['news_id']}")
        hydrated_developments.append({
            "event": news.headline,
            "date": news.date.isoformat(),
            "source": news.source,
            "significance": dev["significance"],
            "sentiment": dev["sentiment"],
        })

    # Inject dual-class caveat post-LLM if flagged (v1.3 — moved out of prompt)
    caveats = list(llm_response["caveats"])
    if data_bundle.research_sources.dual_class_flag:
        caveats.append("Dual-class share structure: voting power concentrated; "
                       "consider governance implications before sizing position.")

    llm_response = deanonymize_text_fields(llm_response, data_bundle)
    structured_data = {**llm_response["structured_data"], "recent_developments": hydrated_developments}

    return {
        "agent_name": "stock_researcher",
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
            "caveats": caveats,
        },
        "key_factors": llm_response["key_factors"],
        "risks": llm_response["risks"],
        "data_quality_assessment": derived["data_quality_assessment"],
        "narrative": llm_response["narrative"],
        "structured_data": structured_data,
        "data_sources_used": derive_sources_researcher(data_bundle),
    }
```
The `deanonymize_text_fields` helper walks every string in the LLM response and replaces `COMPANY_X` → real name, `TICKER_X` → real ticker, `PEER_1_COMPANY` → peer 1 real name, etc. across all text fields (assessment_summary, narrative, caveats, key_factors evidence, risks evidence, and all nested structured_data text fields).
---
# Pass 2 Consumption View
Pass 2 agents (Bull, Bear, Risk, Tax) receive a flat `pass2_view` dict produced by the orchestrator’s `compress_pass1_for_advocate()`, NOT the full `structured_data`. The view is the stable contract between Pass 1 and Pass 2 — Pass 1 schema changes under `structured_data` do not break Pass 2 as long as this view is preserved.
```python
pass2_view = {
    "thesis_archetype": str,                  # structured_data.thesis_archetype (5-value enum as of v1.3)
    "competitive_position": str,              # structured_data.competitive_position
    "overall_moat_durability": str,           # structured_data.moat_assessment.overall_moat_durability
    "moat_trend": str,                        # structured_data.moat_assessment.moat_trend
    "management_assessment": str,             # structured_data.management_assessment
    "top_growth_drivers": [str],              # first 3 of structured_data.growth_drivers
    "top_competitive_threats": [str],         # first 3 of structured_data.competitive_threats
    "top_recent_developments": [              # first 3 of structured_data.recent_developments (hydrated form)
        {"event": str, "date": str, "significance": str, "sentiment": str}
    ],
    "peer_comparison_summary": str,           # structured_data.peer_comparison_summary
}
```
The envelope fields Pass 2 sees alongside `pass2_view` — `reliability_score`, `data_quality_assessment`, `caveats`, `assessment_summary`, truncated `narrative`, top-2 `key_factors`, top-2 `risks` — are defined in the Orchestration Engine’s Pass 1 compression spec.
Note on `thesis_archetype`: the Bull Advocate currently consumes this as the single authoritative classification. Future Sentiment and Macro versions may introduce their own `sentiment_archetype` / `macro_archetype` enums — these are **orthogonal classification dimensions**, not competing labels (see Agent Design Patterns Pattern 13 for the composition rule).
---
# Validation Rules
Enforced by `src/agents/validators.py`. Failure triggers retry with error list appended to user message.
1. **JSON parseable** (strip markdown fences/preamble if present).
2. **Required fields**: `assessment_summary`, `reliability_score`, `analysis_confidence`, `caveats`, `key_factors`, `risks`, `narrative`, `structured_data`.
3. **reliability_score**: integer 0–100, within ±15 of `base_reliability_score`.
4. **analysis_confidence**: one of `[high, medium, low]`.
5. **caveats**: 1–4 non-empty string items. **Changed in v1.3** — was 1–6 in v1.0–v1.2.
6. **key_factors**: 2–4 items, each with `factor`, `importance`, `sentiment`, `evidence`. Evidence must start with a valid source ID token (`N\d+`, `FILING:(Business|MDA)`, `TRANSCRIPT:Q\d+`, `PEER_\d+`, or `SIGNALS`). FILING:RiskFactors removed in v1.3.
7. **risks**: 1–3 items, each with `risk`, `severity`, `likelihood`, `evidence`. Same source-ID rule as key_factors.
8. **narrative**: 720–1,080 characters (\~120–180 words).
9. **business_model_summary** (under `structured_data`): non-empty, 40–400 characters. Must contain at least one source-ID citation.
10. **revenue_mix_notes** (under `structured_data`): non-empty string (“not disclosed” is a valid value).
11. **competitive_position** (under `structured_data`): one of `[dominant, strong, average, weak, deteriorating]`.
12. **peer_comparison_summary** (under `structured_data`): non-empty, must contain a `PEER_\d+` token matching a peer in the payload.
13. **moat_assessment.moats** (under `structured_data`): 1–3 items, each with valid `type`/`strength`/`evidence` (evidence source-ID checked).
14. **moat_assessment.overall_moat_durability** (under `structured_data`): one of `[strong, moderate, weak, none]`.
15. **moat_assessment.moat_trend** (under `structured_data`): one of `[strengthening, stable, eroding]`.
16. **management_assessment** (under `structured_data`): one of `[excellent, competent, concerning, poor, insufficient_data]`.
17. **recent_developments** (under `structured_data`): 0–5 items, each with valid `news_id` (must exist in payload), `significance`, `sentiment`.
18. **competitive_threats** (under `structured_data`): 1–3 items, each with a source-ID citation. **Changed in v1.3** — was 1–4.
19. **growth_drivers** (under `structured_data`): 1–3 items, each with a source-ID citation. **Changed in v1.3** — was 1–4.
20. **thesis_archetype** (under `structured_data`): one of `[secular_grower, dividend_compounder, cyclical_recovery, quality_compounder, value_trap_candidate]`. **Changed in v1.3** — was 8 values; dropped `turnaround`, `disrupted_incumbent`, `special_situation`.
21. **Canadian data rule**: If payload’s `sedar_filing_available` is false, `caveats` must contain the exact string `"Canadian filing depth limited: SEDAR not accessible via API; analysis relies on FMP fundamentals and public news."`.
22. **No real company name in LLM response** (scope: direct-token anonymization check only): Validator checks that neither the canonical ticker nor the full company name appears in any LLM-produced text field before de-anonymization.
**Scope of rule 22**: This rule catches **direct** leakage of the canonical ticker and the full legal name only. It does NOT catch executive names (CEO, CFO), product names (iPhone, Azure), subsidiary names, ticker variations, or partial-name matches. Semantic leakage must be caught retrospectively by the Feedback Analyst. A configurable allowlist of per-company semantic tokens is an Open Question for future versions.
---
# Retry Prompt Injection
```plain text
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember: use COMPANY_X/TICKER_X tokens, cite sources by ID only (N{num}, FILING:Business, FILING:MDA, TRANSCRIPT:Q{n}, PEER_{num}, SIGNALS), do not retype news headlines or dates in recent_developments.
```
---
# Memory Brief Schema
The `{memory_brief}` placeholder is populated by the orchestrator from `StockAnalysisMemory` if the stock has been analyzed before. Schema (injected as a compact block, \~150 tokens max):
```plain text
Prior analysis ({days_since} days ago, timeline={prior_timeline}, score={composite_score}/100):
  thesis_archetype: {prior_thesis_archetype}
  competitive_position: {prior_competitive_position}
  overall_moat_durability: {prior_moat_durability} (trend: {prior_moat_trend})
  management_assessment: {prior_management_assessment}
  key_factors_then: [{top_2_factor_names}]
  top_risks_then: [{top_2_risk_names}]
  outcome: {return_pct if scored else "not yet scored"}
  what_changed_since: {short_list_of_events_since_prior}
```
**Timeline-scaled decay** (loosened in v1.3 to match retail cadence): Memory is included if **\<30 days old at short_term, \<90 days at medium_term, \<180 days at long_term**. Below marginal weight, the orchestrator omits the memory brief entirely rather than injecting a stale prior. v1.0–v1.2 used 14/60/180 — too tight on short_term for a monthly-or-slower retail cadence.
**Scored outcome bias correction**: When `outcome` is populated, it acts as ground truth the LLM can learn from within the run. This is distinct from the post-20-predictions accuracy brief which operates at the aggregate level across all stocks.
---
# Accuracy Brief (Post-20 Predictions)
Injected into system prompt before output schema. \~80 tokens.
```plain text
YOUR ACCURACY: {N} predictions scored. Bias: {bias_description}. Adjust: {calibration_instruction}. Strongest archetype: {best_archetype}. Weakest archetype: {worst_archetype}.
```
---
# Minimum Data Threshold
Per [🤖 Agents Overview](https://www.notion.so/Agents-Overview-325ea06798238004a793fc1720637698?pvs=21): **company name and at least one recent filing or earnings transcript**.
Below threshold: `base_reliability_score` will be \<20. Orchestrator sets `data_quality_assessment` to `"insufficient"` automatically. The LLM should focus narrative on what is missing and refuse to synthesize a thesis from nothing.
---
# Canadian vs US Data Asymmetry
<table header-row="true">
<tr>
<td>Source</td>
<td>US stocks</td>
<td>Canadian stocks</td>
</tr>
<tr>
<td>Filings</td>
<td>SEC 10-K/10-Q via FMP (full text)</td>
<td>SEDAR (no API) — only FMP fundamental digests available</td>
</tr>
<tr>
<td>Transcripts</td>
<td>Finnhub earnings transcripts</td>
<td>Limited — only large-cap TSX names have transcripts</td>
</tr>
<tr>
<td>News</td>
<td>FMP news + Finnhub AI sentiment</td>
<td>FMP news only, no AI sentiment</td>
</tr>
<tr>
<td>Analyst data</td>
<td>Full FMP analyst estimates</td>
<td>Thinner FMP coverage, especially small-caps</td>
</tr>
</table>
**Orchestrator behavior when ****`sedar_filing_available`**** is false**: `sedar_filing_available` injected into system prompt; validation rule 21 forces the exact templated caveat to appear in output. No separate reliability cap — the existing per-source penalties in `compute_base_reliability_researcher()` (filing recency, transcript/news/peer counts) already reflect Canadian coverage gaps without needing a blanket ceiling on top.
---
# Dual-Class and CIK Verification (orchestrator-only in v1.3)
The data pipeline runs a pre-flight check that `filing.cik` (or SEDAR equivalent) matches `stock.cik` before including the filing in the payload. If the company has multiple share classes (dual-class structure), the pipeline resolves the canonical entity ID, loads filings keyed on entity ID (not ticker), and sets `dual_class_flag: true` internally. **Changed in v1.3**: the LLM is no longer told about dual-class structure or CIK verification — both are pure data-pipeline concerns. When `dual_class_flag` is true, the orchestrator appends a standard caveat post-LLM in `merge_output_researcher` (see Orchestrator Merge Logic). For sub-\$250k retail portfolios this affects \<5% of names; the simplification cleans up the prompt for the other 95%.
For cross-listed stocks (e.g., Shopify on both TSX and NYSE) the pipeline prefers the primary listing’s filings (SEC filings for Shopify, since its primary listing is NYSE) and does not blend jurisdictions.
---
# How the System Prompt and User Message Interact
- **Source ID consistency**: System prompt defines exact citation tokens (`N\d+`, `FILING:Business`, `FILING:MDA`, `TRANSCRIPT:Q\d+`, `PEER_\d+`, `SIGNALS`) and user message uses them as labels. Validators 6/7/9/13/18/19 enforce this.
- **Every CoT step has a corresponding structured field**: Step 1 → `business_model_summary` + `revenue_mix_notes`; Step 2 → `competitive_position` + `peer_comparison_summary`; Step 3 → `moat_assessment`; Step 4 → `management_assessment` + `management_notes`; Step 5 → `recent_developments`; Step 6 → `caveats`; Step 7 → `narrative` + `thesis_archetype`.
- **Anonymization boundary**: System prompt and user message both use `COMPANY_X`/`TICKER_X` inside filing/transcript content. News headlines are not anonymized.
- **Non-thinking mode**: `enable_thinking: false`. Thinking mode produces `<think>` blocks that break JSON parsing.
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
<td>`{TICKER_X}`</td>
<td>Literal string</td>
<td>Anonymization token — real ticker is never sent to LLM</td>
</tr>
<tr>
<td>`{COMPANY_X}`</td>
<td>Literal string</td>
<td>Anonymization token — real name is never sent to LLM</td>
</tr>
<tr>
<td>`{sector}`</td>
<td>`DataBundle.company_info["sector"]`</td>
<td>Preserved (not anonymized)</td>
</tr>
<tr>
<td>`{timeline}`</td>
<td>`AnalysisContext.timeline`</td>
<td>Concrete: short_term / medium_term / long_term</td>
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
<td>`{news_window_days}`</td>
<td>Conditional by timeline</td>
<td>30 / 90 / 180</td>
</tr>
<tr>
<td>`{account_instruction}`</td>
<td>Conditional injection table</td>
<td>Selected by `build_messages()`</td>
</tr>
<tr>
<td>`{sector_moat_hint}`</td>
<td>Sector moat hint table (8 rows in v1.3)</td>
<td>Selected by `build_messages()`</td>
</tr>
<tr>
<td>`{base_reliability_score}`</td>
<td>`compute_base_reliability_researcher()`</td>
<td>Orchestrator pre-computed</td>
</tr>
<tr>
<td>`{reliability_reason}`</td>
<td>`compute_base_reliability_researcher()`</td>
<td>Human-readable reason string</td>
</tr>
<tr>
<td>`{sedar_filing_available}`</td>
<td>`DataBundle.research_sources.sedar_filing_available`</td>
<td>“true” or “false”; drives rule 21 (false triggers the caveat)</td>
</tr>
<tr>
<td>`{transcript_mgmt_excerpt}`</td>
<td>`DataBundle.research_sources.transcript_excerpts` — `content` of the current-quarter entry where `type="mgmt"`</td>
<td>Added 2026-08-07 (documentation-completeness gap); “not available in provided data” if no matching excerpt</td>
</tr>
<tr>
<td>`{transcript_guidance_excerpt}`</td>
<td>`DataBundle.research_sources.transcript_excerpts` — `content` of the current-quarter entry where `type="guidance"`</td>
<td>Added 2026-08-07; same fallback as above</td>
</tr>
<tr>
<td>`{transcript_qa_excerpt}`</td>
<td>`DataBundle.research_sources.transcript_excerpts` — `content` of the current-quarter entry where `type="qa"`</td>
<td>Added 2026-08-07; same fallback as above</td>
</tr>
<tr>
<td>`{peer_1_token}` / `{peer_2_token}`</td>
<td>Literal `PEER_1` / `PEER_2`</td>
<td>Used in CoT step 2</td>
</tr>
<tr>
<td>`{memory_brief}`</td>
<td>`StockAnalysisMemory` query</td>
<td>Empty string if none or below decay threshold (30/90/180 in v1.3)</td>
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
<tr>
<td>Payload placeholders</td>
<td>`DataBundle.research_sources.*`</td>
<td>Missing section → “not available in provided data”</td>
</tr>
</table>
---
# Testing Checklist
- [ ] Does `qwen3.6:35b-a3b` in non-thinking mode produce valid JSON matching the schema, with no leaked `<think>` reasoning tokens?
- [ ] Does thinking mode accidentally activate? (Check for `<think>` tags)
- [ ] Does `reliability_score` stay within ±15 of `base_reliability_score`?
- [ ] Does the narrative stay within 120–180 words (720–1,080 chars) and reference source IDs?
- [ ] Does every `key_factors` / `risks` / moat evidence field start with a valid source ID token (no `FILING:RiskFactors` — removed in v1.3)?
- [ ] Does `peer_comparison_summary` contain a `PEER_\d+` reference (rule 12)?
- [ ] Are `recent_developments` entries limited to `news_id`, `significance`, `sentiment` (not re-typed headlines/dates)?
- [ ] Does the orchestrator correctly hydrate news headlines/dates from IDs at merge time?
- [ ] Does the orchestrator correctly de-anonymize `COMPANY_X`/`TICKER_X` in all text fields?
- [ ] Does the LLM produce 1–3 moats, 0–5 recent_developments, 2–4 key_factors, 1–3 risks, 1–3 competitive_threats, 1–3 growth_drivers, 1–4 caveats? (Updated in v1.3.)
- [ ] Does the model refuse to recommend buy/sell?
- [ ] Does the model adapt focus based on timeline instruction?
- [ ] **Canadian asymmetry test**: Pass a Royal Bank of Canada (RY on TSX) payload with `sedar_filing_available=false`. Does the exact templated caveat appear (rule 21)? Does `base_reliability_score` land wherever the per-source penalties (filing/transcript/news/peer counts) put it, with no separate cap applied?
- [ ] **Contrastive analysis test**: Pass AAPL with MSFT + GOOGL peer blocks. Does `peer_comparison_summary` cite both peers with specific contrasts?
- [ ] **Sparse data test**: Pass a small-cap with 2 news items and no transcripts. Does `data_quality_assessment` resolve to “low”/“insufficient”?
- [ ] **Anomaly test**: Inject contradictory sources (filing says growth, news says layoffs). Does the contradiction appear in caveats?
- [ ] **Management insufficient data test**: Pass a payload with empty SIGNALS and no transcripts. Does `management_assessment` return `"insufficient_data"`?
- [ ] **Quiet stock test**: Pass a payload with no material news items. Does `recent_developments` come back empty (allowed) rather than with fabricated “no news” stubs?
- [ ] **Direct-token anonymization leakage test (rule 22)**: Pass AAPL. Does the LLM response (pre-deanonymization) contain the strings “Apple” or “AAPL” anywhere? Rule 22 should catch this.
- [ ] **Semantic anonymization leakage test**: Pass AAPL. Does the LLM response (pre-deanonymization) contain “iPhone”, “Tim Cook”, “Cupertino”, or similar company-specific tokens that rule 22 does NOT catch but indicate training-data leakage? Log for Feedback Analyst review — NOT a validation failure, but a quality signal.
- [ ] **Thesis archetype test (updated for v1.3)**: Pass a mix of 5 stocks (a secular grower, a dividend compounder, a cyclical recovery, a quality compounder, a suspected value trap). Does the model pick the right archetype for each? Confirm no v1.2 archetypes (turnaround, disrupted_incumbent, special_situation) appear.
- [ ] **Phase-narration leakage test**: Verify the narrative does not contain “Step 1:”, “First, looking at the business model…”, or similar CoT-recap phrasing.
- [ ] **pass2_view shape test**: the orchestrator’s compression produces every key listed in the Pass 2 Consumption View section with correct types. Missing keys fail Pass 2 ingestion.
- [ ] **Cross-agent news ID consistency test**: Run Researcher and Sentiment on the same stock. Confirm that a news item that appears in both payloads has the same `N{num}` identifier in both.
- [ ] **Dual-class caveat injection test (new in v1.3)**: Pass Shopify (dual-class) with `dual_class_flag=true`. Confirm: (a) LLM is NOT told about dual-class structure in the prompt; (b) post-merge `caveats` contains the standard dual-class caveat appended by the orchestrator.
- [ ] **Quality tier removal test (new in v1.3)**: Confirm that no headline lines in the payload contain `quality_tier:` labels and that the LLM does not reference tiers in any output field.
- [ ] **RiskFactors removal test (new in v1.3)**: Confirm that no payload contains a `RISK FACTORS DIGEST` section and that no LLM output cites `FILING:RiskFactors`. Validation rule 6 should reject the citation if it appears.
- [ ] **Fixture stocks**: AAPL (US large-cap, rich data, leakage risk), RY on TSX (Canadian financial, thinner data, asymmetry), Shopify on TSX (Canadian growth, dual-listed — uses SEC filings; also dual-class caveat path), a small-cap TSX name (sparse data edge case)
---
# Token Budget
<table header-row="true">
<colgroup>
<col>
<col>
<col width="446">
</colgroup>
<tr>
<td>Component</td>
<td>Budget</td>
<td>Notes</td>
</tr>
<tr>
<td>System prompt</td>
<td>\~700 tokens</td>
<td>CoT + rules + injection + schema + moat hint + memory reconciliation. \~120 tokens lighter than v1.2 (smaller archetype enum, smaller sector hint table, simplified rules 4 and 7, dual-class instructions removed).</td>
</tr>
<tr>
<td>User message</td>
<td>\~2,750 tokens</td>
<td>Filing digests (2×250 — RiskFactors removed) + transcript excerpts (2×400) + news block (\~250 — quality_tier labels removed) + 2 peers (\~400) + signals/flags (\~120) + headers/labels. \~250 tokens lighter than v1.2.</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~1,340 tokens</td>
<td>Smaller arrays (caveats 4 max, threats/drivers 3 max), 5-value archetype enum. \~60 tokens lighter than v1.2.</td>
</tr>
<tr>
<td>Total</td>
<td>\~4,790 tokens</td>
<td>\~430 tokens lighter than v1.2 budget.</td>
</tr>
</table>
**Model configuration**: `num_ctx`: 8192; `enable_thinking`: false (verify suppression is clean on `qwen3.6:35b-a3b` — see Pattern 10 caveat); VRAM: \~18-22GB at Q4_K_M (MoE, partial CPU offload expected); Expected latency: unverified estimate, re-measure after deployment (was \~16–25s per stock on the prior `qwen3:14b`).
**Cloud fallback**: If the payload exceeds 6K input tokens after pre-summarization (rare), fall back to Claude Sonnet 4.6.
---
# Open Questions / Future Improvements
- **Pre-summarization design**: The *implementation* of pre-summarization — particularly whether transcript excerpts use a small LLM classifier or rule-based extraction, and how to preserve exact management quotes without paraphrasing them — needs its own design in the Data Pipeline doc. This is the hardest remaining piece of this agent.
- **Few-shot example**: Start without one. If JSON retry rate \>10% after first batch, add a partial example (\~150 tokens) showing filled `moat_assessment` and `recent_developments`.
- **Anonymization edge cases — semantic-leakage allowlist**: Rule 22 only catches direct ticker/name leakage. If the Feedback Analyst’s retrospective review shows semantic leakage is common on well-known names, add a per-company allowlist that the data pipeline substitutes out before the payload reaches the agent.
- **News deduplication**: Rewritten headlines across wire services can inflate news count. SimHash dedup before payload assembly is a future improvement.
- **Multi-language filings**: Dual-language filings for Canadian stocks (EN/FR). Currently English-only. Flag for Phase 2.
- **Thesis archetype enum evolution**: The 5-entry enum (down from 8 in v1.3) is still a guess. After 50+ predictions, review whether the trim was right or whether `turnaround` / `special_situation` need to come back for any retail-relevant patterns.
- **Orthogonal archetype composition**: When Sentiment v1.1 / Macro v1.1 introduce candidate `sentiment_archetype` / `macro_archetype` enums, the composition rule (these are orthogonal, not competing) must be specified in Agent Design Patterns Pattern 13.
- **RiskFactors re-introduction**: If retrospective review shows that risk-section signals are being missed (e.g., for distressed names or pre-earnings warnings), reconsider re-introducing a tighter (≤100-token) RiskFactors digest. v1.3 removed it on the bet that retail-held names rarely have non-boilerplate RiskFactors content.
---
# Related Documents
- [Agent Design Patterns](https://www.notion.so/Agent-Design-Patterns-33fea067982380e6a055d3ed74c8eada?pvs=21) — the 13 validated patterns this prompt applies, including the anonymization principle (Researcher is the only Pass 1 agent that anonymizes, per the pattern)
- [📊 Fundamental Analyst Agent Prompt](https://www.notion.so/Fundamental-Analyst-Agent-Prompt-333ea06798238063a42bed3e2c132ea7?pvs=21) — reference implementation of the same patterns for a quantitative agent
- [🤖 Agents Overview](https://www.notion.so/Agents-Overview-325ea06798238004a793fc1720637698?pvs=21) — AgentOutput schema, pass structure, token budgets, minimum data thresholds
- [Technical Design](https://www.notion.so/Technical-Design-325ea067982380cb8374da20e14c8923?pvs=21) — DataBundle definition, orchestrator layout
- [🔄 Data Pipeline](https://www.notion.so/Data-Pipeline-341ea06798238191a875f3f4f63082f7?pvs=21) — globally-assigned news IDs (`N{num}`) are specified here; Researcher’s Source ID stability note references this contract
---

_Source: Notion — Project Stock Picker → Agent Prompts → Stock Researcher Agent Prompt. Downloaded 2026-07-02._
