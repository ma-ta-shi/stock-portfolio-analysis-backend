**Agent**: Macro Economist
**Status:** Draft v1.2 — retail-simplified
**Phase:** Phase 2
**Pass**: Pass 1 (Research & Analysis)
**Role**: Assess the macroeconomic environment (rates, inflation, growth, currency, commodities) and translate it into specific headwinds and tailwinds for this stock’s sector and business.
**Prompt Version**: v1.2
**Target Model**: `qwen3.6:35b-a3b` via Ollama (non-thinking mode — verify `enable_thinking:false` fully suppresses reasoning on this model; see Agent Design Patterns Pattern 10 caveat)
**Audience**: Canadian retail investors with total portfolio values under \$250k CAD, holding stocks across TFSA / RRSP / Trading / General accounts. Institutional-grade complexity that doesn’t pay rent at this scale has been trimmed.
**Token Budget**: \~450 system prompt \| \~1,150 data input \| \~750 expected output \| \~2,350 total
---
# What Changed in v1.2
v1.2 simplifies the prompt for the actual user base — individual retail Canadian investors with portfolio values \<\$250k CAD across TFSA / RRSP / Trading accounts. The cuts target institutional-grade complexity that doesn’t change retail decisions at this scale. Behavior on Canadian and US large-cap names should be substantively unchanged; behavior on geopolitically-exposed names will be lighter-touch.
**Removed entirely**:<br>- `geopolitical_risks` structured field with probability / impact / `G{num}` citation grammar. Material geopolitical concerns are now mentioned in `sector_headwinds` with a brief evidence note. Drops the entire G\{num\} citation system, the geopolitical news payload block, cross-validator #1, the dedicated CoT step, and the 0-floor array logic.<br>- DXY (USD trade-weighted index). Redundant for a Canadian investor who already has CAD/USD.<br>- `CB_COMM` as a separate citation token and payload block. Material central bank stance is rolled into the `RATE` block as an optional one-liner. The orchestrator’s mechanical rate-direction classification (`tightening | pausing | easing`) carries the bulk of the signal.
**Compressed**:<br>- Bond yields: 2yr / 5yr / 10yr × US + Canada → 10yr + `curve_shape` (`normal | flat | inverted`) per jurisdiction.<br>- Currency exposure hint: 5 buckets → 3 (`primarily_cad`, `mixed`, `primarily_usd`). Data-unavailable case folds into the conservative default for the primary listing.<br>- `overall_macro_environment`: 5-point scale → 3-point (`favorable | neutral | unfavorable`). Retail at this scale doesn’t change behavior between “very_favorable” and “favorable”.<br>- Narrative: 120–180 words → 80–120 words.<br>- CoT: 9 steps → 6 steps. Geopolitical step folded into headwinds step; data-sanity step folded into Rule 7.<br>- Sector macro sensitivity table: each row trimmed to 3–4 mechanisms.<br>- Account instructions: tightened to \~15 words each.
---
# Design Decisions & Rationale (v1.2)
The numbered decisions from v1.1 carry forward except where noted. Decisions specifically modified by v1.2:
1. Macro data is pre-computed passthrough; LLM produces the causal chain (unchanged)
2. Sector macro sensitivity hint (Pattern 9 adaptation) — trimmed
Each sector row in the hint table is now 3–4 mechanisms rather than 5. Retail-actionable mechanisms only. Full table under Conditional Injection Reference.
1. Citation grammar for structured macro inputs — narrowed
Allowed tokens: `RATE`, `YIELD`, `CPI`, `GDP`, `EMPLOY`, `FX`, `VIX`, `COMMOD`. `CB_COMM` and `G{num}` are removed. Material CB stance (when present) is delivered inside the `RATE` block as a one-liner annotation; cite `RATE` for it.
1. Sector-specific commodity coverage is an availability flag, not a reliability cap (unchanged)
2. Cross-listed currency sensitivity — 3-bucket hint
Three buckets: `primarily_cad` (USD revenue \<30%), `mixed` (30–60%), `primarily_usd` (\>60%). When `usd_revenue_exposure_pct` is unavailable, the orchestrator picks the conservative default for the primary listing (US listing → `primarily_usd`; Canadian listing → `primarily_cad`) and notes “exposure not disclosed” in the hint text.
1. Geopolitical risks — REMOVED as a structured field
Material geopolitical concerns surface as `sector_headwinds` items with the relevant payload citation (typically `COMMOD` for sanctions / supply-chain risks, or none if the concern doesn’t tie to a structured block — in which case the headwind is omitted rather than fabricated). No probability / impact taxonomy. No dedicated payload block. No `G{num}` IDs.
1. `sector_cycle_position` is a classification the LLM owns (unchanged)
2. No anonymization (unchanged)
3. Tighter timeline windows than Researcher — narrowed scope
Timeline still scopes the macro data freshness expectations in `base_reliability_score`. With CB commentary removed as a separate block, the timeline-scoped CB window is gone; macro series freshness windows remain (policy rate ≤14d, CPI ≤45d, GDP ≤120d).
1. Mechanical-vs-qualitative reliability split (unchanged)
2. Memory decay matches Researcher / Sentiment — thresholds corrected
Memory included if \<30 days old at short_term, \<90 days at medium_term, \<180 days at long_term. # v1.2c: updated from stale v1.1 values (14/60/180) to match Researcher v1.3 and Sentiment v1.2c cross-agent standardisation.
1. Local-primary model, no planned escalation path (unchanged)
2. Secondary-order impact rationales are OK, but must be traceable (unchanged)
3. `overall_macro_environment` is a 3-point scale — narrowed
`favorable | neutral | unfavorable`. Pass 2 advocates translate favorability into direction.
1. Jurisdiction selection is informational, not prescriptive (unchanged from v1.1)
---
## Input & Sources
**Raw data inputs** (from Agent Data Mapping):
- BoC rates, CAD/USD, GoC yields — Bank of Canada Valet (CA) / — (US)
- US macro (Fed funds, CPI, GDP, unemployment) — — (CA) / FRED (US)
- Canadian CPI, unemployment — FRED via OECD (CA) / — (US)
- VIX — — (CA) / FRED VIXCLS (US)
- General market news — Finnhub /news?category=general (CA) / Finnhub /news?category=general (US)
- Canadian sector performance — openbb-tmx .index.sectors (CA) / — (US)
**From the pre-computation pipeline** (delivered pre-assembled in the User Message data payload — the agent never calls these APIs directly):
- Policy rate direction (`tightening|pausing|easing`) computed from BoC/Fed 90d delta — agent reads direction + delta, doesn't compute it
- 10yr yield curve shape per jurisdiction (`normal|flat|inverted`) for US and Canada
- CPI trend (`rising|stable|falling`) from 3-month delta, US and Canada in parallel
- GDP QoQ annualized + 4-quarter trend, and unemployment level + 6-month delta, both jurisdictions
- CAD/USD level, 90-day % change, and trend classification (`cad_strengthening|stable|cad_weakening`)
- VIX level plus volatility regime classification (`low|elevated|high`; low\<15, elevated 15-25, high\>25) and 30-day average
- Sector-relevant commodity line (WTI for Energy, copper/gold for Materials, natural gas for Utilities) gated by `sector_commodity_relevant` flag — omitted entirely for sectors with no tracked commodity
- Optional one-line central bank stance annotation (`{cb_stance_note}`) folded into the RATE block when material, citable as `RATE`
- `base_reliability_score` (0-100) with a reasons string, computed from data freshness/availability across policy rate, CPI, GDP, unemployment, FX, VIX, and sector commodity — the LLM only applies a ±15pt qualitative adjustment on top
- Jurisdiction selection (Canadian vs US primary) and the currency-exposure bucket (`primarily_cad|mixed|primarily_usd`) pre-resolved from primary exchange/domicile and `usd_revenue_exposure_pct`, with a conservative default when exposure is undisclosed
---
# System Prompt (v1.2)
Invoke with `enable_thinking: false`.

**Runtime prompt:** see [`prompts/macro_economist/v1.2.txt`](../../prompts/macro_economist/v1.2.txt)
## Conditional Injection Reference
**Timeline** (inject based on `AnalysisContext.timeline`):
<table header-row="true">
<colgroup>
<col>
<col width="568">
</colgroup>
<tr>
<td>Value</td>
<td>`{timeline_instruction}`</td>
</tr>
<tr>
<td>`short_term`</td>
<td>`SHORT-TERM (1-4 wks). Weight upcoming data releases (CPI, employment, central bank decisions) and event-driven catalysts. De-emphasize secular trends.`</td>
</tr>
<tr>
<td>`medium_term`</td>
<td>`MEDIUM-TERM (1-12 mo). Balance sector rotation and business-cycle positioning with near-term catalysts. Rate direction and earnings-cycle sensitivity matter most.`</td>
</tr>
<tr>
<td>`long_term`</td>
<td>`LONG-TERM (1-5+ yrs). Emphasize secular trends (demographics, energy transition, deglobalization, structural inflation) and durable regime shifts. De-emphasize quarterly noise.`</td>
</tr>
</table>
**Account** (inject based on `AnalysisContext.account_type`):
<table header-row="true">
<colgroup>
<col>
<col width="570">
</colgroup>
<tr>
<td>Value</td>
<td>`{account_instruction}`</td>
</tr>
<tr>
<td>`tfsa`</td>
<td>`TFSA. Long horizon, gains tax-free. Weight durable compounding and macro regimes that support it.`</td>
</tr>
<tr>
<td>`rrsp`</td>
<td>`RRSP. Long-horizon retirement vehicle. Weight rate environment's impact on dividend sustainability and long-term sector earnings power.`</td>
</tr>
<tr>
<td>`trading`</td>
<td>`Trading account. 50% capital-gains inclusion. Near-term macro catalysts (CB meetings, CPI, GDP prints) are more actionable.`</td>
</tr>
<tr>
<td>`general`</td>
<td>`General. Note which macro factors matter most for each account type (TFSA / RRSP / Trading) given this stock's sector.`</td>
</tr>
</table>
**Sector macro sensitivity hint** (inject based on `DataBundle.company_info.sector`). The orchestrator also uses this table to determine `sector_commodity_relevant` for reliability scoring.
<table header-row="true">
<colgroup>
<col>
<col width="408">
<col>
</colgroup>
<tr>
<td>Sector</td>
<td>`{sector_macro_hint}`</td>
<td>Relevant commodity</td>
</tr>
<tr>
<td>Financials</td>
<td>`net_interest_margin, yield_curve_slope, credit_loss_provisioning, unemployment_trend`</td>
<td>—</td>
</tr>
<tr>
<td>Real Estate</td>
<td>`10yr_yield (cap_rate_driver), mortgage_rates, inflation_pass_through_to_rents`</td>
<td>—</td>
</tr>
<tr>
<td>Utilities</td>
<td>`10yr_yield (bond_proxy), natural_gas_price, regulatory_rate_base`</td>
<td>natural gas</td>
</tr>
<tr>
<td>Energy</td>
<td>`oil_price (WTI), usd_strength, inventory_levels`</td>
<td>oil (WTI)</td>
</tr>
<tr>
<td>Materials</td>
<td>`industrial_commodity_prices, usd_strength, china_demand`</td>
<td>copper / gold</td>
</tr>
<tr>
<td>Industrials</td>
<td>`PMI, capex_cycle, interest_rates (financing)`</td>
<td>copper (bellwether)</td>
</tr>
<tr>
<td>Consumer Staples</td>
<td>`inflation_pass_through_to_price, input_costs (agriculture), unemployment_trend`</td>
<td>—</td>
</tr>
<tr>
<td>Consumer Discretionary</td>
<td>`unemployment_trend, real_wage_growth, interest_rates (big-ticket financing)`</td>
<td>—</td>
</tr>
<tr>
<td>Healthcare</td>
<td>`demographic_trends, regulatory_environment, usd_strength (if multinational)`</td>
<td>—</td>
</tr>
<tr>
<td>Technology</td>
<td>`interest_rates (duration risk), usd_strength (if multinational), capex_cycle`</td>
<td>—</td>
</tr>
<tr>
<td>Communication Services</td>
<td>`interest_rates (capex financing), advertising_cycle, regulatory_environment`</td>
<td>—</td>
</tr>
<tr>
<td>Default / Unknown</td>
<td>`interest_rates, inflation, economic_growth, usd_strength`</td>
<td>—</td>
</tr>
</table>
**Currency exposure hint** (3 buckets):
<table header-row="true">
<colgroup>
<col>
<col width="468">
</colgroup>
<tr>
<td>Condition</td>
<td>`{currency_exposure_hint}`</td>
</tr>
<tr>
<td>Primary listing US</td>
<td>`Currency exposure: primarily_usd. Reports in USD, primarily US market. CAD/USD moves affect a Canadian investor's after-tax return but are not a direct operational factor.`</td>
</tr>
<tr>
<td>Canadian listing, USD rev \<30% (or unknown)</td>
<td>`Currency exposure: primarily_cad. Reports in CAD, primarily Canadian revenue. CAD/USD moves mainly affect input-cost lines (USD-priced commodities or imports) and price relative to US peers.`</td>
</tr>
<tr>
<td>Canadian listing, USD rev 30-60%</td>
<td>`Currency exposure: mixed. CAD/USD moves partially hedge revenue against cost pressure. Weakening CAD can boost reported revenue but raise USD-priced input costs.`</td>
</tr>
<tr>
<td>Canadian listing, USD rev \>60%</td>
<td>`Currency exposure: primarily_usd. Reports in CAD but earns mostly USD revenue. Weakening CAD boosts CAD-translated revenue and earnings; strengthening CAD is a translation headwind. Often the dominant macro factor for cross-listed Canadian growth names.`</td>
</tr>
</table>
---
# Output Schema (v1.2 — LLM-produced fields only)
The orchestrator populates metadata, derived enum fields, all numeric macro passthroughs, and the orchestrator-computed trend classifications. The LLM produces only the interpretive fields below.
```javascript
{
  "assessment_summary": "≤80 words. Most important macro finding first.",
  "reliability_score": 0-100,
  "analysis_confidence": "high|medium|low",
  "caveats": ["specific data gaps, contradictions, stale series, or qualitative concerns"],
  "key_factors": [
    {"factor": "name", "importance": "high|medium|low", "sentiment": "positive|negative|neutral", "evidence": "≤15 words, starts with citation token"}
  ],
  "risks": [
    {"risk": "name", "severity": "high|medium|low", "likelihood": "high|medium|low", "evidence": "≤15 words, starts with citation token"}
  ],
  "narrative": "80-120 word synthesis. Reference specific citations. Not a step-by-step recap.",
  "structured_data": {
    "interest_rate_environment": {
      "impact_on_stock": "positive|neutral|negative",
      "rationale": "1-2 sentences. Contains RATE or YIELD token and names the sector-specific transmission mechanism."
    },
    "inflation_environment": {
      "impact_on_stock": "positive|neutral|negative",
      "rationale": "1-2 sentences. Contains CPI token and names the sector-specific transmission mechanism."
    },
    "economic_growth": {
      "outlook": "accelerating|stable|decelerating|recessionary",
      "impact_on_stock": "positive|neutral|negative",
      "rationale": "1-2 sentences. Contains GDP or EMPLOY token."
    },
    "currency_impact": {
      "impact_on_stock": "positive|neutral|negative",
      "rationale": "1-2 sentences. Contains FX token. Applies the currency exposure hint from rule 6."
    },
    "sector_cycle_position": "early_cycle|mid_cycle|late_cycle|recession",
    "sector_tailwinds": ["≤20 words per item, contains a citation token"],
    "sector_headwinds": ["≤20 words per item, contains a citation token"],
    "overall_macro_environment": "favorable|neutral|unfavorable"
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
<td>`sector_tailwinds`</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>`sector_headwinds`</td>
<td>1</td>
<td>4</td>
</tr>
</table>
Tailwinds / headwinds have ≥1 minimums because any stock in any macro environment has at least one of each.
### Fields the orchestrator populates (NOT produced by LLM)
- **Metadata**: `agent_name`, `agent_pass`, `analysis_context`, `data_sources_used`
- **Numeric passthrough + trend classifications** (merged into `structured_data`):
	- `interest_rate_environment.current_direction`: `tightening|pausing|easing` from policy-rate 90d delta
	- `interest_rate_environment.current_rate_pct`, `interest_rate_environment.rate_90d_delta_bp`
	- `inflation_environment.trend`: `rising|stable|falling` from CPI YoY 3m delta
	- `inflation_environment.cpi_yoy_pct`, `inflation_environment.core_cpi_yoy_pct`
	- `economic_growth.gdp_qoq_annualized_pct`, `economic_growth.unemployment_rate_pct`
	- `currency_impact.cad_usd_trend`: `cad_strengthening|stable|cad_weakening` from CAD/USD 90d delta
	- `currency_impact.cad_usd_level`, `currency_impact.cad_usd_90d_change_pct`
	- `volatility_regime`: `low|elevated|high` from VIX (VIX\<15=low, 15-25=elevated, \>25=high)
	- `commodity_context.relevant_commodity_name`, `commodity_context.level`, `commodity_context.direction`
	- `yield_context.us_10yr_pct`, `yield_context.ca_10yr_pct`, `yield_context.us_curve_shape`, `yield_context.ca_curve_shape`
- **Derived enums**:
	- `data_quality_assessment`: ≥70=“high”, 40–69=“medium”, 20–39=“low”, \<20=“insufficient”
	- `reliability_factors.data_completeness`: derived from source-presence checklist
	- `reliability_factors.data_freshness`: derived from max age across core macro series
---
# Orchestrator Pre-computation: `base_reliability_score`
```python
def compute_base_reliability_macro(data_bundle: DataBundle) -> tuple[int, str]:
    score = 100
    reasons = []
    src = data_bundle.macro_sources

    # Core macro series freshness (policy rate + CPI are non-negotiable)
    if src.policy_rate_age_days is None:
        score -= 30; reasons.append("no policy rate data")
    elif src.policy_rate_age_days > 14:
        score -= 10; reasons.append(f"policy rate{src.policy_rate_age_days}d old")

    if src.cpi_age_days is None:
        score -= 25; reasons.append("no CPI data")
    elif src.cpi_age_days > 45:
        score -= 10; reasons.append(f"CPI{src.cpi_age_days}d old")

    if src.gdp_age_days is None:
        score -= 10; reasons.append("no GDP data")
    elif src.gdp_age_days > 120:
        score -= 5; reasons.append(f"GDP{src.gdp_age_days}d old")

    if src.unemployment_age_days is None:
        score -= 8; reasons.append("no unemployment data")

    if src.cad_usd_age_days is None:
        score -= 5; reasons.append("no FX data")

    if src.vix_age_days is None:
        score -= 3; reasons.append("no VIX data")

    # Sector-specific commodity availability
    if src.sector_commodity_relevant:
        if src.sector_commodity_age_days is None:
            score -= 12
            reasons.append(f"{src.sector_commodity_name} missing (sector-relevant)")
        elif src.sector_commodity_age_days > 14:
            score -= 5
            reasons.append(f"{src.sector_commodity_name}{src.sector_commodity_age_days}d old")

    # Bond yields (10yr only — curve shape derives from it via separate signal)
    if not src.bond_yields_available:
        score -= 4; reasons.append("10yr yield unavailable")

    reason_str = "; ".join(reasons) if reasons else "all macro series present and current"
    return (max(0, score), reason_str)
```
**Derived quality enums** unchanged in logic from v1.1; the source-presence checklist drops the CB commentary entry.
---
# User Message (Data Payload)
```javascript
{canonical_ticker} ({company_name}) | {sector} | {primary_exchange} | {currency}
As of: {data_timestamp} | Timeline: {timeline}

RATE (policy rates, as of {rate_date}):
  BoC overnight: {boc_rate_pct}% | 90d delta: {boc_rate_90d_delta_bp}bp | direction: {boc_direction}
  Fed Funds: {fed_rate_pct}% | 90d delta: {fed_rate_90d_delta_bp}bp | direction: {fed_direction}
  {cb_stance_note}

YIELD (10yr government bond yields, as of {yield_date}):
  US 10yr: {us_10yr}% | curve: {us_curve_shape}
  Canada 10yr: {ca_10yr}% | curve: {ca_curve_shape}

CPI (inflation, as of {cpi_date}):
  US CPI YoY: {us_cpi_yoy}% | Core CPI YoY: {us_core_cpi_yoy}% | 3m delta: {us_cpi_3m_delta}pp | trend: {us_cpi_trend}
  Canada CPI YoY: {ca_cpi_yoy}% | 3m delta: {ca_cpi_3m_delta}pp | trend: {ca_cpi_trend}

GDP (growth, as of {gdp_date}):
  US GDP QoQ annualized: {us_gdp_qoq}% | last 4Q: {us_gdp_4q_trend}
  Canada GDP QoQ annualized: {ca_gdp_qoq}% | last 4Q: {ca_gdp_4q_trend}

EMPLOY (employment, as of {employ_date}):
  US unemployment: {us_unemp}% | 6m change: {us_unemp_6m_delta}pp
  Canada unemployment: {ca_unemp}% | 6m change: {ca_unemp_6m_delta}pp

FX (currency, as of {fx_date}):
  CAD/USD: {cad_usd} | 90d change: {cad_usd_90d_pct}% | trend: {cad_usd_trend}

VIX (volatility, as of {vix_date}):
  Level: {vix} | regime: {vix_regime} | 30d average: {vix_30d_avg}

COMMOD (commodity relevant to {sector}, as of {commod_date}):
  {commodity_lines}
  {commodity_na_note}

DATA AVAILABILITY FLAGS:
  usd_revenue_exposure_pct: {usd_revenue_exposure_pct}
  sector_commodity_relevant: {sector_commodity_relevant}
  sector_commodity_name: {sector_commodity_name}
  bond_yields_available: {bond_yields_available}
  missing_sources: {missing_sources_list}

{data_pipeline_warnings}
```
**`{cb_stance_note}`**: Optional one-liner inserted into the RATE block when there is a material CB stance signal in the recent window (e.g., `"BoC: dovish pivot signaled at April meeting"`). Cite as `RATE`. If no material signal, the line is empty. Replaces the v1.1 `CB_COMM` block.
**Commodity lines**: Populated only when `sector_commodity_relevant=true`. For Energy: `WTI: {wti_level} | 90d change: {wti_90d_pct}% | direction: {wti_direction}`. For Materials: copper / gold. For Utilities: natural gas. For sectors with no relevant commodity, `{commodity_na_note}` reads `no commodity tracked for {sector}`.
**Pre-summarization budgets**: Core macro blocks are compact numeric records. Total payload target \~1,150 tokens before warnings.
---
# Orchestrator Merge Logic
```python
def merge_output_macro(llm_response: dict, data_bundle: DataBundle,
                       context: AnalysisContext, base_score: int) -> dict:
    derived = derive_quality_enums_macro(base_score, data_bundle)
    src = data_bundle.macro_sources

    # Jurisdiction selection based on primary exchange + domicile, not ticker suffix
    primary_is_canadian = (
        data_bundle.company_info.get("primary_exchange") in {"TSX", "TSXV", "CSE"}
        or data_bundle.company_info.get("country") == "CA"
    )

    orchestrator_rate = {
        "current_direction": compute_rate_direction(
            src.boc_rate_90d_delta_bp if primary_is_canadian else src.fed_rate_90d_delta_bp
        ),
        "current_rate_pct": src.boc_rate_pct if primary_is_canadian else src.fed_rate_pct,
        "rate_90d_delta_bp": src.boc_rate_90d_delta_bp if primary_is_canadian else src.fed_rate_90d_delta_bp,
    }
    orchestrator_inflation = {
        "trend": compute_cpi_trend(
            src.ca_cpi_3m_delta if primary_is_canadian else src.us_cpi_3m_delta
        ),
        "cpi_yoy_pct": src.ca_cpi_yoy if primary_is_canadian else src.us_cpi_yoy,
        "core_cpi_yoy_pct": src.us_core_cpi_yoy,
    }
    orchestrator_growth = {
        "gdp_qoq_annualized_pct": src.ca_gdp_qoq if primary_is_canadian else src.us_gdp_qoq,
        "unemployment_rate_pct": src.ca_unemp if primary_is_canadian else src.us_unemp,
    }
    orchestrator_currency = {
        "cad_usd_trend": compute_fx_trend(src.cad_usd_90d_pct),
        "cad_usd_level": src.cad_usd,
        "cad_usd_90d_change_pct": src.cad_usd_90d_pct,
    }
    orchestrator_volatility = {
        "volatility_regime": classify_vix(src.vix),
    }
    orchestrator_yield = {
        "us_10yr_pct": src.us_10yr,
        "ca_10yr_pct": src.ca_10yr,
        "us_curve_shape": src.us_curve_shape,
        "ca_curve_shape": src.ca_curve_shape,
    }
    orchestrator_commod = {
        "relevant_commodity_name": src.sector_commodity_name if src.sector_commodity_relevant else None,
        "level": src.sector_commodity_level if src.sector_commodity_relevant else None,
        "direction": src.sector_commodity_direction if src.sector_commodity_relevant else None,
        "relevant_for_sector": src.sector_commodity_relevant,
    }

    sd = llm_response["structured_data"]
    merged_structured = {
        "interest_rate_environment": {**sd["interest_rate_environment"], **orchestrator_rate},
        "inflation_environment": {**sd["inflation_environment"], **orchestrator_inflation},
        "economic_growth": {**sd["economic_growth"], **orchestrator_growth},
        "currency_impact": {**sd["currency_impact"], **orchestrator_currency},
        "volatility_regime": orchestrator_volatility["volatility_regime"],
        "yield_context": orchestrator_yield,
        "commodity_context": orchestrator_commod,
        "sector_cycle_position": sd["sector_cycle_position"],
        "sector_tailwinds": sd["sector_tailwinds"],
        "sector_headwinds": sd["sector_headwinds"],
        "overall_macro_environment": sd["overall_macro_environment"],
    }

    return {
        "agent_name": "macro_economist",
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
        "data_sources_used": derive_sources_macro(data_bundle),
    }
```
Geopolitical-risks validation and `G{num}` cross-checking are removed. Jurisdiction selection logic is unchanged from v1.1.
---
# Pass 2 Consumption View
```python
pass2_view = {
    "overall_macro_environment": str,           # favorable|neutral|unfavorable
    "interest_rate_impact": str,
    "interest_rate_direction": str,
    "interest_rate_rationale": str,
    "inflation_impact": str,
    "inflation_trend": str,
    "inflation_rationale": str,
    "growth_outlook": str,
    "growth_impact": str,
    "growth_rationale": str,
    "currency_impact": str,
    "currency_trend": str,
    "currency_rationale": str,
    "volatility_regime": str,
    "sector_cycle_position": str,
    "sector_tailwinds": [str],
    "sector_headwinds": [str],
    "commodity_context": {
        "relevant_commodity_name": "str|None",
        "direction": "str|None",
        "relevant_for_sector": bool,
    },
}
```
`geopolitical_risks` is removed from the view. Pass 2 advocates that previously consumed it now pick up material geopolitical / trade concerns from `sector_headwinds`.
---
# Validation Rules
1. **JSON parseable** (strip markdown fences / preamble if present).
2. **Required fields**: `assessment_summary`, `reliability_score`, `analysis_confidence`, `caveats`, `key_factors`, `risks`, `narrative`, `structured_data`.
3. **reliability_score**: integer 0–100, within ±15 of `base_reliability_score`.
4. **analysis_confidence**: one of `[high, medium, low]`.
5. **caveats**: 1–4 non-empty string items. # v1.2c: was 1–6 in v1.0–v1.2; aligned with Researcher v1.3 standardisation across all Pass 1 agents.
6. **key_factors**: 2–4 items. Evidence must start with a valid citation token (`RATE`, `YIELD`, `CPI`, `GDP`, `EMPLOY`, `FX`, `VIX`, `COMMOD`). Evidence ≤15 words (soft).
7. **risks**: 1–3 items. Same citation + length rules.
8. **narrative**: 480–720 characters (≈80–120 words).
9. **interest_rate_environment.impact_on_stock**: one of `[positive, neutral, negative]`.
10. **interest_rate_environment.rationale**: non-empty, contains `RATE` or `YIELD`.
11. **inflation_environment.impact_on_stock**: one of `[positive, neutral, negative]`.
12. **inflation_environment.rationale**: non-empty, contains `CPI`.
13. **economic_growth.outlook**: one of `[accelerating, stable, decelerating, recessionary]`.
14. **economic_growth.impact_on_stock**: one of `[positive, neutral, negative]`.
15. **economic_growth.rationale**: non-empty, contains `GDP` or `EMPLOY`.
16. **currency_impact.impact_on_stock**: one of `[positive, neutral, negative]`.
17. **currency_impact.rationale**: non-empty, contains `FX`.
18. **sector_cycle_position**: one of `[early_cycle, mid_cycle, late_cycle, recession]`.
19. **sector_tailwinds**: 1–4 items, each containing at least one valid citation token. ≤20 words (soft).
20. **sector_headwinds**: same as sector_tailwinds.
21. **overall_macro_environment**: one of `[favorable, neutral, unfavorable]`.
**Cross-validators**:
1. **Citation tokens match payload availability**: If `missing_sources` includes a source (e.g., `COMMOD` because no sector commodity is tracked), evidence fields and rationales citing that token are rejected.
2. **Sector-commodity conditional**: If `sector_commodity_relevant=true` and the LLM’s `sector_tailwinds` + `sector_headwinds` combined contain no `COMMOD` citation, emit a non-blocking warning.
The v1.1 cross-validator #1 (`G{num}` payload existence check) is removed.
---
# Retry Prompt Injection
```javascript
VALIDATION ERRORS:
{validation_error_list}

Fix these issues. Respond with corrected JSON only. Remember: cite sources by block label only (RATE, YIELD, CPI, GDP, EMPLOY, FX, VIX, COMMOD), do not recompute numeric passthrough fields, every rationale must name the sector-specific transmission mechanism.
```
---
# Memory Brief Schema
```javascript
Prior macro analysis ({days_since} days ago, timeline={prior_timeline}, score={composite_score}/100):
  overall_macro_environment_then: {prior_overall}
  interest_rate_impact_then: {prior_rate_impact}
  inflation_impact_then: {prior_inflation_impact}
  sector_cycle_position_then: {prior_sector_cycle}
  key_tailwinds_then:[{top_2_tailwind_phrases}]
  key_headwinds_then:[{top_2_headwind_phrases}]
  outcome: {return_pct if scored else "not yet scored"}
  what_changed_since: {short_list_of_material_macro_events_since_prior}
```
Decay thresholds: included if \<30d (short), \<90d (medium), \<180d (long). # v1.2c: corrected from stale v1.1 values (\<14d/\<60d/\<180d) to match Researcher v1.3 and Sentiment v1.2c. (See Design Decision #11.)
---
# Accuracy Brief (Post-20 Predictions)
```javascript
YOUR ACCURACY: {N} macro calls scored. Bias: {bias_description}. Adjust: {calibration_instruction}. Strongest signal: {best_signal_type}. Weakest signal: {worst_signal_type}.
```
Empty until ≥20 scored predictions.
---
# Minimum Data Threshold
Unchanged from v1.1: **current interest rate + current inflation**. Below threshold, `base_reliability_score` will be \<20 and `data_quality_assessment` becomes `"insufficient"`.
---
# Token Budget
<table header-row="true">
<colgroup>
<col>
<col>
<col width="417">
</colgroup>
<tr>
<td>Component</td>
<td>Budget</td>
<td>Notes</td>
</tr>
<tr>
<td>System prompt</td>
<td>\~450 tokens</td>
<td>9 rules + 6-step CoT + conditional injections + schema + accuracy brief + memory reconciliation</td>
</tr>
<tr>
<td>User message</td>
<td>\~1,150 tokens</td>
<td>Core macro blocks (\~600) + simplified yield (\~80) + commodity (\~80) + flags (\~80) + headers (\~310)</td>
</tr>
<tr>
<td>Expected output</td>
<td>\~750 tokens</td>
<td>Environment blocks (4 × \~70) + cycle position + tailwinds/headwinds (\~160) + narrative (\~120) + key_factors/risks (\~140) + caveats + metadata</td>
</tr>
<tr>
<td>**Total**</td>
<td>**\~2,350 tokens**</td>
<td>\~25% reduction vs v1.1</td>
</tr>
</table>
**Model configuration**: `num_ctx`: 8192 \| `enable_thinking`: false (verify suppression is clean on `qwen3.6:35b-a3b` — see Pattern 10 caveat) \| VRAM \~18-22GB at Q4_K_M (MoE, partial CPU offload expected) \| latency: unverified estimate, re-measure after deployment (was \~14-17s per stock on the prior `qwen3:14b`).
---
# Testing Checklist (delta from v1.1)
Removed:<br>- Geopolitical citation test (no longer applicable)<br>- No-geopolitical test (no longer applicable)<br>- `G{num}` payload-existence cross-validator test<br>- CB commentary gap test (no separate block)
Modified:<br>- **Currency exposure test**: Pass SHOP.TO. Does `currency_impact.rationale` correctly reference the `primarily_usd` bucket and flag weakening CAD as a revenue tailwind?<br>- **Domestic-currency test**: Pass AQN. Does `currency_impact.rationale` correctly apply `primarily_cad`?<br>- **Overall environment calibration test**: Pass 5 stocks across the favorability spectrum. Do the `overall_macro_environment` enums distribute across `favorable / neutral / unfavorable` rather than clustering at “neutral”?<br>- **Material geopolitical headwind test**: Pass an Energy stock during a period with a relevant supply-chain / trade event. Does the LLM surface it in `sector_headwinds` with a `COMMOD` citation, rather than fabricating a stub for a stock with no payload-supported concern?
Carry-forward (unchanged from v1.1):<br>- Sector-sensitive tests (Financials, Energy, REIT)<br>- Cross-listed currency test, domestic-currency test<br>- Contradiction test, stale data test<br>- Sector-commodity gap test, non-commodity sector test<br>- Timeline adaptation test<br>- CoT-recap leakage test<br>- Jurisdiction passthrough non-mismatch test<br>- pass2_view shape test (note: `geopolitical_risks` no longer in view)
---

_Source: Notion — Project Stock Picker → Agent Prompts → Macro Economist Agent Prompt. Downloaded 2026-07-02._
