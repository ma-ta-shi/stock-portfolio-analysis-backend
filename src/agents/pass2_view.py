"""Builds the flat `pass2_view` each Pass 1 agent owes Pass 2.

WHY THIS EXISTS
---------------
All five Pass 1 prompts attribute `pass2_view` to `compress_pass1_for_advocate()`.
That function does not exist -- not in `backend/src`, not in `simulation`. And no
Pass 1 prompt asks the LLM to produce `pass2_view`; every one lists it under
"Fields the orchestrator populates (NOT produced by LLM)". So nothing built it,
while `compress_pass1_outputs()` *read* it (`output.get("pass2_view", {})`).

The result: `pass2_view` was always `{}`, so `build_pass2_user_message()` rendered
every agent as "NOT AVAILABLE - reason: data_quality_assessment=insufficient,
pass2_view empty" and the function then named `extract_reliability_scores()`
(renamed `extract_confidence_levels()` under 86bbummwp / D6) zeroed every score.
Bull and Bear received nothing at all. See docs/technical/prompt-revision-protocol.md 51.1.

SCOPE
-----
This builds the fields Pass 2 ACTUALLY CONSUMES -- the union of what
`validate_pass2_view_shape()` requires (35) and what Bull's payload template renders
(54) -- not all 89 the prompts declare. That is
deliberate: the required set is the load-bearing subset (FUND's 6 are exactly its
headline judgments), and the full 89 does not fit the ~500 token/agent compression
budget anyway (protocol doc 53.2). Extend per-agent when a consumer needs more.

CONTAINER NAMES DIFFER PER AGENT -- this is the trap
----------------------------------------------------
RSRCH, SENT, MACRO put interpretive output under `structured_data`.
FUND and TECH put it under `interpretive_fields`.
A single generic reader would silently return nothing for two of five agents.

Every mapping below is sourced from the agent's own prompt, not inferred. Only 13
of the 89 declared fields carry a source comment in the prompts, so the rest were
resolved by reading each Output Schema directly.
"""

# Which key each agent nests its interpretive output under.
_CONTAINER = {
    "RSRCH": "structured_data",
    "FUND": "interpretive_fields",
    "TECH": "interpretive_fields",
    "SENT": "structured_data",
    "MACRO": "structured_data",
}


def _dig(obj: dict, *path, default=None):
    """Walk a nested path, returning `default` at the first missing/non-dict step."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _obj(value) -> dict:
    """A nested block, or {} if it is anything else.

    `x or {}` is NOT enough: a model that returns a string where an object is
    specified leaves a str, and the next `.get()` raises AttributeError. The
    Sentiment validator had the mirror-image bug (TypeError on a dict where a
    string was assumed) and it went unnoticed because nothing ran it against real
    output -- so guard the shape rather than trusting the schema.
    """
    return value if isinstance(value, dict) else {}


def _top(seq, n: int) -> list:
    """First `n` items. `top_*` fields are slices of a full array, not separate fields."""
    return list(seq)[:n] if isinstance(seq, list) else []


def build_pass2_view(agent_id: str, agent_output: dict, bundle: dict | None = None) -> dict:
    """Build one agent's flat pass2_view from its output plus orchestrator-owned data.

    `bundle` carries fields the LLM never produces -- precomputed numerics and
    derived enums. Absent or partial is fine: those fields come back None rather
    than raising, matching the graceful-degradation pattern used across precompute/.
    """
    c = _obj(_dig(agent_output, _CONTAINER.get(agent_id, "structured_data"), default={}))
    b = bundle or {}

    if agent_id == "RSRCH":
        moat = _obj(c.get("moat_assessment"))
        return {
            "thesis_archetype": c.get("thesis_archetype"),
            "competitive_position": c.get("competitive_position"),
            "overall_moat_durability": moat.get("overall_moat_durability"),
            "moat_trend": moat.get("moat_trend"),
            "management_assessment": c.get("management_assessment"),
            # top_* are slices of the full arrays the LLM produces
            "top_growth_drivers": _top(c.get("growth_drivers"), 2),
            "top_competitive_threats": _top(c.get("competitive_threats"), 2),
            "top_recent_developments": _top(c.get("recent_developments"), 2),
            "peer_comparison_summary": c.get("peer_comparison_summary"),
        }

    if agent_id == "FUND":
        return {
            # Direct passthrough since the 2026-08-31 rename. Was
            # `pe_vs_sector_avg`, a narrow P/E-vs-sector name for what CoT step 1
            # always computed across P/E, P/B and PEG. Renamed because Pass 2 already
            # receives P/E and the sector median as raw numbers, so an enum restating
            # them is redundant; the field earns its place as the broader judgment.
            # `insufficient_data` is valid only when all three multiples are missing.
            "valuation_vs_sector": c.get("valuation_vs_sector"),
            "health_rating": c.get("health_rating"),
            "guidance_vs_consensus": c.get("guidance_vs_consensus"),
            "dividend_sustainability": c.get("dividend_sustainability"),
            # Numeric passthrough from DataBundle, never the LLM. margin_trend was
            # explicitly moved out of the LLM's interpretive fields in v5.1 and is
            # NET-margin trend specifically (fundamentals.py._margin_trend).
            "pe_ratio": b.get("pe_ratio"),
            "margin_trend": b.get("margin_trend"),
            # Bull's key_metrics / growth / margins lines render these. All are
            # "Numeric passthrough from DataBundle" per the FUND prompt -- the LLM
            # produces only interpretive_fields, never a number.
            "sector_pe_median": b.get("sector_pe_median"),
            "roe": b.get("roe"),
            "debt_to_equity": b.get("debt_to_equity"),
            "net_margin": b.get("net_margin"),
            "operating_margin": b.get("operating_margin"),
            "revenue_growth_yoy": b.get("revenue_growth_yoy"),
            "revenue_growth_3yr_cagr": b.get("revenue_growth_3yr_cagr"),
            "fcf_to_net_income": b.get("fcf_to_net_income"),
        }

    if agent_id == "TECH":
        return {
            "primary_trend": c.get("primary_trend"),
            "trend_strength": c.get("trend_strength"),
            "momentum_zone": c.get("momentum_zone"),
            "momentum_direction": c.get("momentum_direction"),
            "volume_confirmation": c.get("volume_confirmation"),
            "nearest_level_bias": c.get("nearest_level_bias"),
            "confluence_score": c.get("confluence_score"),
            # Orchestrator-owned: support_resistance and technical_indicators from
            # precompute. volatility_regime_derived is computed from atr_14 vs
            # atr_60_avg, not judged by the LLM (TECH prompt rule 18).
            "nearest_support": b.get("nearest_support"),
            "nearest_resistance": b.get("nearest_resistance"),
            "volatility_regime_derived": b.get("volatility_regime_derived"),
        }

    if agent_id == "SENT":
        news = _obj(c.get("news_sentiment"))
        analyst = _obj(c.get("analyst_sentiment"))
        return {
            # Flat names, per the prompt's explicit "a flat pass2_view dict".
            # The validator's old nested "news_sentiment"/"analyst_sentiment" names
            # were a half-finished migration -- corrected 2026-08-31.
            "news_sentiment_overall": news.get("overall"),
            "sentiment_trend": news.get("sentiment_trend"),
            "insider_activity_interpretation": c.get("insider_activity_interpretation"),
            "short_interest_interpretation": c.get("short_interest_interpretation"),
            "peer_sentiment_comparison": c.get("peer_sentiment_comparison"),
            # Orchestrator numeric passthrough, merged into analyst_sentiment.
            "consensus_rating": analyst.get("consensus_rating"),
            "average_price_target": analyst.get("average_price_target"),
        }

    if agent_id == "MACRO":
        # The LLM emits four nested objects; pass2_view is flat. Each contributes
        # impact_on_stock (+ outlook for growth) and rationale.
        rates = _obj(c.get("interest_rate_environment"))
        infl = _obj(c.get("inflation_environment"))
        grow = _obj(c.get("economic_growth"))
        curr = _obj(c.get("currency_impact"))
        return {
            "overall_macro_environment": c.get("overall_macro_environment"),
            "interest_rate_impact": rates.get("impact_on_stock"),
            "interest_rate_rationale": rates.get("rationale"),
            "inflation_impact": infl.get("impact_on_stock"),
            "inflation_rationale": infl.get("rationale"),
            "growth_outlook": grow.get("outlook"),
            "growth_impact": grow.get("impact_on_stock"),
            "growth_rationale": grow.get("rationale"),
            "currency_impact": curr.get("impact_on_stock"),
            "currency_rationale": curr.get("rationale"),
            # Bundle-sourced: trend/direction series the LLM never sees as a verdict.
            "interest_rate_direction": b.get("interest_rate_direction"),
            "inflation_trend": b.get("inflation_trend"),
            "currency_trend": b.get("currency_trend"),
            "commodity_context": b.get("commodity_context"),
            "sector_cycle_position": c.get("sector_cycle_position"),
            "sector_tailwinds": c.get("sector_tailwinds") or [],
            "sector_headwinds": c.get("sector_headwinds") or [],
            # Orchestrator-owned: classify_vix(src.vix), thresholds <15 low,
            # 15-25 elevated, >25 high (MACRO prompt line 352).
            "volatility_regime": b.get("volatility_regime"),
        }

    return {}
