"""The Pass 1 output contract: every field a Pass 1 agent's model writes has a stated purpose and a named consumer, or it is not written.

Why this exists (ledger BB-107, `docs/technical/pass1-output-contract.md`): about 40 to 55% of what the five Pass 1 agents wrote never
reached an agent prompt, a fifth of it `key_factors`, and nothing stated what any field was for. The rule since 2026-10-07: an output
needs a purpose and a consumer that uses it, or it is removed. `tests/agents/test_pass1_output_contract.py` fails if a prompt's output
schema gains or loses a field without this module changing with it, so a new output cannot be added without saying what it is for.

Consumers: Bu Bull, Be Bear, Ri Risk Advisor, Tx Tax Strategist (the Researcher's and Fundamental's confidence and quality only, plus
Fundamental's dividend inputs, which the code adds), CA and Sh the CIO stage A and Shadow CIO (one summary line per agent), API the
result endpoint (narrative and structured_output). Fields code decides and merges after the model answers are not listed: they cost no
output tokens (the register lists them).
"""

_ALL_BUR = "Bu, Be, Ri"

COMMON = {
    "assessment_summary": ("The agent's most important finding in 80 words", "Bu, Be, Ri, Tx (RSRCH, FUND), CA and Sh (the only Pass 1 text they see), API"),
    "analysis_confidence": ("The agent's own confidence in its conclusion", "Bu, Be, Ri, Tx, CA, Sh (reliability warnings and summary line); validators (confidence needs a caveat when data is flagged)"),
    "caveats": ("Specific data gaps and concerns", "Bu, Be, Ri, Tx (RSRCH, FUND); validators (confidence must be justified)"),
    "risks": ("Evidence-cited risks, one to three", "Bu thesis_risks, Be downside_triggers and tail risk, Ri downside_scenarios (delivered 2026-10-07)"),
    "narrative": ("The reasoning and synthesis, carrying the figures behind each claim", "Bu, Be, Ri, Tx (RSRCH, FUND), API"),
}

PASS1_OUTPUT_CONTRACT: dict[str, dict[str, tuple[str, str]]] = {
    "stock_researcher": {
        **COMMON,
        "structured_data.thesis_archetype": ("Classify the investment thesis", "Bu, Be, Ri, Tx (view and system prompt); Pass 2 validators (archetype alignment)"),
        "structured_data.competitive_position": ("Market standing", _ALL_BUR + ", Tx"),
        "structured_data.moat_assessment.overall_moat_durability": ("Moat strength", _ALL_BUR + ", Tx"),
        "structured_data.moat_assessment.moat_trend": ("Moat direction", _ALL_BUR + ", Tx"),
        "structured_data.moat_assessment.moats": ("The moat types and evidence behind the two moat labels; scaffolding: cutting it moved the labels (BB-107)", "none directly; the agent's own moat labels"),
        "structured_data.management_assessment": ("Management label", _ALL_BUR + ", Tx"),
        "structured_data.management_notes": ("The reason for the management label; scaffolding: cutting it moved moat durability (BB-107)", "none directly; the agent's own judgements"),
        "structured_data.business_model_summary": ("What the business does; context the agent's competitive-position and moat calls rest on (cutting it moved them, BB-107)", "none directly; the agent's own judgements"),
        "structured_data.revenue_mix_notes": ("Segment weights, context for the same calls", "none directly; the agent's own judgements"),
        "structured_data.peer_comparison_summary": ("How the company compares with its two closest industry peers", _ALL_BUR + ", Tx (view); never used by name in 30 runs, left to review"),
        "structured_data.recent_developments": ("Cited news events with significance and sentiment", _ALL_BUR + ", Tx (first two, with headline and date)"),
        "structured_data.competitive_threats": ("Cited threats", _ALL_BUR + ", Tx (first two)"),
        "structured_data.growth_drivers": ("Cited growth drivers", _ALL_BUR + ", Tx (first two)"),
    },
    "fundamental_analyst": {
        **COMMON,
        "interpretive_fields.valuation_vs_sector": ("One judgement across all multiples", "Bu (11 of 31 runs by value), CA (3), Be, Ri, Tx"),
        "interpretive_fields.health_rating": ("Balance-sheet and earnings health on the company kind's lens", "Ri (15 of 31 runs by value), Bu (4), Be, Tx"),
        "interpretive_fields.dividend_sustainability": ("Is the dividend covered", "Bu, Be, Ri (28 of 30 runs), CA, Tx (code-added dividend inputs)"),
    },
    "technical_analyst": {
        **COMMON,
        "interpretive_fields.trend_strength": ("Strength of the trend; scaffolds volume_confirmation (cutting it shifted that label, BB-107)", "Bu, Be, Ri (barely used by value)"),
        "interpretive_fields.volume_confirmation": ("Does volume confirm the primary trend", _ALL_BUR),
        "interpretive_fields.pattern_signal": ("A chart pattern, if any (mostly none)", "none: removal candidate once volume_confirmation is decided in code"),
        "interpretive_fields.pattern_confirmed": ("Whether that pattern is confirmed (set in 1 of 40 rows)", "none: removal candidate, as pattern_signal"),
        "interpretive_fields.confluence_score": ("How many of momentum, volume and level proximity agree with the trend", "Bu, Be"),
    },
    "sentiment_analyst": {
        **COMMON,
        "structured_data.news_sentiment.overall": ("Overall tone of the news", _ALL_BUR),
        "structured_data.news_sentiment.sentiment_trend": ("Direction of the tone", _ALL_BUR),
        "structured_data.news_sentiment.dominant_themes": ("The stories driving the news, each tied to a news ID the validator checks; kept after cutting it flipped labels in 3 of 15 runs (BB-107)", "none directly; the agent's own trend and tone judgements"),
        "structured_data.insider_activity_interpretation": ("What the insider net direction signals", _ALL_BUR),
        "structured_data.contrarian_signals": ("Positioning or mood that cuts against the crowd (non-empty in 15 of 40 rows)", "none directly; kept with themes pending more evidence"),
    },
    "macro_economist": {
        **COMMON,
        "structured_data.interest_rate_environment.impact_on_stock": ("Effect of rates on the stock", _ALL_BUR),
        "structured_data.interest_rate_environment.rationale": ("The transmission mechanism, in one or two sentences", _ALL_BUR),
        "structured_data.inflation_environment.impact_on_stock": ("Effect of inflation on the stock", _ALL_BUR),
        "structured_data.inflation_environment.rationale": ("The transmission mechanism", _ALL_BUR),
        "structured_data.economic_growth.outlook": ("Growth direction", _ALL_BUR + ", Tx"),
        "structured_data.economic_growth.impact_on_stock": ("Effect of growth on the stock", _ALL_BUR),
        "structured_data.economic_growth.rationale": ("The transmission mechanism", _ALL_BUR),
        "structured_data.currency_impact.impact_on_stock": ("Effect of the currency on the stock", _ALL_BUR),
        "structured_data.currency_impact.rationale": ("The transmission mechanism", _ALL_BUR),
        "structured_data.sector_cycle_position": ("Where the sector is in its cycle", _ALL_BUR),
        "structured_data.sector_tailwinds": ("Cited macro help for this sector", _ALL_BUR),
        "structured_data.sector_headwinds": ("Cited macro harm for this sector", _ALL_BUR),
        "structured_data.overall_macro_environment": ("Overall macro verdict", _ALL_BUR + " (by value in 8, 7, 5 of 31 runs)"),
    },
}
