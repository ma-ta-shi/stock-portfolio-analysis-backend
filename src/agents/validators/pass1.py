"""Pass 1 validators — one function per agent.

Production port of `simulation/validators/pass1.py` (86bbuhjup) -- mechanical
import-path port only, logic unchanged; proven and live-verified against real
gpt-oss:20b output all session in the harness.

Each returns (passed: bool, errors: list[str]).
All validation rules derived from the agent prompt documentation.
"""
from agents.utils import char_count, word_count
from agents.validators.common import THESIS_ARCHETYPES, _sweep_declared_enums

# PASS1_AGENT_IDS used to be redefined here too, unused anywhere in this file's own body --
# deleted rather than re-imported for the sake of it (agents.utils.PASS1_AGENT_IDS is the
# one canonical definition; pass2.py and cio.py import it because they actually use it).


def _require_field(obj: dict, field: str, errors: list[str]) -> bool:
    if field not in obj or obj[field] is None:
        errors.append(f"Missing required field: {field}")
        return False
    return True


def _check_string(obj: dict, field: str, min_chars: int, max_chars: int, errors: list[str]):
    v = obj.get(field, "")
    if not isinstance(v, str) or not v.strip():
        errors.append(f"{field}: must be a non-empty string")
        return
    c = char_count(v)
    if c < min_chars:
        errors.append(f"{field}: too short ({c} chars, min {min_chars})")
    if c > max_chars:
        errors.append(f"{field}: too long ({c} chars, max {max_chars})")


def _check_key_factors(key_factors: list, errors: list[str], required_sentiments: set | None = None,
                       lo: int = 2, hi: int = 4):
    """Bounds are declared PER AGENT in each prompt's Array-bounds table, so they are
    parameters, not constants. Sentiment lowered its floor to 1 (thin Canadian
    coverage forces invention at a floor of 2 -- see protocol doc 44.4/47.5); the
    others remain 2-4. test_prompt_contract.py asserts these stay in step."""
    if not isinstance(key_factors, list):
        errors.append("key_factors: must be a list")
        return
    if len(key_factors) < lo:
        errors.append(f"key_factors: need >={lo} items, got {len(key_factors)}")
    if len(key_factors) > hi:
        errors.append(f"key_factors: need <={hi} items, got {len(key_factors)}")
    valid_imp = {"high", "medium", "low"}
    valid_sent = {"positive", "negative", "neutral"}
    for i, kf in enumerate(key_factors):
        if not isinstance(kf, dict):
            errors.append(f"key_factors[{i}]: must be a dict")
            continue
        for sub in ("factor", "importance", "sentiment", "evidence"):
            if sub not in kf or not kf[sub]:
                errors.append(f"key_factors[{i}].{sub}: missing or empty")
        if kf.get("importance") not in valid_imp:
            errors.append(f"key_factors[{i}].importance: must be high|medium|low")
        sent = kf.get("sentiment", "")
        if sent not in valid_sent:
            errors.append(f"key_factors[{i}].sentiment: must be positive|negative|neutral")
        if required_sentiments and sent not in required_sentiments:
            errors.append(
                f"key_factors[{i}].sentiment: must be one of {required_sentiments}, got '{sent}'"
            )


def _check_risks(risks: list, errors: list[str], lo: int = 1, hi: int = 3):
    """Per-agent bounds -- see _check_key_factors. Sentiment allows 0 risks: a data
    gap is a caveat, not a risk, and a floor of 1 made the model manufacture one in
    5 of 6 runs (protocol doc 49.1/49.5)."""
    if not isinstance(risks, list):
        errors.append("risks: must be a list")
        return
    if len(risks) < lo:
        errors.append(f"risks: need >={lo} item(s), got {len(risks)}")
    if len(risks) > hi:
        errors.append(f"risks: need <={hi} items, got {len(risks)}")
    for i, r in enumerate(risks):
        if not isinstance(r, dict):
            errors.append(f"risks[{i}]: must be a dict")
            continue
        for sub in ("risk", "severity", "evidence"):
            if sub not in r or not r[sub]:
                errors.append(f"risks[{i}].{sub}: missing or empty")
        if r.get("severity") not in {"high", "medium", "low"}:
            errors.append(f"risks[{i}].severity: must be high|medium|low")


def _check_narrative_chars(obj: dict, min_c: int, max_c: int, errors: list[str]):
    narrative = obj.get("narrative", "")
    c = char_count(narrative)
    if c < min_c:
        errors.append(f"narrative: too short ({c} chars, min {min_c} = ~{min_c//6} words)")
    if c > max_c:
        errors.append(f"narrative: too long ({c} chars, max {max_c} = ~{max_c//6} words)")



# --------------------------------------------------------------------------
# `interpretive_fields` -- the CURRENT documented contract for FUND and TECH.
#
# The prompts were refactored: what this validator enforces as two objects
# (`structured_data` + `pass2_view`) the Output Schema now declares as ONE
# (`interpretive_fields`), with a different vocabulary for several members.
# The validators never followed. Consequences observed 2026-09-02:
#   * FUND terminally failed on JNJ with "Missing required field: structured_data"
#     after 3 attempts, while emitting a correct `interpretive_fields`.
#   * The SAME prompt PASSED on RY.TO only because that model volunteered a
#     redundant `structured_data` key as well -- pass/fail by luck.
#
# Both shapes are accepted deliberately, rather than picking a winner here:
# `simulation/runners/` still emit the legacy shape and pass, while the
# documented prompts emit this one. Which becomes canonical is a project
# decision (see end-to-end-audit-findings.md E12/E18), not a validator's.
IF_SPEC_FUNDAMENTAL = {
    "required": ("valuation_vs_sector", "health_rating", "guidance_vs_consensus",
                 "dividend_sustainability", "peer_comparison_summary"),
    "enums": {
        # Vocabularies quoted from the Fundamental Output Schema, NOT authored here.
        "valuation_vs_sector": {"undervalued", "fair", "overvalued", "insufficient_data"},
        "health_rating": {"healthy", "adequate", "stressed"},
        "guidance_vs_consensus": {"above", "inline", "below", "not_available"},
        "dividend_sustainability": {"strong", "adequate", "at_risk", "not_applicable"},
    },
}

IF_SPEC_TECHNICAL = {
    "required": ("primary_trend", "trend_strength", "momentum_zone"),
    "enums": {
        "primary_trend": {"bullish", "bearish", "mixed"},
        "trend_strength": {"strong", "moderate", "weak"},
        "momentum_zone": {"oversold", "neutral", "overbought"},
    },
}


def _check_interpretive_fields(output: dict, errors: list, spec: dict) -> None:
    """`interpretive_fields` is the sole real container for FUND/TECH as of 86bbdutn6 --
    called unconditionally now, not behind an `if _has_interpretive_fields(output)` branch
    (that branch and its old `structured_data` fallback are gone; see the 86bbuhk82 plan,
    item 3). `.get()` rather than `output["interpretive_fields"]` so a missing/wrong-type
    value (already flagged by the required-field check) doesn't crash this one too."""
    itf = output.get("interpretive_fields")
    if not isinstance(itf, dict):
        return
    for field in spec["required"]:
        if field not in itf:
            errors.append(f"interpretive_fields.{field}: missing")
    for field, allowed in spec["enums"].items():
        val = itf.get(field)
        if val is not None and val not in allowed:
            errors.append(
                f"interpretive_fields.{field}: must be one of "
                f"{'|'.join(sorted(allowed))}, got '{val}'"
            )



def validate_stock_researcher(output: dict) -> tuple[bool, list[str]]:
    """Validate Stock Researcher output."""
    errors: list[str] = []

    # Required top-level fields
    for field in (
        "assessment_summary",
        "analysis_confidence", "caveats", "key_factors", "risks", "narrative",
        "structured_data",
    ):
        _require_field(output, field, errors)

    # assessment_summary <=80 words
    summary = output.get("assessment_summary", "")
    wc = word_count(summary)
    if wc > 80:
        errors.append(f"assessment_summary: too long ({wc} words, max 80)")
    if not summary.strip():
        errors.append("assessment_summary: must be non-empty")

    # narrative: 120-180 words → 720-1080 chars
    _check_narrative_chars(output, 720, 1080, errors)

    # caveats: 1-4 items
    caveats = output.get("caveats", [])
    if not isinstance(caveats, list):
        errors.append("caveats: must be a list")
    elif len(caveats) < 1:
        errors.append("caveats: need >=1 item")
    elif len(caveats) > 4:
        errors.append(f"caveats: need <=4 items, got {len(caveats)}")

    # key_factors: 2-4 items
    _check_key_factors(output.get("key_factors", []), errors)

    # risks: 1-3 items
    _check_risks(output.get("risks", []), errors)

    # structured_data
    sd = output.get("structured_data", {})
    if not isinstance(sd, dict):
        errors.append("structured_data: must be a dict")
    else:
        if sd.get("thesis_archetype") not in THESIS_ARCHETYPES:
            errors.append(
                f"structured_data.thesis_archetype: must be one of {THESIS_ARCHETYPES}, got '{sd.get('thesis_archetype')}'"
            )
        for field in ("competitive_position", "moat_assessment", "management_assessment",
                       "growth_drivers", "competitive_threats", "recent_developments", "peer_comparison_summary"):
            if field not in sd:
                errors.append(f"structured_data.{field}: missing")
        # competitive_position / management_assessment -- declared enums (real schema:
        # 5 values each), previously only presence-checked above, never value-checked.
        # Found reviewing the permanent enum-inventory test's own allowlist: both names were
        # listed in _EXPLICITLY_HANDLED_ELSEWHERE (test_prompt_contract.py) on the assumption
        # an explicit check existed, when only a presence check did -- the allowlist was
        # asserting coverage that wasn't real. Path-scoped rather than added to the shared
        # sweep: only Stock Researcher declares either name across all 13 real templates today,
        # but both are common enough English words that a future prompt could reuse them with
        # different vocabulary, same reasoning as keeping "type" out of the sweep.
        cp = sd.get("competitive_position")
        if cp is not None and cp not in {"dominant", "strong", "average", "weak", "deteriorating"}:
            errors.append(
                f"structured_data.competitive_position: must be one of "
                f"dominant|strong|average|weak|deteriorating, got {cp!r}"
            )
        ma = sd.get("management_assessment")
        if ma is not None and ma not in {
            "excellent", "competent", "concerning", "poor", "insufficient_data"
        }:
            errors.append(
                f"structured_data.management_assessment: must be one of "
                f"excellent|competent|concerning|poor|insufficient_data, got {ma!r}"
            )
        moat = sd.get("moat_assessment", {})
        if isinstance(moat, dict):
            if moat.get("overall_moat_durability") not in {"strong", "moderate", "weak", "none"}:
                errors.append("structured_data.moat_assessment.overall_moat_durability: must be strong|moderate|weak|none")
            # Union, not the prompt's literal 3-value set: the real prompt declares
            # strengthening|stable|eroding, but real model output (checked across 4 live
            # calls plus every historical capture) has only ever produced "weakening" for a
            # declining moat, never "eroding" -- and the only real downstream consumer (Bull
            # Advocate) reads this as free text, not a branched value, so accepting both costs
            # nothing. See the 86bbuhk82 plan for the live-testing record.
            if moat.get("moat_trend") not in {"strengthening", "stable", "eroding", "weakening"}:
                errors.append("structured_data.moat_assessment.moat_trend: must be strengthening|stable|eroding")
            # moats[].type -- declared in the real schema (9-value), unenforced until now.
            # "type" is deliberately not in the shared sweep: too generic a key name to sweep
            # globally without risking a future collision. Surfaced by the permanent
            # enum-inventory test, confirmed live earlier this session (moats[0].type=
            # "cost_advantage" in a real Stock Researcher call).
            valid_moat_type = {
                "network_effects", "switching_costs", "brand", "cost_advantage", "regulatory",
                "scale", "intellectual_property", "data_advantage", "none",
            }
            for i, m in enumerate(moat.get("moats") or []):
                if isinstance(m, dict) and m.get("type") is not None and m["type"] not in valid_moat_type:
                    errors.append(
                        f"structured_data.moat_assessment.moats[{i}].type: must be one of "
                        f"{valid_moat_type}, got {m['type']!r}"
                    )

    # `pass2_view` is deliberately NOT checked here. It's built by the orchestrator's
    # compress_pass1_outputs()/build_pass2_view() from `structured_data`, downstream of this
    # agent -- confirmed directly in compression.py's own docstring ("no Pass 1 agent produces
    # that field... every prompt lists it as orchestrator-populated"), and confirmed live during
    # 86bbdutn6: an isolated first-attempt call produced a `structured_data` block matching the
    # real schema exactly, with zero errors, while this check burned all 3 retries demanding a
    # field the real prompt never asks the LLM to produce. `validate_pass2_view_shape()`
    # (agents/validators/compression.py) is the correct place to check pass2_view's shape,
    # applied to the orchestrator-built view, not raw agent output.

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_fundamental_analyst(output: dict) -> tuple[bool, list[str]]:
    """Validate Fundamental Analyst output."""
    errors: list[str] = []

    # `interpretive_fields` is the sole real container as of 86bbdutn6 -- no longer
    # conditional on `_has_interpretive_fields()` / falling back to `structured_data`
    # (86bbuhk82 item 3: "resolve the structured_data/interpretive_fields split").
    required = [
        "assessment_summary",
        "analysis_confidence", "caveats", "key_factors", "risks", "narrative",
        "interpretive_fields",
    ]
    for field in required:
        _require_field(output, field, errors)

    # assessment_summary <=80 words
    summary = output.get("assessment_summary", "")
    if word_count(summary) > 80:
        errors.append(f"assessment_summary: too long ({word_count(summary)} words, max 80)")

    # This branch used to key on data_quality_assessment=="insufficient" -- that field is
    # gone (86bbummwp / D6: never LLM-produced, mechanically rolled up elsewhere). Repointed
    # to analysis_confidence, the one real, live signal for "this agent cannot produce a
    # useful analysis" (pass1-confidence-model.md). Deliberately an EXACT match, not loosened
    # to include "low" -- confirmed live (86bbummwp) that a real "low"-confidence
    # Fundamental call still produces fully compliant narrative/key_factors content on its own
    # merits, so exempting "low" from these checks would let honestly-thin-but-real output
    # skip checks it can already satisfy. (validate_stub_fundamental in compression.py is the
    # separate, narrower check that does accept "low" for the one fixture that needs it.)
    if output.get("analysis_confidence") == "insufficient":
        # pass2_view must be empty dict
        p2v = output.get("pass2_view")
        if p2v is not None and p2v != {}:
            errors.append("pass2_view: must be {} (empty) when analysis_confidence=insufficient")
        _sweep_declared_enums(output, errors)
        # Insufficient data: the content checks below do not apply. This return was
        # at FUNCTION indent, so it fired unconditionally and made every check after
        # it unreachable -- narrative length, key_factors bounds, risks bounds,
        # interpretive-field enums and the declared-enum sweep. Found 2026-09-03 when
        # a FUND output with 5 key_factors passed a 2-4 bound.
        return len(errors) == 0, errors

    # narrative: 120-180 words → 720-1080 chars
    _check_narrative_chars(output, 720, 1080, errors)

    _check_key_factors(output.get("key_factors", []), errors)
    _check_risks(output.get("risks", []), errors)

    # A second, formerly-unreachable legacy `structured_data`/`pass2_view` block (guarded by
    # an `if _has_interpretive_fields(output):` that returned unconditionally right after)
    # used to sit here, checking a container/vocabulary the real prompt no longer produces.
    # Deleted outright rather than fixed into reachability -- 86bbdutn6 confirmed
    # `interpretive_fields` is the sole real contract, so there is nothing left for it to
    # ever correctly validate.
    _check_interpretive_fields(output, errors, IF_SPEC_FUNDAMENTAL)
    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_technical_analyst(output: dict) -> tuple[bool, list[str]]:
    """Validate Technical Analyst output."""
    errors: list[str] = []

    required = [
        "assessment_summary",
        "analysis_confidence", "caveats", "key_factors", "risks", "narrative",
        "interpretive_fields",
    ]
    for field in required:
        _require_field(output, field, errors)

    summary = output.get("assessment_summary", "")
    if word_count(summary) > 80:
        errors.append(f"assessment_summary: too long ({word_count(summary)} words, max 80)")

    _check_narrative_chars(output, 720, 1080, errors)
    _check_key_factors(output.get("key_factors", []), errors)
    _check_risks(output.get("risks", []), errors)

    # Same deletion as validate_fundamental_analyst above -- the old unreachable
    # `pass2_view`-based legacy block (uptrend/downtrend/sideways vocabulary the real prompt
    # never asks for) is gone, not fixed into reachability.
    _check_interpretive_fields(output, errors, IF_SPEC_TECHNICAL)
    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_sentiment_analyst(output: dict) -> tuple[bool, list[str]]:
    """Validate Sentiment Analyst output."""
    errors: list[str] = []

    for field in (
        "assessment_summary",
        "analysis_confidence", "caveats", "key_factors", "risks", "narrative",
        "structured_data",
    ):
        _require_field(output, field, errors)

    summary = output.get("assessment_summary", "")
    if word_count(summary) > 80:
        errors.append(f"assessment_summary: too long ({word_count(summary)} words, max 80)")

    _check_narrative_chars(output, 720, 1080, errors)
    # Sentiment's declared bounds: key_factors 1-3, risks 0-2 (thin-data path).
    _check_key_factors(output.get("key_factors", []), errors, lo=1, hi=3)
    _check_risks(output.get("risks", []), errors, lo=0, hi=2)

    # structured_data
    sd = output.get("structured_data", {})
    if isinstance(sd, dict):
        # short_interest_interpretation is an OBJECT in prompt schema v1.2
        # ({trend, interpretation}), not a flat string. The previous flat-string
        # check raised TypeError ("unhashable type: 'dict'") on every schema-
        # conformant output -- a crash, not a validation failure -- which means
        # this validator had never been run against real model output.
        valid_si = {"elevated_volatility_risk", "normal", "insufficient_data"}
        valid_trend = {"increasing", "stable", "decreasing", "unknown"}
        si = sd.get("short_interest_interpretation")
        if not isinstance(si, dict):
            errors.append(
                "structured_data.short_interest_interpretation: must be an object with "
                "'trend' and 'interpretation'"
            )
        else:
            if si.get("interpretation") not in valid_si:
                errors.append(
                    "structured_data.short_interest_interpretation.interpretation: "
                    f"must be one of {valid_si}"
                )
            if si.get("trend") not in valid_trend:
                errors.append(
                    "structured_data.short_interest_interpretation.trend: "
                    f"must be one of {valid_trend}"
                )
        # social_sentiment must always be "unknown"
        if sd.get("social_sentiment") not in {"unknown", None}:
            errors.append(
                f"structured_data.social_sentiment: must be 'unknown' (orchestrator-set), got '{sd.get('social_sentiment')}'"
            )

        # news_sentiment.overall -- declared in the real schema (5-value, distinct from the
        # 3-value key_factors[].sentiment vocabulary, hence not in the shared sweep), unenforced
        # until now. Surfaced by the permanent enum-inventory test, not a live failure.
        ns = sd.get("news_sentiment")
        if isinstance(ns, dict):
            valid_overall = {"very_positive", "positive", "neutral", "negative", "very_negative"}
            overall = ns.get("overall")
            if overall is not None and overall not in valid_overall:
                errors.append(
                    f"structured_data.news_sentiment.overall: must be one of {valid_overall}, got {overall!r}"
                )

    # pass2_view must not include social_sentiment
    p2v = output.get("pass2_view", {})
    if isinstance(p2v, dict) and "social_sentiment" in p2v:
        errors.append("pass2_view: must NOT include social_sentiment (excluded from stable contract)")

    # contrarian_signals: 0-2 items
    cs = output.get("contrarian_signals", [])
    if isinstance(cs, list) and len(cs) > 2:
        errors.append(f"contrarian_signals: max 2 items, got {len(cs)}")

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_macro_economist(output: dict) -> tuple[bool, list[str]]:
    """Validate Macro Economist output."""
    errors: list[str] = []

    for field in (
        "assessment_summary",
        "analysis_confidence", "caveats", "key_factors", "risks", "narrative",
        "structured_data",
    ):
        _require_field(output, field, errors)

    summary = output.get("assessment_summary", "")
    if word_count(summary) > 80:
        errors.append(f"assessment_summary: too long ({word_count(summary)} words, max 80)")

    # Macro narrative: 80-120 words → 480-720 chars (SHORTER than other agents)
    _check_narrative_chars(output, 480, 720, errors)

    _check_key_factors(output.get("key_factors", []), errors)
    _check_risks(output.get("risks", []), errors)

    # structured_data
    sd = output.get("structured_data", {})
    if isinstance(sd, dict):
        valid_macro_env = {"favorable", "neutral", "unfavorable"}
        if sd.get("overall_macro_environment") not in valid_macro_env:
            errors.append(
                f"structured_data.overall_macro_environment: must be favorable|neutral|unfavorable (3-value), got '{sd.get('overall_macro_environment')}'"
            )
        valid_cycle = {"early_cycle", "mid_cycle", "late_cycle", "recession"}
        if sd.get("sector_cycle_position") not in valid_cycle:
            errors.append(
                "structured_data.sector_cycle_position: must be early_cycle|mid_cycle|late_cycle|recession"
            )

    # pass2_view
    p2v = output.get("pass2_view", {})
    if isinstance(p2v, dict):
        if p2v.get("overall_macro_environment") not in {"favorable", "neutral", "unfavorable", None}:
            errors.append("pass2_view.overall_macro_environment: must be favorable|neutral|unfavorable")
        if p2v.get("sector_cycle_position") not in {
            "early_cycle", "mid_cycle", "late_cycle", "recession", None
        }:
            errors.append("pass2_view.sector_cycle_position: invalid value")

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_earnings_proximity_caveat(output: dict, earnings_days: int) -> tuple[bool, list[str]]:
    """Check that Technical Analyst includes earnings proximity caveat when <=5 days."""
    errors: list[str] = []
    if earnings_days > 5:
        return True, []

    REQUIRED_CAVEAT = "Earnings in ≤5 days"
    caveats = output.get("caveats", [])
    narrative = output.get("narrative", "")
    combined = " ".join(str(c) for c in caveats) + " " + narrative

    if REQUIRED_CAVEAT.lower() not in combined.lower() and "earnings" not in combined.lower():
        errors.append(
            f"TECH: earnings_proximity_days={earnings_days}<=5 but mandatory earnings proximity caveat not found"
        )

    if output.get("analysis_confidence") == "high":
        errors.append(
            f"TECH: analysis_confidence must be 'medium' (not 'high') when earnings_proximity_days={earnings_days}<=5"
        )

    return len(errors) == 0, errors


def validate_thin_volume_caveat(output: dict, avg_dollar_volume: float) -> tuple[bool, list[str]]:
    """Check that Technical Analyst includes thin volume caveat when avg_dollar_volume_20 < $1M."""
    errors: list[str] = []
    if avg_dollar_volume >= 1_000_000:
        return True, []

    caveats = output.get("caveats", [])
    narrative = output.get("narrative", "")
    combined = " ".join(str(c) for c in caveats) + " " + narrative

    if "volume" not in combined.lower() and "liquid" not in combined.lower():
        errors.append(
            f"TECH: avg_dollar_volume=${avg_dollar_volume:,.0f} < $1M but no thin volume / liquidity caveat found"
        )

    return len(errors) == 0, errors


def validate_canadian_caveat(output: dict, agent_id: str) -> tuple[bool, list[str]]:
    """Check that RSRCH and SENT include the mandatory Canadian data limited caveat.

    The reliability_score<=70 cap this used to also enforce is gone (86bbummwp / D6: the
    field is never LLM-produced anymore, so the old `output.get("reliability_score", 101)`
    would now ALWAYS see the 101 default and ALWAYS fail -- going from an intermittent bug to
    a permanent one if left in place).

    KNOWN, UNFIXED, SEPARATE DEFECT (not D6, found and live-verified during 86bbummwp
    planning): the "Finnhub" phrase below is stale relative to the current real prompts. Ran
    Stock Researcher and Sentiment Analyst live against a canadian_data_limited=true fixture --
    real output caveats match their own prompts' actual declared phrases exactly
    (`{has_filing_digest}`'s "Filing depth limited: ..." for RSRCH -- renamed from
    `{sedar_filing_available}` 2026-09-23, see research_sources_bundle.py, after a live AAPL
    run showed the old wording's "Canadian filing depth limited" phrase leaking into a US
    stock's real output --
    `{canadian_sentiment_inferred}`'s "Article sentiment is scored by a local LLM..." for
    SENT), and neither mentions Finnhub at all. This function will keep failing in a real
    sweep after this fix -- correctly, just for a different reason than the cap did. Recommend
    a fast-follow ticket to rewrite this check against the two real, agent-specific triggers;
    out of scope here (caveat-wording drift, not a D6 field removal)."""
    errors: list[str] = []
    REQUIRED_PHRASE = "Finnhub"
    caveats = output.get("caveats", [])
    combined = " ".join(str(c) for c in caveats)

    if REQUIRED_PHRASE.lower() not in combined.lower():
        errors.append(
            f"{agent_id}: canadian_data_limited=true but mandatory Finnhub caveat phrase not found in caveats"
        )

    return len(errors) == 0, errors
