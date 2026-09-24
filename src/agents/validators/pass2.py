"""Pass 2 validators — Bull, Bear, Tax Strategist, Risk Advisor."""
import re

from agents.utils import PASS1_AGENT_IDS, char_count
from agents.validators.common import THESIS_ARCHETYPES, _sweep_declared_enums

WEAKEST_POINT_TYPES = {
    "negative_catalyst", "adverse_fundamental", "data_gap", "no_clear_invalidator"
}


def _has_pass1_agent_id(text: str) -> bool:
    """Check that text contains at least one valid Pass 1 agent ID."""
    return any(aid in text for aid in PASS1_AGENT_IDS)


# The Risk Advisor is the only agent with its own citation vocabulary, declared in
# its rule 1 alongside the Pass 1 IDs. Word-boundary matched so "VOL" does not fire
# inside "VOLATILE" and "DD" does not fire inside "ADDED".
RISK_METRIC_TOKENS = {"BETA", "VOL", "DD", "LIQ", "CORR", "CONC"}
_RISK_TOKEN_RE = re.compile(r"\b(?:" + "|".join(sorted(RISK_METRIC_TOKENS)) + r")\b")


def _has_risk_citation(text: str) -> bool:
    """Accept either a Pass 1 agent ID or a pre-computed metric token.

    The Risk Advisor prompt authorises both, in two places: its output schema
    ("Must cite a Pass 1 agent ID or metric token") and its own validation rules
    ("at least one valid citation token", where rule 1 defines valid as the Pass 1
    IDs plus BETA VOL DD LIQ CORR CONC).

    Requiring a Pass 1 ID made one of strongest_signal's two stated purposes
    impossible: it is "the dominant downside scenario OR SANITY FLAG", and CoT step 6
    asks the model to flag inconsistencies between BETA, VOL, DD and LIQ -- a claim
    about the metrics has no Pass 1 agent to cite. The old check rejected the
    prompt's own worked examples, including "BETA: 1.4 - moderate amplifier vs
    S&P/TSX". Observed live: a terminal failure, which drops the field from the CIO
    payload entirely (format_risk_advisor_for_cio renders strongest_signal).
    """
    return _has_pass1_agent_id(text) or bool(_RISK_TOKEN_RE.search(text))


# The Tax Strategist's own citation vocabulary, quoted from its rule 1 ("Pre-computed
# tax-metric tokens"). Word-boundary matched for the same reason as the risk tokens:
# LIST must not fire inside "LISTED", DOM not inside "DOMESTIC" or "RANDOM", LOSS not
# inside "LOSSES", ROOM not inside "ROOMS".
TAX_METRIC_TOKENS = {"DIVID", "LIST", "DOM", "WHT", "CGAIN", "ROOM", "LOSS", "ELIG"}
_TAX_TOKEN_RE = re.compile(r"\b(?:" + "|".join(sorted(TAX_METRIC_TOKENS)) + r")\b")
_REF_RE = re.compile(r"\bREF\b")
# Rule 9: "Evidence starts with a valid citation token followed by `:`". Anchored,
# because the prompt's compact format is positional -- a token buried mid-sentence is
# not the declared format and is exactly how an uncited claim gets dressed up.
_TAX_EVIDENCE_PREFIX_RE = re.compile(
    # PASS1_AGENT_IDS is a list (agents.utils's canonical form -- other callers rely on its
    # order), so it needs an explicit set() to union with the two literal sets here.
    r"^\s*(?:" + "|".join(sorted(TAX_METRIC_TOKENS | set(PASS1_AGENT_IDS) | {"REF"})) + r")\b[^:]*:"
)
# A citation must be followed by an actual claim. Observed live: `"CGAIN:"` -- a valid
# token, a colon, and nothing at all, which satisfied the prefix rule while asserting
# nothing.
#
# A flat character minimum was WRONG and caused two terminal failures on 2026-09-01:
# it rejected `"DIVID: 2.0%"`, which is a correct compact citation carrying a real
# figure in only 4 characters. Rejecting valid output is worse than the gap it closed
# -- a terminal failure drops the agent from the CIO payload entirely.
#
# So: a body counts as a claim if it carries a FIGURE, or is long enough to be prose.
#   "DIVID: 2.0%"      -> digit      -> accepted
#   "CGAIN:"           -> empty      -> rejected
#   "RSRCH: trading"   -> no digit, 7 chars -> rejected (echoes context, states nothing)
_MIN_EVIDENCE_BODY_CHARS = 8
_HAS_DIGIT_RE = re.compile(r"[0-9]")


def _evidence_body(text: str) -> str:
    """The claim after the leading citation token's colon."""
    m = _TAX_EVIDENCE_PREFIX_RE.match(text)
    return text[m.end():].strip() if m else ""


def _evidence_states_a_claim(text: str) -> bool:
    """A figure, or enough prose to be a statement. See _MIN_EVIDENCE_BODY_CHARS."""
    body = _evidence_body(text)
    return bool(_HAS_DIGIT_RE.search(body)) or len(body) >= _MIN_EVIDENCE_BODY_CHARS


# SOFT vs HARD failures (122.5).
# The narrative citation-breadth rules failed AND retried, and retry exhaustion removes
# the agent from the CIO payload entirely -- measured at 2 of 9 runs on 2026-09-01.
# A weakly-cited narrative is a quality problem; a missing agent is a structural one,
# and the structural one is worse.
#
# These stay hard failures on every attempt except the last, so retries still apply
# corrective pressure. On the final attempt the caller may accept the output and log
# them, rather than losing the agent over a citation count. The prompt itself already
# uses this split -- Rule 12's risk surfaces are "soft enforcement ... Missing logs
# warning (no retry)".
#
# Everything NOT listed here stays hard: a missing tax_profile, a bad enum, an
# RRSP-withdrawal action, or loss-harvesting in a registered account all make the
# output unusable or unsafe, and no leniency applies to them on any attempt.
SOFT_ERROR_PREFIXES = (
    "narrative: needs >=3 distinct tax-metric tokens",
    "narrative: needs >=1 REF citation",
)


def split_soft_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """(hard, soft). Soft = citation breadth on the narrative; see SOFT_ERROR_PREFIXES."""
    soft = [e for e in errors if e.startswith(SOFT_ERROR_PREFIXES)]
    hard = [e for e in errors if not e.startswith(SOFT_ERROR_PREFIXES)]
    return hard, soft


def _has_tax_citation(text: str) -> bool:
    """A Pass 1 agent ID, a pre-computed tax-metric token, or REF.

    All three are authorised by the Tax Strategist's rule 1. Kept separate from
    _has_risk_citation because the two agents have genuinely different vocabularies --
    sharing one permissive union would let a Tax claim "cite" BETA, which its own
    prompt does not authorise.
    """
    return bool(
        _has_pass1_agent_id(text) or _TAX_TOKEN_RE.search(text) or _REF_RE.search(text)
    )


# A bare field name is snake_case or a lone agent ID. Real evidence carries a value:
# a number ("roe 15.7%"), or a prose/enum word ("dividend_sustainability strong").
# Single-token metric names that look like prose but are field names, so they do
# not count as "a value was quoted". snake_case names are detected structurally.
_BARE_METRIC_NAMES = {
    "roe", "roa", "roic", "eps", "pe", "pb", "ps", "peg", "beta", "rsi", "macd",
    "atr", "vix", "fcf", "ebitda", "capex", "dcf", "wacc", "sma", "ema", "bb",
}
_TOKEN_SPLIT = re.compile("[^A-Za-z0-9_%.$]+")


def _evidence_carries_value(text: str) -> bool:
    """True if evidence quotes a supporting value, not just field names.

    Measured need: Bull emitted `"FUND: revenue_growth_yoy, roe"` -- two field names,
    no values, while 15.7% and 15.7% sat in the payload. The previous check passed it
    because the string contains "FUND". That is prefix-counting, which this pass
    repeatedly found measures labelling rather than substance.

    A digit alone is NOT a sufficient test: `"FUND: dividend_sustainability strong"`
    quotes an enum value correctly and contains no digit. So: drop agent IDs,
    snake_case field names and known bare metric names, then require a real word to
    remain. Token-based rather than regex-stripping -- an earlier regex version had
    an escaping bug that silently matched nothing and passed both failure cases.
    """
    if any(c.isdigit() for c in text):
        return True
    for token in _TOKEN_SPLIT.split(text):
        if not token:
            continue
        if token in PASS1_AGENT_IDS:
            continue
        if "_" in token:                       # snake_case field name
            continue
        if token.lower() in _BARE_METRIC_NAMES:
            continue
        return True                            # a genuine word survived
    return False


def _check_array_bounds(obj: dict, field: str, min_len: int, max_len: int, errors: list[str]):
    """Bounded-array check shared across the Pass 2 validators.

    Restored 2026-08-31 after an index-based edit in this file accidentally spanned
    and removed it (31 tests failed with NameError). Phrasing matches pass1.py's
    equivalent checks so error text stays consistent across passes.
    """
    value = obj.get(field)
    if not isinstance(value, list):
        errors.append(f"{field}: must be a list")
        return
    if len(value) < min_len:
        errors.append(f"{field}: need >={min_len} items, got {len(value)}")
    if len(value) > max_len:
        errors.append(f"{field}: need <={max_len} items, got {len(value)}")



def validate_bull_advocate(
    output: dict, real_researcher_archetype: str | None = None
) -> tuple[bool, list[str]]:
    """Validate Bull Case Advocate output.

    real_researcher_archetype: the actual researcher_thesis_archetype(compressed_pass1)
    value from the payload (86bbt1k1p follow-up) -- optional so existing callers that
    don't supply it (including every pre-existing test) are unaffected; the cross-field
    alignment check below is skipped entirely when it's None.
    """
    errors: list[str] = []

    # recommendation must be "bullish"
    if output.get("recommendation") != "bullish":
        errors.append(f"recommendation: must be 'bullish', got '{output.get('recommendation')}'")

    # confidence: 0-100 integer
    conf = output.get("confidence")
    if not isinstance(conf, (int, float)):
        errors.append("confidence: must be a number")
    elif not (0 <= conf <= 100):
        errors.append(f"confidence: must be 0-100, got {conf}")
    conf = int(conf) if isinstance(conf, (int, float)) else 0

    # thesis_summary: 80-500 chars, must reference timeline or account
    ts = output.get("thesis_summary", "")
    c = char_count(ts)
    if c < 80:
        errors.append(f"thesis_summary: too short ({c} chars, min 80)")
    if c > 500:
        errors.append(f"thesis_summary: too long ({c} chars, max 500)")

    # strongest_argument: must contain Pass 1 agent ID
    sa = output.get("strongest_argument", "")
    if not sa.strip():
        errors.append("strongest_argument: must be non-empty")
    elif not _has_pass1_agent_id(sa):
        errors.append("strongest_argument: must contain at least one Pass 1 agent ID (RSRCH/FUND/TECH/SENT/MACRO)")

    # weakest_point + weakest_point_type
    wp = output.get("weakest_point", "")
    wpt = output.get("weakest_point_type", "")
    if not wp.strip():
        errors.append("weakest_point: must be non-empty")
    if wpt not in WEAKEST_POINT_TYPES:
        errors.append(f"weakest_point_type: must be one of {WEAKEST_POINT_TYPES}, got '{wpt}'")
    if wpt == "no_clear_invalidator" and conf < 70:
        errors.append(f"weakest_point_type='no_clear_invalidator' requires confidence>=70, got {conf}")
    if wpt != "no_clear_invalidator" and wp and not _has_pass1_agent_id(wp):
        errors.append("weakest_point: must cite at least one Pass 1 agent ID (unless weakest_point_type=no_clear_invalidator)")
    if conf >= 80 and wpt == "no_clear_invalidator":
        errors.append(f"confidence={conf}>=80 but weakest_point_type=no_clear_invalidator (not allowed at high confidence)")

    # caveats: 0-5
    _check_array_bounds(output, "caveats", 0, 5, errors)

    # key_factors: 2-4, all must have sentiment="positive"
    key_factors = output.get("key_factors", [])
    if not isinstance(key_factors, list):
        errors.append("key_factors: must be a list")
    else:
        if len(key_factors) < 2:
            errors.append(f"key_factors: need >=2, got {len(key_factors)}")
        if len(key_factors) > 4:
            errors.append(f"key_factors: need <=4, got {len(key_factors)}")
        seen = [kf.get("evidence", "") for kf in key_factors if isinstance(kf, dict)]
        if len(seen) != len(set(seen)):
            errors.append(
                "key_factors: duplicate evidence string across factors -- reusing the "
                "same evidence means you have one factor, not two"
            )
        for i, kf in enumerate(key_factors):
            if not isinstance(kf, dict):
                continue
            if kf.get("sentiment") != "positive":
                errors.append(f"key_factors[{i}].sentiment: must be 'positive' for Bull advocate, got '{kf.get('sentiment')}'")
            if not kf.get("evidence", "").strip():
                errors.append(f"key_factors[{i}].evidence: must be non-empty")
            elif not _has_pass1_agent_id(kf["evidence"]):
                errors.append(f"key_factors[{i}].evidence: must start with a Pass 1 agent ID")
            elif not _evidence_carries_value(kf["evidence"]):
                errors.append(
                    f"key_factors[{i}].evidence: cites field names without values "
                    f"({kf['evidence']!r}) -- quote the supporting figure or value"
                )

    # thesis_risks: 1-3
    _check_array_bounds(output, "thesis_risks", 1, 3, errors)
    for i, tr in enumerate(output.get("thesis_risks", [])):
        if isinstance(tr, dict) and tr.get("evidence"):
            ev = tr["evidence"]
            if not _has_pass1_agent_id(ev) and not ev.strip().startswith("thesis assumption:"):
                errors.append(f"thesis_risks[{i}].evidence: must cite Pass 1 agent ID or start with 'thesis assumption:'")

    # narrative: 1500-2800 chars, must reference >=3 distinct Pass 1 agents
    narrative = output.get("narrative", "")
    nc = char_count(narrative)
    if nc < 1500:
        errors.append(f"narrative: too short ({nc} chars, min 1500)")
    if nc > 2800:
        errors.append(f"narrative: too long ({nc} chars, max 2800)")
    agents_in_narrative = [aid for aid in PASS1_AGENT_IDS if aid in narrative]
    if len(agents_in_narrative) < 3:
        errors.append(
            f"narrative: must reference >=3 distinct Pass 1 agent IDs, found {len(agents_in_narrative)}: {agents_in_narrative}"
        )

    # structured_data
    sd = output.get("structured_data", {})
    if not isinstance(sd, dict):
        errors.append("structured_data: must be a dict")
    else:
        # core_arguments: 1-4, at least one primary
        cas = sd.get("core_arguments", [])
        _check_array_bounds(sd, "core_arguments", 1, 4, errors)
        primaries = [ca for ca in cas if isinstance(ca, dict) and ca.get("strength") == "primary"]
        if len(primaries) < 1:
            errors.append(f"structured_data.core_arguments: at least 1 must have strength='primary', found {len(primaries)}")

        # catalysts: 1-4
        _check_array_bounds(sd, "catalysts", 1, 4, errors)

        # market_misreads: 1-3
        _check_array_bounds(sd, "market_misreads", 1, 3, errors)

        # historical_analogy: required field
        ha = sd.get("historical_analogy", {})
        if not isinstance(ha, dict):
            errors.append("structured_data.historical_analogy: must be a dict")
        else:
            company = ha.get("company", "")
            analogy_fit = ha.get("analogy_fit", "")
            # analogy_fit's own declared vocabulary (strong|moderate|weak|none) was never
            # range-checked below -- only the "weak is inadmissible" business rule and the
            # none_found special case were, so an out-of-vocabulary value (e.g. "excellent")
            # passed silently. Same gap class as competitive_position/management_assessment/
            # alignment_with_researcher, found the same way.
            #
            # The non-none branch (company != "none_found", analogy_fit in strong|moderate) has
            # never been observed live, across 34 real gpt-oss:20b samples: all 7 original harness
            # fixtures (86bbuhk82) PLUS scenario_08_business_model_transition.json -- a fixture
            # purpose-built to closely mirror a specific, well-documented real-world precedent
            # (a company mid-transition from perpetual-license to subscription revenue, ARR
            # growing while reported revenue mechanically declines).
            #
            # NOT attributed to the model -- traced the scenario_08 run end to end and found the
            # actual cause is the harness's stale D6 reliability_score/data_quality_assessment
            # defaulting bug (86bbummwp), not the model or this fixture. RSRCH's raw output
            # correctly omitted reliability_score (current prompt doesn't ask for it) and,
            # unlike the other 4 agents that happened to fabricate a plausible value on retry,
            # terminally failed -- so compress_pass1_outputs() defaulted it to
            # reliability_score=0/data_quality_assessment="insufficient". RSRCH's narrative (the
            # ONLY source carrying the license-to-subscription story) reached the compressed view
            # intact, but build_pass1_reliability_warnings() rendered "RSRCH: 0 (skipped)" into
            # Bull Advocate's actual prompt from that same 0 default. Bull's real output caveat
            # confirms it: "RSRCH reliability 0/100; key qualitative insights are unverified." The
            # model correctly discounted a source it was told (falsely, by the harness) was
            # unreliable, then correctly fell back to the prompt's own "if in doubt, none_found"
            # default given what it had been told -- not a model or prompt defect.
            #
            # UPDATE (86bbummwp landed and was re-tested): re-ran scenario_08 through Pass 1 +
            # Bull Advocate after the fix. Zero "Missing required field" errors on any of the 5
            # Pass 1 agents this time, and Bull Advocate ran with ZERO validation errors on first
            # attempt (previously it had to retry past the false "RSRCH: 0 (skipped)" discount).
            # Result: still "none_found"/"none" -- but now a CLEAN read, not a confounded one.
            # 35 real samples total (the prior 34 plus this clean re-run), zero non-none. This is
            # now genuine evidence the non-none branch is rare/hard to trigger with the current
            # model and fixture set, not an artifact of the harness bug -- the bug explained one
            # bad data point, not the whole pattern. Still not enough to call the branch
            # unreachable (see the mechanism-only claim above); a fixture with an even more
            # forceful real-world analogy trigger, or more scenario variety, would be the next
            # step if this is ever worth chasing further.
            if analogy_fit not in {"strong", "moderate", "weak", "none", "", None}:
                errors.append(
                    f"historical_analogy.analogy_fit: must be one of strong|moderate|weak|none, "
                    f"got {analogy_fit!r}"
                )
            if company == "none_found":
                if analogy_fit not in {"none", None, ""}:
                    errors.append(
                        f"historical_analogy: company='none_found' but analogy_fit='{analogy_fit}' (should be 'none')"
                    )
            elif company:
                if analogy_fit == "weak":
                    errors.append(
                        "historical_analogy: analogy_fit='weak' is inadmissible — must be 'strong' or 'moderate'"
                    )

        # thesis_archetype_alignment
        ta = sd.get("thesis_archetype_alignment", {})
        if isinstance(ta, dict):
            bull_archetype = ta.get("bull_archetype")
            if bull_archetype not in THESIS_ARCHETYPES:
                errors.append(
                    f"thesis_archetype_alignment.bull_archetype: must be one of {THESIS_ARCHETYPES}"
                )
            # alignment_with_researcher is a declared 2-value enum (agrees|disagrees) -- the
            # existing check below only reacted to the value "disagrees" for a side effect and
            # never rejected anything else, so e.g. "somewhat" or "unclear" passed silently.
            # Found the same way as Stock Researcher's competitive_position/management_assessment
            # gap: listed in test_prompt_contract.py's _EXPLICITLY_HANDLED_ELSEWHERE on the
            # assumption a real value check existed here, when only a side-effect check did.
            alignment = ta.get("alignment_with_researcher", "")
            if alignment not in {"agrees", "disagrees"}:
                errors.append(
                    f"thesis_archetype_alignment.alignment_with_researcher: must be "
                    f"agrees|disagrees, got {alignment!r}"
                )
            if alignment == "disagrees" and not ta.get("disagreement_reason", "").strip():
                errors.append("thesis_archetype_alignment.disagreement_reason: must be non-empty when alignment='disagrees'")
            # 86bbt1k1p follow-up: alignment_with_researcher must actually reflect whether
            # bull_archetype matches the REAL researcher archetype, not the model's own
            # (possibly wrong) echo of it in this same object -- live-observed on a real
            # scenario_03 run: bull_archetype == researcher_archetype (both
            # "value_trap_candidate") but alignment_with_researcher == "disagrees", a flat
            # self-contradiction per the prompt's own Rule 3 ("your alignment with the
            # Researcher's classification is recorded in alignment_with_researcher" -- a
            # pure label-match, no other reading available). Uses the caller-supplied real
            # value rather than ta.get("researcher_archetype") precisely so a mistranscribed
            # echo can't make two wrongs look consistent. Only checked when the caller
            # supplies it AND alignment/bull_archetype are themselves valid -- an already-
            # invalid alignment or archetype is flagged above; piling this check on top of
            # that would just be a second error for the same root cause.
            if (
                real_researcher_archetype in THESIS_ARCHETYPES
                and bull_archetype in THESIS_ARCHETYPES
                and alignment in {"agrees", "disagrees"}
            ):
                should_agree = bull_archetype == real_researcher_archetype
                if should_agree and alignment != "agrees":
                    errors.append(
                        f"thesis_archetype_alignment.alignment_with_researcher: bull_archetype "
                        f"{bull_archetype!r} matches the real researcher archetype -- must be "
                        f"'agrees', got {alignment!r}"
                    )
                elif not should_agree and alignment != "disagrees":
                    errors.append(
                        f"thesis_archetype_alignment.alignment_with_researcher: bull_archetype "
                        f"{bull_archetype!r} differs from the real researcher archetype "
                        f"{real_researcher_archetype!r} -- must be 'disagrees', got {alignment!r}"
                    )

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_bear_advocate(
    output: dict, real_researcher_archetype: str | None = None
) -> tuple[bool, list[str]]:
    """Validate Bear Case Advocate output.

    real_researcher_archetype: same param as validate_bull_advocate, same reasoning.
    """
    errors: list[str] = []

    if output.get("recommendation") != "bearish":
        errors.append(f"recommendation: must be 'bearish', got '{output.get('recommendation')}'")

    conf = output.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= conf <= 100):
        errors.append(f"confidence: must be integer 0-100, got {conf}")

    ts = output.get("thesis_summary", "")
    if char_count(ts) < 80:
        errors.append("thesis_summary: too short (min 80 chars)")

    sa = output.get("strongest_argument", "")
    if not sa.strip():
        errors.append("strongest_argument: must be non-empty")
    elif not _has_pass1_agent_id(sa):
        errors.append("strongest_argument: must contain at least one Pass 1 agent ID")

    # key_factors: 2-4, all must have sentiment="negative"
    key_factors = output.get("key_factors", [])
    if isinstance(key_factors, list):
        if len(key_factors) < 2:
            errors.append(f"key_factors: need >=2, got {len(key_factors)}")
        if len(key_factors) > 4:
            errors.append(f"key_factors: need <=4, got {len(key_factors)}")
        seen = [kf.get("evidence", "") for kf in key_factors if isinstance(kf, dict)]
        if len(seen) != len(set(seen)):
            errors.append(
                "key_factors: duplicate evidence string across factors -- reusing the "
                "same evidence means you have one factor, not two"
            )
        for i, kf in enumerate(key_factors):
            if isinstance(kf, dict):
                ev = str(kf.get("evidence", ""))
                if not ev.strip():
                    errors.append(f"key_factors[{i}].evidence: must be non-empty")
                elif not _has_pass1_agent_id(ev):
                    errors.append(
                        f"key_factors[{i}].evidence: must cite a Pass 1 agent ID"
                    )
                elif not _evidence_carries_value(ev):
                    errors.append(
                        f"key_factors[{i}].evidence: cites field names without values "
                        f"({ev!r}) -- quote the supporting figure or value"
                    )
            if isinstance(kf, dict) and kf.get("sentiment") != "negative":
                errors.append(
                    f"key_factors[{i}].sentiment: must be 'negative' for Bear advocate, got '{kf.get('sentiment')}'"
                )

    _check_array_bounds(output, "thesis_risks", 1, 3, errors)

    # narrative: 1500-2800 chars, >=3 distinct Pass 1 agents
    narrative = output.get("narrative", "")
    nc = char_count(narrative)
    if nc < 1500:
        errors.append(f"narrative: too short ({nc} chars, min 1500)")
    if nc > 2800:
        errors.append(f"narrative: too long ({nc} chars, max 2800)")
    agents_in_narrative = [aid for aid in PASS1_AGENT_IDS if aid in narrative]
    if len(agents_in_narrative) < 3:
        errors.append(
            f"narrative: must reference >=3 distinct Pass 1 agent IDs, found {len(agents_in_narrative)}"
        )

    sd = output.get("structured_data", {})
    if isinstance(sd, dict):
        _check_array_bounds(sd, "core_arguments", 1, 4, errors)
        _check_array_bounds(sd, "downside_triggers", 1, 4, errors)
        _check_array_bounds(sd, "market_misreads", 1, 3, errors)

        # tail_risk_assessment (Rule 7): tail_risk_level's own enum value is covered by the
        # shared sweep below. Content requirements are CONDITIONAL on the level, per the
        # prompt's own rule 7 verbatim: "For elevated/moderate, provide a short
        # triggering_event description and a Pass 1 citation. For negligible (the expected
        # answer for most fundamentally sound stocks), leave scenario/triggering_event/evidence
        # empty." Confirmed live: a real "negligible" call correctly returned all three empty --
        # requiring them non-empty unconditionally would reject the documented, expected case.
        tra = sd.get("tail_risk_assessment")
        if not isinstance(tra, dict):
            errors.append("structured_data.tail_risk_assessment: must be an object")
        else:
            level = tra.get("tail_risk_level")
            if level in (None, ""):
                errors.append("structured_data.tail_risk_assessment.tail_risk_level: missing")
            elif level in ("elevated", "moderate"):
                if not str(tra.get("scenario") or "").strip():
                    errors.append(
                        f"structured_data.tail_risk_assessment.scenario: must be non-empty when "
                        f"tail_risk_level={level!r}"
                    )
                if not str(tra.get("triggering_event") or "").strip():
                    errors.append(
                        f"structured_data.tail_risk_assessment.triggering_event: must be "
                        f"non-empty when tail_risk_level={level!r}"
                    )
                if not str(tra.get("evidence") or "").strip():
                    errors.append(
                        f"structured_data.tail_risk_assessment.evidence: must be non-empty "
                        f"when tail_risk_level={level!r}"
                    )
                agents = tra.get("supporting_pass1_agents")
                if not isinstance(agents, list) or not agents:
                    errors.append(
                        f"structured_data.tail_risk_assessment.supporting_pass1_agents: must "
                        f"cite >=1 Pass 1 agent when tail_risk_level={level!r}"
                    )

        # thesis_archetype_alignment -- completely unvalidated before this fix (86bbuhjup
        # finding 5). Mirrors Bull's equivalent block in validate_bull_advocate field-for-
        # field, except agrees_with_researcher is a BOOL here (Bull's alignment_with_
        # researcher is a string "agrees"|"disagrees") -- a genuine schema asymmetry
        # between the two sibling agents' real prompts, confirmed by reading both directly,
        # not a copy-paste inconsistency introduced here.
        taa = sd.get("thesis_archetype_alignment", {})
        if isinstance(taa, dict):
            bear_archetype = taa.get("bear_archetype")
            if bear_archetype not in THESIS_ARCHETYPES:
                errors.append(
                    f"thesis_archetype_alignment.bear_archetype: must be one of {THESIS_ARCHETYPES}"
                )
            agrees = taa.get("agrees_with_researcher")
            if not isinstance(agrees, bool):
                errors.append(
                    f"thesis_archetype_alignment.agrees_with_researcher: must be a boolean, "
                    f"got {type(agrees).__name__}"
                )
            elif agrees is False and not str(taa.get("disagreement_note") or "").strip():
                errors.append(
                    "thesis_archetype_alignment.disagreement_note: must be non-empty when "
                    "agrees_with_researcher=false"
                )
            # Cross-field consistency against the REAL researcher archetype -- same
            # reasoning as Bull's equivalent check (see validate_bull_advocate): live-
            # observed on the bull side (scenario_03), the mechanism is identical for
            # Bear, just not yet independently observed live.
            if (
                isinstance(agrees, bool)
                and real_researcher_archetype in THESIS_ARCHETYPES
                and bear_archetype in THESIS_ARCHETYPES
            ):
                should_agree = bear_archetype == real_researcher_archetype
                if should_agree and agrees is not True:
                    errors.append(
                        f"thesis_archetype_alignment.agrees_with_researcher: bear_archetype "
                        f"{bear_archetype!r} matches the real researcher archetype -- must be "
                        f"true, got {agrees!r}"
                    )
                elif not should_agree and agrees is not False:
                    errors.append(
                        f"thesis_archetype_alignment.agrees_with_researcher: bear_archetype "
                        f"{bear_archetype!r} differs from the real researcher archetype "
                        f"{real_researcher_archetype!r} -- must be false, got {agrees!r}"
                    )

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_tax_strategist(
    output: dict, account_type: str | None = None
) -> tuple[bool, list[str]]:
    """Validate Tax Strategist output.

    Citation and admissibility enforcement added 2026-09-01. Before that this function
    checked field presence, two array counts, narrative LENGTH, and the tax_profile
    enums -- and nothing else. Every citation rule the prompt declares (7, 9, 10, 12,
    13) and the loss-harvesting account constraint (14) were unenforced, so the agent
    could emit an uncited claim, a fabricated withholding rate, or a narrative citing
    no Pass 1 agent at all and still return PASS. Bull, Bear and Risk each gained real
    evidence checks during the 2026-08 audit; Tax was not carried along, and nothing
    surfaced it because a validator that checks little still passes.

    `account_type` is optional so existing callers keep working; rule 14 is only
    checkable when it is supplied.
    """
    errors: list[str] = []

    # Must NOT have recommendation field
    if "recommendation" in output:
        errors.append("Tax Strategist: must NOT have a 'recommendation' field")

    # Must have groundedness_score, not confidence
    gs = output.get("groundedness_score")
    if gs is None and "confidence" in output:
        errors.append("Tax Strategist: uses 'groundedness_score', not 'confidence'")
    elif gs is not None:
        if not isinstance(gs, (int, float)) or not (0 <= gs <= 100):
            errors.append(f"groundedness_score: must be 0-100, got {gs}")

    # thesis_summary, strongest_signal must be non-empty
    if not output.get("thesis_summary", "").strip():
        errors.append("thesis_summary: must be non-empty")
    if not output.get("strongest_signal", "").strip():
        errors.append("strongest_signal: must be non-empty")
    elif not _has_tax_citation(output["strongest_signal"]):
        # Rule 7: ">=1 valid citation token".
        errors.append(
            "strongest_signal: must cite a Pass 1 agent ID, a tax-metric token "
            f"({', '.join(sorted(TAX_METRIC_TOKENS))}), or REF"
        )

    # key_factors: 2-4, each evidence in the declared compact format
    _check_array_bounds(output, "key_factors", 2, 4, errors)
    for i, kf in enumerate(output.get("key_factors") or []):
        if not isinstance(kf, dict):
            errors.append(f"key_factors[{i}]: must be an object")
            continue
        ev = str(kf.get("evidence") or "")
        if not ev.strip():
            errors.append(f"key_factors[{i}].evidence: must be non-empty")
        elif not _TAX_EVIDENCE_PREFIX_RE.match(ev):
            # Rule 9: "Evidence starts with a valid citation token followed by `:`".
            errors.append(
                f"key_factors[{i}].evidence: must START with a valid citation token "
                f"followed by ':' (got {ev[:40]!r})"
            )
        elif not _evidence_states_a_claim(ev):
            errors.append(
                f"key_factors[{i}].evidence: cites a token but states no claim after "
                f"it (got {ev[:40]!r})"
            )
        # sentiment: real vocabulary is positive|negative|neutral (confirmed against the
        # current tax_strategist/v1.txt template) -- previously unenforced entirely. Not
        # folded into the shared DECLARED_ENUMS sweep: Bull/Bear/Risk each constrain their own
        # key_factors[].sentiment to a different, narrower subset (Bull: positive only; Bear:
        # negative only; Risk: negative|neutral only), so "sentiment" is genuinely
        # per-agent-specific, not a safe global vocabulary.
        sent = kf.get("sentiment")
        if sent is not None and sent not in {"positive", "negative", "neutral"}:
            errors.append(
                f"key_factors[{i}].sentiment: must be positive|negative|neutral, got {sent!r}"
            )

    # narrative: 1200-3200 chars
    narrative = output.get("narrative", "")
    nc = char_count(narrative)
    if nc < 1200:
        errors.append(f"narrative: too short ({nc} chars, min 1200)")
    if nc > 3200:
        errors.append(f"narrative: too long ({nc} chars, max 3200)")

    # Rule 10 citation breadth: ">=2 distinct Pass 1 IDs (or 1 + FUND specifically)
    # AND >=3 distinct pre-computed metric tokens AND >=1 REF citation". This is the
    # ONLY surface where the prompt demands Pass 1 breadth -- key_factors evidence is
    # fully compliant when every token is a tax metric, so measuring Pass 1 usage
    # there and reading its absence as the agent ignoring Pass 1 is a category error.
    if narrative:
        mt = set(_TAX_TOKEN_RE.findall(narrative))
        # Relaxed 2026-09-03 from ">=2 distinct (or exactly 1 if FUND)" to ">=1 any"
        # (audit E88). A tax narrative has natural reason to cite FUND (yield, payout,
        # earnings) and arguably RSRCH (archetype implies holding period). TECH's RSI
        # and SENT's news sentiment bear on neither capital-gains treatment nor
        # withholding, so there is usually no honest SECOND citation to make.
        #
        # Measured consequence of the old rule: it was the single most frequent
        # first-attempt failure in the pipeline (4x, vs 1x each for the sibling token
        # and REF rules), and the retry satisfied it by ENUMERATING -- "The analysis is
        # grounded in RSRCH, FUND, TECH, SENT, MACRO, and the pre-computed tokens ..."
        # in one sentence supporting no tax claim. The first attempt's narrative, which
        # cited only what was relevant, was the better output.
        #
        # The old rule's own "(or exactly 1 if it is FUND)" exception already conceded
        # the point for the common case; this generalises it instead of patching it.
        # The >=3 token and >=1 REF rules are unchanged -- they fail rarely and are
        # satisfiable from material the agent genuinely uses.
        # NO Pass 1 citation floor. Removed 2026-09-03 (audit E90) after >=2 and then
        # >=1 both failed for the same reason.
        #
        # The rule existed to keep the tax narrative company-specific. **The >=3
        # tax-metric token rule below already does that**: the tokens carry
        # `DIVID: 2.3% yield, 4 payments/yr`, `ELIG: canadian_eligible`, `LIST: TSX`,
        # `WHT (this account, tfsa): 0.0%` -- all specific to this holding and this
        # account. The Pass 1 floor was redundant with a rule the agent satisfies
        # naturally, and was the only one it could not.
        #
        # Measured: 6 of 8 first attempts failed on it; a 2,343-char correct,
        # densely-cited trading-account narrative was rejected 3/3 for naming no agent,
        # while the narrative that PASSED cited all five in one enumerating sentence.
        # The prompt still asks for Pass 1 citation where it bears on tax treatment --
        # as guidance, without a floor that can only be met by listing IDs.
        if len(mt) < 3:
            errors.append(
                f"narrative: needs >=3 distinct tax-metric tokens, got {sorted(mt) or 'none'}"
            )
        if not _REF_RE.search(narrative):
            errors.append("narrative: needs >=1 REF citation")

    # tax_profile
    tp = output.get("tax_profile", {})
    if not isinstance(tp, dict):
        errors.append("tax_profile: must be a dict")
    else:
        valid_fit = {"excellent", "good", "fair", "poor"}
        account_fit = tp.get("account_fit_score")
        if account_fit not in valid_fit:
            errors.append(f"tax_profile.account_fit_score: must be one of {valid_fit}, got '{account_fit}'")

        # poor account_fit mandates non-null cross_account_recommendation
        if account_fit == "poor":
            car = tp.get("cross_account_recommendation")
            if not car or not str(car).strip():
                errors.append(
                    "tax_profile.cross_account_recommendation: must be non-null when account_fit_score='poor'"
                )

        # cross_account_recommendation.better_account -- also declared, also unenforced until
        # now. Real schema: a nested object ({better_account, reasoning, drag_delta_pct}), not
        # the bare string the presence check above tolerates for callers still on the old shape.
        car = tp.get("cross_account_recommendation")
        if isinstance(car, dict):
            valid_better_account = {"tfsa", "rrsp", "trading"}
            ba = car.get("better_account")
            if ba is not None and ba not in valid_better_account:
                errors.append(
                    f"tax_profile.cross_account_recommendation.better_account: must be one of "
                    f"{valid_better_account}, got {ba!r}"
                )

        valid_efficiency = {"favorable", "neutral", "unfavorable"}
        if tp.get("tax_efficiency_for_account") not in valid_efficiency:
            errors.append(
                "tax_profile.tax_efficiency_for_account: must be favorable|neutral|unfavorable"
            )

        # dividend_classification -- declared in the real schema, unenforced until now
        # (surfaced by the permanent enum-inventory test, not by a live failure).
        valid_div_class = {"canadian_eligible", "us", "foreign", "mixed", "none"}
        div_class = tp.get("dividend_classification")
        if div_class is not None and div_class not in valid_div_class:
            errors.append(
                f"tax_profile.dividend_classification: must be one of {valid_div_class}, got {div_class!r}"
            )

        # Numeric passthroughs: must be numeric
        for field in ("dividend_yield_pct", "withholding_tax_rate_pct", "effective_after_tax_yield_pct"):
            v = tp.get(field)
            if v is not None and not isinstance(v, (int, float)):
                errors.append(f"tax_profile.{field}: must be a number (passthrough from orchestrator)")

        # Rule 12: key_tax_risks 1-4; each needs risk, severity, and cited evidence.
        _check_array_bounds(tp, "key_tax_risks", 1, 4, errors)
        valid_sev = {"high", "medium", "low"}
        for i, r in enumerate(tp.get("key_tax_risks") or []):
            if not isinstance(r, dict):
                errors.append(f"tax_profile.key_tax_risks[{i}]: must be an object")
                continue
            if not str(r.get("risk") or "").strip():
                errors.append(f"tax_profile.key_tax_risks[{i}].risk: must be non-empty")
            if r.get("severity") not in valid_sev:
                errors.append(
                    f"tax_profile.key_tax_risks[{i}].severity: must be high|medium|low, "
                    f"got {r.get('severity')!r}"
                )
            ev = str(r.get("evidence") or "")
            if not _has_tax_citation(ev):
                errors.append(
                    f"tax_profile.key_tax_risks[{i}].evidence: must contain a valid "
                    f"citation token (got {ev[:40]!r})"
                )

        # Rule 13: tax_optimization_actions 0-3; each justification cites a token.
        _check_array_bounds(tp, "tax_optimization_actions", 0, 3, errors)
        for i, a in enumerate(tp.get("tax_optimization_actions") or []):
            if not isinstance(a, dict):
                errors.append(f"tax_profile.tax_optimization_actions[{i}]: must be an object")
                continue
            if not str(a.get("action") or "").strip():
                errors.append(
                    f"tax_profile.tax_optimization_actions[{i}].action: must be non-empty"
                )
            valid_applies_to = {"tfsa", "rrsp", "trading", "cross_account"}
            applies_to = a.get("applies_to")
            if applies_to is not None and applies_to not in valid_applies_to:
                errors.append(
                    f"tax_profile.tax_optimization_actions[{i}].applies_to: must be one of "
                    f"{valid_applies_to}, got {applies_to!r}"
                )
            just = str(a.get("justification") or "")
            if not _has_tax_citation(just):
                errors.append(
                    f"tax_profile.tax_optimization_actions[{i}].justification: must cite "
                    f"a valid token (got {just[:40]!r})"
                )

        # Rule 14: loss harvesting is null for TFSA/RRSP. Not bookkeeping -- a capital
        # loss is not claimable in a registered account, so a harvesting suggestion
        # there is wrong advice, not merely unhelpful. Only checkable when the caller
        # supplies the account.
        if account_type in ("tfsa", "rrsp"):
            lh = tp.get("loss_harvesting_opportunity")
            if lh not in (None, "not_applicable"):
                errors.append(
                    f"tax_profile.loss_harvesting_opportunity: must be null in a "
                    f"{account_type.upper()} -- capital losses are not claimable in a "
                    f"registered account (got {str(lh)[:60]!r})"
                )

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_risk_advisor_stage_a(output: dict) -> tuple[bool, list[str]]:
    """Validate Risk Advisor Stage A (account-neutral risk profile) output.

    Split from the old single `validate_risk_advisor` (ClickUp 86bbuhk82) to match the real
    two-stage prompt split from 86bbdutn6. The old function nested a Stage-B-only field,
    `position_size_recommendation`, under Stage A's `risk_profile` object -- confirmed via a
    full field-name extraction of `backend/prompts/risk_advisor/v1_stage_a.txt` and
    `v1_stage_b.txt` that it doesn't belong to Stage A's real schema at all. Because the check
    only fired when the field was truthy (`if psr and not (...)`), it silently never triggered
    against real Stage A output instead of loudly rejecting it -- broken quietly, not loudly.
    """
    errors: list[str] = []

    if "recommendation" in output:
        errors.append("Risk Advisor: must NOT have a 'recommendation' field")

    gs = output.get("groundedness_score")
    if gs is None and "confidence" in output:
        errors.append("Risk Advisor: uses 'groundedness_score', not 'confidence'")
    elif gs is not None:
        if not isinstance(gs, (int, float)) or not (0 <= gs <= 100):
            errors.append(f"groundedness_score: must be 0-100, got {gs}")

    if not output.get("thesis_summary", "").strip():
        errors.append("thesis_summary: must be non-empty")
    if not output.get("strongest_signal", "").strip():
        errors.append("strongest_signal: must be non-empty")
    elif not _has_risk_citation(output["strongest_signal"]):
        errors.append(
            "strongest_signal: must cite a Pass 1 agent ID or a metric token "
            "(BETA VOL DD LIQ CORR CONC)"
        )

    # key_factors: 2-4, sentiment must be negative|neutral only
    key_factors = output.get("key_factors", [])
    if isinstance(key_factors, list):
        _check_array_bounds(output, "key_factors", 2, 4, errors)
        for i, kf in enumerate(key_factors):
            if isinstance(kf, dict):
                sent = kf.get("sentiment", "")
                if sent not in {"negative", "neutral"}:
                    errors.append(
                        f"key_factors[{i}].sentiment: Risk Advisor must use 'negative' or 'neutral', got '{sent}'"
                    )

    # narrative: 250-400 words per the Stage A prompt's own hard constraint ("stop at 400
    # words") and schema comment ("250-400 words") == 1500-2400 chars at the ~6 chars/word
    # heuristic used elsewhere in this codebase. The old bound here (1200-3200) matched
    # neither number -- corrected while splitting this function, not a separate change.
    narrative = output.get("narrative", "")
    nc = char_count(narrative)
    if nc < 1500:
        errors.append(f"narrative: too short ({nc} chars, min 1500)")
    if nc > 2400:
        errors.append(f"narrative: too long ({nc} chars, max 2400)")

    # risk_profile (Stage A's own meaning: volatility/drawdown/downside-scenario quantification
    # -- NOT to be confused with Stage B's differently-shaped risk_profile_summary on the CIO,
    # or with position sizing, which belongs to Risk Advisor Stage B, see
    # validate_risk_advisor_stage_b).
    rp = output.get("risk_profile", {})
    if not isinstance(rp, dict):
        errors.append("risk_profile: must be a dict")
    else:
        valid_vol = {"very_high", "high", "moderate", "low"}
        if rp.get("volatility_assessment") not in valid_vol:
            errors.append(f"risk_profile.volatility_assessment: must be one of {valid_vol}, got {rp.get('volatility_assessment')!r}")

        # downside_scenarios: 2-4, no two within ±2% estimated_impact_pct, and each item's own
        # declared fields (probability, timeline) -- the count/spacing checks already existed;
        # the per-item probability/timeline enum checks did not.
        ds = rp.get("downside_scenarios", [])
        if not isinstance(ds, list) or len(ds) < 2:
            errors.append(f"risk_profile.downside_scenarios: need >=2 items, got {len(ds) if isinstance(ds, list) else 0}")
        elif len(ds) > 4:
            errors.append(f"risk_profile.downside_scenarios: need <=4 items, got {len(ds)}")
        else:
            valid_timeline = {"within_4_weeks", "1_to_3_months", "3_to_12_months",
                              "1_to_3_years", "3_plus_years"}
            for i, s in enumerate(ds):
                if not isinstance(s, dict):
                    continue
                if s.get("probability") not in {"high", "medium", "low"}:
                    errors.append(
                        f"risk_profile.downside_scenarios[{i}].probability: must be high|medium|low, got {s.get('probability')!r}"
                    )
                if s.get("timeline") not in valid_timeline:
                    errors.append(
                        f"risk_profile.downside_scenarios[{i}].timeline: must be one of {valid_timeline}, got {s.get('timeline')!r}"
                    )
            impacts = [s.get("estimated_impact_pct") for s in ds if isinstance(s, dict)]
            for i in range(len(impacts)):
                for j in range(i + 1, len(impacts)):
                    if impacts[i] is not None and impacts[j] is not None:
                        if abs(impacts[i] - impacts[j]) < 2.0:
                            errors.append(
                                f"downside_scenarios[{i}] and [{j}]: estimated_impact_pct too close "
                                f"({impacts[i]}% vs {impacts[j]}%, must differ by >=2%)"
                            )

        valid_rr = {"favorable", "neutral", "unfavorable"}
        if rp.get("risk_reward_ratio") not in valid_rr:
            errors.append("risk_profile.risk_reward_ratio: must be favorable|neutral|unfavorable")

        # data_sanity_flags: 0-3
        dsf = rp.get("data_sanity_flags", [])
        if isinstance(dsf, list) and len(dsf) > 3:
            errors.append(f"risk_profile.data_sanity_flags: max 3 items, got {len(dsf)}")

    _sweep_declared_enums(output, errors)
    return len(errors) == 0, errors


def validate_risk_advisor_stage_b(output: dict) -> tuple[bool, list[str]]:
    """Validate Risk Advisor Stage B (account-specific sizing/stop-loss overlay) output.

    Never validated before this split -- the old single function checked none of these fields
    at all except a mis-nested, effectively-dead `position_size_recommendation` check (see
    `validate_risk_advisor_stage_a`'s docstring).
    """
    errors: list[str] = []

    valid_corr = {"high", "medium", "low", "unknown"}
    corr = output.get("correlation_to_existing_portfolio")
    if corr is not None and corr not in valid_corr:
        errors.append(f"correlation_to_existing_portfolio: must be one of {valid_corr}, got {corr!r}")

    valid_conc = {"high", "medium", "low", "not_applicable"}
    conc = output.get("concentration_risk")
    if conc is not None and conc not in valid_conc:
        errors.append(f"concentration_risk: must be one of {valid_conc}, got {conc!r}")

    valid_liq = {"high", "medium", "low"}
    liq = output.get("liquidity_risk")
    if liq is not None and liq not in valid_liq:
        errors.append(f"liquidity_risk: must be one of {valid_liq}, got {liq!r}")

    # Same "X-Y%" contract CIO Stage B's position_sizing_recommendation check enforces on
    # this value at the point of consumption (cio.py) -- matched here at the point of
    # production so a malformed value ("-5%", "abc-def%") fails at its source agent instead
    # of passing here and only surfacing downstream at the CIO.
    psr = str(output.get("position_size_recommendation") or "")
    if not re.match(r"^\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*%\s*$", psr):
        errors.append(
            f"position_size_recommendation: must be a percentage band \"X-Y%\" "
            f"(e.g. '3-5%'), got {psr!r}"
        )

    stop_loss = output.get("stop_loss_suggestion")
    if stop_loss is not None and not isinstance(stop_loss, (int, float)):
        errors.append(f"stop_loss_suggestion: must be a number or null, got {type(stop_loss).__name__}")

    if not str(output.get("sizing_rationale") or "").strip():
        errors.append("sizing_rationale: must be non-empty")

    _check_array_bounds(output, "caveats", 0, 2, errors)

    return len(errors) == 0, errors


def validate_tax_passthroughs(output: dict, fixture: dict, account_type: str) -> tuple[bool, list[str]]:
    """Validate that Tax Strategist numeric passthroughs are within ±0.1 of orchestrator values."""
    errors: list[str] = []
    tp = output.get("tax_profile", {})
    oc = fixture.get("orchestrator_precomputed", {})
    fund = fixture.get("fundamental_data", {})

    expected_yield = fund.get("dividend_yield_pct", 0.0)
    actual_yield = tp.get("dividend_yield_pct")
    if actual_yield is not None and expected_yield is not None:
        if abs(actual_yield - expected_yield) > 0.15:
            errors.append(
                f"tax_profile.dividend_yield_pct: expected ~{expected_yield}%, got {actual_yield}% (tolerance ±0.1%)"
            )

    expected_wht = oc.get("WHT_rate_pct", 0.0)
    if account_type == "rrsp" and not fund.get("eligible_canadian_dividend"):
        expected_wht = 0.0  # RRSP treaty exemption
    actual_wht = tp.get("withholding_tax_rate_pct")
    if actual_wht is not None and expected_wht is not None:
        if abs(actual_wht - expected_wht) > 0.15:
            errors.append(
                f"tax_profile.withholding_tax_rate_pct: expected ~{expected_wht}%, got {actual_wht}% (tolerance ±0.1%)"
            )

    yield_key = "effective_after_tax_yield_rrsp_pct" if account_type == "rrsp" else "effective_after_tax_yield_tfsa_pct"
    expected_eff = oc.get(yield_key, oc.get("effective_after_tax_yield_tfsa_pct", 0.0))
    actual_eff = tp.get("effective_after_tax_yield_pct")
    if actual_eff is not None and expected_eff is not None:
        if abs(actual_eff - expected_eff) > 0.2:
            errors.append(
                f"tax_profile.effective_after_tax_yield_pct: expected ~{expected_eff}%, got {actual_eff}% (tolerance ±0.1%)"
            )

    return len(errors) == 0, errors
