"""Shared utilities: disagreement score, reliability warnings, JSON parsing, gates.

Production port of `simulation/utils.py` (86bbuhjup) -- pure functions, no I/O, no
async needed, so this is a direct port unlike the runners that consume it. Ported
wholesale rather than cherry-picked: this project has already been bitten twice by
splitting a shared foundation across independently-maintained copies
(`researcher_thesis_archetype` drifting across 4 copies; `CONFIDENCE_STATUS_LABELS`
maintained separately in two files) -- the harness itself exists at all because of
that lesson (see `simulation/validators/common.py`'s own docstring).

`gate1_check()` is NOT a byte-for-byte port -- caught live while porting: the
harness's own copy still encoded the pre-2026-09-16 `>=3-of-5 + RSRCH-or-FUND` count
threshold, which `docs/decision-log.md`'s 2026-09-16 entry explicitly settled
*against* in favor of completion-only gating. The harness's own function was never
updated when that decision landed (its "Status" line names
`pass1-confidence-model.md`/`agent-architecture.md` as updated, not this file) --
fixed in both places in the same pass so this doesn't resurface a second time.
"""
import json
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class RenderedField:
    """One agent payload field's rendered text plus its own presence flag,
    computed together in the same helper call (86bbwachy Phase 4) -- not two
    separately-maintained functions (one building the string, one re-deriving
    "is this N/A" a second time) that could silently drift apart. Used by
    each in-scope agent's own field-rendering helpers (e.g.
    pass1_stock_researcher.py's `_dividend_context`/`_business_description`);
    `build_user_message()` collects the `.present` flags into the
    `{field_name: bool}` map persisted as `agent_outputs.input_field_coverage`.

    `text` is what actually goes into the prompt (including the "N/A ..."
    wording when absent) -- `present` is the machine-readable fact a human
    would otherwise have to re-derive by grepping the rendered prompt for the
    word "N/A", which isn't reliable (see run-instrumentation planning: this
    is the whole reason a separate signal is worth building).
    """
    text: str
    present: bool


DIRECTION_MAP = {
    "bullish": 2,
    "somewhat_bullish": 1,
    "neutral": 0,
    "somewhat_bearish": -1,
    "bearish": -2,
}

AGENT_ID_MAP = {
    "stock_researcher": "RSRCH",
    "fundamental_analyst": "FUND",
    "technical_analyst": "TECH",
    "sentiment_analyst": "SENT",
    "macro_economist": "MACRO",
}

PASS1_AGENT_IDS = ["RSRCH", "FUND", "TECH", "SENT", "MACRO"]


def compute_disagreement_score(bull_confidence: int, bear_confidence: int) -> tuple[int, str]:
    """Exact formula from Orchestration Engine doc."""
    scores = [2, -2]  # bull always bullish=2, bear always bearish=-2
    confidences = [bull_confidence / 100.0, bear_confidence / 100.0]
    total_weight = sum(confidences)
    if total_weight < 1e-6:
        return 0, "consensus"
    weighted_mean = sum(s * c for s, c in zip(scores, confidences)) / total_weight
    weighted_var = sum(
        c * (s - weighted_mean) ** 2 for s, c in zip(scores, confidences)
    ) / max(total_weight, 1e-6)
    direction_component = min(50, int(weighted_var / 4.0 * 50))
    confidence_spread = int(np.std(confidences) * 50)
    disagreement_score = min(100, direction_component + confidence_spread)
    if disagreement_score <= 19:
        label = "consensus"
    elif disagreement_score <= 39:
        label = "mild_dissent"
    elif disagreement_score <= 59:
        label = "split_decision"
    else:
        label = "high_conflict"
    return disagreement_score, label


def compute_outlook_distance(primary_outlook: str, shadow_outlook: str) -> tuple[int, bool]:
    """Distance between the primary CIO's and Shadow CIO's stock_outlook on
    the 5-point scale, plus the >2-notch high_divergence flag (86bbt1kct).

    Documented as a mechanical, zero-LLM computation ("computed in
    orchestrator merge, not by the LLM" -- docs/agents/shadow_cio.md,
    Design Decision 9): the Shadow's own runner never sees the primary's
    output and cannot compute this itself. Reuses DIRECTION_MAP, already
    the project's one shared bullish..bearish -> int vocabulary (used by
    compute_disagreement_score above) -- no new vocabulary needed.
    """
    distance = abs(DIRECTION_MAP[primary_outlook] - DIRECTION_MAP[shadow_outlook])
    return distance, distance > 2


# The one, shared analysis_confidence -> warning-status mapping (D6's own downstream-rewrite
# table: medium->caution, low->unreliable, insufficient->skipped, high->OK/all-clear).
# validators/compression.py's validate_reliability_warning_format() imports this rather than
# keeping its own copy -- a second, independently-maintained copy is exactly the duplication
# shape that already caused a real, silent bug this project (researcher_thesis_archetype
# drifting across 4 copies until one was caught reading the wrong key).
CONFIDENCE_STATUS_LABELS = {
    "high": "OK",
    "medium": "caution",
    "low": "unreliable",
    "insufficient": "skipped",
}


def render_data_coverage_line(field_presence: dict[str, bool], gap_sentences: dict[str, str]) -> str:
    """Shared DATA COVERAGE line renderer (86bbummwp Tier 1a) for the four Pass 1
    agents -- Fundamental, Technical, Sentiment, Macro Economist -- whose
    `data_coverage_line` was previously hardcoded to the literal "standard." on
    every single run, feeding each agent's own `analysis_confidence`
    self-assessment a false statement about its own inputs. Stock Researcher's
    own `_data_coverage_line()` (pass1_stock_researcher.py) is deliberately NOT
    migrated here: its gap logic reads a `list[str]` (`missing_sources_list`), a
    genuinely different input shape from this function's `dict[str, bool]`, not
    just a naming difference.

    Iterates `field_presence.items()`, not `gap_sentences.items()`: a key ABSENT
    from `field_presence` (e.g. Macro Economist's `statcan` key, only ever
    included for a Canadian stock) means "not applicable, don't mention it,"
    while a key PRESENT but `False` means a real gap worth a sentence -- these
    are semantically different, not two spellings of the same thing, and
    getting this backwards would silently break every caller, not just Macro.
    """
    gaps = [
        gap_sentences[field]
        for field, present in field_presence.items()
        if not present and field in gap_sentences
    ]
    return "standard." if not gaps else "; ".join(gaps) + "."


def to_data_coverage(field_presence: dict[str, bool], gap_sentences: dict[str, str]) -> dict:
    """D6's `data_coverage` mechanical flag (86bbummwp Tier 2), structured
    (`{"present": [...], "absent": [...]}`) rather than prose -- for
    `agent_outputs.data_coverage`, distinct from `render_data_coverage_line()`'s
    prompt-facing sentence above.

    Same two-argument shape and same filtering as `render_data_coverage_line()`
    -- deliberately: neither Fundamental nor Macro Economist has a
    separately-materialized "coarse" presence dict to hand to a bare
    `field_presence`-only function (Fundamental's own full `field_presence` has
    15 granular ratio keys it never wants surfaced here; Macro's own
    `commodities`-relevance filtering happens inline inside its own
    `_data_coverage_line()`, not as a reusable object) -- so this filters to
    `gap_sentences`' own keys exactly the way the prose renderer already does,
    letting every agent call this with the exact same two arguments it already
    passes to `render_data_coverage_line()`, with no new object to build or
    keep in sync between the two representations.
    """
    present = []
    absent = []
    for field, is_present in field_presence.items():
        if field not in gap_sentences:
            continue
        (present if is_present else absent).append(field)
    return {"present": present, "absent": absent}


def render_data_warnings(anomalies: list[str], stale_data: list[str]) -> str:
    """Combines D6's `anomalies` (86bbummwp Tier 2, already full sentences --
    e.g. "US yield curve is inverted") and `stale_data` (bare series/category
    names -- e.g. `["rate", "cpi"]`) into the one `data_warnings` prompt
    placeholder every Pass 1 template already declares.

    Generalizes Technical Analyst's own former `_data_warnings_line()` (which
    only ever handled `anomalies`) to also cover `stale_data`, discovered
    missing during this same investigation: `self.last_stale_data` was
    computed and stored on every one of the 5 agents but never shown to the
    model anywhere, even for Technical Analyst, whose `data_warnings` wiring
    was otherwise the one agent doing this right since Tier 1b.

    Returns `""` when both lists are empty (`"; ".join([])` on its own, no
    `or ""` fallback needed), matching every prior version of this line.
    """
    parts = list(anomalies)
    if stale_data:
        parts.append(f"stale data: {', '.join(stale_data)}")
    return "; ".join(parts)


def build_pass1_reliability_warnings(agent_confidence: dict[str, str]) -> str:
    """Format: RSRCH: high (OK) | FUND: medium (caution) | TECH: low (unreliable) | ...
    Returns empty string when all agents are 'high'.
    """
    parts = []
    all_ok = True
    for agent_id in PASS1_AGENT_IDS:
        confidence = agent_confidence.get(agent_id, "insufficient")
        status = CONFIDENCE_STATUS_LABELS.get(confidence, "skipped")
        if status != "OK":
            all_ok = False
        parts.append(f"{agent_id}: {confidence} ({status})")
    if all_ok:
        return ""
    return " | ".join(parts)


def researcher_thesis_archetype(compressed_pass1: dict) -> str:
    """Stock Researcher's thesis_archetype, as every Pass 2 agent actually
    receives it: nested under RSRCH's `pass2_view` in `compressed_pass1`
    (compress_pass1_outputs() builds it there for every agent, unconditionally
    -- never under a raw-output key like `structured_data`, which belongs to
    the uncompressed agent output Pass 2 runners never see).

    Shared by all four Pass 2 runners that inject this value (Bull, Bear, Tax
    Strategist, Risk Advisor) -- previously four separately-maintained copies
    of this same lookup in the harness, one of which had drifted to reading
    `structured_data` and so always returned "unclassified" regardless of the
    real value. Ported as a shared function from day one here, not
    duplicated per runner a second time.
    """
    rsrch = compressed_pass1.get("RSRCH")
    if rsrch and rsrch.get("pass2_view"):
        return rsrch["pass2_view"].get("thesis_archetype", "unclassified")
    return "unclassified"


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, stripping markdown fences if present.

    Uses raw_decode rather than a bare json.loads so trailing non-whitespace
    after the first balanced JSON value is tolerated and discarded, not a
    hard failure -- defense-in-depth against the trailing-brace failure mode
    documented in docs/technical/two-turn-execution-mechanism.md (a
    continuation successor turn with a flatter schema than its predecessor's
    has a reproducible tendency to append one stray extra `}`). Same logic
    `agents/base.py`'s own `_parse_json_response` already applies to every
    LLM response this pipeline parses; kept here too since some ported
    callers (e.g. compression.py) parse JSON independently of a runner call.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Rough truncation: ~4 chars per token."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def char_count(text: str) -> int:
    return len(text) if text else 0


def extract_numeric_tokens(text: str) -> list[str]:
    """Extract all numeric tokens like 14.2%, $185, 2.4x from text."""
    return re.findall(r'\$?\d+\.?\d*[%x]?', text)


def agent_completed(out) -> bool:
    """Did this agent actually produce usable output?

    Both gates used to test `out is not None`. That test can never fail on a real
    failure: `BaseRunner.call_with_validation`/`call_with_validation_start` end with
    `return last_result, last_errors[, ...]`, handing back the LAST INVALID OUTPUT on
    a terminal validation failure, or `{}` when every attempt raised -- never `None`
    (`call_model`/`_call_generate` catch `JSONDecodeError`/`ClientError` internally).

    Owner ruling, 2026-09-02: **the CIO must never run when it has no agent inputs.**
    This predicate implements that ruling, and nothing beyond it:

      * `None`, a non-dict, or an empty dict -> not completed.
      * every value `None` -> not completed. This is the shape a below-threshold stub
        is specified to take (`docs/technical/pass1-confidence-model.md`) --
        compression's own `compress_pass1_outputs()` renders this case as
        `analysis_confidence: "insufficient"`.

    DELIBERATELY NOT DECIDED HERE: whether an agent that produced real content but
    FAILED VALIDATION should count as complete. SETTLED 2026-09-02 (docs/decision-log.md,
    audit E77): yes, it keeps counting -- validation status is a warning, not a gate.
    The case that would justify gating on it (fabrication from absent evidence) is
    already blocked upstream by this function's own None/empty/all-None check, which
    stops Pass 2 running at all when Pass 1 is empty.
    """
    if not isinstance(out, dict) or not out:
        return False
    return any(v is not None for v in out.values())


def gate1_check(pass1_outputs: dict) -> tuple[bool, str]:
    """Gate 1: completion-only. No multi-agent count-threshold.

    SETTLED 2026-09-16 (docs/decision-log.md) -- resolves the four contradictory
    definitions this gate had across `agent-architecture.md`, `orchestration-engine.md`,
    `pass1-confidence-model.md`, and the harness's own (stale, pre-decision) copy of
    this function. The only run-level gate is: did *any* usable Pass 1 output come
    back at all. Per-agent data insufficiency is signaled entirely through
    `analysis_confidence` (`insufficient`/`low`), which flows downstream like any
    other confidence value -- no separate whole-run block layered on top of it.
    `cleared_min_data_threshold` is not built, per-agent or aggregate;
    `analysis_confidence: insufficient` already carries the identical signal.
    """
    completed = [aid for aid, out in pass1_outputs.items() if agent_completed(out)]
    if not completed:
        return False, "No Pass 1 agent produced usable output (Gate 1 failure)"
    return True, "Gate 1 passed"


def gate2_check(pass2_outputs: dict) -> tuple[bool, str]:
    """Gate 2: Both Bull AND Bear must complete. Tax/Risk failures are warnings only.

    SETTLED 2026-09-02 (docs/decision-log.md, audit E63): bull AND bear are both
    mandatory, resolving a contradiction against `agent-architecture.md:336`'s
    "bull OR bear, >=2/4" wording (that document was corrected to match). A Gate 2
    failure skips the CIO entirely (also SETTLED 2026-09-02, audit E78) -- measured
    that a one-sided CIO run with one advocate missing produces a confident,
    undiscounted one-sided call rather than a properly hedged one.
    """
    if not agent_completed(pass2_outputs.get("bull")):
        return False, "Bull Case Advocate did not complete (Gate 2 failure)"
    if not agent_completed(pass2_outputs.get("bear")):
        return False, "Bear Case Advocate did not complete (Gate 2 failure)"
    return True, "Gate 2 passed"
