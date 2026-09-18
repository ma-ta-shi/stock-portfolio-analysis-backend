"""Runtime prompt template loading (ClickUp 86bbdutn6).

The 15 agent prompt design docs (`Agent Prompts/Current Prompts/*.md`, in the
top-level repo) are documentation, not feedable templates -- changelogs,
rationale, and orchestrator merge notes live alongside the actual System
Prompt / Output Schema content, fenced in Markdown with inconsistent fence
languages across agents. `backend/scripts/split_prompt_docs.py` extracts the
runtime-relevant blocks from those docs once (at authoring time, not on every
call) and writes the result to `backend/prompts/{slug}/v{n}.txt` -- this
module is what reads those already-split files back at call time, plus the
extraction functions the split script itself calls (kept here, not in the
script, so the highest-risk logic in this ticket is unit-tested against real
files rather than buried in a one-off CLI with no test coverage).

`backend/prompts/` is the *only* place a runtime template lives. A top-level
`prompts/` and a `simulation/prompts/` directory both existed before this
ticket and had silently drifted from each other and from this one since their
first commits -- retired 2026-09-18. `backend/` is a separately-deployed repo
(the local worker runs from a `backend/`-only checkout), so this was always
the only copy guaranteed to exist wherever backend code actually runs.
"""

import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Three fence languages are actually used across the 15 real docs (checked
# exhaustively, not sampled: every "System Prompt"/"Output Schema" heading in
# every doc, 30 headings total) -- ```plain text, ```javascript, ```json.
# `python` and an unlabelled fence are accepted too as harmless extra
# tolerance. A single-language pattern is a real, measured failure mode: it
# cost 20 empty Technical Analyst runs and a fully-empty Bear output schema
# extraction before this was fixed in backend/_audit/_bull_rig.py, which this
# pattern is ported from verbatim.
_FENCE = r"```(?:plain text|javascript|json|python)?\n(.*?)```"


def extract_block(md: str, header_prefix: str) -> str:
    """Return the first fenced block under an H1 heading starting with
    `header_prefix` (e.g. "System Prompt", or "System Prompt — Stage A" for
    CIO/Risk Advisor, whose two stages need distinct heading text to
    disambiguate -- a bare "System Prompt" search would silently match
    Stage A's heading as a prefix and always return Stage A.

    Raises ValueError if no such heading+fence pair exists -- a one-time
    authoring tool silently writing an empty template is worse than it
    failing loudly, and every real call site here runs against a known
    prompt doc where the block is expected to exist.
    """
    m = re.search(rf"^# {re.escape(header_prefix)}.*?{_FENCE}", md, re.S | re.M)
    if m is None:
        raise ValueError(f"No {header_prefix!r} block found")
    return m.group(1)


def extract_h1_span(md: str, header_prefix: str) -> str:
    """Isolate one H1 heading's span, from the heading itself to the next H1
    heading or end of document.

    Needed for the CIO/Risk Advisor Output Schema case specifically: both
    have exactly one `# Output Schema` H1, containing two `## Stage A` / `##
    Stage B` H2 sub-schemas nested inside it. `## Stage B` as bare text also
    recurs elsewhere in both documents (Validation Rules, Retry Prompt
    Injection, the token-budget table) -- an unbounded search for it would
    walk past the real schema and silently grab retry-prompt boilerplate
    instead (confirmed live: CIO's Retry Prompt Injection section has its
    own `## Stage B` heading directly above a ```plain text fence). Bounding
    to this H1's span first is what makes the later search unambiguous.
    """
    m = re.search(rf"^# {re.escape(header_prefix)}.*?(?=^# |\Z)", md, re.S | re.M)
    if m is None:
        raise ValueError(f"No {header_prefix!r} H1 section found")
    return m.group(0)


def extract_stage_schema(md: str, output_schema_header: str, stage_label: str) -> str:
    """Extract a stage's fenced schema block, nested as an H2 under the
    shared `# Output Schema` H1 -- see extract_h1_span for why this can't
    just be extract_block with a longer header string.
    """
    span = extract_h1_span(md, output_schema_header)
    m = re.search(rf"^## {re.escape(stage_label)}.*?{_FENCE}", span, re.S | re.M)
    if m is None:
        raise ValueError(f"No {stage_label!r} sub-schema found under {output_schema_header!r}")
    return m.group(1)


def bake_output_schema(system_prompt: str, schema: str, placeholder: str = "output_schema") -> str:
    """Splice a static Output Schema block into its `{output_schema}`
    placeholder, once, at split time -- not a `fill()` call, since nothing
    else in the schema text should resolve yet (per-call placeholders like
    `{account_type}` that happen to live inside the schema text are meant to
    survive this step and get resolved later, at real call time, by fill()).

    The 9 single-stage agents use the literal name `{output_schema}`. CIO and
    Risk Advisor use `{output_schema_stage_a}` / `{output_schema_stage_b}`
    instead -- a real, easy-to-miss difference: grepping for the bare
    `{output_schema}` string (as an earlier pass at this ticket did) finds
    zero matches in either of their prompts and wrongly suggests they don't
    use placeholder-based injection at all. They do; the name just carries
    the stage. Pass `placeholder="output_schema_stage_a"` (etc.) for those.
    """
    return system_prompt.replace(f"{{{placeholder}}}", schema)


def fill(template: str, values: dict[str, str]) -> str:
    """Substitute only known `{key}` placeholders; every other brace
    (including bare `{` that isn't a `\\w+` name, and citation notation like
    `N{num}` in the Sentiment/Researcher prompts) is left untouched.

    `str.format()` cannot be used here -- the prompts embed JSON in their
    Output Schema blocks, and `.format()` reads those braces as placeholders
    too, raising `KeyError` on the first one it doesn't recognize.
    """
    return re.sub(
        r"\{(\w+)\}",
        lambda m: values[m.group(1)] if m.group(1) in values else m.group(0),
        template,
    )


def load_template(slug: str, version: int = 1, stage: str | None = None) -> str:
    """Read `backend/prompts/{slug}/v{version}.txt`, or the
    `v{version}_stage_{stage}.txt` variant when `stage` ("a"/"b") is given.

    Raises FileNotFoundError with an actionable message -- a silent `None`
    return here would just move this exact ticket's own "nothing loads the
    documented prompt" problem one layer down.
    """
    filename = f"v{version}_stage_{stage}.txt" if stage else f"v{version}.txt"
    path = _PROMPTS_DIR / slug / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No runtime prompt template at {path} (slug={slug!r}, version={version}, "
            f"stage={stage!r}). Has backend/scripts/split_prompt_docs.py been run for "
            f"this agent?"
        ) from exc
