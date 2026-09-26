"""Compresses Pass 1 outputs for injection into Pass 2 agent prompts.

Production port of `simulation/runners/compression.py` (86bbuhjup). Mimics the
orchestrator's OutputCompressor.compress_pass1() logic -- builds the data payload
each Pass 2 agent receives.

`compress_pass1_outputs`/`extract_confidence_levels` are clean ports (operate only
on already-produced Pass 1 agent JSON output, no fixture/DataBundle involvement).
`extract_data_quality_levels` (86bbummwp Tier 3) has no harness equivalent at all --
genuinely new, not a port, since the harness predates D6's mechanical
`data_quality_assessment` entirely. `compress_pass1_outputs` itself gained a new,
also-not-ported `mechanical_quality` param at the same time.
`build_pass2_user_message` is NOT a clean port: the harness version takes a
`fixture` dict as its first argument (`fixture["context"]` for
ticker/company_name/sector/etc.); production has no fixture, so this reads the
same fields from a real `DataBundle` instead -- `bundle.stock` (ticker, currency,
exchange), `bundle.company_info` (name, sector -- a plain dict per
data/schemas/data_bundle.py, not yet a typed sub-schema), `bundle.context`
(account_type, timeline), `bundle.data_vintage` (the timestamp fixture.context's
`data_timestamp` was standing in for).
"""
import json

from agents.pass2_view import build_pass2_view
from agents.utils import truncate_to_tokens
from data.schemas.data_bundle import DataBundle

AGENT_KEYS = {
    "RSRCH": "stock_researcher",
    "FUND": "fundamental_analyst",
    "TECH": "technical_analyst",
    "SENT": "sentiment_analyst",
    "MACRO": "macro_economist",
}


def compress_pass1_outputs(
    pass1_results: dict[str, dict | None],
    bundles: dict[str, dict] | None = None,
    mechanical_quality: dict[str, str] | None = None,
) -> dict[str, dict | None]:
    """Returns compressed version of each Pass 1 agent output (~500-800 tokens target).

    `pass2_view` is BUILT here, not read. No Pass 1 agent produces that field
    (every prompt lists it as orchestrator-populated) -- reading it directly off
    `output` would always see `{}` and render every agent as NOT AVAILABLE to
    Pass 2 (see docs/technical/prompt-revision-protocol.md 51.1).

    `bundles` supplies the orchestrator-owned fields the LLM never produces
    (precomputed numerics, derived enums), keyed by agent_id. Optional -- those
    fields come back None without it.

    `mechanical_quality` (86bbummwp Tier 3) supplies D6 section 3's per-agent
    `data_quality_assessment` -- keyed by agent_id, the same
    `runner.last_data_quality_assessment` value already persisted onto that
    agent's own `AgentOutput` row, threaded in here the same way `bundles` is
    rather than read back from the DB (this function only ever sees the raw
    Pass 1 LLM output dict, never the orchestrator-computed mechanical flags,
    confirmed directly -- there is no other path for this value to reach
    Pass 2/CIO within a single run).
    """
    compressed = {}
    bundles = bundles or {}
    mechanical_quality = mechanical_quality or {}
    for agent_id, output in pass1_results.items():
        if output is None:
            compressed[agent_id] = None
            continue
        # An explicit pass2_view on the output wins, so a caller can override;
        # otherwise build it from the agent's own interpretive fields.
        # isinstance, not truthiness: a model can emit `pass2_view` as a STRING
        # (observed live, RY.TO), which then reached build_pass2_user_message and
        # raised AttributeError: 'str' object has no attribute 'items'.
        explicit = output.get("pass2_view")
        view = explicit if isinstance(explicit, dict) and explicit else build_pass2_view(
            agent_id, output, bundles.get(agent_id)
        )
        compressed[agent_id] = {
            "assessment_summary": output.get("assessment_summary", ""),
            # reliability_score retired 86bbummwp (D6) -- not LLM-produced anymore.
            # analysis_confidence is the one real, live model-judged signal; a
            # genuinely-missing value honestly defaults to "insufficient" (a real member of
            # this enum's own vocabulary), unlike the old reliability_score=0 default, which
            # was never a value the field could actually take.
            "analysis_confidence": output.get("analysis_confidence", "insufficient"),
            # data_quality_assessment (86bbummwp Tier 3) -- mechanical, from
            # `mechanical_quality` above, NEVER from `output` (the LLM never
            # produces this field; see D6 section 3's own clean-ownership split).
            # "low" default matches this 3-value enum having no "insufficient"
            # member (unlike analysis_confidence's 4-value one) -- an agent with
            # no mechanical_quality entry is the worst real case for input
            # quality, so "low" is the honest, conservative default.
            "data_quality_assessment": mechanical_quality.get(agent_id, "low"),
            "caveats": output.get("caveats", []),
            "pass2_view": view,
            "narrative_truncated": truncate_to_tokens(output.get("narrative", ""), 300),
        }
    return compressed


def build_pass2_user_message(
    bundle: DataBundle,
    compressed_pass1: dict[str, dict | None],
    account_type: str | None = None,
) -> str:
    """Builds the user message payload for Pass 2 agents from compressed Pass 1.

    `bundle` is the real DataBundle for this analysis run, standing in for the
    harness's `fixture` dict -- see this module's docstring for the field mapping.
    """
    ticker = bundle.stock.ticker
    company = bundle.company_info.get("name")
    sector = bundle.company_info.get("sector")
    exchange = bundle.stock.exchange
    currency = bundle.stock.currency
    timeline = bundle.context.timeline
    acct = account_type or bundle.context.account_type
    timestamp = bundle.data_vintage.isoformat()

    lines = [
        f"{ticker} ({company}) | {sector} | {exchange} | {currency}",
        f"As of: {timestamp} | Timeline: {timeline} | Account: {acct}",
        "",
    ]

    agent_labels = {
        "RSRCH": "STOCK RESEARCHER",
        "FUND": "FUNDAMENTAL ANALYST",
        "TECH": "TECHNICAL ANALYST",
        "SENT": "SENTIMENT ANALYST",
        "MACRO": "MACRO ECONOMIST",
    }

    missing_agents = []
    for agent_id in ["RSRCH", "FUND", "TECH", "SENT", "MACRO"]:
        out = compressed_pass1.get(agent_id)
        label = agent_labels[agent_id]
        if out is None:
            lines.append(f"PASS 1 — {label} ({agent_id}): NOT AVAILABLE — reason: agent failed or data insufficient")
            missing_agents.append(agent_id)
            continue

        p2v = out.get("pass2_view") or {}
        if not isinstance(p2v, dict) or not p2v:
            # Stub / insufficient data
            lines.append(
                f"PASS 1 — {label} ({agent_id}): NOT AVAILABLE — reason: analysis_confidence=insufficient, pass2_view empty"
            )
            missing_agents.append(agent_id)
            continue

        lines.append(f"PASS 1 — {label} ({agent_id}):")
        # Was "reliability_score: {x}/100 | quality: {enum}" (86bbummwp / D6: both retired).
        lines.append(f"  confidence: {out['analysis_confidence']}")
        lines.append(f"  assessment_summary: {out['assessment_summary']}")
        if out.get("caveats"):
            lines.append(f"  caveats: {json.dumps(out['caveats'])}")

        # Serialize pass2_view fields compactly
        for k, v in p2v.items():
            if isinstance(v, dict):
                lines.append(f"  {k}: {json.dumps(v)}")
            elif isinstance(v, list):
                lines.append(f"  {k}: {json.dumps(v)}")
            else:
                lines.append(f"  {k}: {v}")

        lines.append(f"  narrative(key points): {out['narrative_truncated']}")
        lines.append("")

    if missing_agents:
        lines.append(f"DATA GAPS / MISSING PASS 1 AGENTS: {', '.join(missing_agents)}")

    return "\n".join(lines)


def extract_confidence_levels(
    compressed_pass1: dict[str, dict | None],
) -> dict[str, str]:
    """Extract analysis_confidence by agent ID for warning block generation.

    "insufficient" is the honest value for both agent-didn't-run and
    agent-produced-nothing-usable -- both render as the real, valid
    "insufficient" enum member, which build_pass1_reliability_warnings() maps to
    the same "skipped" status label.
    """
    levels = {}
    for agent_id, out in compressed_pass1.items():
        if out is None:
            levels[agent_id] = "insufficient"
        elif not out.get("pass2_view"):
            levels[agent_id] = "insufficient"
        else:
            levels[agent_id] = out.get("analysis_confidence", "insufficient")
    return levels


def extract_data_quality_levels(
    compressed_pass1: dict[str, dict | None],
) -> dict[str, str]:
    """Extract data_quality_assessment by agent ID (86bbummwp Tier 3), for the
    same warning-block generation `extract_confidence_levels()` feeds.

    Deliberately NOT mirroring that function's `pass2_view`-gated second
    branch: `data_quality_assessment` is a mechanical fact about INPUT data,
    computed before the LLM call ever happens, and is unaffected by whether
    the agent's own narrative output was coherent enough to build a
    `pass2_view` -- gating on that would wrongly downgrade an agent with
    genuinely clean input data just because its own output happened to be
    unparseable. `out is None` (the agent produced nothing at all) is the
    one real "unknown" case, and "low" is the honest, conservative value for
    it -- this 3-value enum has no "insufficient" member.
    """
    levels = {}
    for agent_id, out in compressed_pass1.items():
        if out is None:
            levels[agent_id] = "low"
        else:
            levels[agent_id] = out.get("data_quality_assessment", "low")
    return levels
