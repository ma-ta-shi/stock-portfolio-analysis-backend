"""Pass 3 — Chief Investment Officer (CIO) runner (v1.1).

Production port of `simulation/runners/pass3_cio.py` (86bbuhjup, built
earlier this session under 86bbt1kct/86bbt1k1p). Close to a clean port --
the CIO only ever consumes OTHER agents' already-produced outputs
(`compressed_pass1`, `pass2_outputs`) plus a handful of context fields
(ticker/company_name/sector/timeline/account_type); unlike every Pass 1/2
runner in this pipeline, it never reads raw DataBundle financial data
directly, so there's no per-field DataBundle-translation risk here. Only
`fixture["context"]` accesses became `bundle.stock`/`bundle.company_info`/
`bundle.context` reads -- every summary-building function below
(`build_pass1_summaries`, `build_advocate_summary`,
`build_risk_advisor_stage_a_summary`, `_build_general_outlook_summary`) is a byte-for-byte port, since none of
them ever touched `fixture` at all. `_build_tax_strategist_summary` was too,
originally -- no longer accurate as of 86bc8eg3j, which added
`capital_gains_treatment_summary`/`tax_optimization_actions` to it (Stage B
only; see that function's own docstring, and `build_user_message()`'s comment
on why Stage A gets neither).

Synthesis pass (local gpt-oss:20b in v1; see CLAUDE.md LLM Routing). Produces final stock_outlook and prediction.
Input semantics: Bull/Bear confidence = directional conviction.
Risk/Tax groundedness_score = data quality (NOT direction).
Directional proxies: Risk → risk_reward_ratio (Stage A); Tax → tax_efficiency_for_account
(Stage B only -- Stage A has zero Tax Strategist input, see build_user_message()).
Disagreement caps: split_decision → confidence ≤65; high_conflict → outlook must be neutral/somewhat range.
"""
import json

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import compute_disagreement_score
from agents.validators.cio import tail_risk_line, validate_cio_stage_a, validate_cio_stage_b
from data.schemas.data_bundle import DataBundle

# The old embedded system_prompt (independently authored, diverged from the
# real documented prompt) is deleted, not commented out -- the real prompt
# now loads from backend/prompts/cio/v1_stage_a.txt via load_template()
# (ClickUp 86bbdutn6). Stage B's template also exists (backend/prompts/cio/
# v1_stage_b.txt) and this runner makes both calls (run() and run_stage_b()). This is NOT a
# missing-primitive gap the way it is for Risk Advisor: per
# docs/technical/two-turn-execution-mechanism.md's "Why CIO doesn't use this
# mechanism", the CIO's Stage A->B was deliberately designed to NOT use
# context-array continuation at all -- Stage B needs fresh Tax Strategist
# input regardless of mechanism, and otherwise only
# needs Stage A's own compact JSON output pasted as text, which an ordinary
# second call already provides. What's actually missing here is the Stage B
# *invocation itself* (deciding what to paste, building the second call) --
# scoped to 86bbt1k1p's deferred Item C, not to 86bc2d414 (which built the
# continuation primitive Risk Advisor needs, and which this runner correctly
# has no use for).
#
# Stage B (86bbt1k1p Item C): run_stage_b() below builds that ordinary
# second call -- system_prompt filled from v1_stage_b.txt's named
# placeholders (general_outlook_summary/tax_strategist_summary), via the EXISTING, unmodified call_with_validation
# (no continuation context involved, per the reasoning above).
#
# The real Stage A prompt embeds its own placeholders directly (unlike the
# old ad-hoc version, which took all context as a free-form user message):
# {ticker}/{company_name}/{sector}/{timeline}/{timeline_instruction} and
# {disagreement_score}/{disagreement_classification} map onto values this
# runner already computes below. {memory_brief}, {reflexion_brief}, and
# {agent_accuracy_briefs} are real calibration-system inputs (prior scored
# predictions, accuracy history) that this simulation harness has no source
# for yet -- filled empty, which is also the CORRECT real value for an
# early-life system with no accumulated history yet, not a stand-in for
# missing wiring. {cold_start_cap_active} mirrors that: false until a real
# scored-prediction counter exists to flip it.


def build_pass1_summaries(compressed_pass1: dict) -> str:
    """Same per-agent reliability summary build_user_message's own PASS 1
    RELIABILITY SUMMARY section renders, exposed separately because the real
    Stage A prompt wants it as its own {pass1_summaries} placeholder rather
    than folded into the user message. Made public (86bbt1kct) -- the Shadow
    CIO is a second real consumer (docs/agents/shadow_cio.md: reuses
    format_pass1_summaries_for_cio 'verbatim' from the primary CIO), the
    same trigger this project already uses elsewhere to promote a function
    from private/inline to shared rather than duplicate it a second time."""
    lines = []
    for agent_id, out in compressed_pass1.items():
        if out is None:
            lines.append(f"  {agent_id}: NOT AVAILABLE")
        elif not out.get("pass2_view"):
            # quality={enum} (86bbummwp Tier 3) shown here too, not gated on
            # pass2_view -- it's a mechanical fact about input DATA, computed
            # before the LLM call ever happens, unaffected by whether this
            # agent's own narrative output was coherent enough to build one
            # (same reasoning as compression.py's own extract_data_quality_levels()).
            lines.append(
                f"  {agent_id}: INSUFFICIENT DATA (confidence={out.get('analysis_confidence', 'insufficient')}, "
                f"quality={out.get('data_quality_assessment', 'low')})"
            )
        else:
            # confidence={enum}, quality={enum}: this line briefly carried
            # "reliability={score}/100, quality={enum}" pre-D6 (86bbummwp),
            # then dropped quality entirely when reliability_score was
            # removed (audit E58) since nothing mechanical replaced it yet.
            # Tier 3 restores quality, now mechanically sourced from this
            # same agent's own stale_data/anomalies/data_coverage rather
            # than an LLM-derived or avg() value -- narrative context only,
            # not a verification layer, same as confidence alongside it.
            # Deliberately NOT collapsed into one combined verdict here or
            # anywhere else Pass 1 quality reaches Pass 2/CIO -- see
            # agents/utils.py::build_pass1_reliability_warnings()'s own
            # docstring for why both signals are shown side by side instead.
            # finding={assessment_summary}: added 86bbt1k1p -- the CIO cites Pass 1 in its decision
            # factors and falsifier ("FUND: revenue grew 15.7% YoY", not just "FUND"), which confidence
            # alone can never supply. Single pipe-delimited line, matching this file's existing
            # convention elsewhere (e.g. the Risk Advisor block below).
            lines.append(
                f"  {agent_id}: confidence={out['analysis_confidence']}, "
                f"quality={out.get('data_quality_assessment', 'low')} | "
                f"finding: {out.get('assessment_summary', '')}"
            )
    return "\n".join(lines) if lines else "(no Pass 1 output available)"


def _advocate_entries(output: dict, label: str) -> list[tuple[str, list[str]]]:
    """One advocate's (Bull or Bear) Stage A content as (field, lines) entries, the first line being the
    value and any further lines its continuation. `label` must be "BULL" or "BEAR"; the two share almost
    everything but differ in a few field names/shapes (Bull's catalysts/valuation_argument vs Bear's
    downside_triggers/tail_risk_assessment; Bull's alignment_with_researcher is a string, Bear's
    agrees_with_researcher is a bool), matching the branching the reference pseudocode
    (format_advocate_for_cio) already uses for the same reason -- one shared function, not two
    near-duplicates. Rendered either one advocate at a time (build_advocate_summary) or both side by side
    (build_advocates_block).
    """
    default_recommendation = "bullish" if label == "BULL" else "bearish"
    entries: list[tuple[str, list[str]]] = [
        ("recommendation", [f"{output.get('recommendation', default_recommendation)} | confidence: {output.get('confidence', 0)}/100"]),
        ("thesis_summary", [str(output.get("thesis_summary", ""))]),
        ("strongest_argument", [str(output.get("strongest_argument", ""))]),
        ("weakest_point", [str(output.get("weakest_point", ""))]),
    ]
    sd = output.get("structured_data", {})
    # The advocate's case for THIS horizon, its variant view of what the market gets wrong, and its own invalidators were all
    # produced for the CIO but never shown to it (the first port followed the reference pseudocode, which dropped them).
    timeline_arg = str(sd.get("timeline_focused_argument") or "").strip()
    if timeline_arg:
        entries.append(("timeline_focused_argument", [timeline_arg]))
    if sd.get("core_arguments"):
        # Each argument with its strength and the evidence behind it (the design's
        # format_advocate_for_cio passes the whole object; the first port kept only the claim, so
        # the CIO weighed claims stripped of their figures).
        args = [
            f"  - [{arg.get('strength', 'N/A')}] {arg.get('argument', '')} | evidence: {arg.get('evidence', '')}"
            for arg in sd["core_arguments"] if isinstance(arg, dict)
        ]
        entries.append(("core_arguments", [""] + args))
    va = sd.get("valuation_argument")
    if isinstance(va, dict) and (va.get("claim") or va.get("evidence")):
        entries.append(("valuation_argument", [f"{va.get('claim', '')} | evidence: {va.get('evidence', '')}"]))
    misreads = [
        f"  - {m.get('misread', '')} (why it persists: {m.get('why_this_persists', '')}) | evidence: {m.get('evidence', '')}"
        for m in (sd.get("market_misreads") or []) if isinstance(m, dict)
    ]
    if misreads:
        entries.append(("market_misreads (where this side says the market is wrong)", [""] + misreads))
    risks = [
        f"  - {r.get('risk', '')} (severity {r.get('severity', 'N/A')}, likelihood {r.get('likelihood', 'N/A')}) | evidence: {r.get('evidence', '')}"
        for r in (output.get("thesis_risks") or []) if isinstance(r, dict)
    ]
    if risks:
        entries.append(("thesis_risks (what would invalidate this side's case)", [""] + risks))

    if label == "BULL":
        if sd.get("catalysts"):
            entries.append(("catalysts", [json.dumps([c["catalyst"] for c in sd["catalysts"]])]))
        # 86bbt1k1p: Stage A's "Archetype check" step names this field directly, comparing
        # it against Bear's equivalent.
        taa = sd.get("thesis_archetype_alignment")
        if isinstance(taa, dict):
            entries.append((
                "thesis_archetype_alignment",
                [
                    f"bull_archetype={taa.get('bull_archetype', 'N/A')} "
                    f"vs researcher_archetype={taa.get('researcher_archetype', 'N/A')} | "
                    f"alignment={taa.get('alignment_with_researcher', 'N/A')} | "
                    f"reason: {taa.get('disagreement_reason', '')}"
                ],
            ))
    else:
        if sd.get("downside_triggers"):
            entries.append(("downside_triggers", [json.dumps([t["trigger"] for t in sd["downside_triggers"]])]))
        # 86bbt1k1p: Stage A's "Tail-risk cross-check" step explicitly requires comparing
        # this against Risk Advisor's downside_scenarios -- the validator already
        # numerically checks the CIO's answer against this same data (see run()'s
        # tail_inputs_present/bear_tail_risk_level), but the model itself was never
        # shown it to reason from.
        tra = sd.get("tail_risk_assessment")
        if isinstance(tra, dict):
            entries.append((
                "tail_risk_assessment",
                [
                    f"level={tra.get('tail_risk_level', 'N/A')} | "
                    f"scenario: {tra.get('scenario', '')} | trigger: {tra.get('triggering_event', '')} | "
                    f"evidence: {tra.get('evidence', '')} | supporting_pass1_agents: "
                    f"{json.dumps(tra.get('supporting_pass1_agents', []))}"
                ],
            ))
        # 86bbt1k1p: Stage A's "Archetype check" step names this field directly. Bear's
        # schema uses agrees_with_researcher as a BOOL (Bull's equivalent is a string) --
        # normalized to the same agrees/disagrees vocabulary so the CIO doesn't have to
        # reconcile two different representations itself.
        taa = sd.get("thesis_archetype_alignment")
        if isinstance(taa, dict):
            agrees = taa.get("agrees_with_researcher")
            alignment_str = "agrees" if agrees is True else "disagrees" if agrees is False else "N/A"
            entries.append((
                "thesis_archetype_alignment",
                [
                    f"bear_archetype={taa.get('bear_archetype', 'N/A')} "
                    f"vs researcher_archetype={taa.get('researcher_archetype', 'N/A')} | "
                    f"alignment={alignment_str} | "
                    f"reason: {taa.get('disagreement_note', '')}"
                ],
            ))
    # The rest of what the design specifies for the advocate block and the first port dropped: the
    # asymmetry read, the advocate's own data caveats and its written case. The narrative is supporting
    # detail (the structured fields above stay authoritative), the way the Tax narrative is shown.
    asymmetry = str(sd.get("asymmetry_assessment") or "").strip()
    if asymmetry:
        entries.append(("asymmetry_assessment", [asymmetry]))
    caveats = [str(c) for c in (output.get("caveats") or []) if str(c).strip()]
    if caveats:
        entries.append(("caveats", ["; ".join(caveats)]))
    narrative = str(output.get("narrative") or "").strip()
    if narrative:
        entries.append(("narrative (supporting detail; the structured fields above are authoritative)", [narrative]))
    return entries


def build_advocate_summary(output: dict | None, label: str) -> str:
    """Renders one advocate's (Bull or Bear) Stage A block, one after the other. build_advocates_block
    (both side by side) is what the CIO and Shadow CIO now use; this stays for a single advocate."""
    if not output:
        return f"\n{label} CASE ADVOCATE: NOT AVAILABLE"
    lines = [f"\n{label} CASE ADVOCATE:"]
    for key, value_lines in _advocate_entries(output, label):
        head = f"  {key}:" + (f" {value_lines[0]}" if value_lines[0] else "")
        lines.append(head)
        lines.extend(f"  {v}" for v in value_lines[1:])
    return "\n".join(lines)


def build_advocates_block(bull: dict | None, bear: dict | None) -> str:
    """Bull and Bear shown field by field, both sides under each field, the side listed first alternating
    from field to field, so neither case gets the last word or the first impression throughout. Measured when
    this was built: with the cases one after the other, swapping their order moved the CIO's call up to 4 notches;
    shown this way it moved most cases not at all. If either advocate is missing there is no pair to
    interleave, so it falls back to the one-after-the-other rendering."""
    if not bull or not bear:
        return build_advocate_summary(bull, "BULL") + "\n" + build_advocate_summary(bear, "BEAR")
    sides = {"BULL": dict(_advocate_entries(bull, "BULL")), "BEAR": dict(_advocate_entries(bear, "BEAR"))}
    keys = list(sides["BULL"]) + [k for k in sides["BEAR"] if k not in sides["BULL"]]
    lines = ["\nBULL AND BEAR CASES (each field is shown for both advocates; the one listed first alternates):"]
    for n, key in enumerate(keys):
        lines.append(f"{key}:")
        order = ("BULL", "BEAR") if n % 2 == 0 else ("BEAR", "BULL")
        for who in order:
            value_lines = sides[who].get(key)
            if value_lines is None:
                continue
            lines.append(f"  {who}: {value_lines[0]}".rstrip())
            lines.extend(f"    {v.strip()}" for v in value_lines[1:])
    return "\n".join(lines)


def build_risk_advisor_stage_a_summary(risk_output: dict | None) -> str:
    """Risk Advisor's Stage A (general risk profile) block. Extracted
    (86bbt1kct) for the same reason as build_advocate_summary above --
    the Shadow CIO reuses this exact content (docs/agents/shadow_cio.md:
    format_risk_advisor_stage_a_for_cio 'verbatim')."""
    if not risk_output:
        return "\nRISK ADVISOR: NOT AVAILABLE"

    rp = risk_output.get("risk_profile", {})
    lines = [
        "\nRISK ADVISOR:",
        f"  groundedness_score: {risk_output.get('groundedness_score', 0)}/100 (data quality, NOT directional)",
        f"  risk_reward_ratio: {rp.get('risk_reward_ratio', 'N/A')} (use as directional proxy)",
        f"  volatility_assessment: {rp.get('volatility_assessment', 'N/A')}",
        f"  beta: {rp.get('beta', 'N/A')} | max_drawdown_1yr: {rp.get('max_drawdown_1yr', 'N/A')}%",
        f"  thesis_summary: {risk_output.get('thesis_summary', '')}",
        # 86bbt1k1p: real, validator-enforced, cited signal from Risk's existing Stage A
        # call -- previously dropped entirely.
        f"  strongest_signal: {risk_output.get('strongest_signal', '')}",
    ]
    if rp.get("downside_scenarios"):
        # The CIO's stage A prompt treats these "as a probability-weighted distribution", so each
        # scenario carries its probability and timeline (they used to be dropped).
        scenarios_summary = "; ".join(
            f"{s.get('scenario', '')} ({s.get('estimated_impact_pct', 'N/A')}%, "
            f"{s.get('probability', 'N/A')} probability, {s.get('timeline', 'N/A')})"
            for s in rp["downside_scenarios"]
            if isinstance(s, dict)
        )
        lines.append(f"  downside_scenarios: {scenarios_summary}")
    # Only when there is something to say (a clean run carries neither).
    flags = [str(f) for f in (rp.get("data_sanity_flags") or []) if str(f).strip()]
    if flags:
        lines.append(f"  data_sanity_flags: {'; '.join(flags)}")
    caveats = [str(c) for c in (risk_output.get("caveats") or []) if str(c).strip()]
    if caveats:
        lines.append(f"  caveats: {'; '.join(caveats)}")
    # The written risk read. Beyond the structured lines above it carries each scenario's trigger, the beta
    # and drawdown interpretation, liquidity and dividend sustainability, which were computed on every run
    # but never reached the CIO. Supporting detail, like the advocate and Tax narratives.
    narrative = str(risk_output.get("narrative") or "").strip()
    if narrative:
        lines.append(f"  narrative (supporting detail; the structured fields above are authoritative): {narrative}")
    return "\n".join(lines)


def tail_risk_inputs(pass2_outputs: dict) -> tuple[str | None, float | None]:
    """(Bear's tail_risk_level, Risk Advisor's worst estimated_impact_pct), both from the payload rather
    than from the CIO's own output. Shared by the user message (which shows the computed classification)
    and the validator (which checks it)."""
    bear_tra = (pass2_outputs.get("bear") or {}).get("structured_data", {}).get("tail_risk_assessment")
    level = bear_tra.get("tail_risk_level") if isinstance(bear_tra, dict) else None
    scenarios = (pass2_outputs.get("risk") or {}).get("risk_profile", {}).get("downside_scenarios") or []
    impacts = [
        s.get("estimated_impact_pct") for s in scenarios
        if isinstance(s, dict) and isinstance(s.get("estimated_impact_pct"), (int, float))
    ]
    return level, (min(impacts) if impacts else None)


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
    pass2_outputs: dict,
    disagreement_score: int,
    disagreement_category: str,
) -> str:
    ctx = bundle.context

    lines = [
        f"SYNTHESIS REQUEST: {bundle.stock.ticker} ({bundle.company_info.get('name')}) | {bundle.company_info.get('sector')}",
        f"Timeline: {ctx.timeline} | Account: {ctx.account_type}",
        f"Disagreement Score: {disagreement_score}/100 — Category: {disagreement_category}",
        "",
    ]
    # The Pass 1 summaries are in the system prompt ({pass1_summaries}); this message used to repeat them (the same five
    # lines twice in every stage A prompt, about 450 tokens), as the Bull and Risk messages did before they were fixed.
    lines.append("=== PASS 2 OUTPUTS ===")
    lines.append(build_advocates_block(pass2_outputs.get("bull"), pass2_outputs.get("bear")))

    # Tax Strategist has NO input into Stage A, deliberately and by documented
    # design -- confirmed directly: the Stage A runtime prompt states the read
    # must be "the SAME... regardless of which account eventually holds the
    # position," and the Stage A validator's own key_decision_factors[].source
    # check already restricts citations to bull|bear|risk|pass1, with a comment
    # noting "tax" was removed from that set for exactly this reason. This
    # function used to leak account-specific Tax Strategist facts here anyway
    # (account_fit_score, tax_efficiency_for_account, cross_account_recommendation,
    # etc. -- all inherently account-specific, since Tax Strategist only ever runs
    # scoped to one account_type) -- removed. Do not add a Tax Strategist block
    # back here, even a "NOT AVAILABLE" placeholder -- Stage A shouldn't
    # reference Tax Strategist's existence at all, not just its content.
    lines.append(build_risk_advisor_stage_a_summary(pass2_outputs.get("risk")))
    lines.append("\n" + tail_risk_line(*tail_risk_inputs(pass2_outputs)))

    if disagreement_category in ("split_decision", "high_conflict"):
        lines.append(f"\n⚠️ DISAGREEMENT CAP: category={disagreement_category}")
        if disagreement_category == "split_decision":
            lines.append("  → Your confidence must be ≤65")
        else:
            lines.append("  → stock_outlook must be neutral/somewhat range (not full bullish or bearish)")

    lines.append("\nProduce the Stage A JSON now: the reasoning fields first, then the verdict.")
    return "\n".join(lines)


def _build_general_outlook_summary(stage_a_result: dict | None) -> str:
    """Stage A's own decided conclusion, pasted as plain text for Stage B --
    this IS the 'continuation' content for the CIO's own chain, just
    delivered as text rather than via context-array continuation (CIO
    deliberately doesn't use that mechanism, see the module-level comment
    above). Field list and shape follow the reference pseudocode in
    "Agent Prompts/Current Prompts/Chief Investment Officer (CIO) Agent
    Prompt.md" (format_general_outlook_for_stage_b), with one live-verified
    correction: the real Stage A output key is `disagreement_category`, not
    the doc's stale `disagreement_classification`."""
    if not stage_a_result:
        return "GENERAL READ: NOT AVAILABLE"

    r = stage_a_result
    bull_a = r.get("bull_case_assessment") or {}
    bear_a = r.get("bear_case_assessment") or {}
    kdf = r.get("key_decision_factors") or []
    kdf_lines = "\n".join(
        f"  - {kf.get('factor', '')} ({kf.get('direction', 'N/A')}, source={kf.get('source', 'N/A')}): "
        f"{kf.get('evidence', '')}"
        for kf in kdf if isinstance(kf, dict)
    )

    return (
        f"stock_outlook: {r.get('stock_outlook', 'N/A')}\n"
        f"confidence: {r.get('confidence', 'N/A')}/100\n"
        f"expected_return_tier: {r.get('expected_return_tier', 'N/A')}\n"
        f"thesis_summary: {r.get('thesis_summary', '')}\n"
        f"bull_case_assessment: verdict={bull_a.get('verdict', 'N/A')} | {bull_a.get('engagement', '')}\n"
        f"bear_case_assessment: verdict={bear_a.get('verdict', 'N/A')} | {bear_a.get('engagement', '')}\n"
        f"risk_reward_consumed_as: {r.get('risk_reward_consumed_as', 'N/A')}\n"
        f"tail_risk_cross_check: {r.get('tail_risk_cross_check', 'N/A')} | note: {r.get('tail_risk_note', '')}\n"
        f"key_decision_factors:\n{kdf_lines}\n"
        f"disagreement_score: {r.get('disagreement_score', 'N/A')} | "
        f"category: {r.get('disagreement_category', 'N/A')}"
    )


def _render_tax_optimization_actions(actions: list | None) -> str:
    """Used by _build_tax_strategist_summary() (Stage B) only -- Stage A has no
    Tax Strategist input at all (see build_user_message()'s own comment), so this
    is a single-consumer helper despite once being shared with a since-removed
    Stage A rendering (86bc8eg3j originally added it there too; that addition
    itself violated Stage A's account-neutrality and was reverted). Kept as its
    own function anyway -- correct, tested, and reasonably self-contained even
    with one caller. Includes estimated_benefit_pct, not just action/justification
    -- matches this file's own existing convention of surfacing a quantified
    number alongside qualitative text (see _build_tax_strategist_summary()'s
    cross_text/drag_delta_pct)."""
    return "; ".join(
        f"{a.get('action', '')} (benefit={a.get('estimated_benefit_pct', 'N/A')}%/yr, "
        f"justification={a.get('justification', 'N/A')})"
        for a in (actions or []) if isinstance(a, dict)
    )


def _render_loss_harvesting(lh: object) -> str:
    """The Tax Strategist's loss_harvesting_opportunity is an object (or null/"not_applicable"
    -- always null in a TFSA/RRSP, where losses are not claimable). One compact line."""
    if not lh or lh == "not_applicable":
        return ""
    if not isinstance(lh, dict):
        return str(lh)
    return (
        f"available={lh.get('available', 'N/A')}, "
        f"estimated_tax_savings={lh.get('estimated_tax_savings_pct', 'N/A')}%, "
        f"superficial_loss_window_safe={lh.get('superficial_loss_window_safe', 'N/A')}, "
        f"remediation={lh.get('remediation_suggestion') or 'none'}; {lh.get('reasoning', '')}"
    ).strip()


def _build_tax_strategist_summary(tax_result: dict | None) -> str:
    """Tax Strategist's structured fields, for the CIO's own Stage B call --
    the ticket's actual core ask (86bbt1k1p names this "the highest-value
    item"). Field list from the reference pseudocode
    (format_tax_strategist_for_cio), with one correction: the doc's
    `output.data_quality_assessment` reference is omitted -- D6 (86bbummwp)
    already removed that field from the real schema.

    86bc8eg3j: adds capital_gains_treatment_summary and tax_optimization_actions
    -- both real, prompt-governed fields (Step 5/Rule 8, Rules 14/15 respectively
    in prompts/tax_strategist/v1.txt) that this function previously stopped short
    of, so they never reached the CIO's own Stage B system prompt despite being
    computed on every real Tax Strategist run."""
    if not tax_result:
        return "TAX STRATEGIST: NOT AVAILABLE"

    tp = tax_result.get("tax_profile") or {}
    key_risks = tp.get("key_tax_risks") or []
    risks_text = "; ".join(
        f"{kr.get('risk', '')} (severity={kr.get('severity', 'N/A')})"
        for kr in key_risks if isinstance(kr, dict)
    )

    actions_text = _render_tax_optimization_actions(tp.get("tax_optimization_actions"))

    # Fields Tax computes that used to stop here (Tax Strategist Wave 2): the contribution-room
    # impacts, loss-harvesting, the data-gap caveats and the narrative. Each line appears only
    # when there is something to say, so a TFSA/RRSP run with no room action or loss opportunity
    # adds nothing. The structured fields above stay authoritative (v1_stage_b.txt rule 2).
    detail_lines = []
    caveats = [str(c) for c in (tax_result.get("caveats") or []) if str(c).strip()]
    if caveats:
        detail_lines.append(f"caveats: {'; '.join(caveats)}")
    for field in ("tfsa_contribution_room_impact", "rrsp_contribution_room_impact"):
        value = tp.get(field)
        if value and str(value).strip().lower() != "null":
            detail_lines.append(f"{field}: {value}")
    loss_text = _render_loss_harvesting(tp.get("loss_harvesting_opportunity"))
    if loss_text:
        detail_lines.append(f"loss_harvesting_opportunity: {loss_text}")
    detail_block = "".join(f"{line}\n" for line in detail_lines)
    narrative = str(tax_result.get("narrative") or "").strip()
    narrative_line = (
        f"\ntax_narrative (supporting context; the fields above are authoritative): {narrative}"
        if narrative
        else ""
    )

    # What the Tax Strategist wrote for exactly this purpose: which tax mechanics dominate for this account and
    # timeline, and the listing/domicile read that drives withholding. They were computed on every run and never shown.
    lens_lines = "".join(
        f"{field}: {tp[field]}\n"
        for field in ("account_axis_summary", "context_aware_strongest_argument", "listing_summary")
        if str(tp.get(field) or "").strip()
    )

    return (
        f"groundedness_score: {tax_result.get('groundedness_score', 'N/A')}/100\n"
        f"tax_efficiency_for_account: {tp.get('tax_efficiency_for_account', 'N/A')}\n"
        f"account_fit_score: {tp.get('account_fit_score', 'N/A')}\n"
        f"{lens_lines}"
        f"thesis_summary: {tax_result.get('thesis_summary', '')}\n"
        f"strongest_signal: {tax_result.get('strongest_signal', '')}\n"
        f"dividend_classification: {tp.get('dividend_classification', 'N/A')} | "
        f"yield: {tp.get('dividend_yield_pct', 'N/A')}% | "
        f"after_tax_yield: {tp.get('effective_after_tax_yield_pct', 'N/A')}% | "
        f"annual_drag: {tp.get('annual_tax_drag_pct', 'N/A')}%/yr\n"
        f"wht_interpretation: {tp.get('wht_interpretation', '')}\n"
        f"capital_gains_treatment_summary: {tp.get('capital_gains_treatment_summary', 'N/A')}\n"
        f"key_tax_risks: {risks_text}\n"
        f"{detail_block}"
        f"tax_optimization_actions: {actions_text}"
        f"{narrative_line}"
    )


class CIORunner(BaseRunner):
    GROUND_MODE = "log"  # figures not in this agent's input are recorded in validator_errors, never a retry (agents/grounding.py)

    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        pass2_outputs: dict,
    ) -> tuple[dict, list[str]]:
        # Stage-specific, not just "cio" -- matches the same fix orchestrator.py's
        # AgentOutput rows already got this session (agent_name "cio" for both
        # stages meant nothing told them apart). call_site (86bbwachy Phase 2)
        # is derived from current_agent, so this is the one place that needs to
        # know which stage it is, not every call site downstream.
        self.current_agent = "cio_stage_a"
        bull_conf = pass2_outputs.get("bull", {}).get("confidence", 0) if pass2_outputs.get("bull") else 0
        bear_conf = pass2_outputs.get("bear", {}).get("confidence", 0) if pass2_outputs.get("bear") else 0
        disagreement_score, disagreement_category = compute_disagreement_score(bull_conf, bear_conf)

        # Real inputs for the tail-risk cross-check (86bbuhk82 item 2) -- both
        # tail_inputs_present and the numeric mapping's two values come from the actual
        # payload, not the CIO's own output, so they're computed here and handed to the
        # validator rather than inferred from what the CIO claims.
        bear_tail_risk_level, risk_worst_impact = tail_risk_inputs(pass2_outputs)
        tail_inputs_present = bear_tail_risk_level is not None and risk_worst_impact is not None

        # Risk Advisor Stage A's own risk_reward_ratio, for the risk_reward_consumed_as
        # cross-check (prompt rule 11) -- the validator param existed but was never wired here,
        # so that rule never fired against real output. Same shape as tail_inputs_present above:
        # the comparison value lives in the payload, not the CIO's own output.
        risk_reward_source = (pass2_outputs.get("risk") or {}).get("risk_profile", {}).get("risk_reward_ratio")

        user_msg = build_user_message(
            bundle, compressed_pass1, pass2_outputs, disagreement_score, disagreement_category
        )
        ctx = bundle.context
        system_prompt = fill(
            load_template("cio", stage="a"),
            {
                "ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "disagreement_score": str(disagreement_score),
                "disagreement_classification": disagreement_category,
                "cold_start_cap_active": "false",
                "reflexion_brief": "",
                "memory_brief": "",
                "agent_accuracy_briefs": "",
                "pass1_summaries": build_pass1_summaries(compressed_pass1),
            },
        )
        result, errors = await self.call_with_validation(
            system_prompt,
            user_msg,
            lambda out: validate_cio_stage_a(
                out,
                disagreement_category,
                tail_inputs_present=tail_inputs_present,
                risk_reward_source=risk_reward_source,
                bear_tail_risk_level=bear_tail_risk_level,
                risk_worst_estimated_impact_pct=risk_worst_impact,
            ),
            max_tokens=5000,
            temperature=0.0,
        )
        if result:
            result["disagreement_score"] = disagreement_score
            result["disagreement_category"] = disagreement_category
        return result, errors

    async def run_stage_b(
        self,
        bundle: DataBundle,
        stage_a_result: dict,
        tax_result: dict | None,
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        """Account-specific tax read + the synthesis_narrative the user
        actually reads. An ordinary second call (call_with_validation,
        unmodified) -- no continuation context, per the module-level
        comment on why the CIO's own Stage A->B doesn't use that
        mechanism. Caller must have already confirmed stage_a_result came
        from a Stage A call with errors == [] -- unlike Risk Advisor's
        continuation (which structurally can't proceed without a real
        context), this plain second call has no equivalent guard, and a
        Stage A result that failed validation could be missing fields
        entirely.
        """
        self.current_agent = "cio_stage_b"  # see run()'s own comment on why
        ctx = bundle.context
        acct = account_type or ctx.account_type
        system_prompt = fill(
            load_template("cio", stage="b"),
            {
                "account_type": acct,
                "timeline": ctx.timeline,
                "general_outlook_summary": _build_general_outlook_summary(stage_a_result),
                "tax_strategist_summary": _build_tax_strategist_summary(tax_result),
            },
        )
        tax_efficiency_source = (
            (tax_result.get("tax_profile") or {}).get("tax_efficiency_for_account")
            if tax_result else None
        )
        return await self.call_with_validation(
            system_prompt,
            "Produce the Stage B JSON output now.",
            lambda out: validate_cio_stage_b(
                out,
                stage_a_expected_return_tier=stage_a_result.get("expected_return_tier"),
                tax_efficiency_source=tax_efficiency_source,
                stage_a_outlook=stage_a_result.get("stock_outlook"),
            ),
            max_tokens=2000,
            temperature=0.0,
        )
