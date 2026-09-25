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
`build_risk_advisor_stage_a_summary`, `_build_general_outlook_summary`,
`_build_risk_advisor_stage_b_summary`, `_build_tax_strategist_summary`) is a
byte-for-byte port, since none of them ever touched `fixture` at all.

Synthesis pass. Cloud-only. Produces final stock_outlook and prediction.
Input semantics: Bull/Bear confidence = directional conviction.
Risk/Tax groundedness_score = data quality (NOT direction).
Directional proxies: Risk → risk_reward_ratio; Tax → tax_efficiency_for_account.
Disagreement caps: split_decision → confidence ≤65; high_conflict → outlook must be neutral/somewhat range.
"""
import json

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import compute_disagreement_score
from agents.validators.cio import validate_cio_stage_a, validate_cio_stage_b
from data.schemas.data_bundle import DataBundle

# The old embedded system_prompt (independently authored, diverged from the
# real documented prompt) is deleted, not commented out -- the real prompt
# now loads from backend/prompts/cio/v1_stage_a.txt via load_template()
# (ClickUp 86bbdutn6). Stage B's template also exists (backend/prompts/cio/
# v1_stage_b.txt) but this runner still makes only one call. This is NOT a
# missing-primitive gap the way it is for Risk Advisor: per
# docs/technical/two-turn-execution-mechanism.md's "Why CIO doesn't use this
# mechanism", the CIO's Stage A->B was deliberately designed to NOT use
# context-array continuation at all -- Stage B needs fresh Tax Strategist +
# Risk Advisor Stage B input regardless of mechanism, and otherwise only
# needs Stage A's own compact JSON output pasted as text, which an ordinary
# second call already provides. What's actually missing here is the Stage B
# *invocation itself* (deciding what to paste, building the second call) --
# scoped to 86bbt1k1p's deferred Item C, not to 86bc2d414 (which built the
# continuation primitive Risk Advisor needs, and which this runner correctly
# has no use for).
#
# Stage B (86bbt1k1p Item C): run_stage_b() below builds that ordinary
# second call -- system_prompt filled from v1_stage_b.txt's named
# placeholders (general_outlook_summary/risk_advisor_stage_b_summary/
# tax_strategist_summary), via the EXISTING, unmodified call_with_validation
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
            lines.append(
                f"  {agent_id}: INSUFFICIENT DATA (confidence={out.get('analysis_confidence', 'insufficient')})"
            )
        else:
            # confidence={enum}: was "reliability={score}/100, quality={enum}" pre-D6
            # (86bbummwp); narrative context only, not a verification layer (audit E58).
            # finding={assessment_summary}: added 86bbt1k1p -- Stage A's own "Pass 1
            # citation inventory" step requires a concrete finding per agent ("FUND:
            # revenue grew 15.7% YoY", not just "FUND"), which confidence alone can
            # never supply. Single pipe-delimited line, matching this file's existing
            # convention elsewhere (e.g. the Risk Advisor block below).
            lines.append(
                f"  {agent_id}: confidence={out['analysis_confidence']} | "
                f"finding: {out.get('assessment_summary', '')}"
            )
    return "\n".join(lines) if lines else "(no Pass 1 output available)"


def build_advocate_summary(output: dict | None, label: str) -> str:
    """Renders one advocate's (Bull or Bear) Stage A block. Extracted
    (86bbt1kct) from build_user_message's own inline Bull/Bear rendering --
    the Shadow CIO is a second real consumer of this exact content
    (docs/agents/shadow_cio.md: reuses format_advocate_for_cio 'verbatim'
    from the primary CIO). `label` must be "BULL" or "BEAR"; the two share
    almost everything but differ in a few field names/shapes (Bull's
    catalysts/valuation_argument vs Bear's downside_triggers/
    tail_risk_assessment; Bull's alignment_with_researcher is a string,
    Bear's agrees_with_researcher is a bool), matching the branching the
    reference pseudocode (format_advocate_for_cio) already uses for the
    same reason -- one shared function, not two near-duplicates.

    Byte-for-byte equivalence with the pre-extraction inline version
    matters here: build_user_message joins its `lines` list with "\\n",
    and every element below is built the same way (a list of lines
    "\\n".join()-ed internally) so the leading "\\n" on the header line
    produces the exact same blank-line separator between sections as
    before.
    """
    if not output:
        return f"\n{label} CASE ADVOCATE: NOT AVAILABLE"

    default_recommendation = "bullish" if label == "BULL" else "bearish"
    lines = [
        f"\n{label} CASE ADVOCATE:",
        f"  recommendation: {output.get('recommendation', default_recommendation)} | confidence: {output.get('confidence', 0)}/100",
        f"  thesis_summary: {output.get('thesis_summary', '')}",
        f"  strongest_argument: {output.get('strongest_argument', '')}",
        f"  weakest_point: {output.get('weakest_point', '')}",
    ]
    sd = output.get("structured_data", {})
    if sd.get("core_arguments"):
        lines.append(f"  core_arguments: {json.dumps([a['argument'] for a in sd['core_arguments']])}")

    if label == "BULL":
        if sd.get("catalysts"):
            lines.append(f"  catalysts: {json.dumps([c['catalyst'] for c in sd['catalysts']])}")
        if sd.get("valuation_argument"):
            lines.append(f"  valuation_argument: {sd['valuation_argument'].get('claim', '')}")
        # 86bbt1k1p: Stage A's "Archetype check" step names this field directly, comparing
        # it against Bear's equivalent.
        taa = sd.get("thesis_archetype_alignment")
        if isinstance(taa, dict):
            lines.append(
                f"  thesis_archetype_alignment: bull_archetype={taa.get('bull_archetype', 'N/A')} "
                f"vs researcher_archetype={taa.get('researcher_archetype', 'N/A')} | "
                f"alignment={taa.get('alignment_with_researcher', 'N/A')} | "
                f"reason: {taa.get('disagreement_reason', '')}"
            )
    else:
        if sd.get("downside_triggers"):
            lines.append(f"  downside_triggers: {json.dumps([t['trigger'] for t in sd['downside_triggers']])}")
        # 86bbt1k1p: Stage A's "Tail-risk cross-check" step explicitly requires comparing
        # this against Risk Advisor's downside_scenarios -- the validator already
        # numerically checks the CIO's answer against this same data (see run()'s
        # tail_inputs_present/bear_tail_risk_level), but the model itself was never
        # shown it to reason from.
        tra = sd.get("tail_risk_assessment")
        if isinstance(tra, dict):
            lines.append(
                f"  tail_risk_assessment: level={tra.get('tail_risk_level', 'N/A')} | "
                f"scenario: {tra.get('scenario', '')} | trigger: {tra.get('triggering_event', '')} | "
                f"evidence: {tra.get('evidence', '')} | supporting_pass1_agents: "
                f"{json.dumps(tra.get('supporting_pass1_agents', []))}"
            )
        # 86bbt1k1p: Stage A's "Archetype check" step names this field directly. Bear's
        # schema uses agrees_with_researcher as a BOOL (Bull's equivalent is a string) --
        # normalized to the same agrees/disagrees vocabulary so the CIO doesn't have to
        # reconcile two different representations itself.
        taa = sd.get("thesis_archetype_alignment")
        if isinstance(taa, dict):
            agrees = taa.get("agrees_with_researcher")
            alignment_str = "agrees" if agrees is True else "disagrees" if agrees is False else "N/A"
            lines.append(
                f"  thesis_archetype_alignment: bear_archetype={taa.get('bear_archetype', 'N/A')} "
                f"vs researcher_archetype={taa.get('researcher_archetype', 'N/A')} | "
                f"alignment={alignment_str} | "
                f"reason: {taa.get('disagreement_note', '')}"
            )
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
        f"  beta: {rp.get('beta', 'N/A')} | max_drawdown_1yr: {rp.get('max_drawdown_1yr', 'N/A')}%",
        f"  position_size_recommendation: {rp.get('position_size_recommendation', 'N/A')}",
        f"  concentration_risk: {rp.get('concentration_risk', 'N/A')} | liquidity_risk: {rp.get('liquidity_risk', 'N/A')}",
        f"  thesis_summary: {risk_output.get('thesis_summary', '')}",
        # 86bbt1k1p: real, validator-enforced, cited signal from Risk's existing Stage A
        # call -- previously dropped entirely.
        f"  strongest_signal: {risk_output.get('strongest_signal', '')}",
    ]
    if rp.get("downside_scenarios"):
        scenarios_summary = "; ".join(
            f"{s['scenario']} ({s['estimated_impact_pct']}%)" for s in rp["downside_scenarios"]
        )
        lines.append(f"  downside_scenarios: {scenarios_summary}")
    return "\n".join(lines)


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
        "=== PASS 1 RELIABILITY SUMMARY ===",
        build_pass1_summaries(compressed_pass1),
        "",
    ]
    lines.append("=== PASS 2 OUTPUTS ===")
    lines.append(build_advocate_summary(pass2_outputs.get("bull"), "BULL"))
    lines.append(build_advocate_summary(pass2_outputs.get("bear"), "BEAR"))

    # Tax Strategist stays inline here, not extracted -- the Shadow CIO never
    # receives Tax Strategist output at all (docs/agents/shadow_cio.md v2:
    # no account-specific counterpart), so there is no second consumer to
    # share this block with.
    tax = pass2_outputs.get("tax")
    if tax:
        tp = tax.get("tax_profile", {})
        lines.append("\nTAX STRATEGIST:")
        lines.append(f"  groundedness_score: {tax.get('groundedness_score', 0)}/100 (data quality, NOT directional)")
        lines.append(f"  account_fit_score: {tp.get('account_fit_score', 'N/A')}")
        lines.append(f"  tax_efficiency_for_account: {tp.get('tax_efficiency_for_account', 'N/A')} (use as directional proxy)")
        lines.append(f"  effective_after_tax_yield_pct: {tp.get('effective_after_tax_yield_pct', 'N/A')}%")
        lines.append(f"  thesis_summary: {tax.get('thesis_summary', '')}")
        if tp.get("cross_account_recommendation"):
            lines.append(f"  cross_account_recommendation: {tp['cross_account_recommendation']}")
    else:
        lines.append("\nTAX STRATEGIST: NOT AVAILABLE")

    lines.append(build_risk_advisor_stage_a_summary(pass2_outputs.get("risk")))

    if disagreement_category in ("split_decision", "high_conflict"):
        lines.append(f"\n⚠️ DISAGREEMENT CAP: category={disagreement_category}")
        if disagreement_category == "split_decision":
            lines.append("  → Your confidence must be ≤65")
        else:
            lines.append("  → stock_outlook must be neutral/somewhat range (not full bullish or bearish)")

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


def _build_risk_advisor_stage_b_summary(risk_stage_b_result: dict | None) -> str:
    """Risk Advisor's account-specific overlay, for the CIO's own Stage B
    call. Field list from the reference pseudocode
    (format_risk_advisor_stage_b_for_cio)."""
    if not risk_stage_b_result:
        return "RISK ADVISOR (account-specific overlay): NOT AVAILABLE"

    r = risk_stage_b_result
    return (
        f"position_size_recommendation: {r.get('position_size_recommendation', 'N/A')}\n"
        f"stop_loss_suggestion: {r.get('stop_loss_suggestion')}\n"
        f"correlation_to_existing_portfolio: {r.get('correlation_to_existing_portfolio', 'N/A')} | "
        f"concentration_risk: {r.get('concentration_risk', 'N/A')} | "
        f"liquidity_risk: {r.get('liquidity_risk', 'N/A')}\n"
        f"sizing_rationale: {r.get('sizing_rationale', '')}"
    )


def _build_tax_strategist_summary(tax_result: dict | None) -> str:
    """Tax Strategist's structured fields, for the CIO's own Stage B call --
    the ticket's actual core ask (86bbt1k1p names this "the highest-value
    item"). Field list from the reference pseudocode
    (format_tax_strategist_for_cio), with one correction: the doc's
    `output.data_quality_assessment` reference is omitted -- D6 (86bbummwp)
    already removed that field from the real schema."""
    if not tax_result:
        return "TAX STRATEGIST: NOT AVAILABLE"

    tp = tax_result.get("tax_profile") or {}
    cross = tp.get("cross_account_recommendation")
    if isinstance(cross, dict):
        cross_text = (
            f"For new capital, {cross.get('better_account', 'N/A')}: {cross.get('reasoning', '')} "
            f"(drag_delta: {cross.get('drag_delta_pct', 'N/A')}%/yr)"
        )
    else:
        cross_text = "current account is appropriate"

    key_risks = tp.get("key_tax_risks") or []
    risks_text = "; ".join(
        f"{kr.get('risk', '')} (severity={kr.get('severity', 'N/A')})"
        for kr in key_risks if isinstance(kr, dict)
    )

    return (
        f"groundedness_score: {tax_result.get('groundedness_score', 'N/A')}/100\n"
        f"tax_efficiency_for_account: {tp.get('tax_efficiency_for_account', 'N/A')}\n"
        f"account_fit_score: {tp.get('account_fit_score', 'N/A')}\n"
        f"thesis_summary: {tax_result.get('thesis_summary', '')}\n"
        f"strongest_signal: {tax_result.get('strongest_signal', '')}\n"
        f"dividend_classification: {tp.get('dividend_classification', 'N/A')} | "
        f"yield: {tp.get('dividend_yield_pct', 'N/A')}% | "
        f"after_tax_yield: {tp.get('effective_after_tax_yield_pct', 'N/A')}% | "
        f"annual_drag: {tp.get('annual_tax_drag_pct', 'N/A')}%/yr\n"
        f"wht_interpretation: {tp.get('wht_interpretation', '')}\n"
        f"cross_account_recommendation: {cross_text}\n"
        f"key_tax_risks: {risks_text}"
    )


class CIORunner(BaseRunner):
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
        bear_tra = (pass2_outputs.get("bear") or {}).get("structured_data", {}).get("tail_risk_assessment")
        bear_tail_risk_level = bear_tra.get("tail_risk_level") if isinstance(bear_tra, dict) else None
        risk_scenarios = (pass2_outputs.get("risk") or {}).get("risk_profile", {}).get("downside_scenarios") or []
        risk_impacts = [
            s.get("estimated_impact_pct") for s in risk_scenarios
            if isinstance(s, dict) and isinstance(s.get("estimated_impact_pct"), (int, float))
        ]
        risk_worst_impact = min(risk_impacts) if risk_impacts else None
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
            temperature=0.3,
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
        risk_stage_b_result: dict | None,
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        """Account-specific overlay + the synthesis_narrative the user
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
                "account_instruction": f"Account: {acct.upper()}.",
                "general_outlook_summary": _build_general_outlook_summary(stage_a_result),
                "risk_advisor_stage_b_summary": _build_risk_advisor_stage_b_summary(risk_stage_b_result),
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
            ),
            max_tokens=2000,
            temperature=0.3,
        )
