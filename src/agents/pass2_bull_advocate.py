"""Pass 2 — Bull Case Advocate runner (v1.6).

Production port of `simulation/runners/pass2_bull_advocate.py` (86bbuhjup).
Close to a clean port -- `build_pass2_user_message` (agents/compression.py)
already does the real DataBundle-vs-fixture translation work (step 2 of this
ticket's port); this runner just supplies a real `DataBundle` instead of a
fixture dict and reads `bundle.context` for timeline/account_type instead of
`fixture["context"]`.

recommendation: hard-coded to "bullish".
confidence: integer 0-100.
key_factors was cut 2026-10-07 (no consumer; core_arguments carries the evidence).
weakest_point_type: negative_catalyst|adverse_fundamental|data_gap|no_clear_invalidator.
thesis_archetype: secular_grower|dividend_compounder|cyclical_recovery|quality_compounder|value_trap_candidate.
"""
from agents.base import BaseRunner
from agents.compression import build_pass2_user_message, extract_confidence_levels, extract_data_quality_levels
from agents.prompts import fill, load_template
from agents.reference_points import asymmetry_reference
from agents.utils import build_pass1_reliability_warnings, researcher_thesis_archetype
from agents.validators.pass2 import validate_bull_advocate
from data.schemas.data_bundle import DataBundle


def build_user_message(bundle: DataBundle, compressed_pass1: dict) -> str:
    """The Pass 1 summaries only, and account-neutral: Bull is one result per ticker and timeline, shared
    by every account's CIO run. The reliability warnings and the researcher archetype used to be appended
    here as well as sent in the system prompt (so every call paid for them twice), and the account used
    to be named (a header and a FOCUS line) though the prompt never uses it."""
    return build_pass2_user_message(bundle, compressed_pass1, account_neutral=True)


class BullAdvocateRunner(BaseRunner):
    GROUND_MODE = "log"  # figures not in this agent's input are recorded in validator_errors, never a retry (agents/grounding.py)

    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
    ) -> tuple[dict, list[str]]:
        self.current_agent = "bull"
        ctx = bundle.context
        user_msg = build_user_message(bundle, compressed_pass1)
        confidence_levels = extract_confidence_levels(compressed_pass1)
        # 86bbummwp Tier 3 -- D6 section 3's mechanical per-agent data quality,
        # shown alongside analysis_confidence (never collapsed into it) so this
        # agent can catch a Pass 1 agent claiming high confidence while its own
        # data quality is mechanically low.
        quality_levels = extract_data_quality_levels(compressed_pass1)
        # Computed once and reused for both the prompt fill and the validator --
        # not called a second time inline in the fill dict below, redundant with
        # build_user_message()'s own call.
        real_researcher_archetype = researcher_thesis_archetype(compressed_pass1)
        system_prompt = fill(
            load_template("bull_advocate"),
            {
                "ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "memory_brief": "",
                "accuracy_brief": "",
                "winning_patterns_brief": "",
                "pass1_reliability_warnings": build_pass1_reliability_warnings(
                    confidence_levels, agent_quality=quality_levels
                ) or "(none)",
                "researcher_thesis_archetype": real_researcher_archetype,
            },
        )
        result, errors = await self.call_with_validation(
            system_prompt,
            user_msg,
            lambda out: validate_bull_advocate(out, real_researcher_archetype=real_researcher_archetype),
            max_tokens=5000,
            temperature=0.3,
        )
        # Hard-code recommendation (defense in depth)
        if result:
            result["recommendation"] = "bullish"
            # Written by code from the data (agents/reference_points.py): the model invented these numbers when it wrote the field.
            if isinstance(result.get("structured_data"), dict):
                result["structured_data"]["asymmetry_assessment"] = asymmetry_reference(bundle)
        return result, errors
