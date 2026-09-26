"""Pass 2 — Bull Case Advocate runner (v1.6).

Production port of `simulation/runners/pass2_bull_advocate.py` (86bbuhjup).
Close to a clean port -- `build_pass2_user_message` (agents/compression.py)
already does the real DataBundle-vs-fixture translation work (step 2 of this
ticket's port); this runner just supplies a real `DataBundle` instead of a
fixture dict and reads `bundle.context` for timeline/account_type instead of
`fixture["context"]`.

recommendation: hard-coded to "bullish".
confidence: integer 0-100.
key_factors: all sentiment must be "positive".
weakest_point_type: negative_catalyst|adverse_fundamental|data_gap|no_clear_invalidator.
thesis_archetype: secular_grower|dividend_compounder|cyclical_recovery|quality_compounder|value_trap_candidate.
"""
from agents.base import BaseRunner
from agents.compression import build_pass2_user_message, extract_confidence_levels, extract_data_quality_levels
from agents.prompts import fill, load_template
from agents.utils import build_pass1_reliability_warnings, researcher_thesis_archetype
from agents.validators.pass2 import validate_bull_advocate
from data.schemas.data_bundle import DataBundle


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
    account_type: str | None = None,
) -> str:
    base = build_pass2_user_message(bundle, compressed_pass1, account_type)
    confidence_levels = extract_confidence_levels(compressed_pass1)
    quality_levels = extract_data_quality_levels(compressed_pass1)
    warnings = build_pass1_reliability_warnings(confidence_levels, agent_quality=quality_levels)
    if warnings:
        base += f"\n\nRELIABILITY WARNINGS: {warnings}"
    # Same lookup the system prompt's {researcher_thesis_archetype} fill uses
    # below -- one shared implementation (agents/utils.py), not a second,
    # independently-maintained copy.
    archetype = researcher_thesis_archetype(compressed_pass1)
    base += f"\n\nRESEARCHER ARCHETYPE FOR THIS STOCK: {archetype}"
    base += f"\nFOCUS: {bundle.context.timeline} | ACCOUNT: {account_type or bundle.context.account_type}"
    return base


class BullAdvocateRunner(BaseRunner):
    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        self.current_agent = "bull"
        ctx = bundle.context
        user_msg = build_user_message(bundle, compressed_pass1, account_type)
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
        return result, errors
