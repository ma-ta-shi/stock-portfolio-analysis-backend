"""Pass 2 — Bear Case Advocate runner (v2).

Production port of `simulation/runners/pass2_bear_advocate.py` (86bbuhjup).
Same shape as pass2_bull_advocate.py -- see that file's module docstring for
why this is close to a clean port.

recommendation: hard-coded to "bearish".
confidence: integer 0-100 (directional conviction DOWN).
key_factors: all sentiment must be "negative".
thesis_archetype vocabulary: same 5 values as Bull.
At confidence <35: high-bar bear framing + numeric token in evidence required.
"""
from agents.base import BaseRunner
from agents.compression import build_pass2_user_message, extract_confidence_levels
from agents.prompts import fill, load_template
from agents.utils import build_pass1_reliability_warnings, researcher_thesis_archetype
from agents.validators.pass2 import validate_bear_advocate
from data.schemas.data_bundle import DataBundle


def build_user_message(
    bundle: DataBundle,
    compressed_pass1: dict,
    account_type: str | None = None,
) -> str:
    base = build_pass2_user_message(bundle, compressed_pass1, account_type)
    confidence_levels = extract_confidence_levels(compressed_pass1)
    warnings = build_pass1_reliability_warnings(confidence_levels)
    if warnings:
        base += f"\n\nRELIABILITY WARNINGS: {warnings}"
    ctx = bundle.context
    base += f"\nFOCUS: {ctx.timeline} | ACCOUNT: {account_type or ctx.account_type}"
    return base


class BearAdvocateRunner(BaseRunner):
    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        account_type: str | None = None,
    ) -> tuple[dict, list[str]]:
        self.current_agent = "bear"
        ctx = bundle.context
        user_msg = build_user_message(bundle, compressed_pass1, account_type)
        confidence_levels = extract_confidence_levels(compressed_pass1)
        real_researcher_archetype = researcher_thesis_archetype(compressed_pass1)
        system_prompt = fill(
            load_template("bear_advocate"),
            {
                "ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "memory_brief": "",
                "calibration_brief": "",
                "pass1_reliability_warnings": build_pass1_reliability_warnings(confidence_levels) or "(none)",
                "researcher_thesis_archetype": real_researcher_archetype,
            },
        )
        result, errors = await self.call_with_validation(
            system_prompt,
            user_msg,
            lambda out: validate_bear_advocate(out, real_researcher_archetype=real_researcher_archetype),
            max_tokens=5000,
            temperature=0.3,
        )
        if result:
            result["recommendation"] = "bearish"
        return result, errors
