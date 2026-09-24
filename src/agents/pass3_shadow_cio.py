"""Shadow CIO runner (86bbt1kct, ported to production 86bbuhjup).

Production port of `simulation/runners/pass3_shadow_cio.py`. Same shape as
pass3_cio.py's port -- only `fixture["context"]` reads become
`bundle.stock`/`bundle.company_info`/`bundle.context`; the three imported
summary-building functions are the primary CIO's own byte-for-byte-ported
ones (agents/pass3_cio.py), reused verbatim exactly as the harness already did.

Parallel, non-blocking calibration twin of the primary CIO's Stage A --
CLAUDE.md's pipeline description: "Shadow CIO (non-blocking, step 12): same
inputs, bear-biased prompt -> stored to shadow_predictions for calibration".
Not `pass4_`: architecturally this is a second consumer of Stage A's exact
inputs, not a new sequential stage.

No Stage B / account-specific counterpart at all (docs/agents/shadow_cio.md
Design Decision 2, v2 2026-09-15): the shadow never receives Tax Strategist
or Risk Advisor Stage B, and produces no tax/sizing/stop-loss content.

Two things this runner deliberately does NOT do, decided from real live
data during 86bbt1kct planning, not from the ticket's original (partially
stale) numbers -- see that ticket and its plan for the full evidence:

1. Inject pre-computed override-arithmetic numbers (`bull x0.5`/`bear x2`)
   into the prompt. Measured live against the CURRENT prompt (rewritten
   2026-09-15 with an explicit "weighting lens, not a formula" instruction)
   on 2 real scenarios during planning: injecting the numbers produced
   IDENTICAL stock_outlook/confidence to the unmodified prompt in both
   runs. The ticket's "only injecting numbers works" finding predates that
   rewrite. Confirmed at larger scale by this file's own first full
   7-scenario sweep (unmodified prompt, no injection): only 1 of 7 (14%)
   was high_divergence, vs. the pre-rewrite ~90% baseline the change-list
   doc itself predicted would "fall sharply" once the qualitative fix
   landed -- it did, without needing the injection at all.
2. Enforce a Pass 1 citation requirement. This project has already paid
   real cost for citation floors added blind (Tax's now-removed
   >=2-distinct-ID rule, E88/E90; the CIO's own validator-prompt-mismatched
   what_would_change_my_mind, E93). `check_numeric_groundedness`
   (agents/validators/shadow_cio.py) is the deliberate alternative:
   real groundedness evidence without requiring citations at all.
"""
from agents.base import BaseRunner
from agents.pass3_cio import (
    build_advocate_summary,
    build_pass1_summaries,
    build_risk_advisor_stage_a_summary,
)
from agents.prompts import fill, load_template
from agents.utils import compute_disagreement_score
from agents.validators.shadow_cio import validate_shadow_cio
from data.schemas.data_bundle import DataBundle


class ShadowCIORunner(BaseRunner):
    async def run(
        self,
        bundle: DataBundle,
        compressed_pass1: dict,
        pass2_outputs: dict,
    ) -> tuple[dict, list[str]]:
        """`pass2_outputs` here is {"bull", "bear", "risk"} -- no "tax" key,
        ever (see module docstring). Same None-guard idiom the primary CIO's
        own run() already uses: a bare pass2_outputs["bull"]["confidence"]
        would raise if an upstream agent failed and this runner is called
        without a Gate-2 check in front of it -- which, like CIORunner.run()
        itself, it deliberately doesn't do; that's the caller's job.
        """
        self.current_agent = "shadow_cio"
        bull_out = pass2_outputs.get("bull")
        bear_out = pass2_outputs.get("bear")
        risk_out = pass2_outputs.get("risk")

        bull_conf = bull_out.get("confidence", 0) if bull_out else 0
        bear_conf = bear_out.get("confidence", 0) if bear_out else 0
        disagreement_score, disagreement_category = compute_disagreement_score(bull_conf, bear_conf)

        risk_groundedness = (risk_out or {}).get("groundedness_score", 0)
        low_groundedness = bull_conf < 40 and bear_conf < 40 and risk_groundedness < 40

        ctx = bundle.context
        system_prompt = fill(
            load_template("shadow_cio"),
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

        # Bull/Bear/Risk Stage A block -- the exact same rendering the primary
        # CIO's own build_user_message uses (docs/agents/shadow_cio.md: reuses
        # format_advocate_for_cio/format_risk_advisor_stage_a_for_cio
        # "verbatim" from the primary CIO). No disagreement-cap reminder
        # appended here (unlike the primary CIO's user message) -- the
        # shadow's own template already states the split_decision/
        # high_conflict caps directly as prompt text (v1.txt Stage 3),
        # confirmed live: both caps were respected with no reminder in
        # this session's own probe.
        user_message = "\n".join([
            "=== PASS 2 OUTPUTS (Bull, Bear, Risk Advisor general risk profile) ===",
            build_advocate_summary(bull_out, "BULL"),
            build_advocate_summary(bear_out, "BEAR"),
            build_risk_advisor_stage_a_summary(risk_out),
        ])

        return await self.call_with_validation(
            system_prompt,
            user_message,
            lambda out: validate_shadow_cio(out, disagreement_category, low_groundedness),
            max_tokens=1200,
            temperature=0.3,
        )
