"""Tests for agents/pass3_shadow_cio.py (86bbuhjup). Close to a clean port,
same shape as test_pass3_cio.py -- build_advocate_summary/build_pass1_summaries/
build_risk_advisor_stage_a_summary are imported from agents.pass3_cio, not
reimplemented, so this port's own real content is small: the runner class
itself and the low_groundedness computation inside run()."""
import agents.pass3_cio as cio_module
import agents.pass3_shadow_cio as shadow_module
from agents.pass3_shadow_cio import ShadowCIORunner


def test_reuses_primary_cios_summary_functions_verbatim_not_a_duplicate():
    """docs/agents/shadow_cio.md's own design intent: the shadow reuses the
    primary CIO's format_advocate_for_cio/format_pass1_summaries_for_cio/
    format_risk_advisor_stage_a_for_cio "verbatim" -- confirmed here as an
    identity check (same function object), not just equal behavior, so a
    future edit to the primary CIO's rendering can't silently diverge from
    what the shadow sees."""
    assert shadow_module.build_advocate_summary is cio_module.build_advocate_summary
    assert shadow_module.build_pass1_summaries is cio_module.build_pass1_summaries
    assert (
        shadow_module.build_risk_advisor_stage_a_summary
        is cio_module.build_risk_advisor_stage_a_summary
    )


def test_runner_is_constructible():
    runner = ShadowCIORunner()
    assert runner.current_agent is None  # set only once run() actually executes
