"""Pass 2 `key_factors` was cut on 2026-10-07: no code reads it (the CIO builder, the Shadow CIO and the orchestrator skip it) and every
figure in it (49 of 49 in 16 replayed outputs) was already stated elsewhere in the same output. These pin the cut in both directions."""
import pytest

from agents.prompts import load_template

TEMPLATES = [("bull_advocate", None), ("bear_advocate", None), ("risk_advisor", "a"), ("tax_strategist", None)]


@pytest.mark.parametrize("slug,stage", TEMPLATES)
def test_the_prompt_no_longer_asks_for_key_factors(slug, stage):
    assert "key_factors" not in load_template(slug, stage=stage)


def test_the_run_quality_summary_does_not_expect_key_factors_from_pass2():
    from services.orchestrator import _NO_KEY_FACTORS_EXPECTED

    assert {"bull", "bear", "tax", "risk"} <= _NO_KEY_FACTORS_EXPECTED
