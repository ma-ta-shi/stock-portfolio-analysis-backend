"""The Researcher prompt after the 2026-10-06 deep dive: limits stated, filing-depth rule rendered sensibly, definitions present."""

from pathlib import Path

from agents.prompts import fill, load_template

PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "stock_researcher" / "v1.txt").read_text(encoding="utf-8")


def test_every_limit_the_validator_enforces_is_in_the_prompt():
    """32 of 59 first attempts failed raw (22 wrote 5 key_factors, 20 overshot the narrative, 4 had no caveats)."""
    for limit in ("key_factors 2-4", "risks 1-3", "caveats 0-4", "assessment_summary 80 words or fewer",
                  "narrative 760-960 characters"):
        assert limit in PROMPT
    assert "HARD FLOOR" not in PROMPT  # the repeated 900-character narrative warning is gone


def _rendered(has_filing_digest: str) -> str:
    return fill(load_template("stock_researcher"), {
        "sector": "Finance", "timeline": "medium_term", "data_coverage_line": "standard.", "data_warnings": "",
        "memory_brief": "", "has_filing_digest": has_filing_digest, "peer_1_token": "PEER_1", "peer_2_token": "PEER_2",
    })


def test_the_filing_depth_rule_reads_as_a_condition_not_if_true_is_false():
    """It rendered "If true is false", and 10 of 71 outputs wrote the filing-depth caveat with a digest in the payload."""
    for flag in ("true", "false"):
        rendered = _rendered(flag)
        assert f"the payload has a filing digest: {flag}." in rendered
        assert "If true is false" not in rendered and "If false is false" not in rendered
    assert "never write a filing-depth caveat" in PROMPT


def test_the_thesis_archetypes_are_defined():
    """Two of five values were used and 'dividend_compounder' never, for KO, ENB.TO or TD.TO."""
    for archetype in ("secular_grower", "dividend_compounder", "cyclical_recovery", "quality_compounder",
                      "value_trap_candidate"):
        assert f"- {archetype}:" in PROMPT


def test_management_needs_a_stated_reason_and_selling_alone_is_not_one():
    assert "needs a stated reason" in PROMPT and "Insider selling alone is never that reason" in PROMPT


def test_the_profile_fallback_is_citable_but_never_moat_evidence():
    """With no filing digest the payload shows the provider's company profile; the dead sector-moat line stays gone."""
    assert "PROFILE:Business for the company profile shown when there is no filing digest" in PROMPT
    assert "never cite it as moat evidence" in PROMPT
    assert "EXPECTED MOAT TYPES" not in PROMPT
