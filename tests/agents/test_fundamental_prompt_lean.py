"""The Fundamental prompt after the 2026-10-07 deep dive: limits stated, nothing false about the trimmer, the model not asked for what code decides."""

from pathlib import Path

from agents.prompts import fill, load_template

PROMPT = (Path(__file__).resolve().parents[2] / "prompts" / "fundamental_analyst" / "v1.txt").read_text(encoding="utf-8")


def test_every_limit_the_validator_enforces_is_in_the_prompt():
    """52 of 63 first attempts failed raw; most wrote more list items than the cap because only the validator knew it."""
    for limit in ("risks 1-3", "assessment_summary 80 words or fewer", "narrative 900-1,200 characters", "over 1500"):
        assert limit in PROMPT


def test_key_factors_are_not_asked_for():
    """Nothing read them and they cost about a fifth of the output; the narrative carries the evidence (ledger BB-107)."""
    assert "key_factors" not in PROMPT and "at least one is required" in PROMPT


def test_no_false_promise_about_the_trimmer_and_no_narrative_wall():
    assert "auto-trim" not in PROMPT and "HARD FLOOR" not in PROMPT


def test_the_model_is_not_asked_for_what_code_decides_or_nobody_reads():
    assert "guidance_vs_consensus" not in PROMPT and "peer_comparison_summary" not in PROMPT


def test_the_three_judged_fields_are_defined():
    for text in ("healthy = profitable", "strong = well covered"):
        assert text in PROMPT


def test_a_metric_the_payload_does_not_show_is_not_a_risk():
    """TD.TO was written up for D/E, FCF to net income and 'operating margin not applicable', none of which describe a bank."""
    assert "a metric it does not show is not a risk, a gap or a caveat" in PROMPT
    assert "If it has a LENS line, judge on that lens" in PROMPT


def test_it_renders_with_only_the_placeholders_the_runner_fills():
    rendered = fill(load_template("fundamental_analyst"), {
        "canonical_ticker": "TD.TO", "company_name": "Toronto-Dominion Bank", "sector": "Finance", "timeline": "medium_term",
        "data_coverage_line": "standard.", "data_warnings": ""})
    assert "{data_coverage_line}" not in rendered and "{timeline}" not in rendered and "{sector}" not in rendered
