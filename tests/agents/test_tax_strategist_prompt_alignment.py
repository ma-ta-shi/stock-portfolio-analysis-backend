"""The Tax Strategist prompt must state every rule its validator enforces.

Measured 2026-10-01 (Tax Strategist Wave 2): of 23 rejected attempts, most traced to a
validator rule the prompt never stated or worded differently -- `key_factors` 2-4 and
`key_tax_risks` 1-4 existed only in the validator, the narrative floor was never given (the
prompt said "300-500 words"), `MARG` was advertised but not accepted, and "Cite REF/TFSA"
taught the model to write a bare token the evidence rule rejects. Each test here pins one
such pair so the prompt and validator cannot drift apart again.
"""
import re

import pytest

from agents.prompts import load_template
from agents.validators.pass2 import TAX_METRIC_TOKENS, validate_tax_strategist

TEMPLATE = load_template("tax_strategist")


def _section(start: str, end: str) -> str:
    i = TEMPLATE.index(start)
    return TEMPLATE[i : TEMPLATE.index(end, i)]


@pytest.mark.parametrize(
    "stated",
    [
        "key_factors: 2-4",
        "key_tax_risks: 1-4",
        "tax_optimization_actions: 0-3",
        "1,000-3,200 characters",
    ],
)
def test_validator_bounds_are_stated_in_the_prompt(stated):
    assert stated in TEMPLATE


def test_the_narrative_range_the_prompt_asks_for_is_inside_the_validators_bounds():
    """The prompt asks for 1,800-2,500 characters (the model lands near the floor of whatever range it is given: aiming at 1,500 produced 1,050-1,220-character narratives); the validator hard-rejects below 1,000 and
    above 3,200. The ask must sit wholly inside the hard limits."""
    assert "1,800-2,500 characters" in TEMPLATE
    narrative_tokens = " ".join(f"{tok}: claim." for tok in sorted(TAX_METRIC_TOKENS)[:3]) + " REF/TFSA: x."

    def narrative_errors(n_chars: int) -> list[str]:
        filler = "x" * max(0, n_chars - len(narrative_tokens) - 1)
        out = {"narrative": f"{narrative_tokens} {filler}"[:n_chars]}
        return [e for e in validate_tax_strategist(out)[1] if e.startswith("narrative: too")]

    assert narrative_errors(1_000) == [] and narrative_errors(999) != []
    assert narrative_errors(3_200) == [] and narrative_errors(3_201) != []


def test_every_validator_token_is_listed_for_the_narrative():
    step_10 = _section("10-NARRATIVE.", "--- STOCK-SPECIFIC CONTEXT")
    for token in TAX_METRIC_TOKENS:
        assert token in step_10, f"{token} is accepted by the validator but absent from the prompt's narrative token list"


def test_every_validator_token_is_defined_in_rule_1():
    rule_1 = _section("1. CITATIONS.", "2. CONFIDENCE-AWARE WEIGHTING")
    for token in TAX_METRIC_TOKENS:
        assert re.search(rf"^\s*- {token}\b", rule_1, re.M), f"{token} has no definition line in rule 1"


def test_the_evidence_format_names_the_claim_and_never_tells_the_model_to_cite_a_bare_ref():
    """"Cite REF/TFSA." led the model to write evidence = "REF/TFSA", which the key_factors rule
    rejects (7 of 23 rejected attempts)."""
    assert "EVIDENCE FORMAT" in TEMPLATE
    assert not re.search(r"\bCite REF/", TEMPLATE)
    assert "`REF/TFSA: <claim>`" in TEMPLATE


def test_the_caveats_schema_no_longer_invites_the_forbidden_estate_and_memory_notes():
    """The schema line used to list "US-estate-tax soft caveat, memory-drop acknowledgments",
    contradicting rule 11; 40 of 41 outputs carried an estate caveat with no portfolio_context."""
    assert "US-estate-tax soft caveat" not in TEMPLATE
    assert "memory-drop" not in TEMPLATE


def test_the_two_score_enums_are_kept_apart():
    assert re.search(r'"account_fit_score": "excellent\|good\|fair\|poor"', TEMPLATE)
    assert re.search(r'"tax_efficiency_for_account": "favorable\|neutral\|unfavorable"', TEMPLATE)
    assert "never use these words for" in TEMPLATE


def test_the_alternatives_line_is_explained_where_the_fit_score_is_defined():
    rule_4 = _section("4. ACCOUNT-FIT SCORE", "5. TAX EFFICIENCY")
    assert "ALTERNATIVES" in rule_4
