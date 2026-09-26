"""Tests for agents/validators/common.py's validate_confidence_requires_caveat_when_flagged
(86bbummwp follow-on, 2026-09-26) -- the standalone shared rule composed into
all 7 agents that produce a confidence/groundedness self-assessment
(pass1_technical_analyst.py, pass1_fundamental_analyst.py,
pass1_macro_economist.py, pass1_sentiment_analyst.py,
pass1_stock_researcher.py, pass2_risk_advisor.py, pass2_tax_strategist.py --
each file's own _validate_with_caveats() tests cover the composition; this
file tests the shared rule in isolation, in the same spirit as
_sweep_declared_enums's own indirect coverage via test_pass2_validators.py."""
from agents.validators.common import validate_confidence_requires_caveat_when_flagged


def test_not_high_never_flags_regardless_of_gaps():
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []},
        is_high=False,
        material_absent=["filings"],
        anomalies=["something odd"],
        stale_data=["price"],
    )
    assert passed, errors


def test_high_with_no_gaps_at_all_never_flags():
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []}, is_high=True, material_absent=[], anomalies=[], stale_data=[]
    )
    assert passed, errors


def test_high_with_material_absent_and_no_caveat_fails():
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []}, is_high=True, material_absent=["filings"], anomalies=[], stale_data=[]
    )
    assert not passed
    assert any("caveats is empty" in e for e in errors)


def test_high_with_anomaly_and_no_caveat_fails():
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []},
        is_high=True,
        material_absent=[],
        anomalies=["US yield curve is inverted"],
        stale_data=[],
    )
    assert not passed


def test_high_with_stale_data_and_no_caveat_fails():
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []}, is_high=True, material_absent=[], anomalies=[], stale_data=["rate"]
    )
    assert not passed


def test_high_with_gap_but_real_caveat_present_passes():
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": ["Rate data is 55 days old."]},
        is_high=True,
        material_absent=[],
        anomalies=[],
        stale_data=["rate"],
    )
    assert passed, errors


def test_caveats_key_entirely_missing_treated_as_empty():
    """output.get("caveats") on a dict with no such key at all -- must not
    crash, must be treated the same as an explicit empty list."""
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {}, is_high=True, material_absent=["filings"], anomalies=[], stale_data=[]
    )
    assert not passed


def test_anomalies_and_stale_data_default_to_none_safely():
    """Pass 2 call sites (Risk Advisor, Tax Strategist) don't pass anomalies/
    stale_data at all -- must not crash on the defaults."""
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []}, is_high=True, material_absent=["beta"]
    )
    assert not passed
    passed, errors = validate_confidence_requires_caveat_when_flagged(
        {"caveats": []}, is_high=True, material_absent=[]
    )
    assert passed, errors
