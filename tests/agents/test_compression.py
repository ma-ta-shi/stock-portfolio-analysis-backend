"""Tests for agents/compression.py's data_quality_assessment wiring (86bbummwp
Tier 3): compress_pass1_outputs()'s new mechanical_quality param, and the new
extract_data_quality_levels() extractor. No dedicated test file existed for
compression.py before this -- test_pass2_view.py covers build_pass2_view()
and only incidentally exercises compress_pass1_outputs()."""
from agents.compression import compress_pass1_outputs, extract_data_quality_levels


def _fund_output(**overrides) -> dict:
    base = {
        "assessment_summary": "Solid fundamentals.",
        "analysis_confidence": "high",
        "caveats": [],
        "narrative": "A narrative.",
        "structured_data": {},
        "interpretive_fields": {
            "valuation_vs_sector": "fair",
            "health_rating": "healthy",
            "guidance_vs_consensus": "inline",
            "dividend_sustainability": "strong",
            "peer_comparison_summary": "Leads sector.",
        },
    }
    base.update(overrides)
    return base


# ---------- compress_pass1_outputs() mechanical_quality param ----------


def test_mechanical_quality_folded_into_compressed_output():
    compressed = compress_pass1_outputs(
        {"FUND": _fund_output()}, mechanical_quality={"FUND": "medium"}
    )
    assert compressed["FUND"]["data_quality_assessment"] == "medium"


def test_mechanical_quality_defaults_to_low_when_agent_missing_from_dict():
    """A real Pass 1 output exists, but mechanical_quality has no entry for
    it -- e.g. an older runner that hasn't implemented the check. "low" is
    the honest, conservative default, not a silent gap."""
    compressed = compress_pass1_outputs({"FUND": _fund_output()}, mechanical_quality={})
    assert compressed["FUND"]["data_quality_assessment"] == "low"


def test_mechanical_quality_omitted_entirely_still_defaults_to_low():
    compressed = compress_pass1_outputs({"FUND": _fund_output()})
    assert compressed["FUND"]["data_quality_assessment"] == "low"


def test_mechanical_quality_never_read_from_raw_output():
    """The mechanical signal must come from mechanical_quality, never from
    anything the LLM output dict itself might contain -- data_quality_assessment
    is never LLM-produced (D6's own clean-ownership split)."""
    out = _fund_output(data_quality_assessment="high")  # LLM should never emit this key
    compressed = compress_pass1_outputs({"FUND": out}, mechanical_quality={"FUND": "low"})
    assert compressed["FUND"]["data_quality_assessment"] == "low"


def test_none_output_has_no_data_quality_key_at_all():
    """A totally-failed agent compresses to None outright -- nothing to fold
    mechanical_quality into, confirmed directly against compress_pass1_outputs's
    own None-handling branch."""
    compressed = compress_pass1_outputs({"FUND": None}, mechanical_quality={"FUND": "low"})
    assert compressed["FUND"] is None


# ---------- extract_data_quality_levels() ----------


def test_extract_reads_data_quality_assessment_directly():
    compressed = {"FUND": {"pass2_view": {"x": 1}, "data_quality_assessment": "medium"}}
    assert extract_data_quality_levels(compressed) == {"FUND": "medium"}


def test_extract_defaults_none_output_to_low():
    compressed = {"FUND": None}
    assert extract_data_quality_levels(compressed) == {"FUND": "low"}


def test_extract_defaults_missing_key_to_low():
    compressed = {"FUND": {"pass2_view": {"x": 1}}}
    assert extract_data_quality_levels(compressed) == {"FUND": "low"}


def test_extract_does_not_gate_on_pass2_view_unlike_confidence_extractor():
    """A real, deliberate divergence from extract_confidence_levels(): a
    missing/falsy pass2_view must NOT force "low" here -- the mechanical
    quality signal is about input data, computed before the LLM call, and is
    unaffected by whether the agent's own narrative output was coherent
    enough to build a pass2_view."""
    compressed = {"FUND": {"pass2_view": {}, "data_quality_assessment": "high"}}
    assert extract_data_quality_levels(compressed) == {"FUND": "high"}


def test_extract_handles_multiple_agents():
    compressed = {
        "FUND": {"pass2_view": {"x": 1}, "data_quality_assessment": "high"},
        "MACRO": {"pass2_view": {"x": 1}, "data_quality_assessment": "low"},
        "SENT": None,
    }
    assert extract_data_quality_levels(compressed) == {
        "FUND": "high", "MACRO": "low", "SENT": "low",
    }
