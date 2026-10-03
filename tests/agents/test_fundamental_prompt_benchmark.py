"""The Fundamental prompt describes the industry P/E benchmark the payload carries, and nothing it no longer has."""
from agents.prompts import load_template

PROMPT = load_template("fundamental_analyst")


def test_it_teaches_the_position_labels_the_payload_prints():
    assert "within_range" in PROMPT and "above_range" in PROMPT and "below_range" in PROMPT
    assert "Industry P/E" in PROMPT


def test_it_no_longer_refers_to_a_peer_table_or_per_metric_sector_medians():
    """The per-peer table and the margin, growth, ROE and leverage medians were removed 2026-10-03: the prompt must not
    ask for comparisons the payload cannot supply."""
    assert "PEER" not in PROMPT.replace("peer_comparison_summary", "")
    assert "sector_median" not in PROMPT
    assert "vs peers" not in PROMPT
    assert "P/B-vs-sector" not in PROMPT


def test_the_peer_comparison_field_is_kept_but_asks_about_the_industry_benchmark():
    assert '"peer_comparison_summary"' in PROMPT
    assert "industry benchmark" in PROMPT.split('"peer_comparison_summary"')[1][:200]
