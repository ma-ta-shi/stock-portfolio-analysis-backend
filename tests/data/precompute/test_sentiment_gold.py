"""The hand-labelled headlines (tests/data/sentiment_gold.json) and the pattern rules (ledger BB-111)."""
import json
from pathlib import Path

from data.precompute.headline_rules import rule_label

GOLD = json.loads((Path(__file__).resolve().parents[1] / "sentiment_gold.json").read_text(encoding="utf-8"))["items"]


def test_the_fixture_is_well_formed():
    assert len(GOLD) == 157 and {i["set"] for i in GOLD} == {"dev", "fresh", "third"}
    assert {i["label"] for i in GOLD} <= {"positive", "negative", "neutral", "unrelated"}
    assert len({i["id"] for i in GOLD}) == len(GOLD)


def test_every_pattern_rule_that_fires_on_a_labelled_headline_agrees_with_the_hand_label():
    fired = [(i, rule_label(i["headline"], i["text"], i["company"], i["ticker"])) for i in GOLD]
    fired = [(i, r) for i, r in fired if r]
    assert len(fired) >= 15  # 17 when the rules were written
    wrong = [(i["headline"], i["label"], r) for i, r in fired if (r, i["label"]) not in {(r, r), ("neutral", "unrelated")}]
    assert wrong == []
