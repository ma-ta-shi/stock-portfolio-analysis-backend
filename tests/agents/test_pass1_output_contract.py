"""Every field a Pass 1 agent's model writes has a stated purpose and consumer (ledger BB-107): the contract in agents/pass1_contract.py
must match each prompt's output schema in both directions, so an output cannot be added or dropped without saying what it is for."""
import re
from pathlib import Path

import pytest

from agents.pass1_contract import PASS1_OUTPUT_CONTRACT

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"

# Outputs the contract itself marks as removal candidates (no consumer): a new one cannot be added to this list quietly.
UNRESOLVED = {
    ("technical_analyst", "interpretive_fields.pattern_signal"),
    ("technical_analyst", "interpretive_fields.pattern_confirmed"),
}


def schema_paths(text: str) -> list[str]:
    """Field paths in a prompt's output schema: objects are walked, the items of a list of objects are not."""
    lines = text[text.rindex("Respond with ONLY valid JSON"):].split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip() == "{")
    stack: list[tuple[int, str, str]] = []
    paths: list[str] = []
    for line in lines[start + 1:]:
        m = re.match(r'^(\s*)"([A-Za-z_0-9]+)":\s*(.*)$', line)
        if not m:
            continue
        indent, name, rest = len(m.group(1)), m.group(2), m.group(3).rstrip().rstrip(",")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if any(kind == "list" for _, _, kind in stack):
            continue
        path = ".".join([n for _, n, _ in stack] + [name])
        if rest.startswith("{") and not rest.endswith("}"):
            stack.append((indent, name, "obj"))
        elif rest.startswith("[") and not rest.endswith("]"):
            stack.append((indent, name, "list"))
            paths.append(path)
        else:
            paths.append(path)
    return paths


@pytest.mark.parametrize("slug", sorted(PASS1_OUTPUT_CONTRACT))
def test_the_contract_matches_the_prompts_output_schema_exactly(slug):
    schema = set(schema_paths((PROMPTS / slug / "v1.txt").read_text(encoding="utf-8")))
    contract = set(PASS1_OUTPUT_CONTRACT[slug])

    assert schema - contract == set(), f"{slug}: fields in the prompt schema with no purpose and consumer in pass1_contract.py"
    assert contract - schema == set(), f"{slug}: fields in pass1_contract.py that the prompt no longer asks for"


def test_every_field_states_a_purpose_and_a_consumer_and_only_the_known_ones_have_none():
    for slug, fields in PASS1_OUTPUT_CONTRACT.items():
        for path, (purpose, consumers) in fields.items():
            assert purpose.strip() and consumers.strip(), (slug, path)
            if consumers.startswith("none:"):
                assert (slug, path) in UNRESOLVED, f"{slug} {path} has no consumer: remove it or give it one"


def test_key_factors_is_not_a_pass1_output_any_more():
    for slug, fields in PASS1_OUTPUT_CONTRACT.items():
        assert "key_factors" not in fields, slug
