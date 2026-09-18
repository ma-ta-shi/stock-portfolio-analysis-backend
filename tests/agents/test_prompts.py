"""Tests the extraction logic against the real `Agent Prompts/Current
Prompts/*.md` design docs, not synthetic fixtures -- the whole point of this
ticket is that a fixture standing in for these files is exactly what let the
old embedded prompts drift from them unnoticed. `load_template()` is tested
against real generated `backend/prompts/` output for the same reason.
"""

import re
from pathlib import Path

import pytest

from agents.prompts import (
    _FENCE,
    bake_output_schema,
    extract_block,
    extract_h1_span,
    extract_stage_schema,
    fill,
    load_template,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_DOCS = _REPO_ROOT / "Agent Prompts" / "Current Prompts"
if not _PROMPT_DOCS.exists():
    pytest.skip(
        f"top-level repo's prompt docs not found at {_PROMPT_DOCS} -- this test suite "
        "assumes both repos are checked out together",
        allow_module_level=True,
    )

_MULTI_STAGE = {"Chief Investment Officer (CIO)", "Risk Advisor"}

_ALL_AGENT_NAMES = [p.stem.removesuffix(" Agent Prompt") for p in _PROMPT_DOCS.glob("*.md")]


def _doc(name: str) -> str:
    return (_PROMPT_DOCS / f"{name} Agent Prompt.md").read_text(encoding="utf-8")


# ---------- extract_block: real fence-language variety ----------


class TestExtractBlockFenceLanguages:
    def test_javascript_fence_both_blocks(self):
        md = _doc("Fundamental Analyst")
        assert extract_block(md, "System Prompt").strip().startswith("You are a financial analyst")
        assert "{" in extract_block(md, "Output Schema")

    def test_mixed_fences_on_the_same_agent(self):
        """Bear's System Prompt is ```plain text, its Output Schema is
        ```json -- the exact case that cost 20 empty Technical runs and a
        fully-empty Bear extraction before a fence-agnostic pattern existed."""
        md = _doc("Bear Case Advocate")
        system = extract_block(md, "System Prompt")
        schema = extract_block(md, "Output Schema")
        assert system.strip().startswith("You are the Bear")
        assert len(schema) > 0

    def test_raises_on_missing_header(self):
        md = _doc("Fundamental Analyst")
        with pytest.raises(ValueError, match="No .* block found"):
            extract_block(md, "Nonexistent Section")


# ---------- CIO / Risk Advisor: the genuinely new, never-before-exercised case ----------


class TestMultiStageExtraction:
    @pytest.mark.parametrize(
        "agent,stage,expected_field",
        [
            ("Chief Investment Officer (CIO)", "A", "stock_outlook"),
            ("Chief Investment Officer (CIO)", "B", "position_sizing_recommendation"),
            ("Risk Advisor", "A", "groundedness_score"),
            ("Risk Advisor", "B", "correlation_to_existing_portfolio"),
        ],
    )
    def test_system_prompt_stage_disambiguation(self, agent, stage, expected_field):
        md = _doc(agent)
        system = extract_block(md, f"System Prompt — Stage {stage}")
        assert system.strip()

    @pytest.mark.parametrize(
        "agent,stage,expected_field",
        [
            ("Chief Investment Officer (CIO)", "A", "stock_outlook"),
            ("Chief Investment Officer (CIO)", "B", "position_sizing_recommendation"),
            ("Risk Advisor", "A", "groundedness_score"),
            ("Risk Advisor", "B", "correlation_to_existing_portfolio"),
        ],
    )
    def test_nested_output_schema_stage_disambiguation(self, agent, stage, expected_field):
        """The one case with no prior reference implementation anywhere --
        neither _cio_rig.py nor _risk_rig.py reference "Stage A"/"Stage B"
        at all. Each stage's schema is a distinct H2 nested under one shared
        `# Output Schema` H1; asserts the correct, DISTINCT field for each
        stage, not just "something non-empty" -- a bug that silently
        returned Stage A's schema for both stages would still pass a
        non-empty check."""
        md = _doc(agent)
        schema = extract_stage_schema(md, "Output Schema", f"Stage {stage}")
        assert expected_field in schema

    def test_stage_a_and_stage_b_schemas_are_actually_different(self):
        md = _doc("Chief Investment Officer (CIO)")
        stage_a = extract_stage_schema(md, "Output Schema", "Stage A")
        stage_b = extract_stage_schema(md, "Output Schema", "Stage B")
        assert stage_a != stage_b
        assert "position_sizing_recommendation" not in stage_a
        assert "stock_outlook" not in stage_b

    def test_stage_b_heading_recurs_elsewhere_in_the_document(self):
        """`## Stage B` is not unique to the Output Schema section -- it
        also appears, unrelated to the output contract, under Validation
        Rules, Retry Prompt Injection, and the token-budget table. Today
        the Output Schema occurrence happens to come first in the document,
        so an unbounded re.search (leftmost match) would still find the
        right content -- bounding to extract_h1_span isn't rescuing today's
        file from a live bug, it's making the extraction correct BY
        CONSTRUCTION rather than by accidental document ordering that a
        future edit could silently invalidate. Demonstrated concretely
        below rather than just asserted."""
        md = _doc("Chief Investment Officer (CIO)")
        stage_b_headings = [m.start() for m in re.finditer(r"^## Stage B", md, re.M)]
        assert len(stage_b_headings) > 1, "expected multiple recurrences of this heading text"

    def test_bounding_is_correct_even_if_an_earlier_decoy_heading_is_inserted(self):
        """The actual robustness property extract_h1_span buys: unlike an
        unbounded search, correctness here doesn't depend on the real
        schema happening to be the first occurrence in the file."""
        real_md = _doc("Chief Investment Officer (CIO)")
        decoy = "## Stage B\n```plain text\nVALIDATION ERRORS:\n{validation_error_list}\n```\n\n"
        md_with_earlier_decoy = decoy + real_md

        unbounded = re.search(rf"^## Stage B.*?{_FENCE}", md_with_earlier_decoy, re.S | re.M)
        assert "VALIDATION ERRORS" in unbounded.group(1), (
            "the decoy proves unbounded search is order-dependent"
        )

        bounded = extract_stage_schema(md_with_earlier_decoy, "Output Schema", "Stage B")
        assert "position_sizing_recommendation" in bounded
        assert "VALIDATION ERRORS" not in bounded


# ---------- extract_h1_span ----------


class TestExtractH1Span:
    def test_bounds_before_the_next_h1(self):
        md = _doc("Chief Investment Officer (CIO)")
        span = extract_h1_span(md, "Output Schema")
        assert span.startswith("# Output Schema")
        assert "# Orchestrator Input Assembly" not in span

    def test_raises_on_missing_h1(self):
        with pytest.raises(ValueError, match="No .* H1 section found"):
            extract_h1_span("# Something Else\ncontent", "Output Schema")


# ---------- bake_output_schema ----------


class TestBakeOutputSchema:
    def test_splices_schema_into_placeholder(self):
        result = bake_output_schema("Rules.\n{output_schema}\nDone.", '{"a": 1}')
        assert result == 'Rules.\n{"a": 1}\nDone.'

    def test_leaves_other_placeholders_alone(self):
        result = bake_output_schema("{account_type} sees {output_schema}", "SCHEMA")
        assert result == "{account_type} sees SCHEMA"

    def test_real_agents_actually_have_the_placeholder(self):
        """Confirms the precondition this function assumes for the 9
        single-stage agents -- verified by grep against all 15 docs before
        planning; this locks that in as a real check, not a one-time
        observation."""
        for name in _ALL_AGENT_NAMES:
            if name in _MULTI_STAGE:
                continue
            system = extract_block(_doc(name), "System Prompt")
            assert "{output_schema}" in system, f"{name} has no {{output_schema}} placeholder"

    def test_cio_and_risk_advisor_use_stage_qualified_placeholder_names_instead(self):
        """Real correction, found by actually running the split script
        against real data rather than trusting the earlier grep: CIO and
        Risk Advisor don't skip placeholder-based schema injection, they
        just name it differently -- `{output_schema_stage_a}` /
        `{output_schema_stage_b}`, not the bare `{output_schema}` the other
        9 agents use. A grep for only the bare string finds zero matches
        and wrongly implies these two don't use the convention at all."""
        for name in _MULTI_STAGE:
            md = _doc(name)
            assert "{output_schema}" not in md
            assert "{output_schema_stage_a}" in md
            assert "{output_schema_stage_b}" in md

    def test_bakes_into_the_stage_qualified_placeholder_and_leaves_the_other_stage_alone(self):
        template = "{output_schema_stage_a} and also {output_schema_stage_b}"
        result = bake_output_schema(template, '{"a": 1}', placeholder="output_schema_stage_a")
        assert result == '{"a": 1} and also {output_schema_stage_b}'


# ---------- fill() ----------


class TestFill:
    def test_substitutes_known_keys(self):
        assert fill("Hello {name}", {"name": "world"}) == "Hello world"

    def test_leaves_unknown_braces_untouched(self):
        assert (
            fill("Hello {name}, {other} stays", {"name": "world"}) == "Hello world, {other} stays"
        )

    def test_citation_notation_survives(self):
        """N{num} is citation notation in the Sentiment/Researcher prompts
        (e.g. "N1", "N23" -- the MODEL fills these in its own output; fill()
        never sees a real "num" key at any actual call site). The guarantee
        fill() actually provides is narrower than "specially protects N{}"
        -- it's "any brace naming a key not in `values` is left untouched",
        which is what keeps N{num} intact in practice."""
        assert fill("See N{num} for detail", {}) == "See N{num} for detail"

    def test_a_real_placeholder_name_colliding_with_citation_notation_does_resolve(self):
        """Documents the actual boundary of the guarantee above: fill() has
        no positional awareness of what precedes a brace, so if "num" were
        ever a real key, N{num} would resolve like any other placeholder.
        Safety here comes from no real prompt ever using "num" as an actual
        fill-dict key, not from the regex protecting the N-prefixed form."""
        assert fill("See N{num} for detail", {"num": "5"}) == "See N5 for detail"

    def test_does_not_choke_on_embedded_json(self):
        """The exact case str.format() cannot handle: raises KeyError on
        the Stock Researcher prompt because its Output Schema block embeds
        literal JSON, and .format() treats every brace as a placeholder."""
        template = '{"news_id": "{ticker}"}'
        assert fill(template, {"ticker": "AAPL"}) == '{"news_id": "AAPL"}'


# ---------- Full sweep: every real doc, not a sample ----------


class TestFullExtractionSweep:
    @pytest.mark.parametrize("name", _ALL_AGENT_NAMES)
    def test_extracts_cleanly_with_no_stray_fence(self, name):
        """Simulates the exact extraction the split script performs against
        every one of the 15 real docs. A stray ``` inside an extracted block
        would mean an embedded example fence truncated the real content
        early -- checked here, not just for the 2-3 agents spot-checked
        during planning."""
        md = _doc(name)
        headers = (
            ["System Prompt — Stage A", "System Prompt — Stage B"]
            if name in _MULTI_STAGE
            else ["System Prompt"]
        )
        for header in headers:
            block = extract_block(md, header)
            assert block.count("```") == 0, f"{name} / {header}: stray fence in extracted block"

        if name in _MULTI_STAGE:
            for stage in ("Stage A", "Stage B"):
                schema = extract_stage_schema(md, "Output Schema", stage)
                assert schema.count("```") == 0, f"{name} / {stage} schema: stray fence"
        else:
            schema = extract_block(md, "Output Schema")
            assert schema.count("```") == 0, f"{name}: stray fence in Output Schema"


# ---------- load_template: against real generated files ----------


class TestLoadTemplate:
    def test_loads_a_real_single_stage_template(self):
        text = load_template("technical_analyst")
        assert text.strip()
        assert "{output_schema}" not in text, (
            "output_schema should be pre-baked, not left as a placeholder"
        )

    def test_loads_a_real_multi_stage_template(self):
        stage_a = load_template("cio", stage="a")
        stage_b = load_template("cio", stage="b")
        assert stage_a != stage_b

    def test_raises_file_not_found_with_actionable_message(self):
        with pytest.raises(FileNotFoundError, match="split_prompt_docs"):
            load_template("does_not_exist_as_an_agent")

    def test_raises_for_a_stage_that_does_not_exist_on_a_single_stage_agent(self):
        with pytest.raises(FileNotFoundError):
            load_template("technical_analyst", stage="a")
