"""The design docs are the source `scripts/split_prompt_docs.py` bakes runtime prompts from, so a runtime prompt edited without
its doc is silently reverted the next time the script runs. This bakes every doc exactly as the script does and requires the
result to equal the runtime file byte for byte. Skipped when the docs repo is not checked out beside backend/."""
from pathlib import Path

import pytest

from agents.prompts import bake_output_schema, extract_block, extract_stage_schema

_ROOT = Path(__file__).resolve().parents[3]
_DOCS = _ROOT / "Agent Prompts" / "Current Prompts"
_PROMPTS = Path(__file__).resolve().parents[2] / "prompts"

SINGLE = [
    ("Fundamental Analyst", "fundamental_analyst"), ("Technical Analyst", "technical_analyst"),
    ("Sentiment Analyst", "sentiment_analyst"), ("Macro Economist", "macro_economist"), ("Stock Researcher", "stock_researcher"),
    ("Bull Case Advocate", "bull_advocate"), ("Bear Case Advocate", "bear_advocate"), ("Tax Strategist", "tax_strategist"),
    ("Shadow CIO (calibration agent)", "shadow_cio"),
]
MULTI = [("Chief Investment Officer (CIO)", "cio", "ab"), ("Risk Advisor", "risk_advisor", "a")]

pytestmark = pytest.mark.skipif(not _DOCS.exists(), reason="design docs repo not checked out beside backend/")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize(("stem", "slug"), SINGLE)
def test_single_stage_runtime_prompt_equals_its_baked_design_doc(stem, slug):
    doc = _read(_DOCS / f"{stem} Agent Prompt.md")
    baked = bake_output_schema(extract_block(doc, "System Prompt"), extract_block(doc, "Output Schema"))
    assert baked == _read(_PROMPTS / slug / "v1.txt"), f"{slug}: edit the design doc fence, not just the runtime prompt"


@pytest.mark.parametrize(("stem", "slug", "stages"), MULTI)
def test_multi_stage_runtime_prompts_equal_their_baked_design_doc(stem, slug, stages):
    doc = _read(_DOCS / f"{stem} Agent Prompt.md")
    for stage in stages:
        baked = bake_output_schema(
            extract_block(doc, f"System Prompt — Stage {stage.upper()}"),
            extract_stage_schema(doc, "Output Schema", f"Stage {stage.upper()}"),
            placeholder=f"output_schema_stage_{stage}",
        )
        assert baked == _read(_PROMPTS / slug / f"v1_stage_{stage}.txt"), f"{slug} stage {stage}: doc and runtime differ"
