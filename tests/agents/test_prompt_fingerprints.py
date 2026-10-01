"""Tests for prompt_fingerprints() (86bc997wr, ledger BB-045): a run must be able to say
which prompt text it used, because every output says prompt_version "v1" however often a
template is edited in place. Hermetic (a temporary prompts directory), so unlike
test_prompts.py this does not need the top-level repo's design docs.
"""

import re

import pytest

import agents.prompts as prompts_module
from agents.prompts import prompt_fingerprints


@pytest.fixture
def prompts_dir(tmp_path, monkeypatch):
    (tmp_path / "technical_analyst").mkdir()
    (tmp_path / "cio").mkdir()
    (tmp_path / "technical_analyst" / "v1.txt").write_text(
        "tech prompt\nline two\n", encoding="utf-8"
    )
    (tmp_path / "cio" / "v1_stage_a.txt").write_text("cio stage a\n", encoding="utf-8")
    (tmp_path / "cio" / "reference.md").write_text("a reference document\n", encoding="utf-8")
    monkeypatch.setattr(prompts_module, "_PROMPTS_DIR", tmp_path)
    return tmp_path


def test_every_template_and_reference_file_gets_a_short_hash(prompts_dir):
    result = prompt_fingerprints()

    assert set(result["files"]) == {
        "technical_analyst/v1.txt",
        "cio/v1_stage_a.txt",
        "cio/reference.md",
    }
    assert all(re.fullmatch(r"[0-9a-f]{12}", h) for h in result["files"].values())
    assert re.fullmatch(r"[0-9a-f]{12}", result["combined"])


def test_the_same_files_give_the_same_hashes_every_time(prompts_dir):
    assert prompt_fingerprints() == prompt_fingerprints()


def test_editing_one_file_changes_that_hash_and_the_combined_hash_only(prompts_dir):
    before = prompt_fingerprints()
    (prompts_dir / "technical_analyst" / "v1.txt").write_text(
        "tech prompt EDITED\n", encoding="utf-8"
    )
    after = prompt_fingerprints()

    assert after["files"]["technical_analyst/v1.txt"] != before["files"]["technical_analyst/v1.txt"]
    assert after["files"]["cio/v1_stage_a.txt"] == before["files"]["cio/v1_stage_a.txt"]
    assert after["files"]["cio/reference.md"] == before["files"]["cio/reference.md"]
    assert after["combined"] != before["combined"]


def test_adding_or_removing_a_file_changes_the_combined_hash(prompts_dir):
    before = prompt_fingerprints()
    (prompts_dir / "cio" / "v1_stage_b.txt").write_text("cio stage b\n", encoding="utf-8")
    added = prompt_fingerprints()
    (prompts_dir / "cio" / "v1_stage_b.txt").unlink()
    removed = prompt_fingerprints()

    assert added["combined"] != before["combined"]
    assert removed["combined"] == before["combined"]  # back to the original set


def test_windows_and_unix_line_endings_hash_the_same(prompts_dir):
    """The loader reads with universal newlines, so the model sees the same text either
    way, and a Windows checkout must agree with a Linux one."""
    path = prompts_dir / "technical_analyst" / "v1.txt"
    path.write_bytes(b"tech prompt\nline two\n")
    unix = prompt_fingerprints()["files"]["technical_analyst/v1.txt"]
    path.write_bytes(b"tech prompt\r\nline two\r\n")
    windows = prompt_fingerprints()["files"]["technical_analyst/v1.txt"]

    assert unix == windows


def test_other_kinds_of_file_are_ignored(prompts_dir):
    (prompts_dir / "technical_analyst" / "notes.bak").write_text("scratch", encoding="utf-8")
    (prompts_dir / "technical_analyst" / "__pycache__").mkdir()
    (prompts_dir / "technical_analyst" / "__pycache__" / "x.pyc").write_bytes(b"\x00")

    assert set(prompt_fingerprints()["files"]) == {
        "technical_analyst/v1.txt",
        "cio/v1_stage_a.txt",
        "cio/reference.md",
    }


def test_the_real_prompt_directory_is_covered(monkeypatch):
    """Against the real backend/prompts: every agent's template, both stages of the
    two-stage agents, and the tax reference document the Tax Strategist embeds."""
    files = prompt_fingerprints()["files"]

    for expected in (
        "technical_analyst/v1.txt",
        "cio/v1_stage_a.txt",
        "cio/v1_stage_b.txt",
        "risk_advisor/v1_stage_a.txt",
        "tax_strategist/v1.txt",
        "tax_strategist/canadian_tax_rules_reference.md",
    ):
        assert expected in files
