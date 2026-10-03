"""Tests for code_fingerprint() (86bc997wr, ledger BB-045): the prompt hash cannot say
whether two runs are comparable on its own, because much of what the model reads is written
by code. The code hash covers that. Hermetic (a temporary source directory).
"""

import re

import pytest

import services.code_version as code_version
from services.code_version import code_fingerprint


@pytest.fixture
def src_dir(tmp_path, monkeypatch):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "utils.py").write_text("STALE = 45\n", encoding="utf-8")
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "run.py").write_text("print('run')\n", encoding="utf-8")
    monkeypatch.setattr(code_version, "_SRC_DIR", tmp_path)
    return tmp_path


def test_the_hash_is_twelve_hex_digits_and_stable(src_dir):
    first = code_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{12}", first)
    assert code_fingerprint() == first


def test_editing_a_source_file_changes_the_hash(src_dir):
    """The case the prompt hash misses: a threshold in code changes what an agent is told."""
    before = code_fingerprint()
    (src_dir / "agents" / "utils.py").write_text("STALE = 90\n", encoding="utf-8")
    assert code_fingerprint() != before


def test_adding_removing_or_renaming_a_file_changes_the_hash(src_dir):
    before = code_fingerprint()
    extra = src_dir / "services" / "extra.py"
    extra.write_text("x = 1\n", encoding="utf-8")
    added = code_fingerprint()
    extra.rename(src_dir / "services" / "renamed.py")
    renamed = code_fingerprint()
    (src_dir / "services" / "renamed.py").unlink()

    assert added != before
    assert renamed != added  # same content, different path
    assert code_fingerprint() == before


def test_windows_and_unix_line_endings_hash_the_same(src_dir):
    path = src_dir / "agents" / "utils.py"
    path.write_bytes(b"STALE = 45\nOTHER = 1\n")
    unix = code_fingerprint()
    path.write_bytes(b"STALE = 45\r\nOTHER = 1\r\n")
    assert code_fingerprint() == unix


def test_editing_a_json_data_file_changes_the_hash(src_dir):
    """The cross-listing map changes behavior with no Python edit."""
    (src_dir / "data").mkdir()
    crosslisting = src_dir / "data" / "ca_us_crosslisting.json"
    crosslisting.write_text('{"RY.TO": {"us_ticker": "RY"}}', encoding="utf-8")
    before = code_fingerprint()
    crosslisting.write_text('{"RY.TO": {"us_ticker": "RYAA"}}', encoding="utf-8")
    assert code_fingerprint() != before


def test_only_python_source_and_json_data_count(src_dir):
    before = code_fingerprint()
    (src_dir / "agents" / "notes.txt").write_text("scratch", encoding="utf-8")
    (src_dir / "data").mkdir(exist_ok=True)
    (src_dir / "data" / "review.md").write_text("a review note", encoding="utf-8")
    (src_dir / "agents" / "__pycache__").mkdir()
    (src_dir / "agents" / "__pycache__" / "utils.cpython-312.pyc").write_bytes(b"\x00\x01")
    (src_dir / "agents" / "__pycache__" / "cached.py").write_text("stale copy", encoding="utf-8")
    assert code_fingerprint() == before


def test_a_file_that_is_not_valid_utf8_does_not_break_the_hash(src_dir):
    (src_dir / "agents" / "odd.py").write_bytes(b"x = '\xff\xfe'\n")
    assert re.fullmatch(r"[0-9a-f]{12}", code_fingerprint())


def test_the_real_source_tree_is_hashed():
    assert re.fullmatch(r"[0-9a-f]{12}", code_fingerprint())
