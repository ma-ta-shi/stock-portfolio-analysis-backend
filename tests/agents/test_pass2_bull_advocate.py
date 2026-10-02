"""Tests for agents/pass2_bull_advocate.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why.
Close to a clean port, so tests focus on the two real seams: reading
bundle.context instead of fixture["context"], and the hardcoded
recommendation="bullish" defense-in-depth."""
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from agents.pass2_bull_advocate import BullAdvocateRunner, build_user_message


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
    )
    return SimpleNamespace(**{**defaults, **overrides})


def _compressed_pass1():
    return {
        "RSRCH": {
            "assessment_summary": "s", "analysis_confidence": "high", "caveats": [],
            "pass2_view": {"thesis_archetype": "secular_grower"}, "narrative_truncated": "n",
        }
    }


def test_message_names_no_account_and_is_identical_for_every_account():
    msgs = {
        build_user_message(_bundle(context=SimpleNamespace(account_type=a, timeline="medium_term")), _compressed_pass1())
        for a in ("tfsa", "rrsp", "trading")
    }
    assert len(msgs) == 1
    msg = msgs.pop()
    assert "ccount" not in msg and "ACCOUNT" not in msg
    assert "Timeline: medium_term" in msg


def test_message_is_the_pass1_summaries_only():
    """The reliability warnings, the archetype and the focus line are in the system prompt; they used to be
    appended here as well, so every call paid for them twice."""
    msg = build_user_message(_bundle(), _compressed_pass1())
    assert "RELIABILITY WARNINGS" not in msg
    assert "RESEARCHER ARCHETYPE" not in msg
    assert "FOCUS" not in msg
    assert "PASS 1 " in msg


def _captured_system_prompt(compressed):
    runner = BullAdvocateRunner()

    async def _fake_call(system_prompt, user_msg, validator, **_kw):
        captured["system"] = system_prompt
        captured["user"] = user_msg
        return {}, []

    captured = {}
    with patch.object(runner, "call_with_validation", side_effect=_fake_call):
        asyncio.run(runner.run(_bundle(), compressed))
    return captured


def test_warnings_and_archetype_are_sent_once_in_the_system_prompt():
    compressed = _compressed_pass1()
    compressed["RSRCH"]["analysis_confidence"] = "low"
    captured = _captured_system_prompt(compressed)
    assert "RESEARCHER ARCHETYPE: secular_grower" in captured["system"]
    assert "RSRCH: low" in captured["system"]
    assert "RESEARCHER ARCHETYPE" not in captured["user"] and "RELIABILITY WARNINGS" not in captured["user"]


def test_run_takes_no_account_argument():
    import inspect

    assert "account_type" not in inspect.signature(BullAdvocateRunner.run).parameters
