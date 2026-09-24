"""Tests for agents/pass2_bear_advocate.py (86bbuhjup). See
test_pass2_bull_advocate.py's docstring for rationale -- same shape."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass2_bear_advocate import build_user_message


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
            "pass2_view": {"thesis_archetype": "value_trap_candidate"}, "narrative_truncated": "n",
        }
    }


def test_renders_real_context_fields_not_fixture():
    msg = build_user_message(_bundle(), _compressed_pass1())
    assert "FOCUS: medium_term | ACCOUNT: tfsa" in msg


def test_account_type_override_takes_precedence():
    msg = build_user_message(_bundle(), _compressed_pass1(), account_type="rrsp")
    assert "ACCOUNT: rrsp" in msg
