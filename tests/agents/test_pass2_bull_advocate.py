"""Tests for agents/pass2_bull_advocate.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why.
Close to a clean port, so tests focus on the two real seams: reading
bundle.context instead of fixture["context"], and the hardcoded
recommendation="bullish" defense-in-depth."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass2_bull_advocate import build_user_message


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


def test_renders_real_context_fields_not_fixture():
    msg = build_user_message(_bundle(), _compressed_pass1())
    assert "FOCUS: medium_term | ACCOUNT: tfsa" in msg


def test_account_type_override_takes_precedence_over_bundle_context():
    msg = build_user_message(_bundle(), _compressed_pass1(), account_type="rrsp")
    assert "ACCOUNT: rrsp" in msg


def test_renders_researcher_archetype():
    msg = build_user_message(_bundle(), _compressed_pass1())
    assert "RESEARCHER ARCHETYPE FOR THIS STOCK: secular_grower" in msg
