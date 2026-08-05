import pytest
from pydantic import ValidationError

from data.schemas.context import AnalysisContext


def test_analysis_context_valid_construction():
    context = AnalysisContext(account_type="tfsa", timeline="medium_term")
    assert context.account_type == "tfsa"
    assert context.timeline == "medium_term"


@pytest.mark.parametrize("account_type", ["tfsa", "rrsp", "trading", "general"])
def test_analysis_context_accepts_all_real_account_types(account_type):
    """Values confirmed live against src/api/tables/analysis_runs.py:13's
    ORM column comment, not assumed from the archived Phase 1 doc (which
    uses different, stale values)."""
    AnalysisContext(account_type=account_type, timeline="short_term")


@pytest.mark.parametrize("timeline", ["short_term", "medium_term", "long_term"])
def test_analysis_context_accepts_all_real_timelines(timeline):
    """Values confirmed live against src/api/tables/analysis_runs.py:14."""
    AnalysisContext(account_type="general", timeline=timeline)


def test_analysis_context_rejects_stale_archived_timeline_values():
    """Regression guard: the archived Phase 1 doc used "1mo"/"3mo"/"12mo"
    — must NOT accept those, only the current short/medium/long_term set."""
    with pytest.raises(ValidationError):
        AnalysisContext(account_type="tfsa", timeline="3mo")


def test_analysis_context_rejects_stale_archived_account_type_value():
    """Regression guard: the archived doc used "taxable" instead of the
    real "trading" value."""
    with pytest.raises(ValidationError):
        AnalysisContext(account_type="taxable", timeline="medium_term")


def test_analysis_context_is_frozen():
    context = AnalysisContext(account_type="tfsa", timeline="medium_term")
    with pytest.raises(ValidationError):
        context.timeline = "long_term"


def test_analysis_context_forbids_extra_fields():
    with pytest.raises(ValidationError):
        AnalysisContext(account_type="tfsa", timeline="medium_term", user_id="some-id")
