"""Pass 1 views reach the Pass 2 agents with their long floats shortened (0.9986728136350936 was copied
into user-visible evidence digit for digit, and each costs about 10 tokens)."""
import pytest

from agents.compression import _rounded


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.9986728136350936, 0.9987),  # real KO fcf_to_net_income
        (0.6939682588514888, 0.6940),  # real RDDT revenue_growth_yoy
        (0.00031234, 0.0003123),  # small values keep their significant digits, never 0.0
        (148.75, 148.75),  # 4 or fewer decimals: untouched
        (25.9, 25.9),
        (1234567.891234, 1234567.9),  # large numbers keep their integer part
        (-0.9986728136350936, -0.9987),
        (0.0, 0.0),
    ],
)
def test_floats_are_shortened_without_losing_the_meaning(value, expected):
    assert _rounded(value) == pytest.approx(expected)


def test_integers_bools_strings_and_none_are_untouched():
    assert _rounded(5) == 5 and _rounded(True) is True and _rounded("0.9986728136350936") == "0.9986728136350936"
    assert _rounded(None) is None


def test_nested_structures_are_rounded_in_place_of_their_floats_only():
    assert _rounded({"a": [0.9986728136350936, "x", 3], "b": {"c": 0.123456789}}) == {"a": [0.9987, "x", 3], "b": {"c": 0.1235}}


def test_nan_and_inf_pass_through():
    import math

    assert math.isnan(_rounded(float("nan"))) and _rounded(float("inf")) == float("inf")


def test_the_builder_shows_rounded_figures():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from agents.compression import build_pass2_user_message

    bundle = SimpleNamespace(
        stock=SimpleNamespace(ticker="KO", currency="USD", exchange="NYSE"),
        company_info={"name": "Coca-Cola", "sector": "Consumer Defensive"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 10, 1, tzinfo=UTC),
    )
    compressed = {
        "FUND": {
            "assessment_summary": "s", "analysis_confidence": "high", "caveats": [],
            "pass2_view": {"fcf_to_net_income": 0.9986728136350936, "roe": 0.396, "detail": {"x": 0.3333333333333}},
            "narrative_truncated": "n",
        }
    }
    msg = build_pass2_user_message(bundle, compressed, agent_ids=("FUND",), account_neutral=True)
    assert "fcf_to_net_income: 0.9987" in msg and "roe: 0.396" in msg and '"x": 0.3333' in msg
    assert "0.9986728" not in msg


def test_numpy_floats_are_handled_like_plain_floats():
    """A real MSFT run crashed all three Pass 2 agents: pandas/numpy values reach the view as np.float64,
    whose repr ("np.float64(0.998...)") Decimal cannot parse."""
    import numpy as np

    assert _rounded(np.float64(0.9986728136350936)) == 0.9987
    assert _rounded(np.float64(514.27)) == 514.27
    assert type(_rounded(np.float64(0.5))) is float
    assert _rounded({"a": [np.float64(1234.56789)]}) == {"a": [1234.6]}


def test_pass2_reads_350_tokens_of_a_narrative_so_a_1300_character_one_is_whole():
    """The Researcher and Fundamental narrative ceiling is 1300 characters; the old 300 token (1200 character) cut
    would have removed the closing synthesis."""
    from agents.compression import NARRATIVE_VIEW_TOKENS
    from agents.utils import truncate_to_tokens

    assert NARRATIVE_VIEW_TOKENS == 350
    narrative = "x" * 1300
    assert truncate_to_tokens(narrative, NARRATIVE_VIEW_TOKENS) == narrative
    assert truncate_to_tokens("x" * 1500, NARRATIVE_VIEW_TOKENS).endswith("...")


def test_a_metric_the_fundamental_view_leaves_out_is_not_printed_as_none():
    """A bank's D/E and FCF/NI are not passed to Pass 2: no "debt_to_equity: None" or "not_applicable" line either."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from agents.compression import build_pass2_user_message

    bundle = SimpleNamespace(
        stock=SimpleNamespace(ticker="TD.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Toronto-Dominion Bank", "sector": "Finance"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 10, 1, tzinfo=UTC),
    )
    view = {"valuation_vs_sector": "fair", "pb_ratio": 2.16, "roe": 0.1275, "debt_to_equity": None, "fcf_to_net_income": None,
            "valuation_lens": "Value on P/B against ROE, with P/E."}
    compressed = {"FUND": {"assessment_summary": "s", "analysis_confidence": "high", "caveats": [], "pass2_view": view,
                           "narrative_truncated": "n"}}

    msg = build_pass2_user_message(bundle, compressed, agent_ids=("FUND",), account_neutral=True)

    assert "pb_ratio: 2.16" in msg and "valuation_lens: Value on P/B against ROE, with P/E." in msg
    assert "debt_to_equity" not in msg and "fcf_to_net_income" not in msg and "None" not in msg
