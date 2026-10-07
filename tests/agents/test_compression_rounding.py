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
            "narrative": "n",
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


def test_a_pass1_narrative_reaches_pass2_whole():
    """The 350 token cut bound for 1 of 200 stored narratives and every Pass 1 validator caps the narrative at 1300
    characters, so it is gone: a 1500 character narrative arrives unchanged."""
    from agents.compression import compress_pass1_outputs

    narrative = "Sentence one. " * 107
    out = compress_pass1_outputs({"RSRCH": {"narrative": narrative, "structured_data": {"thesis_archetype": "quality_compounder"}}})

    assert out["RSRCH"]["narrative"] == narrative and len(narrative) > 1400


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
                           "narrative": "n"}}

    msg = build_pass2_user_message(bundle, compressed, agent_ids=("FUND",), account_neutral=True)

    assert "pb_ratio: 2.16" in msg and "valuation_lens: Value on P/B against ROE, with P/E." in msg
    assert "debt_to_equity" not in msg and "fcf_to_net_income" not in msg and "None" not in msg


def test_a_none_value_is_left_out_for_every_agent_not_printed():
    """Macro's commodity_context and Sentiment's insider_materiality printed as "None" in every real Bull prompt."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from agents.compression import build_pass2_user_message

    bundle = SimpleNamespace(
        stock=SimpleNamespace(ticker="ENB.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Enbridge", "sector": "Energy"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 10, 1, tzinfo=UTC),
    )
    compressed = {
        "MACRO": {"assessment_summary": "s", "analysis_confidence": "medium", "caveats": [], "narrative": "n",
                  "pass2_view": {"overall_macro_environment": "favorable", "commodity_context": None}},
        "SENT": {"assessment_summary": "s", "analysis_confidence": "medium", "caveats": [], "narrative": "n",
                 "pass2_view": {"news_sentiment_overall": "neutral", "insider_materiality": None}},
    }

    msg = build_pass2_user_message(bundle, compressed, agent_ids=("MACRO", "SENT"), account_neutral=True)

    assert "overall_macro_environment: favorable" in msg and "news_sentiment_overall: neutral" in msg
    assert "commodity_context" not in msg and "insider_materiality" not in msg and "None" not in msg


def test_a_pass1_agents_risks_reach_pass2_but_not_its_key_factors():
    """Delivered after a replay: Bull, Bear and Risk used them (15 to 20% of their unique words against 2 to 5% by chance)."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from agents.compression import build_pass2_user_message, compress_pass1_outputs

    bundle = SimpleNamespace(
        stock=SimpleNamespace(ticker="TD.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Toronto-Dominion Bank", "sector": "Finance"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 10, 1, tzinfo=UTC),
    )
    output = {"assessment_summary": "s", "analysis_confidence": "high", "caveats": [], "narrative": "n",
              "structured_data": {"thesis_archetype": "quality_compounder"},
              "key_factors": [{"factor": "ROE", "evidence": "PROF: ROE 12.7%"}],
              "risks": [{"risk": "US retail competition", "severity": "medium", "likelihood": "high", "evidence": "RSRCH: smaller than peers"}]}

    msg = build_pass2_user_message(bundle, compress_pass1_outputs({"RSRCH": output}), agent_ids=("RSRCH",), account_neutral=True)

    assert "risks: US retail competition (medium severity, high likelihood): RSRCH: smaller than peers" in msg
    assert "key_factors" not in msg and "ROE 12.7%" not in msg
