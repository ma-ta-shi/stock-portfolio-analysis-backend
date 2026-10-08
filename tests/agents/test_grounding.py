"""The Pass 1 grounding check: a figure must be in the data the agent was given."""
import pytest

from agents.base import MAX_RETRIES, BaseRunner
from agents.grounding import grounding_errors, ungrounded_figures
from tests.agents.test_base import FakeResponse, FakeSession, _ollama_chat_response

PAYLOAD = (
    "ROE: 12.7% | Net margin: 24.9% | Market cap: 274.44B CAD | Volume today: 3.52M | Payout 0.482 | FCF to NI 0.2407\n"
    "Revenue growth YoY: 8.4% | Price vs SMA200: -8.42% | Nearest support: 167.30 (0.46%, 0.3 ATR) | Headline: $150 Billion pledge"
)


@pytest.mark.parametrize("text", [
    "ROE of 12.7% and a 24.9% net margin",           # as written
    "a market cap of $274.4 billion",                # unit conversion, rounded
    "volume of 3.52 million shares",                 # M against million
    "a payout of 48.2% and 24% of net income as FCF",  # fraction against percent, rounded
    "the price sits 8.4% below its 200-day average",  # sign and a window
    "a $150 billion commitment",                     # case of the unit
    "In 2026 the 20-day and 3 quarters of data show 1 trend",  # a year, a window and counts are not figures
])
def test_a_figure_the_payload_carries_is_grounded(text):
    assert ungrounded_figures(text, PAYLOAD) == []


def test_an_invented_figure_is_flagged():
    """The real case: the payload said support was 0.46% away and the narrative wrote 0.07%."""
    assert ungrounded_figures("The price sits 0.07% above the nearest support and ROE is 12.7%", PAYLOAD) == ["0.07%"]


def test_a_derived_figure_is_flagged_because_the_prompts_say_not_to_recompute():
    assert ungrounded_figures("a 54% discount to the sector median", PAYLOAD) == ["54%"]


def test_errors_name_the_field_and_the_figure_and_are_capped():
    out = {"narrative": "up 77.7%, down 66.6%, flat 55.5%, and 44.4%", "assessment_summary": "a 99.1% jump", "caveats": ["only 12.7%"]}

    errors = grounding_errors(out, PAYLOAD)

    assert len(errors) == 3 and errors[0].startswith("narrative: the figure '77.7%'") and "data you were given" in errors[0]
    assert grounding_errors({"narrative": "n"}, "") == [] and grounding_errors("not a dict", PAYLOAD) == []


class _OnceRunner(BaseRunner):
    GROUND_MODE = "once"


class _LogRunner(BaseRunner):
    GROUND_MODE = "log"


def _runner(cls, *outputs):
    runner = cls()
    runner.session = FakeSession([FakeResponse(200, _ollama_chat_response(o)) for o in outputs])
    return runner


BAD = {"narrative": "support is 0.07% away", "assessment_summary": ""}
GOOD = {"narrative": "support is 0.46% away", "assessment_summary": ""}


@pytest.mark.asyncio
async def test_once_mode_retries_an_ungrounded_figure_with_the_error_named():
    runner = _runner(_OnceRunner, BAD, GOOD)

    result, errors = await runner.call_with_validation("sys", PAYLOAD, lambda out: (True, []))

    assert errors == [] and result["narrative"] == "support is 0.46% away"
    assert len(runner.session.calls) == 2
    assert any("0.07%" in e for e in runner.call_log[0]["errors"])


@pytest.mark.asyncio
async def test_once_mode_retries_a_grounding_flag_only_once_then_records_it():
    """A second bad answer is accepted with the flag recorded: the model is not retried again for the same kind of error."""
    runner = _runner(_OnceRunner, BAD, BAD)

    result, errors = await runner.call_with_validation("sys", PAYLOAD, lambda out: (True, []))

    assert errors == [] and result["narrative"] == BAD["narrative"] and len(runner.session.calls) == 2
    assert runner.call_log[0]["validator_passed"] is False
    assert runner.call_log[1]["validator_passed"] is True
    assert any("0.07%" in e for e in runner.call_log[1]["validator_errors"])


@pytest.mark.asyncio
async def test_once_mode_never_costs_the_output_on_the_last_attempt():
    """If the first answer that parses is the last attempt, a grounding flag cannot fail it."""
    bad_json = FakeResponse(200, {"message": {"content": "not json"}, "done": True})
    runner = _OnceRunner()
    runner.session = FakeSession([bad_json, bad_json, FakeResponse(200, _ollama_chat_response(BAD))])

    result, errors = await runner.call_with_validation("sys", PAYLOAD, lambda out: (True, []))

    assert errors == [] and result["narrative"] == BAD["narrative"]


@pytest.mark.asyncio
async def test_structural_errors_keep_their_full_retry_budget_in_once_mode():
    runner = _runner(_OnceRunner, GOOD, GOOD, GOOD)

    result, errors = await runner.call_with_validation("sys", PAYLOAD, lambda out: (False, ["narrative: too short"]))

    assert errors == ["narrative: too short"] and len(runner.session.calls) == MAX_RETRIES


@pytest.mark.asyncio
async def test_log_mode_never_retries_and_records_the_flag_without_failing_the_call():
    runner = _runner(_LogRunner, BAD)

    result, errors = await runner.call_with_validation("sys", PAYLOAD, lambda out: (True, []))

    assert errors == [] and result["narrative"] == BAD["narrative"] and len(runner.session.calls) == 1
    entry = runner.call_log[0]
    assert entry["validator_passed"] is True and entry["passed"] is True
    assert any("0.07%" in e for e in entry["validator_errors"])


@pytest.mark.asyncio
async def test_a_grounded_answer_records_nothing_in_either_mode():
    for cls in (_OnceRunner, _LogRunner):
        runner = _runner(cls, GOOD)
        await runner.call_with_validation("sys", PAYLOAD, lambda out: (True, []))
        assert runner.call_log[0]["validator_errors"] == []


@pytest.mark.parametrize("text", [
    "ROE of 12% and 24% net margin",              # truncated, not rounded (12.7, 24.9)
    "a Price vs SMA200 of -8.4%",                 # one decimal of -8.42%
    "a market cap of $274 billion",               # truncated
])
def test_a_truncated_figure_is_grounded(text):
    assert ungrounded_figures(text, PAYLOAD) == []


def test_a_range_written_with_one_unit_shares_it_across_both_ends():
    payload = "Revenue: 1,200,000,000 | Guidance: 1,300,000,000 | Growth: 10% | Prior growth: 9%"
    assert ungrounded_figures("revenue of $1.2-1.3bn", payload) == []
    assert ungrounded_figures("growth of 9-10%", payload) == []


def test_a_recited_threshold_from_the_system_prompt_is_not_a_flag():
    out = {"narrative": "avg dollar volume is below the $1M threshold", "assessment_summary": ""}
    assert grounding_errors(out, "rule: thin volume means avg dollar volume > $1M\n" + PAYLOAD) == []


def test_thesis_summary_is_checked_too():
    assert grounding_errors({"thesis_summary": "a 54% discount"}, PAYLOAD)[0].startswith("thesis_summary:")


@pytest.mark.asyncio
async def test_an_ungrounded_runner_does_not_check_figures():
    session = FakeSession([FakeResponse(200, _ollama_chat_response({"narrative": "up 12345.6%"}))])
    runner = BaseRunner()
    runner.session = session

    result, errors = await runner.call_with_validation("sys", PAYLOAD, lambda out: (True, []))

    assert errors == [] and len(session.calls) == 1
    assert MAX_RETRIES >= 2


@pytest.mark.asyncio
async def test_a_trimmable_overshoot_does_not_let_a_flagged_figure_skip_its_retry():
    """The trimmer would pass the first attempt and the invented figure would ship with its one retry spent."""
    bad = {**BAD, "key_factors": [1, 2, 3]}
    runner = _runner(_OnceRunner, bad, GOOD)

    result, errors = await runner.call_with_validation(
        "sys", PAYLOAD, lambda out: (False, ["key_factors: need <=2 items, got 3"]) if "key_factors" in out else (True, [])
    )

    assert errors == [] and result["narrative"] == GOOD["narrative"] and len(runner.session.calls) == 2
