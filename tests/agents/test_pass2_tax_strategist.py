"""Tests for agents/pass2_tax_strategist.py (86bbuhjup, 86bc8efvb). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why.

Focus: _tax_metrics_block() always rebuilds fresh via build_tax_metrics_field()
now (86bc8efvb removed the old bundle.tax_metrics fast path -- that field
could never stay correct once account_state/user_tax_profile carry real
per-user data DataBundle doesn't hold, so every test here controls the
rendered text by mocking build_tax_metrics_field, not a bundle attribute),
and the field-presence/material-absent gating this drives for MARG/ROOM."""
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from agents.pass2_tax_strategist import (
    TaxStrategistRunner,
    _tax_metrics_block,
    _validate_with_caveats,
    build_user_message,
)


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="RY.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Royal Bank of Canada", "sector": "Financials"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        dividend_history=[],
        price_info={"current_price": 100.0},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def _mock_tax_metrics(text: str):
    return patch("agents.pass2_tax_strategist.build_tax_metrics_field", return_value=text)


def test_tax_metrics_block_always_calls_build_tax_metrics_field_fresh():
    bundle = _bundle()
    with _mock_tax_metrics("DIVID: 4.1% yield, 4 payments/yr") as mock_build:
        result = _tax_metrics_block(bundle, "tfsa")
    mock_build.assert_called_once_with(
        "RY.TO", "tfsa", bundle, account_state=None, user_tax_profile=None
    )
    assert result == "DIVID: 4.1% yield, 4 payments/yr"


def test_tax_metrics_block_threads_account_state_and_user_tax_profile_through():
    bundle = _bundle()
    account_state = {"tfsa_room_remaining_cents": 750_000}
    user_tax_profile = {"province": "ON", "income_annual": 95_000.0}
    with _mock_tax_metrics("ROOM: TFSA remaining $7,500") as mock_build:
        _tax_metrics_block(bundle, "tfsa", account_state, user_tax_profile)
    mock_build.assert_called_once_with(
        "RY.TO", "tfsa", bundle, account_state=account_state, user_tax_profile=user_tax_profile
    )


def test_build_user_message_is_the_pass1_summaries_only():
    """The tax block, reliability warnings and target account live in the system prompt;
    repeating them here cost every call ~1k tokens twice over."""
    bundle = _bundle()
    with _mock_tax_metrics(
        "DIVID: 4.1% yield, 4 payments/yr\nWHT (this account, tfsa): 15.0%, ..."
    ):
        msg, _ = build_user_message(bundle, {}, "tfsa")
    assert "DIVID: 4.1% yield" not in msg
    assert "TAX-RELEVANT DATA" not in msg
    assert "RELIABILITY WARNINGS" not in msg
    assert "ANALYSIS TARGET ACCOUNT" not in msg
    assert msg.startswith("RY.TO (Royal Bank of Canada)")


def test_build_user_message_for_override_account_type():
    bundle = _bundle()  # context.account_type == "tfsa"
    with _mock_tax_metrics(
        "DIVID: 4.1%\nWHT (this account, rrsp): 0.0%, treaty exempt"
    ) as mock_build:
        msg, _ = build_user_message(bundle, {}, "rrsp")
    mock_build.assert_called_once_with(
        "RY.TO", "rrsp", bundle, account_state=None, user_tax_profile=None
    )
    assert "Account: rrsp" in msg  # the Pass 1 header already names the analysed account


# ---------- field_presence (86bbwachy Phase 4, extended 86bc8efvb for marg/room) ----------


def test_field_presence_true_for_both_when_dividend_and_wht_are_real():
    bundle = _bundle()  # DIVID has a real yield, WHT has a real 15.0% rate
    with _mock_tax_metrics(
        "DIVID: 4.1% yield, 4 payments/yr\nWHT (this account, tfsa): 15.0%, ..."
    ):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence == {"divid": True, "wht": True, "marg": False, "room": False}


def test_field_presence_divid_false_when_no_dividend_history():
    bundle = _bundle()
    with _mock_tax_metrics(
        "DIVID: no dividend history\nWHT (this account, tfsa): 0.0%, no withholding"
    ):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["divid"] is False


def test_field_presence_wht_false_when_not_modelled():
    bundle = _bundle()
    with _mock_tax_metrics(
        "DIVID: 3.0% yield, 4 payments/yr\n"
        "WHT (this account, tfsa): NOT MODELLED per REF withholding grid"
    ):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["wht"] is False


def test_field_presence_both_false_for_etf_branch():
    """The ETF early-return path renders neither a DIVID nor a WHT line at
    all -- both must read False, not raise a KeyError/crash on a missing
    substring."""
    bundle = _bundle()
    with _mock_tax_metrics(
        "DOM: not applicable - this precompute module covers individual "
        "equities and REITs/trusts only; XIU.TO is asset_type=etf"
    ):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["divid"] is False
    assert presence["wht"] is False


def test_field_presence_marg_true_when_line_present():
    bundle = _bundle()
    with _mock_tax_metrics(
        "MARG: 31.15% combined federal+Ontario marginal rate on the next dollar"
    ):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["marg"] is True


def test_field_presence_marg_false_when_absent():
    bundle = _bundle()
    with _mock_tax_metrics("DIVID: no dividend history"):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["marg"] is False


def test_field_presence_room_true_when_fresh():
    bundle = _bundle()
    with _mock_tax_metrics("ROOM: TFSA remaining $7,500"):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["room"] is True


def test_field_presence_room_false_when_stale_even_though_line_is_present():
    """The real point of the stale-vs-absent distinction (86bc8efvb): a
    present-but-stale ROOM line must read the same as fully absent for
    material_absent/confidence-caveat gating -- staleness isn't left to the
    LLM to notice on its own."""
    bundle = _bundle()
    with _mock_tax_metrics("ROOM: TFSA remaining $7,500 (stale, last updated 430 days ago)"):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["room"] is False


def test_field_presence_room_false_when_absent_entirely():
    bundle = _bundle()
    with _mock_tax_metrics("DIVID: no dividend history"):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["room"] is False


# ---------- TaxStrategistRunner.run() material_absent gating (86bc8efvb) ----------


def _run_with_mocked_call(bundle, account_type, tax_metrics_text):
    """Runs TaxStrategistRunner.run() with build_tax_metrics_field mocked and
    call_with_validation short-circuited (no real LLM call) -- returns the
    partial() handed to call_with_validation so tests can inspect exactly
    what material_absent was computed as."""
    runner = TaxStrategistRunner()
    with (
        _mock_tax_metrics(tax_metrics_text),
        patch.object(runner, "call_with_validation") as mock_call,
    ):

        async def _fake_call(*_args, **_kwargs):
            return {}, []

        mock_call.side_effect = _fake_call
        asyncio.run(runner.run(bundle, {}, account_type))
    return mock_call.call_args[0][2]  # the partial(_validate_with_caveats, ...) positional arg


def test_run_material_absent_excludes_room_for_trading_account():
    """ROOM never renders for a Trading account at all (build_precomputed_tax_metrics's
    own ROOM branch only handles tfsa/rrsp) -- architecturally correct absence,
    not missing data, same treatment wht already gets for classifications
    WHT_GRID has no entry for."""
    bundle = _bundle(context=SimpleNamespace(account_type="trading", timeline="medium_term"))
    validate_partial = _run_with_mocked_call(
        bundle, "trading", "DIVID: 3.0% yield, 4 payments/yr\nCGAIN: ..."
    )
    assert "room" not in validate_partial.keywords["material_absent"]


def test_run_material_absent_includes_room_for_tfsa_when_genuinely_missing():
    """The exclusion above is trading-specific, not a blanket permanent
    exclusion -- for tfsa/rrsp, a genuinely-missing ROOM line still counts."""
    bundle = _bundle(context=SimpleNamespace(account_type="tfsa", timeline="medium_term"))
    validate_partial = _run_with_mocked_call(bundle, "tfsa", "DIVID: 3.0% yield, 4 payments/yr")
    assert "room" in validate_partial.keywords["material_absent"]


# ---------- _validate_with_caveats (86bbummwp follow-on -- new here, this agent
# previously called validate_tax_strategist bare, no composition and no
# account_type at all, so Rule 14 (loss harvesting null for TFSA/RRSP) had
# never actually fired in production) ----------


def _valid_tax_output(**overrides) -> dict:
    base = {
        "groundedness_score": 82,
        "thesis_summary": "Holding this stock in a TFSA incurs a 15% US withholding tax on the dividend that is not recoverable under the Canada-US tax treaty.",
        "strongest_signal": "WHT: 15% US withholding tax on dividends is NOT recoverable in a TFSA.",
        "caveats": ["Tax rules verified against CRA guidance as of March 2026."],
        "key_factors": [
            {"factor": "US dividend WHT drag", "importance": "medium", "sentiment": "negative", "evidence": "WHT: 15% non-recoverable in TFSA"},
            {"factor": "Capital gains tax-free in TFSA", "importance": "high", "sentiment": "positive", "evidence": "REF: TFSA capital gains are completely tax-free"},
        ],
        "narrative": (
            "This US-domiciled company (DOM) trades on NASDAQ (LIST). The dividend yield "
            "triggers US withholding tax of 15% (WHT) when held in a TFSA. Unlike an RRSP, a "
            "TFSA is not recognized as a retirement account under the Canada-US tax treaty, so "
            "the withholding exemption does not apply (REF/Cross-border). This results in a "
            "modest effective after-tax yield reduction (ELIG). The tax drag from withholding "
            "is modest and should not be the primary factor in the investment decision for a "
            "capital-appreciation-oriented thesis. For a medium-term TFSA investor, the "
            "dominant tax advantage is the capital gains exemption (CGAIN) -- any price "
            "appreciation would be entirely tax-free in a TFSA, compared to a 50% inclusion "
            "rate in a taxable account. The ROOM consideration is relevant: holding a "
            "high-growth stock in TFSA maximizes the tax-free compounding advantage, and "
            "contribution room is restored on January 1 of the year following a sell (REF/TFSA), "
            "so an exit is not permanently costly. FUND reports a modest payout ratio, so the "
            "dividend is a small component of total return and the withholding drag stays "
            "secondary; MACRO notes no pending treaty change that would alter this treatment "
            "over the holding period, and RSRCH's own archetype supports a longer holding "
            "period consistent with this account's own tax treatment favoring patience."
        ),
        "tax_profile": {
            "account_fit_score": "good",
            "tax_efficiency_for_account": "favorable",
            "dividend_yield_pct": 1.8,
            "withholding_tax_rate_pct": 15.0,
            "effective_after_tax_yield_pct": 1.53,
            "is_eligible_canadian_dividend": False,
            # 86bc8eg3j: real schema field is capital_gains_treatment_summary, not
            # capital_gains_treatment -- same stale-fixture mismatch fixed in
            # test_pass2_validators.py's own _valid_tax_output().
            "capital_gains_treatment": "Tax-free in TFSA -- all capital gains are completely exempt from Canadian tax.",
            "capital_gains_treatment_summary": "Tax-free in TFSA -- all capital gains are completely exempt from Canadian tax.",
            "loss_risk_assessment": "Capital losses in TFSA permanently reduce available TFSA room.",
            "cross_account_recommendation": None,
            "recommended_account": "current_account_is_optimal",
            "loss_harvesting_opportunity": None,
            "key_tax_risks": [
                {
                    "risk": "The 15% withholding is permanently lost in a TFSA and cannot be reclaimed as a foreign tax credit.",
                    "severity": "medium",
                    "evidence": "WHT: 15% non-recoverable in TFSA per REF/TFSA",
                },
            ],
            "tax_optimization_actions": [],
        },
    }
    base.update(overrides)
    return base


def test_validate_with_caveats_passes_when_schema_valid_and_below_threshold():
    """groundedness_score=82 is below the 85 threshold -- the new confidence/
    data-quality rule never applies regardless of gaps."""
    passed, errors = _validate_with_caveats(
        _valid_tax_output(), material_absent=["divid"], account_type="tfsa"
    )
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_gap():
    out = _valid_tax_output(tax_profile={
        **_valid_tax_output()["tax_profile"], "account_fit_score": "excellent-ish",
    })
    passed, errors = _validate_with_caveats(out, material_absent=[], account_type="tfsa")
    assert not passed
    assert any("account_fit_score" in e for e in errors)


def test_validate_with_caveats_flags_high_groundedness_with_gap_and_no_caveat():
    out = _valid_tax_output(groundedness_score=92, caveats=[])
    passed, errors = _validate_with_caveats(out, material_absent=["divid"], account_type="tfsa")
    assert not passed
    assert any("caveats is empty" in e for e in errors)


def test_validate_with_caveats_passes_high_groundedness_with_gap_when_caveat_present():
    out = _valid_tax_output(
        groundedness_score=92, caveats=["No dividend history available for this stock."]
    )
    passed, errors = _validate_with_caveats(out, material_absent=["divid"], account_type="tfsa")
    assert passed, errors


def test_validate_with_caveats_enforces_rule_14_loss_harvesting_null_for_tfsa():
    """Rule 14, real and now actually wired: loss_harvesting_opportunity must
    be null in a TFSA/RRSP -- previously dead because account_type was never
    supplied to validate_tax_strategist at all."""
    out = _valid_tax_output(tax_profile={
        **_valid_tax_output()["tax_profile"], "loss_harvesting_opportunity": "harvest now",
    })
    passed, errors = _validate_with_caveats(out, material_absent=[], account_type="tfsa")
    assert not passed
    assert any("loss_harvesting_opportunity" in e for e in errors)


def test_validate_with_caveats_rule_14_not_checked_for_trading_account():
    out = _valid_tax_output(tax_profile={
        **_valid_tax_output()["tax_profile"], "loss_harvesting_opportunity": "harvest now",
    })
    passed, errors = _validate_with_caveats(out, material_absent=[], account_type="trading")
    assert passed, errors

# ---------- one copy of the block, warnings and account (Tax Strategist Wave 2) ----------


def _capture_run(bundle, account_type, tax_metrics_text, compressed_pass1=None):
    """Runs TaxStrategistRunner.run() with the tax block mocked and the model call
    short-circuited; returns (system_prompt, user_message, build_tax_metrics_field mock)."""
    runner = TaxStrategistRunner()
    with (
        _mock_tax_metrics(tax_metrics_text) as mock_build,
        patch.object(runner, "call_with_validation") as mock_call,
    ):

        async def _fake_call(*_args, **_kwargs):
            return {}, []

        mock_call.side_effect = _fake_call
        asyncio.run(runner.run(bundle, compressed_pass1 or {}, account_type))
    system_prompt, user_message = mock_call.call_args[0][0], mock_call.call_args[0][1]
    return system_prompt, user_message, mock_build


def test_run_sends_the_tax_block_once_and_builds_it_once():
    block = "DIVID: 4.1% yield, 4 payments/yr\nWHT (this account, tfsa): 15.0%, non-recoverable"
    system_prompt, user_message, mock_build = _capture_run(_bundle(), "tfsa", block)
    assert (system_prompt + user_message).count("DIVID: 4.1% yield, 4 payments/yr") == 1
    assert "DIVID: 4.1% yield" in system_prompt
    mock_build.assert_called_once()


def test_run_sends_the_reliability_warnings_once():
    compressed = {
        "RSRCH": {
            "pass2_view": {"thesis_archetype": "dividend_compounder"},
            "analysis_confidence": "low",
            "data_quality_assessment": "low",
            "assessment_summary": "x",
            "narrative_truncated": "x",
        }
    }
    system_prompt, user_message, _ = _capture_run(_bundle(), "tfsa", "DIVID: 1.0%", compressed)
    warning = "RSRCH: low"
    assert (system_prompt + user_message).count(warning) >= 1
    assert user_message.count("RELIABILITY WARNINGS") == 0
    assert system_prompt.count("RSRCH: low") == 1


def test_system_prompt_names_the_analysed_account_and_has_no_unfilled_placeholder():
    import re

    system_prompt, _, _ = _capture_run(_bundle(), "rrsp", "DIVID: 1.0%")
    assert "Account: RRSP." in system_prompt
    assert re.findall(r"\{[a-z_]+\}", system_prompt) == []


# ---------- ROOM presence reads the analysed account's own item (both rooms are rendered now) ----------


def test_field_presence_room_reads_the_analysed_accounts_item_not_the_other_accounts():
    bundle = _bundle()
    line = "ROOM: TFSA remaining $32,000 | RRSP remaining $28,000 (stale, last updated 430 days ago)"
    with _mock_tax_metrics(line):
        _, tfsa = build_user_message(bundle, {}, "tfsa")
        _, rrsp = build_user_message(bundle, {}, "rrsp")
    assert tfsa["room"] is True  # the stale one is RRSP's
    assert rrsp["room"] is False


def test_field_presence_room_false_when_only_the_other_accounts_room_is_on_file():
    bundle = _bundle()
    with _mock_tax_metrics("ROOM: RRSP remaining $28,000"):
        _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["room"] is False


def test_field_presence_room_stays_false_for_trading_even_though_the_registered_rooms_render():
    bundle = _bundle()
    with _mock_tax_metrics("ROOM: TFSA remaining $32,000 | RRSP remaining $28,000"):
        _, presence = build_user_message(bundle, {}, "trading")
    assert presence["room"] is False

def test_user_message_holds_only_the_rsrch_and_fund_views():
    compressed = {
        agent: {
            "pass2_view": {"x": 1},
            "analysis_confidence": "high",
            "assessment_summary": f"{agent} summary",
            "narrative_truncated": "n",
        }
        for agent in ("RSRCH", "FUND", "TECH", "SENT", "MACRO")
    }
    with _mock_tax_metrics("DIVID: 1.0%"):
        msg, _ = build_user_message(_bundle(), compressed, "tfsa")
    assert "(RSRCH)" in msg and "(FUND)" in msg
    for dropped in ("(TECH)", "(SENT)", "(MACRO)"):
        assert dropped not in msg


def test_system_prompt_reliability_warnings_cover_only_the_agents_tax_receives():
    compressed = {
        agent: {
            "pass2_view": {"thesis_archetype": "x"},
            "analysis_confidence": "low",
            "data_quality_assessment": "low",
            "assessment_summary": "s",
            "narrative_truncated": "n",
        }
        for agent in ("RSRCH", "FUND", "TECH", "SENT", "MACRO")
    }
    system_prompt, _, _ = _capture_run(_bundle(), "tfsa", "DIVID: 1.0%", compressed)
    assert "RSRCH: low" in system_prompt and "FUND: low" in system_prompt
    assert "TECH: low" not in system_prompt and "MACRO: low" not in system_prompt


# ---------- the runner checks the output against the code's expectation ----------


def test_validate_with_caveats_applies_the_expectation_when_given():
    from data.precompute.tax_metrics import compute_account_verdict, resolve_withholding

    expectation = {
        "yield_pct": 1.8,
        "withholding": resolve_withholding("us", "tfsa"),
        "verdict": compute_account_verdict("tfsa", "us", 1.8, 29.65, 6.39),
    }
    ok = _valid_tax_output(tax_profile={
        **_valid_tax_output()["tax_profile"], "annual_tax_drag_pct": 0.27, "effective_after_tax_yield_pct": 1.53,
        "account_fit_score": "good", "tax_efficiency_for_account": "favorable",
    })
    assert _validate_with_caveats(ok, material_absent=[], account_type="tfsa", expectation=expectation)[0]
    wrong = _valid_tax_output(tax_profile={**ok["tax_profile"], "account_fit_score": "excellent"})
    passed, errors = _validate_with_caveats(wrong, material_absent=[], account_type="tfsa", expectation=expectation)
    assert not passed and any("account_fit_score" in e for e in errors)
    # without an expectation the old behaviour is unchanged
    assert _validate_with_caveats(wrong, material_absent=[], account_type="tfsa")[0]


def test_run_hands_the_code_expectation_to_the_validator():
    bundle = _bundle(context=SimpleNamespace(account_type="trading", timeline="medium_term"))
    validate_partial = _run_with_mocked_call(bundle, "trading", "DIVID: 3.0% yield, 4 payments/yr")
    expectation = validate_partial.keywords["expectation"]
    assert expectation["verdict"] is not None and expectation["classification"] == "canadian_eligible"
