"""Tests for agents/pass2_tax_strategist.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why.

Focus: the real simplification this port makes (bundle.tax_metrics is
already the fully-rendered block, not re-derived) and its one real wrinkle
(an account_type override that differs from bundle.context.account_type must
not silently use the wrong account's tax_metrics)."""
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from agents.pass2_tax_strategist import _tax_metrics_block, _validate_with_caveats, build_user_message


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="RY.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Royal Bank of Canada", "sector": "Financials"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        tax_metrics="DIVID: 4.1% yield, 4 payments/yr\nWHT (this account, tfsa): 15.0%, ...",
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_uses_bundle_tax_metrics_directly_when_account_type_matches():
    bundle = _bundle()
    result = _tax_metrics_block(bundle, "tfsa")
    assert result == bundle.tax_metrics


def test_rebuilds_fresh_when_account_type_override_differs():
    """The real wrinkle: bundle.tax_metrics was built for 'tfsa'
    (bundle.context.account_type) -- requesting 'rrsp' must not silently
    reuse the tfsa-scoped string."""
    bundle = _bundle()  # context.account_type == "tfsa", tax_metrics is tfsa-scoped
    with patch("agents.pass2_tax_strategist.build_tax_metrics_field") as mock_build:
        mock_build.return_value = "DIVID: 4.1% yield\nWHT (this account, rrsp): 0.0%, treaty exempt"
        result = _tax_metrics_block(bundle, "rrsp")
    mock_build.assert_called_once_with("RY.TO", "rrsp", bundle)
    assert "rrsp" in result
    assert result != bundle.tax_metrics


def test_build_user_message_renders_real_tax_metrics_block():
    bundle = _bundle()
    msg, _ = build_user_message(bundle, {}, "tfsa")
    assert "DIVID: 4.1% yield, 4 payments/yr" in msg
    assert "WHT (this account, tfsa): 15.0%" in msg
    assert "ANALYSIS TARGET ACCOUNT: TFSA" in msg


def test_build_user_message_uses_fresh_block_for_override_account():
    bundle = _bundle()
    with patch("agents.pass2_tax_strategist.build_tax_metrics_field") as mock_build:
        mock_build.return_value = "DIVID: 4.1%\nWHT (this account, rrsp): 0.0%, treaty exempt"
        msg, _ = build_user_message(bundle, {}, "rrsp")
    assert "WHT (this account, rrsp): 0.0%, treaty exempt" in msg
    assert "ANALYSIS TARGET ACCOUNT: RRSP" in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_field_presence_true_for_both_when_dividend_and_wht_are_real():
    bundle = _bundle()  # DIVID has a real yield, WHT has a real 15.0% rate
    _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence == {"divid": True, "wht": True}


def test_field_presence_divid_false_when_no_dividend_history():
    bundle = _bundle(tax_metrics="DIVID: no dividend history\nWHT (this account, tfsa): 0.0%, no withholding")
    _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["divid"] is False


def test_field_presence_wht_false_when_not_modelled():
    # account_type matches bundle.context.account_type ("tfsa") so
    # _tax_metrics_block reads bundle.tax_metrics directly rather than
    # taking the override path (which would need a real DataBundle's
    # dividend_history/price_info to rebuild fresh).
    bundle = _bundle(
        tax_metrics="DIVID: 3.0% yield, 4 payments/yr\n"
        "WHT (this account, tfsa): NOT MODELLED per REF withholding grid"
    )
    _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence["wht"] is False


def test_field_presence_both_false_for_etf_branch():
    """The ETF early-return path renders neither a DIVID nor a WHT line at
    all -- both must read False, not raise a KeyError/crash on a missing
    substring."""
    bundle = _bundle(
        tax_metrics="DOM: not applicable - this precompute module covers individual "
        "equities and REITs/trusts only; XIU.TO is asset_type=etf"
    )
    _, presence = build_user_message(bundle, {}, "tfsa")
    assert presence == {"divid": False, "wht": False}


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
            "capital_gains_treatment": "Tax-free in TFSA -- all capital gains are completely exempt from Canadian tax.",
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
