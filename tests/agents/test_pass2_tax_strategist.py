"""Tests for agents/pass2_tax_strategist.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why.

Focus: the real simplification this port makes (bundle.tax_metrics is
already the fully-rendered block, not re-derived) and its one real wrinkle
(an account_type override that differs from bundle.context.account_type must
not silently use the wrong account's tax_metrics)."""
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from agents.pass2_tax_strategist import _tax_metrics_block, build_user_message


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
