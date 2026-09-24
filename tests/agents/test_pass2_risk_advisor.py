"""Tests for agents/pass2_risk_advisor.py (86bbuhjup). No harness
equivalent -- see test_pass1_stock_researcher.py's docstring for why.

The beta/max_drawdown_1yr orchestrator-injection in run() and the Stage A->B
continuation wiring are plumbing already covered by test_base.py's own
call_with_validation_start/call_with_validation_continue suite; these tests
focus on this runner's own real seam -- the DataBundle field mapping in
_precomputed_risk_metrics/build_user_message, which is what actually differs
from the harness."""
from datetime import UTC, datetime
from types import SimpleNamespace

from agents.pass2_risk_advisor import _precomputed_risk_metrics, build_user_message


def _bundle(**overrides) -> SimpleNamespace:
    defaults = dict(
        stock=SimpleNamespace(ticker="SHOP.TO", currency="CAD", exchange="TSX"),
        company_info={"name": "Shopify Inc", "sector": "Technology"},
        context=SimpleNamespace(account_type="tfsa", timeline="medium_term"),
        data_vintage=datetime(2026, 9, 23, tzinfo=UTC),
        risk_metrics={"beta": 1.8, "annualized_vol_pct": 42.5, "max_drawdown_1yr_pct": -28.4,
                      "recovery_1yr_days": 120, "max_drawdown_3yr_pct": -45.1,
                      "recovery_3yr_days": 300, "adv_millions": 85.2, "adv_currency": "CAD"},
        price_position={"high_52w": 120.0, "low_52w": 60.0},
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_precomputed_risk_metrics_renders_real_fields():
    text = _precomputed_risk_metrics(_bundle()).text
    assert "BETA: beta = 1.8" in text
    assert "Annualized vol=42.5%" in text
    assert "Max drawdown (1yr) = -28.4%" in text
    assert "recovery: 120 days" in text
    assert "Max drawdown (3yr) = -45.1%" in text
    assert "Avg dollar volume (20d) = 85.2M CAD" in text


def test_missing_risk_metrics_render_honestly_not_fabricated():
    bundle = _bundle(risk_metrics={"beta": None, "annualized_vol_pct": None,
                                    "max_drawdown_1yr_pct": None, "recovery_1yr_days": None,
                                    "max_drawdown_3yr_pct": None, "recovery_3yr_days": None,
                                    "adv_millions": None, "adv_currency": None})
    text = _precomputed_risk_metrics(bundle).text
    assert "BETA: beta = N/A" in text
    assert "Max drawdown (1yr) = N/A" in text


def test_no_ytd_return_fabricated():
    """No precompute source for YTD return anywhere -- must never appear."""
    text = _precomputed_risk_metrics(_bundle()).text
    assert "YTD" not in text
    assert "ytd_return" not in text


def test_correlation_and_concentration_stay_honest_placeholders():
    """No portfolio system exists yet -- these must stay explicit
    placeholders, not fabricated numbers."""
    text = _precomputed_risk_metrics(_bundle()).text
    assert "CORR: Portfolio correlation = unknown" in text
    assert "CONC: Standard concentration risk assessment" in text


def test_stop_loss_note_null_for_tfsa_medium_term():
    msg, _ = build_user_message(_bundle(), {})
    assert "STOP-LOSS NOTE: null is appropriate" in msg


def test_stop_loss_note_numeric_for_trading_short_term():
    bundle = _bundle(context=SimpleNamespace(account_type="trading", timeline="short_term"))
    msg, _ = build_user_message(bundle, {})
    assert "STOP-LOSS NOTE: numeric stop-loss may be appropriate" in msg


# ---------- field_presence (86bbwachy Phase 4) ----------


def test_precomputed_risk_metrics_present_true_when_all_five_fields_real():
    assert _precomputed_risk_metrics(_bundle()).present is True


def test_precomputed_risk_metrics_present_false_when_any_field_missing():
    bundle = _bundle(risk_metrics={"beta": None, "annualized_vol_pct": 42.5,
                                    "max_drawdown_1yr_pct": -28.4, "recovery_1yr_days": 120,
                                    "max_drawdown_3yr_pct": -45.1, "recovery_3yr_days": 300,
                                    "adv_millions": 85.2, "adv_currency": "CAD"})
    assert _precomputed_risk_metrics(bundle).present is False


def test_field_presence_all_true_with_default_bundle():
    _, presence = build_user_message(_bundle(), {})
    assert presence == {
        "beta": True,
        "annualized_vol": True,
        "max_drawdown_1yr": True,
        "max_drawdown_3yr": True,
        "avg_dollar_volume": True,
    }


def test_field_presence_beta_false_when_missing():
    bundle = _bundle(risk_metrics={**_bundle().risk_metrics, "beta": None})
    _, presence = build_user_message(bundle, {})
    assert presence["beta"] is False


def test_field_presence_annualized_vol_false_when_missing():
    bundle = _bundle(risk_metrics={**_bundle().risk_metrics, "annualized_vol_pct": None})
    _, presence = build_user_message(bundle, {})
    assert presence["annualized_vol"] is False


def test_field_presence_max_drawdown_1yr_false_when_missing():
    bundle = _bundle(risk_metrics={**_bundle().risk_metrics, "max_drawdown_1yr_pct": None})
    _, presence = build_user_message(bundle, {})
    assert presence["max_drawdown_1yr"] is False


def test_field_presence_max_drawdown_3yr_false_when_missing():
    """3yr drawdown genuinely degrades independently of 1yr -- confirmed by
    risk_metrics.py's own module docstring (3yr needs ~3yr of history,
    1yr needs ~1yr) -- a stock with 1-2 years of listed history is the
    real-world case this covers."""
    bundle = _bundle(risk_metrics={**_bundle().risk_metrics, "max_drawdown_3yr_pct": None})
    _, presence = build_user_message(bundle, {})
    assert presence["max_drawdown_3yr"] is False


def test_field_presence_avg_dollar_volume_false_when_missing():
    bundle = _bundle(risk_metrics={**_bundle().risk_metrics, "adv_millions": None})
    _, presence = build_user_message(bundle, {})
    assert presence["avg_dollar_volume"] is False


