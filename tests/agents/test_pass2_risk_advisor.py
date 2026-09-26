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

from agents.pass2_risk_advisor import _precomputed_risk_metrics, _validate_with_caveats, build_user_message


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
    text, _ = _precomputed_risk_metrics(_bundle())
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
    text, _ = _precomputed_risk_metrics(bundle)
    assert "BETA: beta = N/A" in text
    assert "Max drawdown (1yr) = N/A" in text


def test_no_ytd_return_fabricated():
    """No precompute source for YTD return anywhere -- must never appear."""
    text, _ = _precomputed_risk_metrics(_bundle())
    assert "YTD" not in text
    assert "ytd_return" not in text


def test_correlation_and_concentration_stay_honest_placeholders():
    """No portfolio system exists yet -- these must stay explicit
    placeholders, not fabricated numbers."""
    text, _ = _precomputed_risk_metrics(_bundle())
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


def test_precomputed_risk_metrics_returns_field_presence_directly():
    """_precomputed_risk_metrics is the one source of truth for both the
    rendered text and the per-field presence map -- build_user_message()
    doesn't re-derive presence separately, so this locks in that the
    function's own second return value is real and correct on its own."""
    _, field_presence = _precomputed_risk_metrics(_bundle())
    assert field_presence == {
        "beta": True,
        "annualized_vol": True,
        "max_drawdown_1yr": True,
        "max_drawdown_3yr": True,
        "avg_dollar_volume": True,
    }


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


# ---------- _validate_with_caveats (86bbummwp follow-on -- new here, this agent
# previously called validate_risk_advisor_stage_a bare, no composition at all) ----------


def _valid_stage_a_output(**overrides) -> dict:
    base = {
        "groundedness_score": 78,
        "thesis_summary": "A moderate risk profile. Beta amplifies market moves but FCF is strong enough to manage current leverage.",
        "strongest_signal": "FUND: D/E of 0.42 is the highest in 5 years.",
        "caveats": [],
        "key_factors": [
            {"factor": "Elevated leverage", "importance": "high", "sentiment": "negative", "evidence": "FUND: D/E 0.42"},
            {"factor": "Technical trend weakness", "importance": "medium", "sentiment": "neutral", "evidence": "TECH: RSI 51.3"},
        ],
        "narrative": (
            "This stock presents a moderate-to-acceptable risk profile for a medium-term "
            "investment. The primary risk is post-acquisition leverage, which is not alarming "
            "in absolute terms but represents a real increase from the historical baseline. In "
            "an elevated rate environment, higher leverage amplifies interest expense "
            "sensitivity, and a further rate increase would compound refinancing risk on any "
            "floating-rate debt taken on to fund the acquisition, particularly if the company "
            "needs to refinance any of it before maturity in a less favorable rate environment "
            "than when it was originally issued. The beta means the stock will experience "
            "meaningfully more volatility than the broader market on both the upside and "
            "downside, which matters more for a shorter holding period than a longer one. The "
            "52-week range demonstrates real downside exposure, though the stock is currently "
            "trading well above the 52-week low, suggesting the worst of the drawdown has "
            "likely already occurred absent a fresh negative catalyst emerging over the coming "
            "quarters. Two distinct downside scenarios exist and are not meaningfully "
            "correlated with each other, providing a fair stress test of the risk framework "
            "rather than two versions of the same underlying event playing out twice. Overall, "
            "position sizing should reflect the elevated but not extreme nature of this risk "
            "profile, favoring a moderate rather than aggressive allocation given the balance "
            "of leverage, volatility, and the still-intact underlying competitive position that "
            "supports the broader investment thesis over a multi-quarter horizon."
        ),
        "risk_profile": {
            "volatility_assessment": "moderate",
            "beta_interpretation": "A beta above 1 means the stock moves more than the broader market on average.",
            "max_drawdown_interpretation": "The peak-to-trough decline took roughly two quarters to recover from.",
            "downside_scenarios": [
                {"scenario": "Competitive disruption triggers multiple compression", "probability": "medium", "estimated_impact_pct": -18.0, "trigger": "RSRCH: competitive threat", "timeline": "3_to_12_months"},
                {"scenario": "Integration costs exceed guidance", "probability": "medium", "estimated_impact_pct": -10.0, "trigger": "FUND: integration risk noted in filing", "timeline": "1_to_3_months"},
            ],
            "risk_reward_ratio": "neutral",
            "data_sanity_flags": [],
        },
    }
    base.update(overrides)
    return base


def test_validate_with_caveats_passes_when_schema_valid_and_below_threshold():
    """groundedness_score=78 is below the 85 threshold -- the new rule never
    applies regardless of gaps, matching the default fixture's own real,
    empty caveats list."""
    passed, errors = _validate_with_caveats(_valid_stage_a_output(), material_absent=["beta"])
    assert passed, errors


def test_validate_with_caveats_fails_on_base_schema_error_regardless_of_gap():
    out = _valid_stage_a_output(risk_profile={
        **_valid_stage_a_output()["risk_profile"], "volatility_assessment": "extreme",
    })
    passed, errors = _validate_with_caveats(out, material_absent=[])
    assert not passed
    assert any("volatility_assessment" in e for e in errors)


def test_validate_with_caveats_flags_high_groundedness_with_gap_and_no_caveat():
    out = _valid_stage_a_output(groundedness_score=92, caveats=[])
    passed, errors = _validate_with_caveats(out, material_absent=["max_drawdown_3yr"])
    assert not passed
    assert any("caveats is empty" in e for e in errors)


def test_validate_with_caveats_passes_high_groundedness_with_gap_when_caveat_present():
    out = _valid_stage_a_output(
        groundedness_score=92,
        caveats=["Insufficient price history for a real 3yr drawdown figure."],
    )
    passed, errors = _validate_with_caveats(out, material_absent=["max_drawdown_3yr"])
    assert passed, errors


def test_validate_with_caveats_exactly_at_threshold_counts_as_high():
    out = _valid_stage_a_output(groundedness_score=85, caveats=[])
    passed, errors = _validate_with_caveats(out, material_absent=["beta"])
    assert not passed


