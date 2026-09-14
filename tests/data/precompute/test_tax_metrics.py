import os
import tempfile
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from data.precompute.tax_metrics import (
    CGAIN_BY_ACCOUNT,
    WHT_GRID,
    TaxReferenceUnavailable,
    build_precomputed_tax_metrics,
    build_tax_rule_snapshot,
    classify_dividend,
    compute_effective_after_tax_yield,
    compute_trailing_dividend,
    is_canadian_dual_listed,
    load_tax_rules_reference,
    resolve_withholding,
)


def _company_info(
    name="", industry="", country="", asset_type="equity", primary_exchange=""
) -> dict:
    return {
        "name": name,
        "industry": industry,
        "country": country,
        "asset_type": asset_type,
        "primary_exchange": primary_exchange,
    }


def _div_record(ex_date: str, amount: float) -> dict:
    return {"ex_date": ex_date, "payment_date": None, "amount_per_share": amount}


def _days_ago(days: int) -> str:
    return (datetime.now().date() - timedelta(days=days)).isoformat()


def _fake_bundle(
    company_info: dict, dividend_history: list, current_price: float | None
) -> SimpleNamespace:
    """Only company_info/dividend_history/price_info are ever read by
    build_precomputed_tax_metrics - a lightweight stand-in, not a full
    schema-valid DataBundle, matching this codebase's own fakes-not-full-
    instances convention for tests."""
    return SimpleNamespace(
        company_info=company_info,
        dividend_history=dividend_history,
        price_info={"current_price": current_price},
    )


# --- classify_dividend ---


def test_classify_dividend_ca_ordinary_corporation():
    info = _company_info(name="Royal Bank of Canada", industry="Banking")
    assert classify_dividend("RY.TO", info) == ("canadian_eligible", "corporation")


def test_classify_dividend_us_ordinary_corporation():
    info = _company_info(name="Apple Inc.", industry="Consumer Electronics", country="US")
    assert classify_dividend("AAPL", info) == ("us", "corporation")


def test_classify_dividend_ca_reit():
    info = _company_info(name="RioCan Real Estate Investment Trust", industry="REITs")
    assert classify_dividend("REI-UN.TO", info) == ("trust_distribution", "REIT")


def test_classify_dividend_us_reit():
    info = _company_info(name="Realty Income Corporation", industry="REIT - Retail", country="US")
    assert classify_dividend("O", info) == ("us_reit", "REIT")


def test_classify_dividend_genuine_limited_partnership():
    """Real, live-confirmed case: Brookfield Infrastructure Partners L.P.
    Its name contains "Partners" (catches on the PARTNERS substring) but
    literally "L.P." with periods - confirming PARTNERS, not " LP", is
    the real anchor for this heuristic.

    country="BM" (Bermuda) is BIP's real, live-confirmed value, not "US"
    as an earlier version of this fixture assumed - re-checked live
    during a later review pass and corrected. That drift never affected
    this test's own pass/fail (the partnership check on NAME fires before
    country is ever consulted), but it does mean this case genuinely
    proves the partnership check must win BEFORE the ADR check: with
    country="BM" (not "US", not ""), BIP would otherwise misclassify as
    "adr" - a real, live-confirmed reason for this function's own
    checked-in-order design, not a hypothetical one."""
    info = _company_info(
        name="Brookfield Infrastructure Partners L.P.",
        industry="Diversified Utilities",
        country="BM",
    )
    classification, structure = classify_dividend("BIP", info)
    assert classification == "limited_partnership"
    assert structure == "limited partnership"


def test_classify_dividend_ca_partnership_via_un_suffix():
    info = _company_info(
        name="Brookfield Infrastructure Partners L.P.", industry="Regulated Utilities"
    )
    classification, _ = classify_dividend("BIP-UN.TO", info)
    assert classification == "limited_partnership"


def test_classify_dividend_mlp_label_when_name_says_both_partners_and_mlp():
    """Synthetic, not live-confirmed - no real ticker checked this session
    combines both "Partners" (needed to trigger the partnership branch at
    all) and "MLP" in its name; AMLP itself never reaches this branch
    (caught by the ETF check first) and BIP has "Partners" but not "MLP".
    Exercises the label-selection logic specifically, disclosed as
    illustrative."""
    info = _company_info(
        name="Example MLP Partners L.P.", industry="Oil & Gas Midstream", country="US"
    )
    _, structure = classify_dividend("EXMLP", info)
    assert structure == "MLP"


def test_classify_dividend_etf_checked_first_even_with_mlp_in_name():
    """Real, live-confirmed case: AMLP (Alerian MLP ETF) - its own name
    contains "MLP", but asset_type=="etf" must win, not the partnership
    branch, since ETF is checked first."""
    info = _company_info(
        name="Alerian MLP ETF", industry="Asset Management", asset_type="etf", country="US"
    )
    _, structure = classify_dividend("AMLP", info)
    assert structure == "etf"


def test_classify_dividend_reit_themed_etf_still_resolves_as_etf():
    """Real, live-confirmed case: VNQ (Vanguard Real Estate ETF) - its own
    industry is "Asset Management", not "REIT", so this doesn't actually
    exercise the REIT-vs-ETF ordering risk, but confirms no regression."""
    info = _company_info(
        name="Vanguard Real Estate ETF", industry="Asset Management", asset_type="etf", country="US"
    )
    _, structure = classify_dividend("VNQ", info)
    assert structure == "etf"


def test_classify_dividend_adr():
    """Real, live-confirmed ADR examples spanning three countries."""
    for ticker, country, name in (
        ("BABA", "CN", "Alibaba Group Holding Limited"),
        ("ASML", "NL", "ASML Holding N.V."),
        ("TM", "JP", "Toyota Motor Corporation"),
    ):
        info = _company_info(name=name, industry="Specialty Retail", country=country)
        assert classify_dividend(ticker, info) == ("adr", "ADR")


def test_classify_dividend_resolves_from_structure_alone_with_no_dividend():
    """SHOP.TO never pays a dividend, but its classification (a real
    structural fact) must not depend on dividend history at all."""
    info = _company_info(name="Shopify Inc.", industry="Software & IT Services")
    assert classify_dividend("SHOP.TO", info) == ("canadian_eligible", "corporation")


# --- compute_trailing_dividend ---


def test_compute_trailing_dividend_quarterly_payer():
    history = [_div_record(_days_ago(d), 0.5) for d in (10, 100, 190, 280)]
    yield_pct, count = compute_trailing_dividend(history, 100.0)
    assert count == 4
    assert yield_pct == pytest.approx(2.0)


def test_compute_trailing_dividend_monthly_payer_not_understated():
    """Regression for the fundamentals.py bug class (finding #8): a
    monthly payer's trailing yield must reflect all ~12 payments in the
    window, not just the last 4."""
    history = [_div_record(_days_ago(d * 30), 0.0965) for d in range(12)]
    yield_pct, count = compute_trailing_dividend(history, 20.63)
    assert count == 12
    assert yield_pct == pytest.approx(12 * 0.0965 / 20.63 * 100, rel=1e-3)


def test_compute_trailing_dividend_excludes_records_outside_window():
    history = [_div_record(_days_ago(10), 0.5), _div_record(_days_ago(400), 0.5)]
    yield_pct, count = compute_trailing_dividend(history, 100.0)
    assert count == 1
    assert yield_pct == pytest.approx(0.5)


def test_compute_trailing_dividend_no_history():
    assert compute_trailing_dividend([], 100.0) == (None, 0)


def test_compute_trailing_dividend_no_price():
    history = [_div_record(_days_ago(10), 0.5)]
    assert compute_trailing_dividend(history, None) == (None, 0)


def test_compute_trailing_dividend_no_records_in_window():
    history = [_div_record(_days_ago(400), 0.5)]
    assert compute_trailing_dividend(history, 100.0) == (None, 0)


# --- resolve_withholding ---


@pytest.mark.parametrize(
    "classification,account_type,expected_rate",
    [
        ("canadian_eligible", "tfsa", 0.0),
        ("canadian_eligible", "rrsp", 0.0),
        ("canadian_eligible", "trading", 0.0),
        ("trust_distribution", "tfsa", 0.0),
        ("trust_distribution", "rrsp", 0.0),
        ("us", "tfsa", 15.0),
        ("us", "rrsp", 0.0),
        ("us", "trading", 15.0),
    ],
)
def test_resolve_withholding_modelled_combinations(classification, account_type, expected_rate):
    result = resolve_withholding(classification, account_type)
    assert result is not None
    assert result[0] == expected_rate


def test_resolve_withholding_trust_distribution_taxable_not_modelled():
    """The rig's own real bug (finding #3): a CA REIT/trust in a taxable
    account must be None, not silently inherit the registered-account
    rate or the fully-not-modelled treatment every other structure gets."""
    assert resolve_withholding("trust_distribution", "trading") is None


@pytest.mark.parametrize(
    "classification",
    ["us_reit", "us_mlp", "limited_partnership", "foreign", "adr"],
)
def test_resolve_withholding_not_modelled_classifications(classification):
    for account_type in ("tfsa", "rrsp", "trading"):
        assert resolve_withholding(classification, account_type) is None


def test_wht_grid_has_exactly_the_modelled_entries():
    """Pins the grid's own size - a real regression check that nothing
    silently gets added or dropped."""
    assert len(WHT_GRID) == 8


# --- compute_effective_after_tax_yield ---


def test_compute_effective_after_tax_yield_matches_prompt_testing_checklist():
    """The live prompt's own testing checklist: drag_delta_pct ≈
    -(yield × 0.15) for a 15%-WHT US dividend in a TFSA. Uses 2.0% yield,
    not 3.0% - the latter's exact drag (0.45) sits on a floating-point
    rounding boundary (confirmed live: round(0.45, 1) == 0.5 in Python,
    a representation quirk, not a production bug) - picking a
    non-boundary value tests the same formula without that noise."""
    effective, drag = compute_effective_after_tax_yield(2.0, 15.0)
    assert drag == pytest.approx(0.3)
    assert effective == pytest.approx(1.7)


def test_compute_effective_after_tax_yield_none_when_no_dividend():
    assert compute_effective_after_tax_yield(None, 15.0) == (None, None)


def test_compute_effective_after_tax_yield_none_when_not_modelled():
    assert compute_effective_after_tax_yield(3.0, None) == (None, None)


def test_compute_effective_after_tax_yield_zero_wht():
    effective, drag = compute_effective_after_tax_yield(4.0, 0.0)
    assert effective == 4.0
    assert drag == 0.0


# --- CGAIN_BY_ACCOUNT ---


def test_cgain_by_account_has_all_three_accounts_and_no_shared_line():
    assert set(CGAIN_BY_ACCOUNT) == {"tfsa", "rrsp", "trading"}
    assert len({CGAIN_BY_ACCOUNT[a] for a in CGAIN_BY_ACCOUNT}) == 3


# --- is_canadian_dual_listed ---


def test_is_canadian_dual_listed_true_for_real_pairs(monkeypatch):
    monkeypatch.setattr(
        "data.precompute.tax_metrics.is_crosslisted",
        lambda ticker: ticker in ("RY.TO", "BNS.TO", "ENB.TO"),
    )
    assert is_canadian_dual_listed("RY.TO") is True
    assert is_canadian_dual_listed("BNS.TO") is True
    assert is_canadian_dual_listed("ENB.TO") is True


def test_is_canadian_dual_listed_false_for_unmapped_ca_ticker(monkeypatch):
    monkeypatch.setattr("data.precompute.tax_metrics.is_crosslisted", lambda ticker: False)
    assert is_canadian_dual_listed("WELL.TO") is False


def test_is_canadian_dual_listed_false_for_us_ticker(monkeypatch):
    monkeypatch.setattr("data.precompute.tax_metrics.is_crosslisted", lambda ticker: False)
    assert is_canadian_dual_listed("AAPL") is False


def test_is_canadian_dual_listed_known_miss_for_us_side_of_real_pair(monkeypatch):
    """Documented gap (finding #13), not a passing test pretending
    otherwise: is_crosslisted() only accepts a CA-suffixed ticker, so the
    US-side ticker of a real dual-listed pair is missed."""
    monkeypatch.setattr(
        "data.precompute.tax_metrics.is_crosslisted", lambda ticker: ticker == "RY.TO"
    )
    assert is_canadian_dual_listed("RY") is False


# --- load_tax_rules_reference / build_tax_rule_snapshot ---


def test_load_tax_rules_reference_against_the_real_shipped_file():
    """Real call against the actual, already-committed reference file -
    confirms the real header parses, not a fixture standing in for it."""
    text, last_verified = load_tax_rules_reference()
    assert last_verified == date(2026, 3, 1)
    assert "Withholding tax grid" in text


def test_load_tax_rules_reference_missing_file_raises():
    with pytest.raises(TaxReferenceUnavailable):
        load_tax_rules_reference("does/not/exist.md")


def test_load_tax_rules_reference_unparseable_header_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# No last-verified header here\n")
        path = f.name
    try:
        with pytest.raises(TaxReferenceUnavailable):
            load_tax_rules_reference(path)
    finally:
        os.unlink(path)


def test_build_tax_rule_snapshot_matches_real_prompt_format():
    line = build_tax_rule_snapshot(date(2026, 3, 1))
    assert (
        line
        == f"TAX_RULE_SNAPSHOT: snapshot_date={date.today().isoformat()}, reference_last_verified=2026-03-01"
    )


# --- build_precomputed_tax_metrics ---


def test_build_precomputed_tax_metrics_ca_ordinary_full_render():
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking", primary_exchange="TSX"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics("RY.TO", "trading", bundle)
    assert "DIVID: 6.0% yield, 4 payments/yr" in block
    assert "ELIG: canadian_eligible" in block
    assert "WHT (this account, trading): 0.0%" in block
    assert "not modelled" not in block.lower()


def test_build_precomputed_tax_metrics_ca_reit_registered_account_not_not_modelled():
    """The fix over the audit rig's own real bug: a CA REIT/trust in a
    registered account must NOT say "not modelled"."""
    bundle = _fake_bundle(
        _company_info(name="RioCan REIT", industry="REITs"),
        [_div_record(_days_ago(d * 30), 0.1) for d in range(12)],
        20.0,
    )
    block = build_precomputed_tax_metrics("REI-UN.TO", "tfsa", bundle)
    assert "WHT (this account, tfsa): 0.0%" in block
    assert "NOT MODELLED" not in block


def test_build_precomputed_tax_metrics_adr_not_modelled():
    bundle = _fake_bundle(
        _company_info(
            name="Alibaba Group Holding Limited", industry="Specialty Retail", country="CN"
        ),
        [_div_record(_days_ago(30), 1.05)],
        109.3,
    )
    block = build_precomputed_tax_metrics("BABA", "rrsp", bundle)
    assert "WHT (this account, rrsp): NOT MODELLED per REF withholding grid" in block


def test_build_precomputed_tax_metrics_no_dividend_still_renders_structural_fields():
    bundle = _fake_bundle(
        _company_info(name="Shopify Inc.", industry="Software & IT Services"), [], 178.4
    )
    for account_type in ("tfsa", "rrsp", "trading"):
        block = build_precomputed_tax_metrics("SHOP.TO", account_type, bundle)
        assert "DIVID: no dividend history" in block
        assert "ELIG: canadian_eligible" in block
        assert "WHT (this account" in block
        assert "Effective after-tax yield" not in block


def test_build_precomputed_tax_metrics_etf_short_circuits():
    bundle = _fake_bundle(
        _company_info(
            name="Alerian MLP ETF", industry="Asset Management", asset_type="etf", country="US"
        ),
        [_div_record(_days_ago(30), 1.03)],
        55.9,
    )
    block = build_precomputed_tax_metrics("AMLP", "trading", bundle)
    assert "not applicable" in block.lower()
    assert "DIVID" not in block
    assert "WHT" not in block


def test_build_precomputed_tax_metrics_rejects_unrecognized_account_type():
    bundle = _fake_bundle(_company_info(), [], None)
    with pytest.raises(ValueError):
        build_precomputed_tax_metrics("AAPL", "general", bundle)  # type: ignore[arg-type]


def test_build_precomputed_tax_metrics_room_loss_us_situs_absent_without_account_state():
    bundle = _fake_bundle(
        _company_info(name="Apple Inc.", industry="Consumer Electronics", country="US"),
        [_div_record(_days_ago(30), 0.26)],
        332.0,
    )
    block = build_precomputed_tax_metrics("AAPL", "tfsa", bundle)
    assert "ROOM" not in block
    assert "LOSS" not in block
    assert "US_SITUS" not in block


def test_build_precomputed_tax_metrics_tax_rule_snapshot_omitted_without_it():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics("AAPL", "trading", bundle)
    assert "TAX_RULE_SNAPSHOT" not in block


def test_build_precomputed_tax_metrics_tax_rule_snapshot_appended_when_given():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "trading", bundle, reference_last_verified=date(2026, 3, 1)
    )
    assert block.splitlines()[-1] == build_tax_rule_snapshot(date(2026, 3, 1))


def test_build_precomputed_tax_metrics_full_gate1_render_with_real_reference():
    """End to end: the full block, DIVID through TAX_RULE_SNAPSHOT, using
    a real load_tax_rules_reference() call, not a hardcoded test date."""
    _, last_verified = load_tax_rules_reference()
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking", primary_exchange="TSX"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics(
        "RY.TO", "trading", bundle, reference_last_verified=last_verified
    )
    for required in ("DIVID", "ELIG", "LIST", "DOM", "WHT", "TAX_RULE_SNAPSHOT"):
        assert required in block


# --- build_precomputed_tax_metrics with account_state (plausible, synthetic values) ---


def test_build_precomputed_tax_metrics_tfsa_room_remaining():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "tfsa", bundle, account_state={"tfsa_room_remaining_cents": 750_000}
    )
    assert "ROOM: TFSA remaining $7,500" in block


def test_build_precomputed_tax_metrics_rrsp_room_remaining():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "rrsp", bundle, account_state={"rrsp_room_remaining_cents": 3_156_000}
    )
    assert "ROOM: RRSP remaining $31,560" in block


def test_build_precomputed_tax_metrics_trading_blocked_superficial_loss_with_dual_listed(
    monkeypatch,
):
    monkeypatch.setattr("data.precompute.tax_metrics.is_crosslisted", lambda ticker: True)
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics(
        "RY.TO",
        "trading",
        bundle,
        account_state={
            "superficial_loss_blocked": True,
            "trading_ytd_realized_losses_cents": 250_000,
        },
    )
    assert "LOSS: superficial-loss window: BLOCKED" in block
    assert "DUAL_LISTED" in block
    assert "YTD realized losses: $2,500" in block


def test_build_precomputed_tax_metrics_trading_loss_available_not_blocked(monkeypatch):
    """The other half of loss_status's own if/else - only ever exercised
    via blocked=True until now. superficial_loss_blocked=False is not
    the same as omitted (None) - it's an explicit "checked, not
    blocked" state, and must still render the LOSS line (the surrounding
    `if blocked is not None or ...` condition is satisfied by an explicit
    False), just with "available" instead of "BLOCKED"."""
    monkeypatch.setattr("data.precompute.tax_metrics.is_crosslisted", lambda ticker: True)
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics(
        "RY.TO",
        "trading",
        bundle,
        account_state={"superficial_loss_blocked": False, "trading_ytd_realized_losses_cents": 0},
    )
    assert "LOSS: superficial-loss window: available" in block
    assert "BLOCKED" not in block


def test_build_precomputed_tax_metrics_trading_loss_omits_dual_listed_for_non_crosslisted(
    monkeypatch,
):
    """The other half of dual_listed_flag's own if/else - only ever
    exercised for a real dual-listed ticker until now. A non-crosslisted
    US ticker in the same LOSS-triggering branch must render the LOSS
    line without the DUAL_LISTED marker."""
    monkeypatch.setattr("data.precompute.tax_metrics.is_crosslisted", lambda ticker: False)
    bundle = _fake_bundle(
        _company_info(name="Apple Inc.", industry="Consumer Electronics", country="US"),
        [_div_record(_days_ago(30), 0.26)],
        332.0,
    )
    block = build_precomputed_tax_metrics(
        "AAPL",
        "trading",
        bundle,
        account_state={
            "superficial_loss_blocked": True,
            "trading_ytd_realized_losses_cents": 100_000,
        },
    )
    assert "LOSS: superficial-loss window: BLOCKED" in block
    assert "DUAL_LISTED" not in block


def test_build_precomputed_tax_metrics_trading_loss_triggered_by_losses_alone():
    """The `blocked is not None or ytd_losses_cents is not None` OR
    condition - only ever exercised with both keys present together until
    now. Supplying ytd_losses_cents alone (blocked entirely absent from
    account_state, not even explicitly None) must still trigger the LOSS
    line, defaulting to "available" since an absent blocked reads as
    falsy, not as its own missing-data case."""
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics(
        "RY.TO",
        "trading",
        bundle,
        account_state={"trading_ytd_realized_losses_cents": 75_000},
    )
    assert "LOSS: superficial-loss window: available" in block
    assert "YTD realized losses: $750" in block


def test_build_precomputed_tax_metrics_trading_ytd_gains_appends_to_cgain():
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics(
        "RY.TO", "trading", bundle, account_state={"trading_ytd_realized_gains_cents": 500_000}
    )
    assert "CGAIN:" in block
    assert "YTD realized in Trading: $5,000" in block


def test_build_precomputed_tax_metrics_trading_cgain_no_ytd_clause_when_gains_key_absent():
    """The other half of the ytd_gains_cents if-check - only ever
    exercised with the key present until now. account_state supplied for
    a trading account but WITHOUT trading_ytd_realized_gains_cents must
    render the plain CGAIN line, no YTD clause appended."""
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_precomputed_tax_metrics("RY.TO", "trading", bundle, account_state={})
    assert "CGAIN:" in block
    assert "YTD realized in Trading" not in block


def test_build_precomputed_tax_metrics_room_omitted_when_key_absent_but_account_state_given():
    """The other half of ROOM's own `if room is not None` check for both
    tfsa and rrsp - only ever exercised with the room key present until
    now (the account_state=None case exercises a different code path
    entirely, since the whole `if account_state is not None:` block is
    skipped, not this inner check)."""
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    for account_type in ("tfsa", "rrsp"):
        block = build_precomputed_tax_metrics("AAPL", account_type, bundle, account_state={})
        assert "ROOM" not in block


def test_build_precomputed_tax_metrics_us_situs_renders_raw_dollar_no_percentage():
    bundle = _fake_bundle(
        _company_info(name="Apple Inc.", industry="Consumer Electronics", country="US"),
        [],
        332.0,
    )
    block = build_precomputed_tax_metrics(
        "AAPL", "trading", bundle, account_state={"us_situs_aggregate_usd": 45_000.0}
    )
    assert "US_SITUS aggregate: $45,000 USD" in block
    assert "%" not in block.split("US_SITUS")[1]


def test_build_precomputed_tax_metrics_us_situs_omitted_when_absent_or_zero():
    bundle = _fake_bundle(
        _company_info(name="Apple Inc.", industry="Consumer Electronics", country="US"), [], 332.0
    )
    block = build_precomputed_tax_metrics("AAPL", "trading", bundle, account_state={})
    assert "US_SITUS" not in block
    block_zero = build_precomputed_tax_metrics(
        "AAPL", "trading", bundle, account_state={"us_situs_aggregate_usd": 0.0}
    )
    assert "US_SITUS" not in block_zero


def test_build_precomputed_tax_metrics_us_situs_gates_on_market_not_classification():
    """Real bug caught on review, not shipped: an earlier version gated
    US_SITUS on `classification in ("us", "us_reit", "us_mlp", "adr")` -
    "us_mlp" is dead (classify_dividend never produces it), and
    "limited_partnership" (which DOES need to count when it's the US
    side) was missing entirely, since that one classification value is
    deliberately market-ambiguous. Gating on the ticker's own real market
    fixes both: a US-side LP correctly renders US_SITUS..."""
    bundle_us_lp = _fake_bundle(
        _company_info(
            name="Example MLP Partners L.P.", industry="Oil & Gas Midstream", country="US"
        ),
        [],
        30.0,
    )
    block = build_precomputed_tax_metrics(
        "EXMLP", "trading", bundle_us_lp, account_state={"us_situs_aggregate_usd": 10_000.0}
    )
    assert "US_SITUS aggregate: $10,000 USD" in block

    # ...while a CA-side partnership/trust ticker correctly does NOT, even
    # though it shares the exact same "limited_partnership" classification
    # value - the whole reason a classification-only check can't work here.
    bundle_ca_lp = _fake_bundle(
        _company_info(
            name="Brookfield Infrastructure Partners L.P.", industry="Regulated Utilities"
        ),
        [],
        30.0,
    )
    block_ca = build_precomputed_tax_metrics(
        "BIP-UN.TO", "trading", bundle_ca_lp, account_state={"us_situs_aggregate_usd": 10_000.0}
    )
    assert "US_SITUS" not in block_ca


# --- LIST line's country fallback ---


def test_build_precomputed_tax_metrics_list_line_falls_back_to_ca_for_blank_country():
    """Real, live-caught gap: country is blank for every CA ticker
    through the real Router path (a disclosed provider-data gap), which
    rendered an uninformative "LIST: TSX ()" - falls back to the 2-letter
    code "CA", matching the real, populated values' own 2-letter-code
    convention ("US", "CN", "NL", "JP")."""
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking", primary_exchange="TSX"),
        [],
        100.0,
    )
    block = build_precomputed_tax_metrics("RY.TO", "trading", bundle)
    assert "LIST: TSX (CA)" in block


def test_build_precomputed_tax_metrics_list_line_uses_real_country_when_present():
    bundle = _fake_bundle(
        _company_info(
            name="Apple Inc.",
            industry="Consumer Electronics",
            country="US",
            primary_exchange="NASDAQ",
        ),
        [],
        332.0,
    )
    block = build_precomputed_tax_metrics("AAPL", "trading", bundle)
    assert "LIST: NASDAQ (US)" in block
