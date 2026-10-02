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
    build_tax_metrics_field,
    build_tax_rule_snapshot,
    _EFFICIENCY_BY_FIT,
    _fit_label,
    classify_dividend,
    compute_account_verdict,
    compute_eligible_dividend_tax_rate,
    compute_marginal_tax_rate,
    compute_trailing_dividend,
    dividend_tax_rate_pct,
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
    confirms the real header parses, not a fixture standing in for it.
    Date updated 2026-10-01 (Ontario eligible-dividend credit added; earlier 86bc8efkg (PR #73) real 2026 CRA data refresh, no longer
    the stale 2026-03-01 placeholder."""
    text, last_verified = load_tax_rules_reference()
    assert last_verified == date(2026, 10, 1)
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
    block = build_precomputed_tax_metrics(
        "RY.TO", "trading", bundle, user_tax_profile={"province": "ON", "income_annual": 85000.0}
    )
    assert "DIVID: 6.0% yield, 4 payments/yr" in block
    assert "ELIG: canadian_eligible" in block
    assert "WHT (this account, trading): 0.0%" in block
    assert "not modelled" not in block.lower()
    assert "TAXCOST (this account, trading): annual dividend tax 0.38% of the holding" in block


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
        assert "effective after-tax yield" not in block
        assert "ALTERNATIVES" not in block


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


# --- build_tax_metrics_field (86bbztxpj) ---


@pytest.mark.parametrize("account_type", ["tfsa", "rrsp", "trading"])
def test_build_tax_metrics_field_happy_path_all_accounts(account_type):
    """End to end through the wrapper, real reference file, all three real
    account types — matches test_build_precomputed_tax_metrics_full_gate1_render_with_real_reference's
    own real-reference pattern, just through build_tax_metrics_field() instead of the two-step
    load_tax_rules_reference() + build_precomputed_tax_metrics() a caller would otherwise do itself."""
    bundle = _fake_bundle(
        _company_info(name="Royal Bank of Canada", industry="Banking", primary_exchange="TSX"),
        [_div_record(_days_ago(d), 1.5) for d in (10, 100, 190, 280)],
        100.0,
    )
    block = build_tax_metrics_field("RY.TO", account_type, bundle)
    for required in ("DIVID", "ELIG", "LIST", "DOM", "WHT", "TAX_RULE_SNAPSHOT"):
        assert required in block


def test_build_tax_metrics_field_rejects_general_account_type():
    """The real runtime risk this test targets: AnalysisContext.account_type's Literal
    still structurally allows "general" (ClickUp 86bbzqud4, not yet fixed) — Python doesn't
    enforce Literal at runtime, so a caller passing context.account_type straight through can
    genuinely reach this ValueError, not just a type-checker-only concern."""
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    with pytest.raises(ValueError, match="general"):
        build_tax_metrics_field("AAPL", "general", bundle)


def test_build_tax_metrics_field_propagates_tax_reference_unavailable():
    """reference_path passed straight through to load_tax_rules_reference(), same pattern as
    test_load_tax_rules_reference_missing_file_raises — no monkeypatching needed."""
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    with pytest.raises(TaxReferenceUnavailable):
        build_tax_metrics_field("AAPL", "trading", bundle, reference_path="does/not/exist.md")


# --- compute_marginal_tax_rate / MARG (86bc8efvb) ---
# Expected values hand-derived from the finalized 86bc8efkg formula (federal +
# Ontario brackets, Ontario surtax approximated via the BPA credit only) —
# see that ticket for the full sourcing/derivation. pytest.approx with a small
# absolute tolerance, not exact float equality, since these are independently
# hand-computed, not copy-pasted from the implementation.


def test_compute_marginal_tax_rate_low_income_no_surtax():
    # federal 14.0% (< $58,523) + ontario 5.05% (< $53,891), T4_approx well
    # under the first surtax threshold -> multiplier 1.0
    assert compute_marginal_tax_rate("ON", 30_000.0) == pytest.approx(19.05, abs=0.01)


def test_compute_marginal_tax_rate_just_below_first_surtax_threshold():
    assert compute_marginal_tax_rate("ON", 94_000.0) == pytest.approx(29.65, abs=0.01)


def test_compute_marginal_tax_rate_just_above_first_surtax_threshold():
    # Same brackets as the case just below, but T4_approx now crosses $5,818
    # -> 20% surtax multiplier kicks in on the Ontario portion only.
    assert compute_marginal_tax_rate("ON", 96_000.0) == pytest.approx(31.48, abs=0.01)


def test_compute_marginal_tax_rate_past_second_surtax_threshold():
    # federal 26.0% + ontario 11.16% x 1.56 (both surtax tiers stacked)
    assert compute_marginal_tax_rate("ON", 150_000.0) == pytest.approx(43.41, abs=0.01)


def test_compute_marginal_tax_rate_federal_bracket_boundary_inclusive():
    """Exactly at a threshold reads the LOWER bracket's rate -- the same
    inclusive-lower convention the 86bc8efkg bracket tables themselves use
    ("$0-$58,523: 14%")."""
    at_threshold = compute_marginal_tax_rate("ON", 58_523.0)
    just_above = compute_marginal_tax_rate("ON", 58_524.0)
    assert at_threshold < just_above  # federal rate ticks up crossing the boundary


def test_compute_marginal_tax_rate_override_takes_precedence():
    """Override wins even with a clearly-inconsistent income figure -- the
    override exists specifically for someone who knows their real rate; the
    bracket calculation is the fallback, never consulted when one is given."""
    assert compute_marginal_tax_rate("ON", 999_999.0, override_pct=25.0) == 25.0


def test_compute_marginal_tax_rate_override_works_with_no_province_or_income():
    assert compute_marginal_tax_rate(None, None, override_pct=40.0) == 40.0


def test_compute_marginal_tax_rate_none_for_non_ontario_province():
    """Real, current data only exists for Ontario (v1 scope) -- a non-ON
    province is genuinely absent, not silently computed against Ontario's
    brackets."""
    assert compute_marginal_tax_rate("BC", 95_000.0) is None


def test_compute_marginal_tax_rate_none_when_province_missing():
    assert compute_marginal_tax_rate(None, 95_000.0) is None


def test_compute_marginal_tax_rate_none_when_income_missing():
    assert compute_marginal_tax_rate("ON", None) is None


# --- MARG token rendering (86bc8efvb) ---


def test_build_precomputed_tax_metrics_marg_present_with_real_profile():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "tfsa", bundle, user_tax_profile={"province": "ON", "income_annual": 95_000.0}
    )
    assert "MARG:" in block
    assert "combined federal+Ontario marginal rate" in block


def test_build_precomputed_tax_metrics_marg_absent_when_user_tax_profile_none():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics("AAPL", "tfsa", bundle)
    assert "MARG:" not in block


def test_build_precomputed_tax_metrics_marg_absent_when_profile_present_but_incomplete():
    """user_tax_profile itself being supplied doesn't guarantee MARG renders
    -- compute_marginal_tax_rate's own absence rules still apply (e.g. income
    missing)."""
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "tfsa", bundle, user_tax_profile={"province": "ON"}
    )
    assert "MARG:" not in block


def test_build_precomputed_tax_metrics_marg_uses_override():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "tfsa", bundle, user_tax_profile={"marginal_tax_rate_override_pct": 33.5}
    )
    assert "MARG: 33.50%" in block


# --- ROOM staleness (86bc8efvb) ---


def test_build_precomputed_tax_metrics_room_fresh_no_staleness_suffix():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    now = datetime(2026, 9, 28)
    block = build_precomputed_tax_metrics(
        "AAPL",
        "tfsa",
        bundle,
        account_state={
            "tfsa_room_remaining_cents": 750_000,
            "tfsa_room_as_of": now - timedelta(days=30),
        },
        now=now,
    )
    assert "ROOM: TFSA remaining $7,500" in block
    assert "(stale" not in block


def test_build_precomputed_tax_metrics_room_stale_gets_suffix():
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    now = datetime(2026, 9, 28)
    block = build_precomputed_tax_metrics(
        "AAPL",
        "tfsa",
        bundle,
        account_state={
            "tfsa_room_remaining_cents": 750_000,
            "tfsa_room_as_of": now - timedelta(days=430),
        },
        now=now,
    )
    assert "ROOM: TFSA remaining $7,500 (stale, last updated 430 days ago)" in block


def test_build_precomputed_tax_metrics_room_missing_as_of_not_treated_as_stale():
    """A missing as_of shouldn't itself manufacture a staleness caveat -- the
    ROOM value's own presence/absence already covers "no data at all"."""
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    block = build_precomputed_tax_metrics(
        "AAPL", "tfsa", bundle, account_state={"tfsa_room_remaining_cents": 750_000}
    )
    assert "ROOM: TFSA remaining $7,500" in block
    assert "(stale" not in block


def test_build_precomputed_tax_metrics_room_exactly_at_threshold_not_stale():
    """age_days > _ROOM_STALE_DAYS (strictly greater), so exactly 400 days
    reads as fresh, not stale -- the boundary itself, not an approximation."""
    bundle = _fake_bundle(_company_info(name="Apple Inc.", country="US"), [], 332.0)
    now = datetime(2026, 9, 28)
    block = build_precomputed_tax_metrics(
        "AAPL",
        "rrsp",
        bundle,
        account_state={
            "rrsp_room_remaining_cents": 3_000_000,
            "rrsp_room_as_of": now - timedelta(days=400),
        },
        now=now,
    )
    assert "(stale" not in block

# --- WHT-labelled after-tax lines, ALTERNATIVES and both contribution rooms (Tax Strategist Wave 2) ---


def _us_payer(price: float = 100.0, quarterly: float = 0.6) -> SimpleNamespace:
    """2.4% yield at the defaults: 4 x 0.6 over 100."""
    return _fake_bundle(
        _company_info(name="Coca-Cola Company", industry="Beverages", country="US"),
        [_div_record(_days_ago(d * 90 + 10), quarterly) for d in range(4)],
        price,
    )


def test_the_tax_lines_use_the_taxcost_token_not_unlabelled_line_names():
    """The old `Effective after-tax yield:` / `Annual tax drag:` lines carried no token name, so the
    model cited the line label, which the validator does not accept; and the later WHT-labelled
    versions counted withholding only (wrong in a Trading account)."""
    block = build_precomputed_tax_metrics("KO", "tfsa", _us_payer())
    assert (
        "TAXCOST (this account, tfsa): annual dividend tax 0.36% of the holding (15% withholding, "
        "non-recoverable); yield after annual dividend tax 2.04%"
    ) in block
    assert "TAXCOST verdict: fit=good, efficiency=favorable" in block
    assert "WHT: effective after-tax yield" not in block and "WHT: annual tax drag" not in block
    assert "Effective after-tax yield:" not in block and "Annual tax drag:" not in block


_PROFILE = {"province": "ON", "income_annual": 85000.0}


def test_the_block_assesses_this_account_only_and_never_compares_accounts():
    for account in ("tfsa", "rrsp", "trading"):
        block = build_precomputed_tax_metrics("KO", account, _us_payer(), user_tax_profile=_PROFILE)
        assert "ALTERNATIVES" not in block and "better_account" not in block and "cross_account" not in block
        assert sum(line.startswith("TAXCOST (this account") for line in block.splitlines()) == 1


def test_us_dividend_in_trading_is_taxed_as_income():
    block = build_precomputed_tax_metrics("KO", "trading", _us_payer(), user_tax_profile=_PROFILE)
    # 2.4% x 29.65%: the 15% withholding is credited against the Canadian tax on the dividend
    assert "annual dividend tax 0.71% of the holding (ordinary income at 29.65%; the 15% withholding is credited" in block
    assert "yield after annual dividend tax 1.69%" in block
    assert "TAXCOST verdict: fit=fair, efficiency=neutral" in block


def test_us_dividend_in_an_rrsp_has_no_annual_tax():
    block = build_precomputed_tax_metrics("KO", "rrsp", _us_payer(), user_tax_profile=_PROFILE)
    assert "annual dividend tax 0.00% of the holding (no annual tax; later withdrawals taxed as ordinary income, not scored)" in block
    assert "TAXCOST verdict: fit=excellent, efficiency=favorable" in block


def _td_bundle():
    return _fake_bundle(
        _company_info(name="Toronto-Dominion Bank", industry="Banking"),
        [_div_record(_days_ago(d * 90 + 10), 1.05) for d in range(4)],
        165.0,
    )


def test_canadian_eligible_dividend_in_trading_is_taxed_after_the_credit():
    block = build_precomputed_tax_metrics("TD.TO", "trading", _td_bundle(), user_tax_profile=_PROFILE)
    # 4 x 1.05 / 165 = 2.545% yield x 6.39% eligible-dividend rate = 0.16
    assert "annual dividend tax 0.16% of the holding (6.39% after the dividend tax credit)" in block
    assert "TAXCOST verdict: fit=good, efficiency=favorable" in block


def test_canadian_dividend_in_trading_without_income_is_not_computed_with_fixed_defaults():
    block = build_precomputed_tax_metrics("TD.TO", "trading", _td_bundle())
    assert "TAXCOST verdict: not computed (the eligible-dividend rate needs the province and income" in block
    assert "use fit=fair, efficiency=neutral" in block
    assert "TAXCOST (this account" not in block


def test_a_class_outside_the_grid_is_not_computed():
    bundle = _fake_bundle(
        _company_info(name="Alibaba Group Holding Limited", industry="Specialty Retail", country="CN"),
        [_div_record(_days_ago(d * 90 + 10), 0.5) for d in range(4)],
        100.0,
    )
    block = build_precomputed_tax_metrics("BABA", "tfsa", bundle, user_tax_profile=_PROFILE)
    assert "TAXCOST verdict: not computed" in block


def test_no_dividend_has_the_best_fit_the_account_allows():
    bundle = _fake_bundle(_company_info(name="Shopify Inc", industry="Software"), [], 100.0)
    for account, fit, eff in (("tfsa", "excellent", "favorable"), ("rrsp", "excellent", "favorable"), ("trading", "good", "favorable")):
        block = build_precomputed_tax_metrics("SHOP", account, bundle, user_tax_profile=_PROFILE)
        assert f"TAXCOST verdict: fit={fit}, efficiency={eff}" in block
        assert "the annual dividend tax is 0.00% in this account" in block


# --- the model itself: numbers worked out from the published brackets, not from the code ---


def test_eligible_dividend_rate_matches_hand_calculation_and_the_published_top_rate():
    # $85,000: federal 20.5%, Ontario 9.15%, no surtax. 1.38 x (20.5 - 15.0198) + 1.38 x (9.15 - 10) = 6.39
    assert compute_eligible_dividend_tax_rate("ON", 85000.0) == 6.39
    # Top bracket: 1.38 x (33 - 15.0198) + 1.38 x (13.16 x 1.56 - 10) = 39.34, the published Ontario top
    # rate on eligible dividends.
    assert compute_eligible_dividend_tax_rate("ON", 300000.0) == 39.34
    # Low income: the credits exceed the tax; they are non-refundable, so the rate is floored at zero.
    assert compute_eligible_dividend_tax_rate("ON", 40000.0) == 0.0


@pytest.mark.parametrize(
    "province, income, override",
    [("ON", None, None), (None, 85000.0, None), ("BC", 85000.0, None), ("ON", 85000.0, 30.0)],
)
def test_eligible_dividend_rate_is_not_modelled_without_inputs_or_with_an_override(province, income, override):
    assert compute_eligible_dividend_tax_rate(province, income, override) is None


@pytest.mark.parametrize(
    "classification, account, marg, elig, expected",
    [
        ("us", "tfsa", 29.65, 6.39, 15.0),
        ("us", "rrsp", 29.65, 6.39, 0.0),
        ("us", "trading", 29.65, 6.39, 29.65),
        ("us", "trading", 10.0, 6.39, 15.0),  # the withholding exceeds the Canadian tax: 15% stays
        ("us", "trading", None, 6.39, None),
        ("canadian_eligible", "tfsa", None, None, 0.0),
        ("canadian_eligible", "rrsp", None, None, 0.0),
        ("canadian_eligible", "trading", 29.65, 6.39, 6.39),
        ("canadian_eligible", "trading", 29.65, None, None),
        ("trust_distribution", "tfsa", None, None, 0.0),
        ("trust_distribution", "trading", 29.65, 6.39, None),
        ("us_reit", "tfsa", 29.65, 6.39, None),
    ],
)
def test_dividend_tax_rate_by_class_and_account(classification, account, marg, elig, expected):
    result = dividend_tax_rate_pct(classification, account, marg, elig)
    assert (None if result is None else result[0]) == expected


_M, _E = 29.65, 6.39


def _verdict(account, classification, yield_pct):
    return compute_account_verdict(account, classification, yield_pct, _M, _E)


@pytest.mark.parametrize(
    "classification, yield_pct, account, cost, fit, efficiency",
    [
        ("us", 2.43, "tfsa", 0.36, "good", "favorable"),  # KO
        ("us", 2.43, "rrsp", 0.00, "excellent", "favorable"),
        ("us", 2.43, "trading", 0.72, "fair", "neutral"),
        ("us", 0.7, "tfsa", 0.10, "excellent", "favorable"),  # MSFT: 0.105 rounds into the top band
        ("us", 0.7, "trading", 0.21, "good", "favorable"),
        ("canadian_eligible", 5.8, "rrsp", 0.00, "excellent", "favorable"),  # ENB
        ("canadian_eligible", 5.8, "trading", 0.37, "good", "favorable"),
        ("canadian_eligible", 2.6, "trading", 0.17, "good", "favorable"),  # TD
        ("us", 6.0, "trading", 1.78, "poor", "unfavorable"),
    ],
)
def test_verdict_for_representative_holdings(classification, yield_pct, account, cost, fit, efficiency):
    v = _verdict(account, classification, yield_pct)
    assert (v["mode"], v["this_cost"], v["fit"], v["efficiency"]) == ("computed", cost, fit, efficiency)


def test_trading_is_never_excellent():
    assert _verdict("trading", "us", 0.05)["fit"] == "good"  # a cost of 0.01, capped: Trading shelters nothing


def test_not_computed_defaults_are_fixed_not_judged():
    v = compute_account_verdict("trading", "us", 2.43, None, None)
    assert (v["mode"], v["fit"], v["efficiency"]) == ("not_computed", "fair", "neutral")
    assert "marginal rate" in v["reason"]
    v = compute_account_verdict("tfsa", "us_reit", 4.0, _M, _E)
    assert v["mode"] == "not_computed" and "outside the modelled combinations" in v["reason"]


def test_a_registered_account_is_computed_without_income_on_file():
    v = compute_account_verdict("tfsa", "us", 2.43, None, None)
    assert v["mode"] == "computed" and v["this_cost"] == 0.36 and v["fit"] == "good"


@pytest.mark.parametrize(
    "cost, label",
    [(0.0, "excellent"), (0.10, "excellent"), (0.11, "good"), (0.50, "good"), (0.51, "fair"), (1.50, "fair"), (1.51, "poor")],
)
def test_fit_band_boundaries(cost, label):
    assert _fit_label(cost) == label


def test_efficiency_summarises_the_fit():
    assert _EFFICIENCY_BY_FIT == {"excellent": "favorable", "good": "favorable", "fair": "neutral", "poor": "unfavorable"}


def test_room_shows_both_registered_rooms_with_the_analysed_account_first():
    bundle = _us_payer()
    state = {"tfsa_room_remaining_cents": 3_200_000, "rrsp_room_remaining_cents": 2_800_000}
    assert "ROOM: TFSA remaining $32,000 | RRSP remaining $28,000" in (
        build_precomputed_tax_metrics("KO", "tfsa", bundle, account_state=state)
    )
    assert "ROOM: RRSP remaining $28,000 | TFSA remaining $32,000" in (
        build_precomputed_tax_metrics("KO", "rrsp", bundle, account_state=state)
    )


def test_room_is_shown_for_a_trading_analysis_so_a_move_to_a_registered_account_can_be_sized():
    state = {"tfsa_room_remaining_cents": 3_200_000, "rrsp_room_remaining_cents": 2_800_000}
    block = build_precomputed_tax_metrics("KO", "trading", _us_payer(), account_state=state)
    assert "ROOM: TFSA remaining $32,000 | RRSP remaining $28,000" in block


def test_room_shows_only_the_rooms_on_file():
    block = build_precomputed_tax_metrics(
        "KO", "trading", _us_payer(), account_state={"rrsp_room_remaining_cents": 2_800_000}
    )
    assert "ROOM: RRSP remaining $28,000" in block
    assert "TFSA remaining" not in block


def test_room_staleness_is_flagged_per_account_not_for_the_whole_line():
    now = datetime(2026, 9, 28)
    block = build_precomputed_tax_metrics(
        "KO",
        "tfsa",
        _us_payer(),
        account_state={
            "tfsa_room_remaining_cents": 3_200_000,
            "tfsa_room_as_of": now - timedelta(days=30),
            "rrsp_room_remaining_cents": 2_800_000,
            "rrsp_room_as_of": now - timedelta(days=430),
        },
        now=now,
    )
    assert "ROOM: TFSA remaining $32,000 | RRSP remaining $28,000 (stale, last updated 430 days ago)" in block
