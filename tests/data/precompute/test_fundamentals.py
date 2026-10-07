import pytest

from data.precompute.fundamentals import (
    metric_group,
    _peg,
    _ttm_metric,
    compute_all,
    compute_balance_sheet_metrics,
    compute_dividend_info,
    compute_growth_metrics,
    compute_profitability_metrics,
    compute_valuation_metrics,
)
from data.providers.base import NormalizedDividendRecord, NormalizedFinancials, NormalizedQuote


def _quarter(revenue, net_income, eps, operating_income=None, **overrides):
    row = {
        "period_end": "2026-06-30",
        "revenue": revenue,
        "net_income": net_income,
        "eps": eps,
        "operating_income": operating_income,
        "interest_expense": 10.0,
        "tax_expense": 40.0,
        "cost_of_revenue": revenue - 400 if revenue else None,
        "depreciation_amortization": 50.0,
        "dividends_paid": None,
        "shares_outstanding": 100.0,
        "operating_cash_flow": None,
        "capital_expenditures": None,
    }
    row.update(overrides)
    return row


# 5 quarters, newest first — revenue/net_income/eps trending up, enough for
# a 4-quarter TTM sum and a quarters[4] YoY margin_trend comparison. capex is
# negative (a cash outflow), matching both providers' real sign.
_QUARTERS = [
    _quarter(1000.0, 150.0, 1.5, 200.0, operating_cash_flow=180.0, capital_expenditures=-40.0),
    _quarter(950.0, 140.0, 1.4, 190.0, operating_cash_flow=170.0, capital_expenditures=-38.0),
    _quarter(900.0, 130.0, 1.3, 180.0, operating_cash_flow=160.0, capital_expenditures=-36.0),
    _quarter(880.0, 120.0, 1.2, 170.0, operating_cash_flow=150.0, capital_expenditures=-34.0),
    _quarter(800.0, 100.0, 1.0, 150.0, operating_cash_flow=140.0, capital_expenditures=-30.0),
]

# 4 years, newest first — for revenue_growth_yoy (annual[0] vs [1]) and
# revenue_growth_3yr_cagr (annual[0] vs [3]).
_ANNUAL = [
    _quarter(3700.0, 500.0, 5.3, period_end="2025-12-31"),
    _quarter(3200.0, 420.0, 4.5, period_end="2024-12-31"),
    _quarter(2900.0, 380.0, 4.0, period_end="2023-12-31"),
    _quarter(2500.0, 320.0, 3.5, period_end="2022-12-31"),
]

_BALANCE_SHEET = {
    "total_assets": 5000.0,
    "total_liabilities": 3000.0,
    "total_equity": 2000.0,
    "total_debt": 1000.0,
    "cash_and_equivalents": 500.0,
    "current_assets": 1500.0,
    "current_liabilities": 700.0,
}


def _fin(**overrides) -> NormalizedFinancials:
    defaults = dict(
        quarters=_QUARTERS, annual=_ANNUAL, balance_sheet=_BALANCE_SHEET, currency="USD"
    )
    defaults.update(overrides)
    return NormalizedFinancials(**defaults)


def _price_info(**overrides) -> NormalizedQuote:
    defaults = NormalizedQuote(
        current_price=50.0, market_cap=5000.0, currency="USD", high_52w=60.0, low_52w=40.0
    )
    defaults.update(overrides)
    return defaults


def _dividend_record(ex_date: str, amount: float) -> NormalizedDividendRecord:
    return NormalizedDividendRecord(ex_date=ex_date, payment_date=None, amount_per_share=amount)


def _dividend_history() -> list[NormalizedDividendRecord]:
    # 6 years x 4 quarterly payments, amount increasing slightly each year —
    # enough for dividend_growth_5yr (needs >5 distinct years) and a clean
    # consecutive_years_paid count.
    records = []
    for year, amount in zip(range(2020, 2026), [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]):
        for month in ("02", "05", "08", "11"):
            records.append(_dividend_record(f"{year}-{month}-15", amount))
    return records


# --- compute_growth_metrics ---


def test_compute_growth_metrics_happy_path():
    result = compute_growth_metrics(_fin())

    assert result["revenue_growth_yoy"] == pytest.approx((3700.0 - 3200.0) / 3200.0)
    assert result["revenue_growth_3yr_cagr"] == pytest.approx((3700.0 / 2500.0) ** (1 / 3) - 1)
    assert result["eps_growth_yoy"] == pytest.approx((5.3 - 4.5) / 4.5)
    assert result["earnings_surprises"] is None
    # forward_pe is a VAL-block field (compute_valuation_metrics), not GROWTH —
    # real bug caught on review, was briefly duplicated in both dicts.
    assert "forward_pe" not in result


# --- _ttm_metric (86bbxuj9e) ---


def test_ttm_metric_prefers_explicit_ttm():
    """A provider-supplied fin.ttm wins over summing quarters (edgartools,
    whose quarterly history is too shallow to sum)."""
    fin = _fin(ttm={"net_income": 999.0})
    assert _ttm_metric(fin, "net_income") == 999.0


def test_ttm_metric_falls_back_to_quarter_sum_when_no_ttm():
    """yfinance / CA path: fin.ttm is None, sum the trailing 4 quarters —
    identical to the old _ttm(fin.quarters, field)."""
    fin = _fin(ttm=None)
    assert _ttm_metric(fin, "net_income") == pytest.approx(150.0 + 140.0 + 130.0 + 120.0)


def test_ttm_metric_falls_back_when_field_absent_from_ttm():
    """fin.ttm exists but lacks the field (e.g. 'eps' is never in it) — fall
    through to the quarter sum, which is None on a 2-quarter US frame."""
    fin = _fin(ttm={"net_income": 500.0}, quarters=_QUARTERS[:2])
    assert _ttm_metric(fin, "eps") is None


def test_compute_growth_metrics_insufficient_annual_data_returns_none():
    result = compute_growth_metrics(_fin(annual=[_ANNUAL[0]]))

    assert result["revenue_growth_yoy"] is None
    assert result["revenue_growth_3yr_cagr"] is None


def test_compute_growth_metrics_earnings_surprises_passed_through():
    """86bbdu04a: passed through directly, not collapsed to None when
    empty — None means no data source wired, [] means fetched-but-empty,
    and compute_all()'s missing_fields scan only flags None as missing."""
    surprises = [{"period_end": "2026-07-30", "eps_actual": 2.02, "eps_estimated": 1.89}]

    result = compute_growth_metrics(_fin(), earnings_surprises=surprises)

    assert result["earnings_surprises"] == surprises


def test_compute_growth_metrics_empty_earnings_surprises_stays_empty_list_not_none():
    result = compute_growth_metrics(_fin(), earnings_surprises=[])

    assert result["earnings_surprises"] == []


def test_compute_growth_metrics_cagr_none_when_base_year_non_positive():
    annual = [
        _ANNUAL[0],
        _ANNUAL[1],
        _ANNUAL[2],
        _quarter(-100.0, 0.0, 0.0, period_end="2022-12-31"),
    ]
    result = compute_growth_metrics(_fin(annual=annual))

    assert result["revenue_growth_3yr_cagr"] is None


# --- compute_profitability_metrics ---


def test_compute_profitability_metrics_happy_path():
    result = compute_profitability_metrics(_fin())

    # margins are trailing-twelve-month (86bbxuj9e), not one quarter
    ttm_rev = 1000.0 + 950.0 + 900.0 + 880.0  # 3730
    ttm_cost = 600.0 + 550.0 + 500.0 + 480.0  # cost_of_revenue = revenue - 400
    ttm_op = 200.0 + 190.0 + 180.0 + 170.0  # 740
    ttm_net_income = 150.0 + 140.0 + 130.0 + 120.0  # 540
    assert result["gross_margin"] == pytest.approx((ttm_rev - ttm_cost) / ttm_rev)
    assert result["operating_margin"] == pytest.approx(ttm_op / ttm_rev)
    assert result["net_margin"] == pytest.approx(ttm_net_income / ttm_rev)
    assert result["roe"] == pytest.approx(ttm_net_income / 2000.0)
    # TTM FCF = ttm(ocf) - |ttm(capex)| = 660 - 148 = 512; over TTM net income
    ttm_fcf = (180.0 + 170.0 + 160.0 + 150.0) - (40.0 + 38.0 + 36.0 + 34.0)
    assert result["fcf_to_net_income"] == pytest.approx(ttm_fcf / ttm_net_income)
    # latest net_margin 0.15 vs quarters[4] net_margin 100/800=0.125 -> +2.5pp -> expanding
    assert result["margin_trend"] == "expanding"


def test_compute_profitability_metrics_margins_from_ttm_on_thin_quarters():
    """US / edgartools shape: 2 quarters + a provider ttm dict. Margins still
    resolve (from ttm), and are the trailing-twelve-month figure."""
    fin = _fin(
        quarters=_QUARTERS[:2],
        ttm={
            "revenue": 4000.0,
            "cost_of_revenue": 2400.0,
            "operating_income": 800.0,
            "net_income": 500.0,
        },
    )
    result = compute_profitability_metrics(fin)

    assert result["gross_margin"] == pytest.approx((4000.0 - 2400.0) / 4000.0)
    assert result["operating_margin"] == pytest.approx(800.0 / 4000.0)
    assert result["net_margin"] == pytest.approx(500.0 / 4000.0)


def test_compute_profitability_metrics_roe_none_when_book_equity_negative():
    """Negative book equity makes ROE meaningless — a net loss over negative
    equity reads as a positive ROE (86bbq04wm)."""
    result = compute_profitability_metrics(
        _fin(balance_sheet={**_BALANCE_SHEET, "total_equity": -500.0})
    )

    assert result["roe"] is None


def test_compute_profitability_metrics_empty_quarters_returns_all_none():
    result = compute_profitability_metrics(_fin(quarters=[]))

    assert result == {
        "gross_margin": None,
        "operating_margin": None,
        "net_margin": None,
        "roe": None,
        "margin_trend": None,
        "fcf_to_net_income": None,
    }


def test_margin_trend_contracting():
    quarters = [
        _quarter(1000.0, 50.0, 0.5),  # net_margin 0.05
        _quarter(1000.0, 50.0, 0.5),
        _quarter(1000.0, 50.0, 0.5),
        _quarter(1000.0, 50.0, 0.5),
        _quarter(1000.0, 150.0, 1.5),  # 4 quarters back: net_margin 0.15
    ]
    result = compute_profitability_metrics(_fin(quarters=quarters))

    assert result["margin_trend"] == "contracting"


def test_margin_trend_stable_within_deadband():
    quarters = [_quarter(1000.0, 150.0, 1.5), _quarter(1000.0, 149.0, 1.49)]
    result = compute_profitability_metrics(_fin(quarters=quarters))

    assert result["margin_trend"] == "stable"


# --- compute_balance_sheet_metrics ---


def test_compute_balance_sheet_metrics_happy_path():
    result = compute_balance_sheet_metrics(_fin())

    assert result == {
        "debt_to_equity": pytest.approx(0.5),
        "current_ratio": pytest.approx(1500.0 / 700.0),
        "interest_coverage": pytest.approx(20.0),
        # TTM: ttm(ocf) - |ttm(capex)| = 660 - 148
        "free_cash_flow": pytest.approx(512.0),
        "cash_position": 500.0,
    }
    assert "health_rating" not in result


def test_compute_balance_sheet_metrics_missing_interest_expense_omits_only_coverage():
    quarters = [_quarter(1000.0, 150.0, 1.5, 200.0, interest_expense=None)]
    result = compute_balance_sheet_metrics(_fin(quarters=quarters, annual=[]))

    assert result["interest_coverage"] is None
    assert result["debt_to_equity"] == pytest.approx(0.5)  # unaffected


def test_compute_balance_sheet_metrics_missing_total_debt_omits_only_de():
    result = compute_balance_sheet_metrics(
        _fin(balance_sheet={**_BALANCE_SHEET, "total_debt": None})
    )

    assert result["debt_to_equity"] is None
    assert result["current_ratio"] is not None  # unaffected


# --- compute_valuation_metrics ---


def test_compute_valuation_metrics_happy_path():
    growth = compute_growth_metrics(_fin())
    result = compute_valuation_metrics(_fin(), _price_info(), growth)

    ttm_eps = 1.5 + 1.4 + 1.3 + 1.2
    assert result["pe_ratio"] == pytest.approx(50.0 / ttm_eps)
    assert result["pb_ratio"] == pytest.approx(50.0 / (2000.0 / 100.0))
    ttm_revenue = 1000.0 + 950.0 + 900.0 + 880.0
    assert result["ps_ratio"] == pytest.approx(5000.0 / ttm_revenue)
    # EBITDA is TTM: ttm(operating_income) + ttm(D&A) = 740 + 200 = 940
    ttm_ebitda = (200.0 + 190.0 + 180.0 + 170.0) + (50.0 * 4)
    assert result["ev_ebitda"] == pytest.approx((5000.0 + 1000.0 - 500.0) / ttm_ebitda)
    assert result["forward_pe"] is None
    assert result["peg_ratio"] == pytest.approx(
        result["pe_ratio"] / (growth["eps_growth_yoy"] * 100)
    )


def test_peg_none_when_growth_is_a_base_year_artifact():
    """A prior-year EPS near zero (impairment year, cyclical trough) yields a
    multi-hundred-percent YoY that collapses PEG toward a meaningless ~0.
    _peg caps the growth input rather than emitting the garbage ratio."""
    assert _peg(4.8, 36.7) is None  # BCE.TO-shaped: P/E ~4.8, "growth" ~3670%
    assert _peg(20.0, 0.25) == pytest.approx(20.0 / 25.0)  # 25% growth still fine
    assert _peg(20.0, 1.0) == pytest.approx(20.0 / 100.0)  # exactly 100% is the boundary, kept


def test_compute_valuation_metrics_forward_pe_with_analyst_estimates():
    """86bbdu04a: forward_pe mirrors pe_ratio's own guard style — current
    price divided by an EPS-like denominator, None if either is missing."""
    growth = compute_growth_metrics(_fin())
    analyst_estimates = {"forward_eps": 5.0}

    result = compute_valuation_metrics(_fin(), _price_info(), growth, analyst_estimates)

    assert result["forward_pe"] == pytest.approx(50.0 / 5.0)


def test_compute_valuation_metrics_forward_pe_none_when_forward_eps_missing():
    growth = compute_growth_metrics(_fin())
    analyst_estimates = {"forward_eps": None}

    result = compute_valuation_metrics(_fin(), _price_info(), growth, analyst_estimates)

    assert result["forward_pe"] is None


def test_compute_valuation_metrics_forward_pe_none_with_bare_empty_dict():
    """Real providers now return a bare {} on no data (fmp.py/openbb_tmx.py/
    yfinance.py, 86bbdu04a review) — not {"forward_eps": None} — matching
    get_company_info's/get_quote's own convention. Confirm this actual
    real-world shape degrades gracefully too, not just the {"forward_eps":
    None} shape."""
    growth = compute_growth_metrics(_fin())

    result = compute_valuation_metrics(_fin(), _price_info(), growth, {})

    assert result["forward_pe"] is None


def test_compute_valuation_metrics_thin_quarters_pe_ratio_none_not_approximated():
    """Real design decision: _ttm() returns None below 4 quarters rather
    than approximating from fewer — matches _cagr()'s own precedent."""
    growth = compute_growth_metrics(_fin())
    thin_fin = _fin(quarters=_QUARTERS[:2])

    result = compute_valuation_metrics(thin_fin, _price_info(), growth)

    assert result["pe_ratio"] is None
    assert result["ps_ratio"] is None


def test_compute_valuation_metrics_uses_explicit_ttm_on_thin_quarters():
    """The US / edgartools shape: only 2 quarters, but a provider-supplied
    fin.ttm carries the trailing-twelve-month aggregates (86bbxuj9e). P/E
    (via net income), P/S and PEG all resolve where the 4-quarter sum can't."""
    growth = compute_growth_metrics(_fin())
    fin = _fin(quarters=_QUARTERS[:2], ttm={"revenue": 3800.0, "net_income": 560.0})

    result = compute_valuation_metrics(fin, _price_info(), growth)

    assert result["pe_ratio"] == pytest.approx(5000.0 / 560.0)  # market_cap / ttm net income
    assert result["ps_ratio"] == pytest.approx(5000.0 / 3800.0)
    assert result["peg_ratio"] is not None


def test_ev_ebitda_always_computed_no_sector_conditional():
    """ev_ebitda is computed unconditionally here — sector-based suppression
    is the payload-template's job, not this module's."""
    growth = compute_growth_metrics(_fin())
    result = compute_valuation_metrics(_fin(), _price_info(), growth)

    assert result["ev_ebitda"] is not None


# yfinance's per-quarter Diluted EPS is sporadically NaN for essentially every
# Canadian filer, so _ttm("eps") is None sector-wide. pe_ratio then derives from
# market_cap / TTM net income to common (falling back to total net income).
_QUARTERS_NO_EPS = [
    _quarter(1000.0, 150.0, None, 200.0, net_income_common=140.0),
    _quarter(950.0, 140.0, None, 190.0, net_income_common=130.0),
    _quarter(900.0, 130.0, None, 180.0, net_income_common=120.0),
    _quarter(880.0, 120.0, None, 170.0, net_income_common=110.0),
]


def test_compute_valuation_metrics_pe_ratio_derives_from_net_income_common_when_eps_nan():
    growth = compute_growth_metrics(_fin())
    fin = _fin(quarters=_QUARTERS_NO_EPS)

    result = compute_valuation_metrics(fin, _price_info(), growth)

    ttm_nic = 140.0 + 130.0 + 120.0 + 110.0  # 500.0
    assert result["pe_ratio"] == pytest.approx(5000.0 / ttm_nic)


def test_compute_valuation_metrics_pe_ratio_falls_back_to_total_net_income():
    """A filer that doesn't disclose the common split — net_income_common
    absent — uses total net_income."""
    growth = compute_growth_metrics(_fin())
    quarters = [
        _quarter(q["revenue"], q["net_income"], None, q["operating_income"])
        for q in _QUARTERS_NO_EPS
    ]

    result = compute_valuation_metrics(_fin(quarters=quarters), _price_info(), growth)

    ttm_ni = 150.0 + 140.0 + 130.0 + 120.0  # 540.0
    assert result["pe_ratio"] == pytest.approx(5000.0 / ttm_ni)


def test_compute_valuation_metrics_pe_ratio_none_on_ttm_loss():
    """A real trailing loss leaves pe_ratio None — same as a negative real-EPS
    P/E — rather than emitting a negative multiple."""
    growth = compute_growth_metrics(_fin())
    quarters = [_quarter(1000.0, -50.0, None, -40.0, net_income_common=-55.0) for _ in range(4)]

    result = compute_valuation_metrics(_fin(quarters=quarters), _price_info(), growth)

    assert result["pe_ratio"] is None
    assert result["peg_ratio"] is None


def test_compute_valuation_metrics_peg_ratio_uses_derived_pe_ratio():
    growth = compute_growth_metrics(_fin())
    assert growth["eps_growth_yoy"] > 0  # _ANNUAL trends up
    fin = _fin(quarters=_QUARTERS_NO_EPS)

    result = compute_valuation_metrics(fin, _price_info(), growth)

    expected_pe = 5000.0 / 500.0
    assert result["peg_ratio"] == pytest.approx(expected_pe / (growth["eps_growth_yoy"] * 100))


# --- compute_dividend_info ---


def test_compute_dividend_info_happy_path():
    result = compute_dividend_info(_fin(), _price_info(), _dividend_history())

    trailing_annual_dividend = 0.20 * 4
    assert result["dividend_yield"] == pytest.approx(trailing_annual_dividend / 50.0)
    ttm_eps = 1.5 + 1.4 + 1.3 + 1.2
    assert result["payout_ratio"] == pytest.approx(trailing_annual_dividend / ttm_eps)
    assert result["dividend_growth_5yr"] == pytest.approx((0.80 / 0.60) ** (1 / 5) - 1)
    assert result["dividend_regularity"] == "regular"
    assert result["consecutive_years_paid"] == 6


def test_compute_dividend_info_payout_ratio_derives_from_net_income_when_eps_nan():
    """CA filers hit the same yfinance quarterly-EPS-NaN problem here — payout
    falls back to abs(TTM cash dividends paid) / TTM total net income."""
    quarters = [_quarter(1000.0, 100.0, None, 200.0, dividends_paid=-30.0) for _ in range(4)]

    result = compute_dividend_info(_fin(quarters=quarters), _price_info(), _dividend_history())

    assert result["payout_ratio"] == pytest.approx(120.0 / 400.0)


def test_compute_dividend_info_payout_ratio_none_when_no_fallback_inputs():
    """eps NaN and no dividends_paid on the statements — payout stays None
    rather than guessing."""
    quarters = [_quarter(1000.0, 100.0, None, 200.0) for _ in range(4)]

    result = compute_dividend_info(_fin(quarters=quarters), _price_info(), _dividend_history())

    assert result["payout_ratio"] is None


def test_compute_dividend_info_no_history_returns_none_regularity():
    result = compute_dividend_info(_fin(), _price_info(), [])

    assert result == {
        "dividend_yield": None,
        "payout_ratio": None,
        "dividend_growth_5yr": None,
        "dividend_regularity": "none",
        "consecutive_years_paid": 0,
    }


def test_compute_dividend_info_irregular_regularity_from_special_dividend():
    history = [_dividend_record(f"2025-{m:02d}-15", 0.20) for m in (2, 5, 8, 11)]
    history.append(_dividend_record("2025-12-01", 5.00))  # special dividend spike

    result = compute_dividend_info(_fin(), _price_info(), history)

    assert result["dividend_regularity"] == "irregular"


def test_compute_dividend_info_consecutive_years_paid_stops_at_gap():
    history = [
        _dividend_record("2025-06-15", 0.20),
        _dividend_record("2023-06-15", 0.18),
    ]  # gap: no 2024

    result = compute_dividend_info(_fin(), _price_info(), history)

    assert result["consecutive_years_paid"] == 1


def test_compute_dividend_info_partial_current_year_does_not_distort_growth():
    """Real bug caught testing through the real code path against live
    AAPL data: comparing the most recent CALENDAR year's total (which is
    very likely partial — however many payments have happened so far this
    year) against a complete prior year systematically understates growth,
    even producing a false negative rate. trailing_annual_dividend (a
    genuine trailing-12-month total) must be used as the "now" figure
    instead of by_year[latest_year]."""
    history = [
        _dividend_record(f"{year}-{m:02d}-15", 0.20)
        for year in range(2020, 2026)
        for m in (2, 5, 8, 11)
    ]
    # 2026: only 3 of 4 payments so far (partial year) — would show as a
    # decline if compared directly against a complete 2025.
    history += [_dividend_record(f"2026-{m:02d}-15", 0.20) for m in (2, 5, 8)]

    result = compute_dividend_info(_fin(), _price_info(), history)

    # trailing 4 payments = last 4 chronologically = 2025-11, 2026-02,
    # 2026-05, 2026-08 = 0.80; reference year 2026-5=2021 total = 0.80.
    # Real, correctly-flat growth, not a fabricated decline.
    assert result["dividend_growth_5yr"] == pytest.approx(0.0)


def test_compute_dividend_info_unsorted_input_handled_correctly():
    """dividend_history ordering isn't guaranteed by the caller — this
    must sort internally rather than assume order."""
    shuffled = list(reversed(_dividend_history()))

    result = compute_dividend_info(_fin(), _price_info(), shuffled)

    assert result["consecutive_years_paid"] == 6
    assert result["dividend_regularity"] == "regular"


# --- compute_all ---


def _fin_scaled(factor: float, currency: str):
    """_fin() with every monetary amount multiplied by `factor`, as a company reporting in `currency` would show."""
    from data.precompute.currency import convert_financials

    return convert_financials(_fin(currency="USD"), factor, currency)



def test_compute_all_happy_path_returns_full_shape():
    result = compute_all(_fin(), _price_info(), _dividend_history())

    assert set(result.keys()) == {
        "valuation_metrics",
        "growth_metrics",
        "profitability_metrics",
        "balance_sheet_metrics",
        "dividend_info",
        "quarters_available",
        "missing_fields",
        "currency_mismatch",
        "not_applicable",
        "metric_profile",
        "latest_financials_period_end",
    }
    assert result["quarters_available"] == 5
    assert result["valuation_metrics"]["pe_ratio"] is not None
    assert result["dividend_info"]["dividend_yield"] is not None
    assert result["currency_mismatch"] is None
    assert result["latest_financials_period_end"] == "2026-06-30"  # newest quarter, _QUARTERS[0]


def test_compute_all_converts_a_usd_reporter_on_a_cad_quote_so_the_multiples_are_right():
    """SHOP.TO 2026-10-02: USD statements, CAD quote. P/E was 144 against a true 103 because a CAD market cap was
    divided by a USD profit (BB-021). With the Bank of Canada rate the statements are converted first, so the result
    equals a company that reported the same numbers in CAD."""
    rate = 1.4
    usd_result = compute_all(_fin(currency="USD"), _price_info(currency="CAD"), [], usd_cad=rate)
    cad_equivalent = compute_all(_fin_scaled(rate, "CAD"), _price_info(currency="CAD"), [])

    assert usd_result["currency_mismatch"] == {
        "financials_currency": "USD", "quote_currency": "CAD", "converted": True, "usd_cad": rate,
    }
    for key in ("pe_ratio", "pb_ratio", "ps_ratio"):
        assert usd_result["valuation_metrics"][key] == pytest.approx(cad_equivalent["valuation_metrics"][key])
    # the distortion removed is exactly the exchange rate: the naive P/E (USD profit read as CAD) divided by the rate
    naive = compute_all(_fin(currency="CAD"), _price_info(currency="CAD"), [])
    assert usd_result["valuation_metrics"]["pe_ratio"] == pytest.approx(naive["valuation_metrics"]["pe_ratio"] / rate)
    assert usd_result["valuation_metrics"]["pb_ratio"] == pytest.approx(naive["valuation_metrics"]["pb_ratio"] / rate)


def test_compute_all_drops_price_based_multiples_when_a_mismatch_cannot_be_converted():
    result = compute_all(_fin(currency="USD"), _price_info(currency="CAD"), [], usd_cad=None)

    assert result["currency_mismatch"]["converted"] is False
    for key in ("pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "peg_ratio"):
        assert result["valuation_metrics"][key] is None
    assert result["dividend_info"]["payout_ratio"] is None
    # ratios that do not divide a price by a statement amount are still there
    assert result["profitability_metrics"]["net_margin"] is not None


def test_compute_all_missing_fields_lists_none_valued_keys():
    thin_fin = _fin(quarters=_QUARTERS[:1], annual=[])
    result = compute_all(thin_fin, _price_info(), [])

    assert "valuation_metrics.pe_ratio" in result["missing_fields"]
    assert "growth_metrics.revenue_growth_yoy" in result["missing_fields"]
    # No analyst_estimates/earnings_surprises passed -> both stay None,
    # correctly flagged as missing (86bbdu04a).
    assert "valuation_metrics.forward_pe" in result["missing_fields"]


def test_compute_all_latest_financials_period_end_none_when_no_quarters():
    """86bbummwp Tier 2 -- nothing to report a period_end from when the
    company has zero reported quarters at all."""
    empty_fin = _fin(quarters=[], annual=[])
    result = compute_all(empty_fin, _price_info(), [])
    assert result["latest_financials_period_end"] is None


def test_compute_all_wires_analyst_estimates_and_earnings_surprises():
    """86bbdu04a: end-to-end through compute_all(), not just the two
    compute_* functions in isolation."""
    analyst_estimates = {"forward_eps": 5.0}
    earnings_surprises = [{"period_end": "2026-07-30", "eps_actual": 2.02, "eps_estimated": 1.89}]

    result = compute_all(
        _fin(), _price_info(), _dividend_history(), analyst_estimates, earnings_surprises
    )

    assert result["valuation_metrics"]["forward_pe"] == pytest.approx(50.0 / 5.0)
    assert result["growth_metrics"]["earnings_surprises"] == earnings_surprises
    assert "valuation_metrics.forward_pe" not in result["missing_fields"]


# --- revenue growth over the most recent period (2026-10-02 accuracy audit) ---


def _growth_fin(quarters, annual):
    return NormalizedFinancials(quarters=quarters, annual=annual, balance_sheet={}, currency="USD")


def _q(period_end, revenue):
    return {"period_end": period_end, "revenue": revenue}


def test_growth_uses_the_latest_quarter_when_it_is_newer_than_the_last_fiscal_year():
    """KO: fiscal year ended 2025-12-31 grew 1.87%, the quarter ended 2026-07-03 grew 6.74%."""
    fin = _growth_fin([_q("2026-07-03", 106.74), _q("2025-06-27", 100.0)],
                      [_q("2025-12-31", 101.87), _q("2024-12-31", 100.0)])
    g = compute_growth_metrics(fin)
    assert g["revenue_growth_yoy"] == pytest.approx(0.0674)
    assert g["revenue_growth_yoy_basis"] == "quarter ended 2026-07-03 vs 2025-06-27"
    assert g["revenue_growth_annual"] == pytest.approx(0.0187)


def test_growth_uses_the_fiscal_year_when_it_ended_after_the_last_quarter():
    """MSFT: fiscal year ended 2026-06-30 is newer than the last 10-Q quarter, 2026-03-31."""
    fin = _growth_fin([_q("2026-03-31", 117.0), _q("2025-03-31", 100.0)],
                      [_q("2026-06-30", 117.79), _q("2025-06-30", 100.0)])
    g = compute_growth_metrics(fin)
    assert g["revenue_growth_yoy"] == pytest.approx(0.1779)
    assert g["revenue_growth_yoy_basis"] == "fiscal year ended 2026-06-30"


def test_growth_finds_the_year_earlier_quarter_even_when_it_is_not_the_next_row():
    """yfinance gives five quarters, newest first: the comparison is index 4, not 1."""
    fin = _growth_fin([_q("2026-07-31", 109.0), _q("2026-04-30", 105.0), _q("2026-01-31", 103.0),
                       _q("2025-10-31", 101.0), _q("2025-07-31", 100.0)],
                      [_q("2025-10-31", 120.0), _q("2024-10-31", 100.0)])
    g = compute_growth_metrics(fin)
    assert g["revenue_growth_yoy"] == pytest.approx(0.09)
    assert g["revenue_growth_annual"] == pytest.approx(0.20)


def test_a_quarter_not_about_a_year_before_is_not_used_as_the_comparison():
    fin = _growth_fin([_q("2026-07-03", 110.0), _q("2026-04-02", 105.0)], [_q("2025-12-31", 101.0), _q("2024-12-31", 100.0)])
    g = compute_growth_metrics(fin)
    assert g["revenue_growth_yoy_basis"] == "fiscal year ended 2025-12-31"


def test_growth_is_none_with_no_usable_periods_and_the_label_says_so():
    g = compute_growth_metrics(_growth_fin([], []))
    assert g["revenue_growth_yoy"] is None and g["revenue_growth_yoy_basis"] is None and g["revenue_growth_annual"] is None


def test_latest_financials_period_end_is_the_newest_period_any_statement_covers():
    fin = _fin(quarters=[_q("2026-03-31", 1.0)] + [{**q} for q in _QUARTERS[1:2]], annual=[_q("2026-06-30", 1.0), _q("2025-06-30", 1.0)])
    assert compute_all(fin, _price_info(), [])["latest_financials_period_end"] == "2026-06-30"


# --- debt metrics and metrics that do not exist for a kind of company (2026-10-02 accuracy audit) ---


def _bs_fin(*, debt, equity=1000.0, quarter=None, annual=None):
    return NormalizedFinancials(
        quarters=[quarter] if quarter else [], annual=[annual] if annual else [],
        balance_sheet={"total_debt": debt, "total_equity": equity}, currency="USD",
    )


def test_a_company_with_no_debt_line_and_no_interest_expense_has_zero_debt_to_equity():
    """RDDT: edgartools finds no debt tags and there is no interest expense anywhere; Yahoo's D/E is 0.006."""
    fin = _bs_fin(debt=None, quarter={"period_end": "2026-06-30", "operating_income": 200.0},
                  annual={"period_end": "2025-12-31", "operating_income": 400.0})
    assert compute_balance_sheet_metrics(fin)["debt_to_equity"] == 0.0


def test_missing_debt_stays_missing_when_the_company_pays_interest():
    fin = _bs_fin(debt=None, quarter={"period_end": "2026-06-30", "operating_income": 200.0, "interest_expense": 10.0})
    assert compute_balance_sheet_metrics(fin)["debt_to_equity"] is None


def test_interest_coverage_falls_back_to_the_fiscal_year_when_the_quarter_has_no_interest_line():
    """MSFT: the 10-Q carries no interest expense, the annual report does (3.05B against 155B operating income)."""
    fin = _bs_fin(debt=40.0, quarter={"period_end": "2026-03-31", "operating_income": 40.0, "interest_expense": None},
                  annual={"period_end": "2026-06-30", "operating_income": 155.0, "interest_expense": 3.1})
    assert compute_balance_sheet_metrics(fin)["interest_coverage"] == pytest.approx(155.0 / 3.1)


def test_interest_coverage_prefers_the_latest_quarter_when_it_has_one():
    fin = _bs_fin(debt=40.0, quarter={"period_end": "2026-03-31", "operating_income": 40.0, "interest_expense": 2.0},
                  annual={"period_end": "2025-12-31", "operating_income": 155.0, "interest_expense": 3.1})
    assert compute_balance_sheet_metrics(fin)["interest_coverage"] == pytest.approx(20.0)


def test_a_reit_has_its_earnings_multiples_marked_not_applicable_when_empty():
    no_earnings = _fin()
    for entry in [*no_earnings.quarters, *no_earnings.annual]:
        entry["net_income"] = None
        entry["net_income_common"] = None
        entry["eps"] = None
    out = compute_all(no_earnings, _price_info(), [], industry="REITs")
    assert out["not_applicable"]["reason"] == "REIT" and "pe_ratio" in out["not_applicable"]["fields"]
    assert "valuation_metrics.pe_ratio" not in out["missing_fields"]


def test_other_industries_have_no_not_applicable_list():
    assert compute_all(_fin(), _price_info(), [], industry="Software - Infrastructure")["not_applicable"] is None


# --- the company-kind profile (ledger BB-106) ---


@pytest.mark.parametrize("industry,pe,ni,group", [
    # industry strings both providers return, from a 55 ticker probe (yfinance for Canada, SEC/FMP for the US)
    ("Banking", 17.6, 5.0, "financials"), ("Banks - Diversified", 14.0, 5.0, "financials"),
    ("Insurance", 16.2, 5.0, "financials"), ("Insurance - Life", 17.9, 5.0, "financials"),
    ("Insurance - Diversified", 12.6, 5.0, "financials"),
    ("Regulated Utilities", 22.0, 5.0, "capital_intensive"), ("Regulated Electric", 17.0, 5.0, "capital_intensive"),
    ("Telecommunications", 4.1, 5.0, "capital_intensive"), ("Telecom Services", 8.0, 5.0, "capital_intensive"),
    ("Oil & Gas Exploration and Production", 12.0, 5.0, "capital_intensive"),
    ("Oil & Gas Integrated", 20.0, 5.0, "capital_intensive"), ("Oil & Gas Storage and Transportation", 22.7, 5.0, "capital_intensive"),
    ("Software - Application", None, -100.0, "pre_profit"), ("Biotechnology", None, -500.0, "pre_profit"),
    ("Software - Application", 109.0, 50.0, "standard"), ("Semiconductors", 30.0, 60.0, "standard"),
    ("Insurance Brokers", 25.0, 5.0, "standard"), ("Oil & Gas Equipment & Services", 15.0, 5.0, "standard"),
    ("Aerospace & Defense", None, -1000.0, "pre_profit"),
    ("Asset Management", 25.0, 5.0, "standard"), ("Financial - Credit Services", 30.0, 5.0, "standard"),
    ("Discount Stores", 47.0, 5.0, "standard"), (None, None, None, "standard"),
    # REITs are not tuned: they stay "standard" even with a trailing loss (existing REIT handling applies)
    ("REITs", None, -10.0, "standard"), ("REIT - Retail", 40.0, 5.0, "standard"),
])
def test_metric_group_from_the_industry_text_both_providers_use(industry, pe, ni, group):
    assert metric_group(industry, pe, ni) == group


def _own_quarters(net_income=150.0, ocf=180.0, capex=-40.0):
    """Fresh rows: the module-level _QUARTERS dicts are shared and other tests mutate them."""
    return [_quarter(1000.0 - 20 * i, net_income, 1.5, 200.0, operating_cash_flow=ocf, capital_expenditures=capex)
            for i in range(5)]


def _bank_fin():
    return _fin(quarters=_own_quarters(), annual=[_quarter(3700.0, 500.0, 5.3, period_end="2025-12-31")] * 4,
                balance_sheet={**_BALANCE_SHEET, "total_assets": 50000.0, "total_equity": 3000.0, "total_debt": 20000.0})


def test_a_bank_hides_the_metrics_that_do_not_describe_it_and_gets_roa_and_equity_to_assets():
    """TD.TO risks read "High leverage: D/E=3.58" and "earnings quality concerns: FCF to net income=0.31"."""
    result = compute_all(_bank_fin(), _price_info(), _dividend_history(), industry="Banking")

    profile = result["metric_profile"]
    assert profile["group"] == "financials"
    for field in ("debt_to_equity", "fcf_to_net_income", "current_ratio", "interest_coverage", "operating_margin"):
        assert field in profile["hidden"]
    assert "ROE" in profile["lens"] and profile["limits"]
    assert result["profitability_metrics"]["roa"] == pytest.approx(600.0 / 50000.0)  # TTM net income (4 x 150) over total assets
    assert result["balance_sheet_metrics"]["equity_to_assets"] == pytest.approx(3000.0 / 50000.0)
    assert not any(f.endswith(("debt_to_equity", "fcf_to_net_income")) for f in result["missing_fields"])


def test_the_new_ratios_exist_only_for_the_group_that_is_judged_on_them():
    standard = compute_all(_fin(quarters=_own_quarters(), annual=[_quarter(3700.0, 500.0, 5.3, period_end="2025-12-31")] * 4),
                           _price_info(), _dividend_history(), industry="Software - Application")

    assert "roa" not in standard["profitability_metrics"] and "equity_to_assets" not in standard["balance_sheet_metrics"]
    assert "cash_runway_quarters" not in standard["balance_sheet_metrics"]
    assert standard["metric_profile"]["group"] == "standard" and standard["metric_profile"]["hidden"] == ["ev_ebitda", "free_cash_flow"]


def test_a_pre_profit_company_gets_a_cash_runway_and_hides_pe_roe_and_payout():
    losing = [_quarter(100.0, -50.0, -0.5, -40.0, operating_cash_flow=-30.0, capital_expenditures=-10.0) for _ in range(5)]
    fin = _fin(quarters=losing, annual=[_quarter(400.0, -200.0, -2.0, period_end="2025-12-31")] * 4,
               balance_sheet={**_BALANCE_SHEET, "cash_and_equivalents": 400.0})

    result = compute_all(fin, _price_info(), _dividend_history(), industry="Software - Application")

    assert result["metric_profile"]["group"] == "pre_profit"
    assert "pe_ratio" in result["metric_profile"]["hidden"] and "roe" in result["metric_profile"]["hidden"]
    # TTM free cash flow = 5 quarters' worth summed over 4: (-30 - 10) * 4 = -160; 400 of cash / (160 / 4) = 10 quarters
    assert result["balance_sheet_metrics"]["cash_runway_quarters"] == pytest.approx(10.0)


def test_a_company_with_positive_cash_flow_has_no_runway():
    fin = _fin(quarters=[_quarter(100.0, -5.0, -0.1, 5.0, operating_cash_flow=30.0, capital_expenditures=-10.0)] * 5,
               annual=[_quarter(400.0, -20.0, -0.4, period_end="2025-12-31")] * 4)
    result = compute_all(fin, _price_info(), _dividend_history(), industry="Biotechnology")
    assert result["metric_profile"]["group"] == "pre_profit"
    assert result["balance_sheet_metrics"]["cash_runway_quarters"] is None


def test_interest_coverage_uses_the_interest_expense_magnitude_whichever_sign_the_filer_tags():
    """COST -32M, V -39M and BLK -134M gave profitable companies a coverage of -88, -158 and -18."""
    for sign in (1, -1):
        quarters = [_quarter(1000.0, 150.0, 1.5, 200.0, interest_expense=sign * 10.0)]
        assert compute_balance_sheet_metrics(_fin(quarters=quarters))["interest_coverage"] == pytest.approx(20.0)


def test_ebitda_and_coverage_fall_back_to_net_income_tax_and_interest_when_there_is_no_operating_income_line():
    """XOM, COP and CVX have no OperatingIncomeLoss, so EV/EBITDA and interest cover were empty for all three."""
    year = _quarter(8000.0, 600.0, 6.0, None, tax_expense=200.0, interest_expense=-100.0, depreciation_amortization=500.0,
                    period_end="2025-12-31")
    quarters = [_quarter(2000.0, 150.0, 1.5, None, tax_expense=50.0, interest_expense=25.0, depreciation_amortization=None)
                for _ in range(5)]
    fin = _fin(quarters=quarters, annual=[year, *_ANNUAL[1:]])

    coverage = compute_balance_sheet_metrics(fin)["interest_coverage"]
    ev_ebitda = compute_valuation_metrics(fin, _price_info(), compute_growth_metrics(fin))["ev_ebitda"]

    # fiscal-year basis: EBIT = 600 + 200 + 100 = 900; coverage 900 / 100; EBITDA = 900 + 500 = 1400
    assert coverage == pytest.approx(9.0)
    assert ev_ebitda == pytest.approx((5000.0 + 1000.0 - 500.0) / 1400.0)


def test_earnings_surprises_are_newest_first_whatever_order_the_provider_returns():
    """yfinance (every Canadian name) is oldest first and the payload took the first two, so TD.TO showed its two OLDEST
    quarters of four."""
    oldest_first = [{"period_end": p, "eps_surprise_pct": 1.0} for p in ("2025-10-31", "2026-01-31", "2026-04-30", "2026-07-31")]

    result = compute_growth_metrics(_fin(), oldest_first)

    assert [s["period_end"] for s in result["earnings_surprises"]] == ["2026-07-31", "2026-04-30", "2026-01-31", "2025-10-31"]
    assert compute_growth_metrics(_fin(), [])["earnings_surprises"] == []
    assert compute_growth_metrics(_fin(), None)["earnings_surprises"] is None


# --- guidance_vs_consensus decided from the earnings surprises ---


def _surprises(*pcts):
    return [{"period_end": f"2026-0{i + 1}-01", "eps_surprise_pct": p} for i, p in enumerate(pcts)]


@pytest.mark.parametrize("pcts,expected", [
    ((5.3, 2.7, 5.9, 4.0), "above"),        # KO: four beats
    ((4.0, -3.9, 7.8, 5.9), "above"),       # JPM: three beats and one small miss
    ((5.3, 0.8, 0.1, 3.4), "inline"),       # COST: beats too small to count
    ((-0.6, -5.8, 8.5, 5.2), "inline"),     # SHOP.TO: mixed
    ((-9.7, -19.8, 5.5, -20.2), "below"),   # T.TO: three misses
    ((3162.4, 81.5, 2108.7, 94.6), "above"),  # INTC: absurd percentages from near-zero estimates count as beats, not as a size
    ((8.0,), "not_available"),
    ((), "not_available"),
])
def test_the_earnings_surprise_pattern(pcts, expected):
    """The model wrote "above" 22, "not_available" 25, "inline" 4 and "below" 2 for the same all-beats input."""
    from data.precompute.fundamentals import earnings_surprise_pattern

    assert earnings_surprise_pattern(_surprises(*pcts)) == expected


def test_the_surprise_pattern_with_no_history_is_not_available():
    from data.precompute.fundamentals import earnings_surprise_pattern

    assert earnings_surprise_pattern(None) == "not_available"
    assert earnings_surprise_pattern([{"period_end": "2026-01-01", "eps_surprise_pct": None}] * 3) == "not_available"
