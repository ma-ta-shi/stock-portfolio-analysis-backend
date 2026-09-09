"""Valuation/growth/profitability/balance-sheet/dividend metrics for the
Fundamental Analyst agent (ClickUp 86ban0wcr).

Pure Python, deterministic, no LLM involved — the pipeline does the math,
the LLM does the thinking (same principle as macro_sources.py/technicals.py).

Input contract: NormalizedFinancials/NormalizedQuote/NormalizedDividendRecord
(src/data/providers/base.py, 86bbb001k/86bbb17pw) — already provider-agnostic,
this module never branches on which provider the data came from.

Real gaps found auditing this ticket's own design against the live
Fundamental Analyst Agent Prompt (not guessed, see docs/decision-log.md and
ClickUp 86ban0wcr for the full reasoning):
- health_rating is NOT computed here — it's an LLM interpretive_fields value
  (sector-relative judgment on raw ratios this module does supply).
- The ANALYST block (consensus_rating/avg_price_target/num_analysts) is not
  this module's job — DataBundle.analyst_consensus is Sentiment Analyst's
  field, cross-read by the Fundamental prompt, not computed here.
- forward_pe/earnings_surprises were real, deferred gaps — no provider
  normalized get_analyst_estimates() and no provider method existed for
  historical actual-vs-estimate earnings data. Closed in 86bbdu04a:
  compute_valuation_metrics() takes an optional NormalizedAnalystEstimates
  and compute_growth_metrics() takes an optional earnings_surprises list;
  both default to None (graceful degradation) so an existing caller with
  no analyst data still works unchanged.
- guidance_vs_consensus stays LLM-interpretive (no reliable structured
  source), a separate, still-pending prompt-file inconsistency.
"""

import statistics

from data.providers.base import (
    NormalizedAnalystEstimates,
    NormalizedDividendRecord,
    NormalizedFinancials,
    NormalizedQuote,
)

# Fields both the sector-median PEER block and each peer_records entry need —
# a subset of valuation/profitability/balance_sheet_metrics, not the full
# dicts (peers don't need e.g. free_cash_flow/cash_position/dividend fields).
# Single source of truth: keys() doubles as the peer-metric whitelist (was
# a separate _PEER_METRIC_KEYS tuple until review — two structures that had
# to be kept in sync by hand for no reason, since dicts are insertion-ordered).
# Maps each metric key -> the prompt's own sector_median_* placeholder name
# (macro_sources.py convention: field names match prompt placeholders 1:1,
# avoids a renaming step at the payload-builder layer).
_SECTOR_MEDIAN_KEYS = {
    "pe_ratio": "sector_median_pe",
    "pb_ratio": "sector_median_pb",
    "revenue_growth_yoy": "sector_median_rev_growth",
    "gross_margin": "sector_median_gross_margin",
    "operating_margin": "sector_median_op_margin",
    "roe": "sector_median_roe",
    "debt_to_equity": "sector_median_de",
    "ev_ebitda": "sector_median_ev_ebitda",
}

# Peer-median guard (ClickUp 86bbq04wm). The unguarded statistics.median() over
# whatever survived a None filter was feeding the Fundamental Analyst corrupt
# sector medians (a -98 P/B median from one peer's negative book equity, a
# peer-wide D/E of exactly 0.0, a 105% revenue-growth median from a spinoff
# stub) — used by the model in 3 of 3 live runs as both a key_factor and a
# risk. Guard here at the source, not the payload renderer: this is a data
# defect (a median that shouldn't exist), and peer_metrics feeds more than the
# prompt payload once DataPipeline.prepare() exists.

# Checked per-metric against the count of VALID contributors, not len(peer_data):
# each median stands on its own evidence. 2 not 3 — real peer sets are ~3
# tickers and after validity filtering a 3 floor would render N/A almost
# always (the block must stay useful, not just uncorrupt).
_MIN_PEERS_FOR_SECTOR_MEDIAN = 2

# Per-metric plausibility ranges (exclusive both ends). The 0.0 lower bound on
# pe/pb/ev_ebitda/debt_to_equity is a principled sign check — these go negative
# only on negative earnings/equity/EBITDA, not a meaningful multiple — needing
# no calibration; the strict `<` also drops an exact-0.0 debt_to_equity (the
# ticket's named missing-data signature, and — accepted — a genuinely
# debt-free peer). gross_margin is positive-only for the same reason (a real
# negative is nearly always COGS concept mis-mapping); operating_margin and roe
# genuinely go negative and only the magnitude is gated.
#
# The outer caps and the growth/margin/roe bands are first-pass, not doc-sourced
# — same disclosure convention as macro_sources.py's trend thresholds — checked
# 2026-09-08 against a 20-name multi-sector US basket on deep (yfinance) data:
# healthy names (AAPL/MSFT/JPM/XOM/NVDA/KO/F/PFE/UNH/HD/CRM/CVX) + loss-makers
# (PLUG/RIVN/BYND) + the audit's problem set (DELL/SNDK/WDC) + GME/CVNA. Nothing
# from a healthy name was nulled (real values topped out well inside every
# band — P/E ~42, ROE ~1.2, worst real revenue decline -40% for MRNA); what
# these ranges reject is DELL's negative-equity P/B and D/E, RIVN/BYND's
# negative P/E and EV/EBITDA, PLUG/BYND's tiny-equity ROE blow-ups, and the
# SNDK/PARA +170% spinoff/restructuring revenue-growth stubs (SNDK's 1.75 is
# the exact value behind the corrupt 105% median). PLUG's ~-1% gross margin is
# also excluded — real but not a useful comparable. NaN (if a provider ever
# leaks one) fails every comparison and is rejected too.
#
# revenue_growth_yoy's lower bound is the least adversarially tested — no
# artifact below -40% appeared in the basket; -0.9 is the "lost nearly all
# revenue YoY = a corporate action, not organic performance" line.
_PEER_METRIC_VALID_RANGE: dict[str, tuple[float, float]] = {
    "pe_ratio": (0.0, 500.0),
    "pb_ratio": (0.0, 100.0),
    "ev_ebitda": (0.0, 200.0),
    "debt_to_equity": (0.0, 50.0),
    "revenue_growth_yoy": (-0.9, 1.5),  # 1.5 catches SNDK's 1.75 / PARA's 1.69
    "gross_margin": (0.0, 0.99),
    "operating_margin": (-1.0, 0.9),
    "roe": (-2.0, 3.0),  # a real buyback-shrunk-equity ROE reaches ~1.2 (AAPL/HD live)
}
# _PEER_METRIC_VALID_RANGE must stay 1:1 with _SECTOR_MEDIAN_KEYS — a missing
# entry KeyErrors in _valid_peer_value (asserted in test_fundamentals.py).


def _valid_peer_value(metric: str, value: float | None) -> float | None:
    """None (renders N/A downstream) unless value is present AND strictly
    inside the metric's plausibility range. Per-metric because the metrics
    fail in different ways — a negative P/B from negative book equity, a D/E
    of exactly 0.0 as a missing-data signature, a spinoff's absurd first-year
    revenue growth."""
    if value is None:
        return None
    low, high = _PEER_METRIC_VALID_RANGE[metric]
    return value if low < value < high else None


def _ttm(periods: list[dict], field: str) -> float | None:
    """Trailing-twelve-month sum over the most recent 4 quarters. None if
    fewer than 4 quarters exist — matches _cagr()'s own precedent (below)
    of returning None on insufficient data rather than approximating with
    a shorter/different period."""
    if len(periods) < 4:
        return None
    values = [p.get(field) for p in periods[:4]]
    if any(v is None for v in values):
        return None
    return sum(values)


def _cagr(annual: list[dict], field: str, years: int = 3) -> float | None:
    """Undefined when the base year is negative or zero (common for
    turnaround names with a loss year several years back) — return None
    rather than a nonsense value."""
    if len(annual) <= years:
        return None
    end, start = annual[0].get(field), annual[years].get(field)
    if start is None or end is None or start <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _ebitda(quarter: dict) -> float | None:
    """EBITDA isn't a raw line item — derive it."""
    operating_income = quarter.get("operating_income")
    depreciation = quarter.get("depreciation_amortization")
    if operating_income is None or depreciation is None:
        return None
    return operating_income + depreciation


def _ev_ebitda(fin: NormalizedFinancials, price_info: NormalizedQuote) -> float | None:
    """Always computed when data allows, not conditionally skipped for
    non-capital-intensive sectors — the live prompt's own CAPITAL_INTENSIVE
    sector conditional is a payload-template/orchestrator rendering
    decision, not a data-computation concern."""
    if not fin.quarters:
        return None
    ebitda = _ebitda(fin.quarters[0])
    if not ebitda:
        return None
    market_cap = price_info.get("market_cap")
    total_debt = fin.balance_sheet.get("total_debt")
    cash = fin.balance_sheet.get("cash_and_equivalents")
    if market_cap is None or total_debt is None or cash is None:
        return None
    enterprise_value = market_cap + total_debt - cash
    return enterprise_value / ebitda


_PEG_MAX_GROWTH = 1.0  # 100% YoY EPS growth


def _peg(pe_ratio: float | None, eps_growth_yoy: float | None) -> float | None:
    """PEG undefined for negative/zero growth, and meaningless when the YoY
    rate is a base-year artifact (a prior-year EPS near zero — an impairment
    year, a cyclical trough) rather than a sustainable rate: that produces a
    multi-hundred-percent "growth" and collapses PEG toward zero. Cap at
    _PEG_MAX_GROWTH — above it the denominator isn't the growth PEG is meant
    to normalise against. First-pass bound, same disclosure convention as
    _PEER_METRIC_VALID_RANGE. Guarding eps_growth_yoy itself (and
    revenue_growth_yoy) in compute_growth_metrics is a separate, pre-existing
    concern this doesn't try to solve."""
    if (
        pe_ratio is None
        or not eps_growth_yoy
        or eps_growth_yoy <= 0
        or eps_growth_yoy > _PEG_MAX_GROWTH
    ):
        return None
    return pe_ratio / (eps_growth_yoy * 100)


def _margin_trend(quarters: list[dict]) -> str | None:
    """Plain trend classifier over net_margin — same ±deadband-threshold
    shape as macro_sources.py's _cpi_trend/_rate_trend. Compares the latest
    quarter to 4 quarters back (YoY) when available, else the oldest
    available quarter — quarters may only have the ticket's own minimum
    of 2."""
    if len(quarters) < 2:
        return None
    latest = quarters[0]
    latest_margin = _net_margin(latest)
    if latest_margin is None:
        return None
    comparison_idx = 4 if len(quarters) > 4 else len(quarters) - 1
    prior_margin = _net_margin(quarters[comparison_idx])
    if prior_margin is None:
        return None
    delta_pp = (latest_margin - prior_margin) * 100
    if delta_pp > 1:
        return "expanding"
    if delta_pp < -1:
        return "contracting"
    return "stable"


def _net_margin(quarter: dict) -> float | None:
    revenue = quarter.get("revenue")
    net_income = quarter.get("net_income")
    if not revenue or net_income is None:
        return None
    return net_income / revenue


def _dividend_regularity(dividend_history: list[NormalizedDividendRecord]) -> str:
    """Classifies payout *regularity* ("regular" | "irregular" | "none") so the
    agent doesn't treat a lumpy/special-heavy history as sustainable yield.

    Named for what it measures: this is NOT the dividend's tax character
    (eligible Canadian / US-qualified / foreign / return-of-capital) — that
    classification has no provider source and is security-tax-classification.py's
    job. The Fundamental prompt's DIV block reads this as {dividend_regularity}.

    Assumes newest-last (ascending) ordering — callers must sort first, see
    compute_dividend_info()."""
    if not dividend_history:
        return "none"
    amounts = [d["amount_per_share"] for d in dividend_history[-8:]]  # ~2yrs quarterly
    if max(amounts) > 2 * min(amounts):
        return "irregular"  # includes special dividends
    return "regular"


def _consecutive_years_paid(years_with_payments: set[int]) -> int:
    """Counts backward from the most recent year while every year has at
    least one payment — stops at the first gap year."""
    if not years_with_payments:
        return 0
    latest_year = max(years_with_payments)
    count = 0
    year = latest_year
    while year in years_with_payments:
        count += 1
        year -= 1
    return count


def compute_growth_metrics(
    fin: NormalizedFinancials, earnings_surprises: list[dict] | None = None
) -> dict:
    """revenue_growth_yoy/eps_growth_yoy use annual[0] vs annual[1] — the
    simplest well-defined YoY, always available once 2+ years of annual
    data exist. Quarterly YoY would need quarters[4], not guaranteed by
    the ticket's own minimum-data threshold (>=2 quarters).

    earnings_surprises (86bbdu04a) is passed through directly, not
    collapsed to None when empty — None means "no data source wired for
    this call," [] means "fetched, genuinely zero rows." compute_all()'s
    missing_fields scan only flags None as missing; collapsing [] into
    None would misreport a real, empty result as missing data. forward_pe
    is a VAL-block field (compute_valuation_metrics), not GROWTH — it was
    briefly duplicated here by mistake (caught on review: it showed
    up twice in missing_fields, once per dict, and would have been a
    silent last-write-wins collision in compute_peer_comparison's
    dict merge)."""
    annual = fin.annual
    revenue_growth_yoy = None
    eps_growth_yoy = None
    if len(annual) >= 2:
        prev_revenue = annual[1].get("revenue")
        if prev_revenue and annual[0].get("revenue") is not None:
            revenue_growth_yoy = (annual[0]["revenue"] - prev_revenue) / prev_revenue
        prev_eps = annual[1].get("eps")
        if prev_eps and annual[0].get("eps") is not None:
            eps_growth_yoy = (annual[0]["eps"] - prev_eps) / prev_eps
    return {
        "revenue_growth_yoy": revenue_growth_yoy,
        "revenue_growth_3yr_cagr": _cagr(annual, "revenue"),
        "eps_growth_yoy": eps_growth_yoy,
        "earnings_surprises": earnings_surprises,
    }


def compute_profitability_metrics(fin: NormalizedFinancials) -> dict:
    if not fin.quarters:
        return {
            "gross_margin": None, "operating_margin": None, "net_margin": None,
            "roe": None, "margin_trend": None, "fcf_to_net_income": None,
        }
    latest = fin.quarters[0]
    revenue = latest.get("revenue")
    gross_margin = None
    if revenue and latest.get("cost_of_revenue") is not None:
        gross_margin = (revenue - latest["cost_of_revenue"]) / revenue
    operating_margin = None
    if revenue and latest.get("operating_income") is not None:
        operating_margin = latest["operating_income"] / revenue
    net_margin = _net_margin(latest)

    roe = None
    ttm_net_income = _ttm(fin.quarters, "net_income")
    total_equity = fin.balance_sheet.get("total_equity")
    # total_equity > 0, not just truthy: negative book equity makes ROE
    # meaningless (a net loss over negative equity reads as a positive ROE) —
    # None is the honest answer for both peers and the subject (86bbq04wm).
    if ttm_net_income is not None and total_equity is not None and total_equity > 0:
        roe = ttm_net_income / total_equity

    fcf_to_net_income = None
    ocf, capex, net_income = (
        latest.get("operating_cash_flow"),
        latest.get("capital_expenditures"),
        latest.get("net_income"),
    )
    if ocf is not None and capex is not None and net_income:
        fcf_to_net_income = (ocf - capex) / net_income

    return {
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "roe": roe,
        "margin_trend": _margin_trend(fin.quarters),
        "fcf_to_net_income": fcf_to_net_income,
    }


def compute_balance_sheet_metrics(fin: NormalizedFinancials) -> dict:
    """No health_rating — resolved as an LLM interpretive_fields value,
    not this module's job (docs/decision-log.md 2026-08-07)."""
    bs = fin.balance_sheet
    total_debt, total_equity = bs.get("total_debt"), bs.get("total_equity")
    debt_to_equity = total_debt / total_equity if total_debt is not None and total_equity else None

    current_assets, current_liabilities = bs.get("current_assets"), bs.get("current_liabilities")
    current_ratio = (
        current_assets / current_liabilities
        if current_assets is not None and current_liabilities
        else None
    )

    interest_coverage = None
    free_cash_flow = None
    if fin.quarters:
        latest = fin.quarters[0]
        operating_income, interest_expense = latest.get("operating_income"), latest.get(
            "interest_expense"
        )
        if operating_income is not None and interest_expense:
            interest_coverage = operating_income / interest_expense
        ocf, capex = latest.get("operating_cash_flow"), latest.get("capital_expenditures")
        if ocf is not None and capex is not None:
            free_cash_flow = ocf - capex

    return {
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "interest_coverage": interest_coverage,
        "free_cash_flow": free_cash_flow,
        "cash_position": bs.get("cash_and_equivalents"),
    }


def compute_valuation_metrics(
    fin: NormalizedFinancials,
    price_info: NormalizedQuote,
    growth: dict,
    analyst_estimates: NormalizedAnalystEstimates | None = None,
) -> dict:
    current_price = price_info.get("current_price")
    market_cap = price_info.get("market_cap")

    ttm_eps = _ttm(fin.quarters, "eps")
    pe_ratio = current_price / ttm_eps if current_price is not None and ttm_eps else None
    if pe_ratio is None:
        # yfinance's per-quarter Diluted EPS is sporadically NaN for essentially
        # every Canadian filer, so _ttm("eps") is None sector-wide. price / EPS
        # is identically market_cap / net income to common, and net income is
        # reliably present per quarter. Income to common (not total) because P/E
        # is a per-common-share metric: this tracks .info trailingPE within ~2%
        # for preferred-heavy names where total net income runs 5-9% low. Falls
        # back to total net_income for filers that don't disclose the common
        # split (identical for names with no preferred). A real TTM loss leaves
        # pe_ratio None, same as a negative real-EPS P/E. compute_peer_comparison
        # runs this per peer, so sector_median_pe is derived the same way.
        ttm_ni = _ttm(fin.quarters, "net_income_common") or _ttm(fin.quarters, "net_income")
        if ttm_ni is not None and ttm_ni > 0 and market_cap is not None:
            pe_ratio = market_cap / ttm_ni

    shares_outstanding = fin.quarters[0].get("shares_outstanding") if fin.quarters else None
    total_equity = fin.balance_sheet.get("total_equity")
    pb_ratio = None
    if current_price is not None and total_equity is not None and shares_outstanding:
        book_value_per_share = total_equity / shares_outstanding
        pb_ratio = current_price / book_value_per_share if book_value_per_share else None

    ttm_revenue = _ttm(fin.quarters, "revenue")
    ps_ratio = market_cap / ttm_revenue if market_cap is not None and ttm_revenue else None

    # forward_pe (86bbdu04a) mirrors pe_ratio's own guard style exactly —
    # current_price / an EPS-like denominator, None if either is missing.
    forward_eps = analyst_estimates.get("forward_eps") if analyst_estimates else None
    forward_pe = current_price / forward_eps if current_price is not None and forward_eps else None

    return {
        "pe_ratio": pe_ratio,
        "forward_pe": forward_pe,
        "pb_ratio": pb_ratio,
        "ps_ratio": ps_ratio,
        "peg_ratio": _peg(pe_ratio, growth.get("eps_growth_yoy")),
        "ev_ebitda": _ev_ebitda(fin, price_info),
    }


def compute_dividend_info(
    fin: NormalizedFinancials,
    price_info: NormalizedQuote,
    dividend_history: list[NormalizedDividendRecord],
) -> dict:
    """Ordering of dividend_history isn't guaranteed consistent across
    providers (confirmed live: FMP and openbb_tmx both return oldest-first,
    yfinance's isn't independently verified) — sort explicitly rather than
    assume the caller's order."""
    if not dividend_history:
        return {
            "dividend_yield": None,
            "payout_ratio": None,
            "dividend_growth_5yr": None,
            "dividend_regularity": "none",
            "consecutive_years_paid": 0,
        }

    sorted_history = sorted(dividend_history, key=lambda d: d["ex_date"])
    trailing_4 = sorted_history[-4:]
    trailing_annual_dividend = sum(d["amount_per_share"] for d in trailing_4)

    current_price = price_info.get("current_price")
    dividend_yield = (
        trailing_annual_dividend / current_price if current_price else None
    )

    by_year: dict[int, float] = {}
    for record in sorted_history:
        year = int(record["ex_date"][:4])
        by_year[year] = by_year.get(year, 0.0) + record["amount_per_share"]
    years_sorted = sorted(by_year)

    # Real bug caught testing through the real code path against live AAPL
    # data: the most recent calendar year in dividend_history is very
    # likely a partial year (however many payments have happened so far),
    # so comparing it directly against a full prior year systematically
    # understates growth — confirmed live (AAPL 2026: only 3 of 4 payments
    # so far, $0.80 vs a complete 2025's $1.03, producing a false negative
    # "growth" rate). trailing_annual_dividend (already a genuine trailing-
    # 12-month total, not calendar-bound) is the correct "now" figure;
    # anchor "5 years ago" to the latest payment's own year rather than an
    # ordinal by_year index, which silently picks the wrong year if any
    # year in between had no payments at all.
    #
    # Known, disclosed, unfixed limitation: a stock split distorts a
    # calendar year's raw per-share total if it falls mid-year (confirmed
    # live: AAPL's 2020 total is ~2.5x any other year, from its August 2020
    # 4:1 split mixing pre-/post-split per-share amounts) — NormalizedDividendRecord
    # carries no split-ratio data to correct this. If the reference year
    # happens to be a split year, dividend_growth_5yr will be skewed; not
    # fixed here.
    dividend_growth_5yr = None
    if years_sorted:
        latest_year = int(sorted_history[-1]["ex_date"][:4])
        reference_year = latest_year - 5
        reference_total = by_year.get(reference_year)
        if reference_total and reference_total > 0:
            dividend_growth_5yr = (trailing_annual_dividend / reference_total) ** (1 / 5) - 1

    ttm_eps = _ttm(fin.quarters, "eps")
    payout_ratio = trailing_annual_dividend / ttm_eps if ttm_eps else None
    if payout_ratio is None:
        # Same yfinance quarterly-EPS-NaN problem as compute_valuation_metrics:
        # for CA filers the per-share path never resolves. Fall back to total
        # cash dividends paid / total net income. Total net_income here (not
        # net income to common, unlike the P/E path) because "Cash Dividends
        # Paid" is total cash dividends including preferred, so the consistent
        # denominator is total net income — this also matches how .info
        # payoutRatio is computed (verified within 0.1pp for RY.TO / SU.TO).
        ttm_ni = _ttm(fin.quarters, "net_income")
        ttm_dividends_paid = _ttm(fin.quarters, "dividends_paid")
        if ttm_ni is not None and ttm_ni > 0 and ttm_dividends_paid is not None:
            payout_ratio = abs(ttm_dividends_paid) / ttm_ni

    return {
        "dividend_yield": dividend_yield,
        "payout_ratio": payout_ratio,
        "dividend_growth_5yr": dividend_growth_5yr,
        "dividend_regularity": _dividend_regularity(sorted_history),
        "consecutive_years_paid": _consecutive_years_paid(set(years_sorted)),
    }


def compute_peer_comparison(
    peer_data: list[tuple[str, NormalizedFinancials, NormalizedQuote]],
) -> dict:
    """Reuses compute_valuation_metrics/compute_profitability_metrics/
    compute_balance_sheet_metrics per peer rather than a parallel
    calculation path. Returns both peer placeholders the live prompt
    needs: sector_medians (keyed to the prompt's own sector_median_*
    placeholder names) and peer_records (per-peer, for {peer_data_block}'s
    individual "PEER MSFT: ..." lines).

    Real, confirmed dead parameters removed on review: the original
    ticket's own proposed signature also took the primary stock's fin/
    valuation/profitability/balance_sheet, copied here without
    independently checking whether they were actually needed — they
    weren't referenced anywhere in this function. The prompt's PEER block
    only needs sector medians and individual peer records; any "P/E 25 vs
    median 18" comparison is the LLM's own SYNTHESIS-step reasoning
    (Design Decision #2: LLM produces interpretive fields only), not a
    precomputed field."""
    if not peer_data:
        return {"sector_medians": {}, "peer_records": []}

    peer_records = []
    for ticker, peer_fin, peer_price_info in peer_data:
        peer_growth = compute_growth_metrics(peer_fin)
        peer_valuation = compute_valuation_metrics(peer_fin, peer_price_info, peer_growth)
        peer_profitability = compute_profitability_metrics(peer_fin)
        peer_balance_sheet = compute_balance_sheet_metrics(peer_fin)
        merged = {**peer_valuation, **peer_growth, **peer_profitability, **peer_balance_sheet}
        record = {"ticker": ticker}
        # Sanitize once, here (86bbq04wm): the median loop reads from
        # peer_records, so this fixes both the medians and the per-peer
        # {peer_data_block} lines. A rejected value stores None (renders N/A).
        record.update({key: _valid_peer_value(key, merged.get(key)) for key in _SECTOR_MEDIAN_KEYS})
        peer_records.append(record)

    sector_medians = {}
    for key, placeholder_name in _SECTOR_MEDIAN_KEYS.items():
        values = [r[key] for r in peer_records if r[key] is not None]
        sector_medians[placeholder_name] = (
            statistics.median(values) if len(values) >= _MIN_PEERS_FOR_SECTOR_MEDIAN else None
        )

    return {"sector_medians": sector_medians, "peer_records": peer_records}


def compute_all(
    fin: NormalizedFinancials,
    price_info: NormalizedQuote,
    dividend_history: list[NormalizedDividendRecord],
    peer_data: list[tuple[str, NormalizedFinancials, NormalizedQuote]],
    analyst_estimates: NormalizedAnalystEstimates | None = None,
    earnings_surprises: list[dict] | None = None,
) -> dict:
    """Public entry point. Order matters: growth before valuation (PEG
    needs eps_growth_yoy). analyst_estimates/earnings_surprises (86bbdu04a)
    default to None — an existing caller with no analyst data still works
    unchanged, same graceful-degradation convention as every other input
    here.

    Returns a `currency_mismatch` key: None when the statement currency and
    the quote currency agree, else {financials_currency, quote_currency}. The
    price-vs-statement multiples are still computed in that case but are
    FX-distorted — see the note below and ClickUp 86bbxucf0."""
    # A Canadian-listed company that reports in USD (ATD.TO, NTR.TO, BN.TO,
    # CSU.TO, ...) has USD statements but a CAD quote. Every price-vs-statement
    # metric below (pe_ratio, pb_ratio, ps_ratio, ev_ebitda, peg_ratio, and the
    # eps-path payout_ratio) is then off by the CAD/USD rate (~1.37). We still
    # emit them for now — coverage first — and mark the mismatch so the payload
    # builder / reliability scorer can flag or suppress the affected multiples.
    # FX-aware reconciliation is ClickUp 86bbxucf0.
    currency_mismatch = None
    if fin.currency != price_info.get("currency"):
        currency_mismatch = {
            "financials_currency": fin.currency,
            "quote_currency": price_info.get("currency"),
        }

    growth = compute_growth_metrics(fin, earnings_surprises)
    profitability = compute_profitability_metrics(fin)
    balance_sheet = compute_balance_sheet_metrics(fin)
    valuation = compute_valuation_metrics(fin, price_info, growth, analyst_estimates)
    dividend = compute_dividend_info(fin, price_info, dividend_history)
    peer = compute_peer_comparison(peer_data)

    missing_fields = [
        f"{bucket_name}.{key}"
        for bucket_name, bucket in (
            ("valuation_metrics", valuation),
            ("growth_metrics", growth),
            ("profitability_metrics", profitability),
            ("balance_sheet_metrics", balance_sheet),
        )
        for key, value in bucket.items()
        if value is None
    ]

    return {
        "valuation_metrics": valuation,
        "growth_metrics": growth,
        "profitability_metrics": profitability,
        "balance_sheet_metrics": balance_sheet,
        "dividend_info": dividend,
        "peer_metrics": peer,
        "quarters_available": len(fin.quarters),
        "missing_fields": missing_fields,
        "currency_mismatch": currency_mismatch,
    }
