"""One presentation currency per ticker: US tickers in USD, Canadian tickers in CAD.

Two real sources break that without conversion (found 2026-10-02 comparing the pipeline with independent figures):

- A CAD-listed company that reports in USD (SHOP.TO, ATD.TO, NTR.TO, BN.TO, CSU.TO ...) has USD statements but a CAD
  quote, so every price-versus-statement multiple was off by the exchange rate: SHOP.TO showed a P/E of 144 and a P/B of
  22 against a true 103 and 15.6 (BB-021). Both advocates argued from the 144.
- yfinance reports the value of every Canadian insider trade in USD, for cross-listed and TSX-only names alike (the
  implied price per share matched the US close for SHOP.TO, RY.TO, CAR-UN.TO, ATD.TO, REI-UN.TO, L.TO and WCN.TO).

The fix is a single conversion point per source, at the Bank of Canada's USD/CAD rate fetched once per run, applied to
every monetary figure and to none of the per-share counts or ratios. All periods use the one current rate (a ratio such
as growth is unchanged by it); the rate and its source are recorded so the payload can say what was done. Only USD and
CAD are converted: any other statement currency is reported as a mismatch and its price-based multiples are dropped
rather than shown distorted.
"""
from dataclasses import replace

from data.providers.base import NormalizedFinancials

# Monetary fields of NormalizedFinancials.quarters / .annual / .ttm entries (per-share EPS is monetary too;
# shares_outstanding is a count and is left alone).
_PERIOD_MONEY = (
    "revenue", "net_income", "net_income_common", "eps", "operating_income", "interest_expense", "tax_expense",
    "depreciation_amortization", "dividends_paid", "cost_of_revenue", "operating_cash_flow", "capital_expenditures",
)
_BALANCE_MONEY = (
    "total_assets", "total_liabilities", "total_equity", "total_debt", "cash_and_equivalents",
    "current_assets", "current_liabilities",
)
# Valuation multiples that divide a quote-currency price by a statement-currency amount. forward_pe is not one of
# them: forward EPS is already in the quote currency (ENB 2.26 USD, SHOP.TO 3.49 CAD; both forward P/Es match Yahoo).
PRICE_BASED_MULTIPLES = ("pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "peg_ratio")


def fx_factor(from_ccy: str | None, to_ccy: str | None, usd_cad: float | None) -> float | None:
    """Multiplier turning an amount in `from_ccy` into `to_ccy`; 1.0 when equal; None when it cannot be converted
    (an unsupported pair, or no rate). `usd_cad` is the Bank of Canada rate: CAD per 1 USD."""
    if from_ccy == to_ccy:
        return 1.0
    if not usd_cad or usd_cad <= 0:
        return None
    if (from_ccy, to_ccy) == ("USD", "CAD"):
        return usd_cad
    if (from_ccy, to_ccy) == ("CAD", "USD"):
        return 1.0 / usd_cad
    return None


def _scale(entry: dict | None, keys: tuple[str, ...], factor: float) -> dict | None:
    if entry is None:
        return None
    scaled = dict(entry)
    for key in keys:
        if isinstance(scaled.get(key), (int, float)):
            scaled[key] = scaled[key] * factor
    return scaled


def convert_financials(fin: NormalizedFinancials, factor: float, to_ccy: str) -> NormalizedFinancials:
    """A copy of `fin` with every monetary amount multiplied by `factor` and `currency` set to `to_ccy`."""
    return replace(
        fin,
        quarters=[_scale(q, _PERIOD_MONEY, factor) for q in fin.quarters],
        annual=[_scale(a, _PERIOD_MONEY, factor) for a in fin.annual],
        balance_sheet=_scale(fin.balance_sheet, _BALANCE_MONEY, factor) or {},
        ttm=_scale(fin.ttm, _PERIOD_MONEY, factor),
        currency=to_ccy,
    )


def to_quote_currency(
    fin: NormalizedFinancials, quote_ccy: str | None, usd_cad: float | None
) -> tuple[NormalizedFinancials, dict | None]:
    """`fin` expressed in the quote currency, plus a mismatch record (None when the currencies already agree).

    The record is {financials_currency, quote_currency, converted, usd_cad}; `converted` is False when the pair is
    unsupported or no rate was available, in which case `fin` is returned unchanged and callers drop the
    price-based multiples."""
    if fin.currency == quote_ccy:
        return fin, None
    factor = fx_factor(fin.currency, quote_ccy, usd_cad)
    info = {"financials_currency": fin.currency, "quote_currency": quote_ccy, "converted": factor is not None,
            "usd_cad": usd_cad if factor is not None else None}
    if factor is None:
        return fin, info
    return convert_financials(fin, factor, quote_ccy), info


def convert_insider_values(rows: list[dict], to_ccy: str | None, usd_cad: float | None) -> tuple[list[dict], bool]:
    """Insider rows with `value` in `to_ccy`, given that the source reports it in USD. Returns (rows, converted):
    unchanged when already USD, or when no rate is available (a row's value is then left in USD and `converted`
    is False so the caller can say so)."""
    factor = fx_factor("USD", to_ccy, usd_cad)
    if factor is None:
        return rows, False
    if factor == 1.0:
        return rows, True
    return [{**r, "value": r["value"] * factor if isinstance(r.get("value"), (int, float)) else r.get("value")}
            for r in rows], True


_SURPRISE_MONEY = ("eps_actual", "eps_estimated", "revenue_actual", "revenue_estimated")


def convert_earnings_surprises(rows: list[dict] | None, mismatch: dict | None) -> list[dict] | None:
    """Earnings-surprise rows with their per-share and revenue amounts in the quote currency when the statements were
    converted. The sources report them in the statement currency (SHOP.TO's 0.34 EPS was USD inside a CAD stock);
    `eps_surprise_pct` is a ratio and is left alone."""
    if not rows or not mismatch or not mismatch.get("converted"):
        return rows
    factor = fx_factor(mismatch["financials_currency"], mismatch["quote_currency"], mismatch["usd_cad"])
    if factor is None or factor == 1.0:
        return rows
    return [{**r, **{k: r[k] * factor for k in _SURPRISE_MONEY if isinstance(r.get(k), (int, float))}} for r in rows]
