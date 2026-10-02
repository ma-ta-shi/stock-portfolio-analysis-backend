"""Independent accuracy audit of the numbers Pass 1 hands to Pass 2 (scripts/audit_pass1_data.py).

Built from the 2026-10-02 Pass 1 data audit (docs/technical/pass1-data-audit-2026-10-02.md) so that "do the numbers come
through right" can be re-run after any data change, for US and Canadian tickers alike. Each check recomputes a figure
from a source other than the one the pipeline used, or from the raw statements, and compares:

- fundamentals: ratios hand-calculated from raw yfinance quarterly statements (in the quote currency), and a Yahoo
  `.info` cross-check; for cross-listed Canadian names, trailing revenue and net income against the SEC
- snapshot: short interest, analyst count and target, 52-week range, volume and next earnings date against Yahoo
- technicals: SMA, RSI and ATR recomputed from raw price history
- macro: policy rate, yields, unemployment and CPI against FRED and the Bank of Canada; Canadian CPI recomputed from
  StatCan's raw index values
- currency: Canadian insider values are in CAD (implied price per share against the CAD close)
- gaps: Pass 2 fields that come through empty and are not marked not_applicable

A finding is one of: ok (agrees within tolerance), known (differs for a documented reason, see KNOWN), mismatch
(unexplained: the exit code), gap (an empty field), skipped (the check could not run). The pure helpers at the top are
unit-tested; the fetching functions below need the network and a local model for the pipeline's precompute calls.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Differences that are definitions, not errors (see the audit doc). key: field -> why.
KNOWN = {
    "debt_to_equity": "pipeline counts financial debt only; Yahoo's total debt includes lease liabilities",
    "operating_margin": "pipeline uses SEC XBRL operating income; Yahoo's differs",
    "net_margin": "pipeline profit is before preferred dividends (banks)",
    "roe": "ROE conventions differ: Yahoo uses average equity; banks: pipeline profit is before preferred dividends",
    "pb_ratio": "Yahoo uses common-equity book value per share; the pipeline uses total equity (banks carry preferred shares)",
    "fcf_to_net_income": "pipeline profit is before preferred dividends (banks)",
    "free_cash_flow": "pipeline is trailing twelve months; Yahoo's .info figure is the stale annual one",
    "revenue_growth_yoy": "pipeline is the most recent period; Yahoo's is the latest quarter",
    "forward_pe": "FMP's next-fiscal-year estimate against Yahoo's forwardEps (two estimate bases)",
    "macd_line": "EMA seeding conventions differ between libraries",
    "macd_signal": "EMA seeding conventions differ between libraries",
    "analyst": "Canadian analyst data comes from TMX, not Yahoo (different sources)",
    "fed_funds_rate": "pipeline uses the daily rate, FRED FEDFUNDS is the monthly average",
    "ps_ratio": "Yahoo's P/S divides a CAD market cap by USD revenue for USD reporters",
}
# Pass 2 fields that are empty for a good reason and so are not reported as gaps.
EXPECTED_EMPTY = {"commodity_context", "peer_pe_median", "pe_vs_peer_median_pct"}


@dataclass
class Finding:
    ticker: str
    check: str
    field: str
    status: str  # ok | known | mismatch | gap | skipped
    ours: object = None
    theirs: object = None
    note: str = ""


# ---------------------------------------------------------------------------------------------- pure helpers


def rel_diff(ours: float | None, theirs: float | None) -> float | None:
    """(ours / theirs) - 1, or None when either is missing or theirs is zero."""
    if ours is None or theirs is None or theirs == 0:
        return None
    return ours / theirs - 1


def trailing_sum(values: list[float] | None) -> float | None:
    """Sum of the newest four quarterly values (newest first), or None with fewer than four."""
    if not values or len(values) < 4:
        return None
    return float(sum(values[:4]))


def hand_ratios(s: dict, market_cap: float, factor: float = 1.0) -> dict:
    """Ratios from raw statement numbers `s` (trailing sums and latest balance, in the statement currency), with
    `factor` converting statement amounts to the quote currency. Missing inputs leave a ratio out."""
    out: dict = {}
    ni, rev, opi, gp = s.get("net_income"), s.get("revenue"), s.get("operating_income"), s.get("gross_profit")
    eq, debt = s.get("equity"), s.get("total_debt")
    if ni and ni > 0:
        out["pe_ratio"] = market_cap / (ni * factor)
    if eq:
        out["pb_ratio"] = market_cap / (eq * factor)
    if rev:
        out["ps_ratio"] = market_cap / (rev * factor)
        if ni is not None:
            out["net_margin"] = ni / rev
        if opi is not None:
            out["operating_margin"] = opi / rev
        if gp is not None:
            out["gross_margin"] = gp / rev
    if ni is not None and eq:
        out["roe"] = ni / eq
    if eq and debt is not None:
        out["debt_to_equity"] = debt / eq
    if s.get("current_assets") is not None and s.get("current_liabilities"):
        out["current_ratio"] = s["current_assets"] / s["current_liabilities"]
    ocf, capex = s.get("operating_cash_flow"), s.get("capex")
    if ocf is not None and capex is not None:
        out["free_cash_flow"] = (ocf + capex) * factor
        if ni:
            out["fcf_to_net_income"] = (ocf + capex) / ni
    return out


def compare(
    ticker: str, check: str, ours: dict, theirs: dict, tol: float = 0.02, known: dict | None = None,
    abs_tol: dict[str, float] | None = None,
) -> list[Finding]:
    """One Finding per field present in `theirs`: ok within `tol`, known when it differs for a documented reason,
    mismatch otherwise, gap when ours is missing. `abs_tol` gives a field an absolute tolerance instead (RSI is a 0 to
    100 scale: two points apart is normal between implementations)."""
    known = KNOWN if known is None else known
    abs_tol = abs_tol or {}
    findings = []
    for field, expected in theirs.items():
        value = ours.get(field)
        if value is None:
            findings.append(Finding(ticker, check, field, "gap", None, expected, "ours is empty"))
            continue
        diff = rel_diff(value, expected)
        if field in abs_tol and abs(value - expected) <= abs_tol[field]:
            findings.append(Finding(ticker, check, field, "ok", value, expected))
            continue
        if diff is None or (field not in abs_tol and abs(diff) <= tol):
            findings.append(Finding(ticker, check, field, "ok", value, expected))
        elif field in known:
            findings.append(Finding(ticker, check, field, "known", value, expected, known[field]))
        else:
            findings.append(Finding(ticker, check, field, "mismatch", value, expected, f"{diff * 100:+.1f}%"))
    return findings


def without_not_applicable(theirs: dict, not_applicable: dict | None) -> dict:
    """`theirs` minus the fields the bundle marks not applicable (a bank's operating margin is not a gap)."""
    skip = set((not_applicable or {}).get("fields", []))
    return {k: v for k, v in theirs.items() if k not in skip}


def empty_pass2_fields(views: dict[str, dict]) -> list[tuple[str, str]]:
    """(agent, field) for every Pass 2 passthrough that is empty and neither marked not_applicable nor expected empty."""
    gaps = []
    for agent, view in views.items():
        for field, value in view.items():
            if value is None and field not in EXPECTED_EMPTY:
                gaps.append((agent, field))
    return gaps


def summarize(findings: list[Finding]) -> dict[str, int]:
    counts = {"ok": 0, "known": 0, "mismatch": 0, "gap": 0, "skipped": 0}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    return counts


def format_report(findings: list[Finding], show_ok: bool = False) -> str:
    """Findings grouped by ticker; ok lines only when `show_ok`. Ends with the counts."""
    lines: list[str] = []
    for ticker in dict.fromkeys(f.ticker for f in findings):
        mine = [f for f in findings if f.ticker == ticker]
        lines.append(f"\n=== {ticker} ===")
        for f in mine:
            if f.status == "ok" and not show_ok:
                continue
            tag = f.status.upper()
            detail = f"ours={_fmt(f.ours)} theirs={_fmt(f.theirs)}" if f.status in ("ok", "known", "mismatch", "gap") else ""
            lines.append(f"  {tag:8s} {f.check}/{f.field} {detail} {f.note}".rstrip())
        if all(f.status == "ok" for f in mine) and not show_ok:
            lines.append("  all checks agree")
    c = summarize(findings)
    lines.append(f"\nok {c['ok']} | known differences {c['known']} | MISMATCH {c['mismatch']} | gaps {c['gap']} | skipped {c['skipped']}")
    return "\n".join(lines)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) < 1e6 else f"{value:,.0f}"
    return str(value)


# ----------------------------------------------------------------------------------------------- fetching (network)


DEFAULT_TICKERS = ["MSFT", "KO", "RDDT", "RY.TO", "SHOP.TO", "TD.TO", "CAR-UN.TO"]
BUNDLE_MAX_AGE_SECONDS = 12 * 3600


async def build_bundle(ticker: str, cache_dir: Path | None = None, refresh: bool = False):
    """A real DataBundle from DataPipeline.prepare() in an isolated in-memory database (the real app.db is never
    touched). Cached as a pickle under `cache_dir` for 12 hours: building one takes about a minute because precompute
    makes local model calls (news scoring, filing summaries)."""
    path = cache_dir / f"{ticker.replace('.', '_')}.pkl" if cache_dir else None
    if path and path.exists() and not refresh and time.time() - path.stat().st_mtime < BUNDLE_MAX_AGE_SECONDS:
        return pickle.loads(path.read_bytes())

    from uuid import uuid4

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from api.tables.stock import Stock
    from data.pipeline import DataPipeline
    from data.providers.router import Router
    from data.schemas.context import AnalysisContext

    async with Router(ticker=ticker) as router:
        info = await router.get_company_info(ticker)
    if not info:
        raise ValueError(f"no company info for {ticker}")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Stock.__table__.create(c))
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        stock = Stock(stock_id=uuid4(), canonical_ticker=ticker, company_name=info.get("name") or ticker,
                      primary_exchange=info.get("primary_exchange") or "", currency=info.get("currency") or "",
                      sector=info.get("sector"), industry=info.get("industry"))
        db.add(stock)
        await db.flush()
        bundle = await DataPipeline().prepare(stock.stock_id, AnalysisContext(account_type="tfsa", timeline="medium_term"), db)
        await db.rollback()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(bundle))
    return bundle


def _series(df, *names):
    for name in names:
        if name in df.index:
            return [float(v) for v in df.loc[name].dropna().sort_index(ascending=False).tolist()]
    return None


def check_fundamentals(bundle, usd_cad: float | None) -> list[Finding]:
    """Hand-calculated ratios from raw quarterly statements, plus a Yahoo .info cross-check."""
    import yfinance as yf

    from data.precompute.currency import fx_factor

    t = bundle.stock.ticker
    tk = yf.Ticker(t)
    info = tk.info
    quote_ccy = bundle.price_info["currency"]
    fin_ccy = info.get("financialCurrency") or quote_ccy
    factor = fx_factor(fin_ccy, quote_ccy, usd_cad) or 1.0
    inc, bal, cf = tk.quarterly_income_stmt, tk.quarterly_balance_sheet, tk.quarterly_cashflow
    first = lambda xs: xs[0] if xs else None  # noqa: E731
    raw = {
        "net_income": trailing_sum(_series(inc, "Net Income Common Stockholders", "Net Income")),
        "revenue": trailing_sum(_series(inc, "Total Revenue")),
        "operating_income": trailing_sum(_series(inc, "Operating Income")),
        "gross_profit": trailing_sum(_series(inc, "Gross Profit")),
        "equity": first(_series(bal, "Stockholders Equity", "Common Stock Equity")),
        "total_debt": first(_series(bal, "Total Debt")),
        "current_assets": first(_series(bal, "Current Assets")),
        "current_liabilities": first(_series(bal, "Current Liabilities")),
        "operating_cash_flow": trailing_sum(_series(cf, "Operating Cash Flow")),
        "capex": trailing_sum(_series(cf, "Capital Expenditure")),
    }
    ours = {**bundle.valuation_metrics, **bundle.profitability_metrics, **bundle.balance_sheet_metrics, **bundle.growth_metrics}
    hand = without_not_applicable(hand_ratios(raw, bundle.price_info["market_cap"], factor), bundle.not_applicable)
    findings = compare(t, "fundamentals/hand", ours, hand)
    de = info.get("debtToEquity")
    yahoo = {"pe_ratio": info.get("trailingPE"), "forward_pe": info.get("forwardPE"), "pb_ratio": info.get("priceToBook"),
             "roe": info.get("returnOnEquity"), "operating_margin": info.get("operatingMargins"),
             "revenue_growth_yoy": info.get("revenueGrowth"), "debt_to_equity": de / 100 if de is not None else None}
    yahoo = without_not_applicable({k: v for k, v in yahoo.items() if v is not None}, bundle.not_applicable)
    findings += compare(t, "fundamentals/yahoo", ours, yahoo, tol=0.05)
    return findings


async def check_sec(bundle, usd_cad: float | None) -> list[Finding]:
    """Cross-listed Canadian names: trailing revenue and net income against the SEC (the independent source)."""
    import json

    from data.providers.router import Router

    t = bundle.stock.ticker
    path = Path(__file__).resolve().parents[1] / "data" / "ca_us_crosslisting.json"
    entry = json.loads(path.read_text(encoding="utf-8")).get(t) if path.exists() else None
    if not t.endswith((".TO", ".V")) or not entry:
        return []
    try:
        async with Router(ticker=entry["us_ticker"]) as router:
            fin = await router._providers["edgartools"].normalize_financials(entry["us_ticker"])
    except Exception as exc:  # 40-F filers have no XBRL in edgartools
        return [Finding(t, "sec", "revenue", "skipped", note=f"no SEC statements via edgartools ({type(exc).__name__})")]
    ttm = fin.ttm or {}
    if not ttm.get("revenue") or not ttm.get("net_income"):
        return [Finding(t, "sec", "revenue", "skipped", note="SEC TTM unavailable")]
    from data.precompute.currency import fx_factor

    factor = fx_factor(fin.currency, bundle.price_info["currency"], usd_cad) or 1.0
    cap, v = bundle.price_info["market_cap"], bundle.valuation_metrics
    ours = {}
    if v.get("ps_ratio"):
        ours["revenue"] = cap / v["ps_ratio"]
    if v.get("pe_ratio"):
        ours["net_income"] = cap / v["pe_ratio"]
    theirs = {"revenue": ttm["revenue"] * factor, "net_income": ttm["net_income"] * factor}
    return compare(t, "sec", ours, theirs)


def check_snapshot(bundle) -> list[Finding]:
    """Short interest, analyst count and target, 52-week range, volume and next earnings date against Yahoo."""
    import yfinance as yf

    t = bundle.stock.ticker
    is_ca = t.endswith((".TO", ".V"))
    tk = yf.Ticker(t)
    info = tk.info
    si, ac, pi, ti = bundle.short_interest or {}, bundle.analyst_consensus or {}, bundle.price_info, bundle.technical_indicators
    findings = compare(t, "snapshot", {"shares_short": si.get("shares_short"), "days_to_cover": si.get("days_to_cover"),
                                       "high_52w": pi.get("high_52w"), "low_52w": pi.get("low_52w")},
                       {k: v for k, v in {"shares_short": info.get("sharesShort"), "days_to_cover": info.get("shortRatio"),
                                          "high_52w": info.get("fiftyTwoWeekHigh"), "low_52w": info.get("fiftyTwoWeekLow")}.items() if v})
    findings += compare(t, "snapshot", {"analyst": ac.get("num_analysts"), "target": ac.get("target_mean")},
                        {k: v for k, v in {"analyst": info.get("numberOfAnalystOpinions"), "target": info.get("targetMeanPrice")}.items() if v},
                        tol=0.05, known={"analyst": KNOWN["analyst"], "target": KNOWN["analyst"]} if is_ca else {})
    hist = tk.history(period="3mo", auto_adjust=False)
    findings += compare(t, "snapshot", {"volume_avg_20": ti.get("volume_avg_20")}, {"volume_avg_20": float(hist["Volume"].iloc[-20:].mean())}, tol=0.10)
    nxt = (bundle.earnings_proximity or {}).get("next_earnings_date")
    table = tk.get_earnings_dates(limit=8)
    upcoming = sorted(str(i.date()) for i in table.index if str(i.date()) >= time.strftime("%Y-%m-%d")) if table is not None else []
    if nxt is None:
        findings.append(Finding(t, "snapshot", "next_earnings_date", "gap", None, upcoming[:1] or None))
    elif upcoming and abs((date.fromisoformat(nxt) - date.fromisoformat(upcoming[0])).days) > 3:
        findings.append(Finding(t, "snapshot", "next_earnings_date", "mismatch", nxt, upcoming[0], "more than 3 days apart"))
    else:
        findings.append(Finding(t, "snapshot", "next_earnings_date", "ok", nxt, upcoming[0] if upcoming else None))
    return findings


def check_technicals(bundle) -> list[Finding]:
    """SMA 20/50/200, RSI 14, ATR 14 and MACD recomputed from raw yfinance price history."""
    import pandas as pd
    import yfinance as yf

    t = bundle.stock.ticker
    h = yf.Ticker(t).history(period="2y", auto_adjust=False)
    c, hi, lo = h["Close"], h["High"], h["Low"]
    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    theirs = {"sma_20": c.rolling(20).mean().iloc[-1], "sma_50": c.rolling(50).mean().iloc[-1], "sma_200": c.rolling(200).mean().iloc[-1],
              "rsi_14": (100 - 100 / (1 + rs)).iloc[-1], "atr_14": tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]}
    findings = compare(t, "technicals", bundle.technical_indicators, {k: float(v) for k, v in theirs.items()}, tol=0.05,
                       abs_tol={"rsi_14": 3.0})
    findings += compare(t, "technicals", bundle.technical_indicators, {"macd_line": float(macd.iloc[-1]), "macd_signal": float(macd.ewm(span=9, adjust=False).mean().iloc[-1])}, tol=0.3)
    return findings


async def check_macro(bundle, us: bool = True) -> list[Finding]:
    """US and Canadian macro against FRED and the Bank of Canada; Canadian CPI year-over-year from StatCan raw index values.
    The values are the same for every ticker, so `us` lets a caller run the FRED and Bank of Canada part once."""
    import os

    import requests
    from fredapi import Fred

    t = bundle.stock.ticker
    ms = bundle.macro_sources
    findings: list[Finding] = []
    fred = Fred(api_key=os.environ["SP_FRED_API_KEY"])
    if us:
        ours = {"fed_funds_rate": ms.fed_funds_rate, "treasury_2y": ms.treasury_2y, "treasury_10y": ms.treasury_10y,
                "unemployment": ms.unemployment, "cpi": ms.cpi}
        # A value published after the bundle was built is not an error: ours is right if it equals any of the last
        # five published observations (the latest one is used when none does, so the mismatch shows the real gap).
        theirs = {}
        for name, sid in (("fed_funds_rate", "FEDFUNDS"), ("treasury_2y", "DGS2"), ("treasury_10y", "DGS10"),
                          ("unemployment", "UNRATE"), ("cpi", "CPIAUCSL")):
            recent = [float(v) for v in fred.get_series(sid).dropna().iloc[-5:].tolist()]
            theirs[name] = next((v for v in reversed(recent) if ours[name] is not None and abs(ours[name] / v - 1) <= 0.001), recent[-1])
        findings += compare(t, "macro/fred", ours, theirs, tol=0.001, known={"fed_funds_rate": KNOWN["fed_funds_rate"]})
        boc = requests.get("https://www.bankofcanada.ca/valet/observations/V39079/json?recent=1", timeout=20).json()["observations"][-1]
        findings += compare(t, "macro/boc", {"boc_rate": ms.boc_rate}, {"boc_rate": float(boc["V39079"]["v"])}, tol=0.001)
    if t.endswith((".TO", ".V")):
        from data.providers import stats_canada as sc

        async with sc.StatsCanadaProvider() as provider:
            rows = await provider._fetch_vectors([sc._CPI_NATIONAL_VECTOR], latest_n=14)
        points = sorted(rows[0]["object"]["vectorDataPoint"], key=lambda x: x["refPer"], reverse=True)
        values = [float(p["value"]) for p in points]
        if len(values) >= 13:
            findings += compare(t, "macro/statcan", {"ca_cpi_yoy": ms.ca_cpi_yoy}, {"ca_cpi_yoy": (values[0] / values[12] - 1) * 100}, tol=0.005)
    return findings


def check_currency(bundle) -> list[Finding]:
    """Canadian insider values are in CAD: the implied price per share is near the CAD close, not the USD close."""
    import yfinance as yf

    t = bundle.stock.ticker
    if not t.endswith((".TO", ".V")):
        return []
    # sales and purchases only: an option exercise's price is its strike, not the market price
    rows = [r for r in bundle.insider_activity.get("transactions", [])
            if r.get("value") and r.get("shares") and r.get("date") and r.get("transaction_type") in ("sale", "purchase")]
    if not rows:
        return [Finding(t, "currency", "insider_value", "skipped", note="no valued insider rows")]
    close = yf.Ticker(t).history(period="1y", auto_adjust=False)["Close"]
    row = rows[0]
    day = close[close.index.strftime("%Y-%m-%d") == row["date"]]
    if day.empty:
        return [Finding(t, "currency", "insider_value", "skipped", note=f"no close on {row['date']}")]
    ok = bundle.insider_activity.get("value_currency") == "CAD"
    return compare(t, "currency", {"insider_price_per_share": row["value"] / row["shares"]}, {"insider_price_per_share": float(day.iloc[0])}, tol=0.1) if ok else \
        [Finding(t, "currency", "insider_value", "mismatch", bundle.insider_activity.get("value_currency"), "CAD", "values not converted to CAD")]


def downgrade_if_stale(findings: list[Finding], bundle, hours: float = 6.0) -> list[Finding]:
    """Mismatches in a bundle older than `hours` become known: the published value may have moved since it was built."""
    from datetime import UTC, datetime

    age = (datetime.now(UTC) - bundle.data_vintage).total_seconds() / 3600
    if age <= hours:
        return findings
    return [Finding(f.ticker, f.check, f.field, "known", f.ours, f.theirs, f"bundle is {age:.0f}h old")
            if f.status == "mismatch" else f for f in findings]


def check_gaps(bundle) -> list[Finding]:
    """Pass 2 fields that come through empty (not marked not_applicable, not expected empty)."""
    from services.orchestrator import _build_pass2_view_bundles

    return [Finding(bundle.stock.ticker, "gaps", f"{agent}.{field}", "gap", None, None, "empty in what Pass 2 receives")
            for agent, field in empty_pass2_fields(_build_pass2_view_bundles(bundle))]
