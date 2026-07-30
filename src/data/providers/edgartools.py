import asyncio
import json
import os
import re
from pathlib import Path

import pandas as pd
import structlog
from edgar import Company, set_identity

from data.providers.base import StockDataProvider
from data.providers.yfinance import YFinanceDataProvider, correct_alignment

logger = structlog.get_logger(__name__)

# Written by src/data/tools/resolve_ca_crosslisting.py (ClickUp 86bb47560) —
# never guess this mapping live, see that tool's module docstring for why
# (confirmed live: naive ticker guessing produces confident wrong-company
# matches, not just misses).
_CROSSLISTING_PATH = Path(__file__).resolve().parents[1] / "ca_us_crosslisting.json"

_EDGAR_IDENTITY_ENV = "SP_EDGAR_IDENTITY"
_identity = os.environ.get(_EDGAR_IDENTITY_ENV)
if _identity:
    set_identity(_identity)
# If unset, edgartools raises its own error on first EDGAR request rather than
# here at import — see financial-data-api-research.md §3.

# (period, statement) -> Financials accessor. Mirrors yfinance.py's MAPPING pattern.
STATEMENT_MAP = {
    "income": lambda financials: financials.income_statement,
    "balance": lambda financials: financials.balance_sheet,
    "cashflow": lambda financials: financials.cashflow_statement,  # no underscore
}

# Gap 5b (financial-data-api-research.md) describes a "fewer than 5 key line items"
# XBRL-quality check for small-caps. Deliberately not implemented: live-tested
# against ~25 real small/micro-cap and pre-revenue tickers (including deliberately
# thin filers like pre-revenue biotechs) and it never fired below 3-of-5 on any of
# them — while a literal "all 5 required" reading misfired on JPM, Visa, and P&G
# (mega-caps whose sector legitimately has no GrossProfit/OperatingIncome line,
# not an XBRL tagging problem). The only fallback trigger kept is get_financials()
# returning None outright (no computed Financials object — e.g. no XBRL-tagged
# filings), which is real and cheap to check.

# Title phrases identifying the MD&A-equivalent and Business-equivalent exhibits
# in a 40-F filing. Checked two ways (see _find_40f_exhibit_text): first against
# each attachment's `description` (works for filers like RY that label exhibits
# descriptively — "EX-2 FINANCIAL REVIEW"), then, if nothing matches, against the
# first ~2000 characters of exhibit text (needed for filers like TD that use
# generic "EX-99.1"-style labels with no description at all — confirmed live
# both patterns exist across the 13 confirmed cross-listed tickers).
#
# Two different marker sets, not one, confirmed necessary live: TD's "Annual
# Information Form" exhibit (ex991.htm) defines "MD&A" as a shorthand term on
# its own cover page ('...are disclosed in... and management's discussion and
# analysis... (the "2025 MD&A")... incorporated by reference...') — that's
# genuinely within the first ~1100 raw characters, so the bare "md&a" marker
# false-matched the WRONG exhibit before ever reaching the real one. A short,
# curated `description` field ("EX-2 FINANCIAL REVIEW") doesn't have this
# problem — nobody labels an unrelated exhibit that way — so the abbreviation
# is safe there; it's only unsafe against full document body text.
_MDA_DESCRIPTION_MARKERS = ("financial review", "management's discussion and analysis", "md&a")
_MDA_CONTENT_MARKERS = ("financial review", "management's discussion and analysis")
_BUSINESS_TITLE_MARKERS = ("annual information form",)

# Exhibits that are never the MD&A/Business content — cheap to rule out by
# description before spending an HTTP call fetching the full text.
_NON_CANDIDATE_DESC_MARKERS = ("certification", "consent", "xbrl")

# How close to the start of a candidate exhibit's (normalized) text a title
# marker must appear to count — see _find_40f_exhibit_text tier 2. Confirmed
# live against TD's AIF exhibit: its real title ("ANNUAL INFORMATIONFORM")
# lands at normalized offset 142 (after border-noise stripping — see
# _normalize), while that same exhibit's own cover-page cross-reference to
# the MD&A ("...and management's discussion and analysis... incorporated by
# reference...") lands at offset 276. 220 sits between the two with margin
# on both sides.
_TITLE_WINDOW_CHARS = 220

_BORDER_NOISE_RE = re.compile(r"[+\-|=]{3,}")
# Table-cell pipe separators, any count — not just runs of 3+. Confirmed live
# (BNS's AIF): a title can be rendered as isolated single "|" characters
# between EVERY word from PDF-table-to-text conversion — "annual | |
# information | form" — which _BORDER_NOISE_RE doesn't touch (it only
# matches runs of 3+) and which breaks an exact-substring marker match even
# after whitespace collapsing, since the pipes become their own tokens.
_TABLE_PIPE_RE = re.compile(r"\|")


def _load_crosslisting_map() -> dict:
    if not _CROSSLISTING_PATH.exists():
        return {}
    return json.loads(_CROSSLISTING_PATH.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    # Curly apostrophe (’) -> straight: confirmed live that filers' own
    # PDF-to-text conversion uses "Management’s", which a plain ASCII
    # "management's" marker silently fails to match without this.
    #
    # Strip ASCII-art table/box borders (runs of 3+ +/-/|/= characters):
    # confirmed live these consume hundreds of characters of "distance" even
    # after whitespace collapsing (a single "+---...---+" border line is one
    # long non-whitespace token), which pushed a genuine boxed title past
    # _TITLE_WINDOW_CHARS before this stripping was added.
    text = _BORDER_NOISE_RE.sub(" ", text)
    text = _TABLE_PIPE_RE.sub(" ", text)
    return " ".join(text.replace("’", "'").split()).lower()


def _is_candidate_40f_exhibit(document_name: str, description: str) -> bool:
    if not document_name.lower().endswith(".htm"):
        return False
    normalized_desc = _normalize(description)
    if normalized_desc in ("graphic", "idea: xbrl document"):
        return False
    return not any(marker in normalized_desc for marker in _NON_CANDIDATE_DESC_MARKERS)


async def _find_40f_exhibit_text(
    cik: int, description_markers: tuple[str, ...], content_markers: tuple[str, ...]
) -> str | None:
    company = await asyncio.to_thread(Company, cik)
    filings = await asyncio.to_thread(
        lambda: company.get_filings(form="40-F", amendments=False).head(1)
    )
    if not filings:
        return None
    attachments = await asyncio.to_thread(lambda: list(filings[0].attachments))

    # Tier 1: description-based match — cheap, no extra HTTP calls.
    for att in attachments:
        desc = _normalize(getattr(att, "description", "") or "")
        if any(marker in desc for marker in description_markers):
            return await asyncio.to_thread(att.text)

    # Tier 2: content-based fallback for filers that don't label exhibits
    # descriptively — check each remaining candidate's own document title.
    # Confirmed live this needs a TIGHT window, not "anywhere in the first
    # 2000 characters": TD's AIF exhibit references the MD&A by name in its
    # own cover-page "incorporated by reference" boilerplate at normalized
    # offset ~756 ("...consolidated financial statements and management's
    # discussion and analysis... incorporated by reference..."), while the
    # real MD&A exhibit's own title appears at offset ~37. A 2000-char
    # window catches both and returns whichever comes first in filing
    # order — wrong. _TITLE_WINDOW_CHARS sits well below that 756 gap.
    # (Slice first, normalize second: normalizing the whole document before
    # slicing would collapse hundreds of KB of box-drawing whitespace into a
    # deceptively short distance, hiding how far into the document a
    # false-positive phrase actually occurs.)
    #
    # Known limitation, accepted deliberately: this only finds exhibits that
    # self-title themselves near the top. Confirmed live that BNS's actual
    # ~1MB MD&A exhibit has no title header at all — it opens directly with
    # substantive content ("Enhanced Disclosure Task Force (EDTF)
    # recommendations...") — so this returns None for it rather than the
    # real content. Widening the window to catch cases like this would
    # reopen the false-positive risk above (TD's cross-reference sits at
    # offset ~756, not far past a wider window). False negative (a caller
    # gets None and can fall back to the existing canadian_data_limited
    # treatment) is the deliberately safer failure mode here, not false
    # positive (a caller silently gets the wrong exhibit's content).
    for att in attachments:
        if not _is_candidate_40f_exhibit(att.document, getattr(att, "description", "") or ""):
            continue
        try:
            text = await asyncio.to_thread(att.text)
        except Exception:
            continue
        if not text:
            continue
        title_area = _normalize(text[:2000])[:_TITLE_WINDOW_CHARS]
        if any(marker in title_area for marker in content_markers):
            return text
    return None


async def _get_10k_item(cik: int, attr: str) -> str | None:
    """`attr` is an edgar.company_reports.ten_k.TenK property name
    ("management_discussion" or "business") — edgartools parses these
    natively for 10-K filers, no exhibit-hunting needed. amendments=False is
    required: a 10-K/A only carries the amended sections (usually just Part
    III exec-comp items), not the full filing — confirmed live this drops
    Item 7 (MD&A) entirely if the amendment is fetched by mistake."""
    company = await asyncio.to_thread(Company, cik)
    filings = await asyncio.to_thread(
        lambda: company.get_filings(form="10-K", amendments=False).head(1)
    )
    if not filings:
        return None
    obj = await asyncio.to_thread(filings[0].obj)
    return getattr(obj, attr, None)


class EdgarToolsDataProvider(StockDataProvider):
    """US-stock fundamentals and insider transactions, sourced directly from SEC
    EDGAR (edgartools) — no ticker whitelist, no tier restriction, unlike FMP's
    free-tier annual-only fundamentals. Covers only the US-stock routing branch:
    prices, profile, estimates, ratings, peers, earnings calendar, and dividends
    are fmp.py's job, not this provider's."""

    def __init__(self, fallback: YFinanceDataProvider | None = None) -> None:
        self._fallback = fallback or YFinanceDataProvider()

    async def get_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        key = statement.lower()
        if key not in STATEMENT_MAP:
            raise ValueError("Invalid statement. Use 'income', 'balance', or 'cashflow'")

        company = await asyncio.to_thread(Company, ticker)
        get_financials_fn = (
            company.get_financials
            if period.lower() == "annual"
            else company.get_quarterly_financials
        )
        financials = await asyncio.to_thread(get_financials_fn)
        if financials is None:
            logger.warning(
                "edgar_financials_missing", ticker=ticker, statement=statement, period=period
            )
            return await self._fallback_financials(ticker, statement, period)

        statement_fn = STATEMENT_MAP[key](financials)
        return await asyncio.to_thread(lambda: statement_fn().to_dataframe())

    async def _fallback_financials(self, ticker: str, statement: str, period: str) -> pd.DataFrame:
        df = await self._fallback.get_financials(ticker, statement, period)
        return correct_alignment(df)

    async def get_insider_trading(self, ticker: str, days: int = 90) -> list[dict]:
        """Transactional-level Form 4 data (date, shares, price, insider name) —
        not the aggregate-only data openbb-tmx gives for Canadian stocks."""
        company = await asyncio.to_thread(Company, ticker)
        filings = await asyncio.to_thread(lambda: company.get_filings(form="4").head(20))
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)

        records: list[dict] = []
        for filing in filings:
            try:
                form4 = await asyncio.to_thread(filing.obj)
                df = await asyncio.to_thread(form4.to_dataframe)
            except Exception:
                logger.warning(
                    "edgar_form4_parse_failed",
                    ticker=ticker,
                    accession=filing.accession_no,
                    exc_info=True,
                )
                continue
            if df is None or df.empty or "Date" not in df.columns:
                continue
            for _, row in df[df["Date"] >= cutoff].iterrows():
                records.append(
                    {
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "shares": float(row["Shares"]) if pd.notna(row["Shares"]) else None,
                        "price": float(row["Price"]) if pd.notna(row["Price"]) else None,
                        "insider_name": row.get("Insider"),
                        "transaction_type": row.get("Transaction Type"),
                        "code": row.get("Code"),
                    }
                )
        return records

    # --- Canadian cross-listed filing coverage (ClickUp 86bb47560) ---
    # Not on StockDataProvider — these are additive methods for the 13
    # confirmed cross-listed CA tickers in data/ca_us_crosslisting.json, not
    # part of the US-stock routing contract. Returns None/[] for any ticker
    # not in that mapping — the caller's existing canadian_data_limited
    # fallback handles that case; this class never guesses a cross-listing.

    async def get_crosslisted_mda(self, ca_ticker: str) -> str | None:
        """MD&A-equivalent narrative: a 40-F's "Financial Review" exhibit, or
        a 10-K's Item 7, depending on the entity's actual SEC filer status —
        see data/ca_us_crosslisting.json's `form_type` field. Filer status
        isn't fixed (CP's 2021 merger changed it from 40-F to 10-K), so this
        always reads the mapping rather than assuming 40-F for every entry."""
        entry = _load_crosslisting_map().get(ca_ticker)
        if entry is None:
            logger.info("crosslisting_not_mapped", ca_ticker=ca_ticker)
            return None
        if entry["form_type"] == "10-K":
            return await _get_10k_item(entry["cik"], "management_discussion")
        return await _find_40f_exhibit_text(
            entry["cik"], _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS
        )

    async def get_crosslisted_business_overview(self, ca_ticker: str) -> str | None:
        """Business-section equivalent: a 40-F's "Annual Information Form"
        exhibit, or a 10-K's Item 1 (Business)."""
        entry = _load_crosslisting_map().get(ca_ticker)
        if entry is None:
            logger.info("crosslisting_not_mapped", ca_ticker=ca_ticker)
            return None
        if entry["form_type"] == "10-K":
            return await _get_10k_item(entry["cik"], "business")
        return await _find_40f_exhibit_text(
            entry["cik"], _BUSINESS_TITLE_MARKERS, _BUSINESS_TITLE_MARKERS
        )

    async def get_crosslisted_interim_exhibits(self, ca_ticker: str, limit: int = 3) -> list[dict]:
        """Recent 6-K interim exhibits (press releases, some earnings-related)
        — supplementary, near-monthly cadence, not a verbatim transcript.
        Only 40-F filers use 6-K; a 10-K filer like CP uses 8-K instead,
        which this method doesn't cover — out of scope for this ticket."""
        entry = _load_crosslisting_map().get(ca_ticker)
        if entry is None or entry["form_type"] != "40-F":
            return []
        company = await asyncio.to_thread(Company, entry["cik"])
        filings = await asyncio.to_thread(lambda: company.get_filings(form="6-K").head(limit))
        return [
            {"filing_date": str(filing.filing_date), "accession_no": filing.accession_no}
            for filing in filings
        ]

    # Out of scope for edgartools — SEC EDGAR has no price, profile, estimates,
    # ratings, peer, earnings-calendar, or dividend data. That's fmp.py's job
    # per the documented routing rule; fake it here rather than fabricating it.
    async def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError("Price history is out of scope for edgartools — use fmp.py")

    async def get_company_info(self, ticker: str) -> dict:
        raise NotImplementedError("Company info is out of scope for edgartools — use fmp.py")

    async def get_analyst_estimates(self, ticker: str) -> dict:
        raise NotImplementedError("Analyst estimates are out of scope for edgartools — use fmp.py")

    async def get_analyst_ratings(self, ticker: str) -> dict:
        raise NotImplementedError("Analyst ratings are out of scope for edgartools — use fmp.py")

    async def get_peers(self, ticker: str, limit: int = 5) -> list[str]:
        raise NotImplementedError("Peers are out of scope for edgartools — use fmp.py")

    async def get_earnings_calendar(self, ticker: str) -> list[dict]:
        raise NotImplementedError("Earnings calendar is out of scope for edgartools — use fmp.py")

    async def get_dividend_history(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        raise NotImplementedError("Dividend history is out of scope for edgartools — use fmp.py")
