"""SEC cross-listing filing extraction for Canadian companies (ClickUp 86bb47560).

13 of the watchlist's Canadian tickers are also NYSE-cross-listed foreign
private issuers, so their MD&A/business-overview content is reachable
through a US SEC filing (40-F, or 10-K for entities like CP that have
become US domestic filers) instead of the SEDAR+ scraper originally
planned (rejected — no official API, confirmed too fragile).

Lives in providers/ but deliberately does not implement StockDataProvider —
every other file in this directory does. These functions do the same kind
of work as edgartools.py's other methods (fetch from SEC EDGAR, locate the
right piece of content, return it), just for a different filing shape (a
40-F/10-K's MD&A/business exhibit for a Canadian entity) and a fixed,
CA-only ticker subset — not a general US-stock capability, so it was
carved out of EdgarToolsDataProvider rather than added to it. Free
functions, not a class: never on any ABC, nothing calls them through a
class today.

Relationship to sibling files:
- data/tools/resolve_ca_crosslisting.py — the offline resolver that WRITES
  ca_us_crosslisting.json (CA ticker -> CIK + filer form type). Never run
  from here; this module only reads what that tool already verified.
- data/ca_us_crosslisting.json — the mapping this module reads.
- data/ca_us_crosslisting_review.md — tickers the resolver couldn't
  cleanly verify; not in the mapping, so they resolve to None/[] here,
  same as any other unmapped ticker.
"""

import asyncio
import json
import os
import re
from pathlib import Path

from edgar import Company, set_identity
import structlog

logger = structlog.get_logger(__name__)

# Written by src/data/tools/resolve_ca_crosslisting.py — never guess this
# mapping live, see that tool's module docstring for why (confirmed live:
# naive ticker guessing produces confident wrong-company matches, not just
# misses).
_CROSSLISTING_PATH = Path(__file__).resolve().parents[1] / "ca_us_crosslisting.json"

_EDGAR_IDENTITY_ENV = "SP_EDGAR_IDENTITY"
_identity = os.environ.get(_EDGAR_IDENTITY_ENV)
if _identity:
    set_identity(_identity)
# If unset, edgartools raises its own error on first EDGAR request rather than
# here at import — see financial-data-api-research.md §3. Duplicated from
# edgartools.py rather than shared: this is 4 lines, calling set_identity()
# from both modules is harmless (it just sets a process-global env var and
# resets cached httpx clients — no module-scoped state, no ordering risk).

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


def is_crosslisted(ca_ticker: str) -> bool:
    """Mapping-only check, no extraction attempt, no I/O. Lets a caller
    distinguish "never cross-listed" from "cross-listed but this specific
    section's extraction failed" (the known BNS/TRI/CNR-style gap) — both
    of those currently collapse to the same None from get_crosslisted_mda/
    get_crosslisted_business_overview, which is ambiguous on its own."""
    return _load_crosslisting_map().get(ca_ticker) is not None


def get_us_ticker(ca_ticker: str) -> str | None:
    """The mapped US symbol for a cross-listed CA ticker, or None — same
    "mapping-only, no I/O beyond the JSON read" contract as is_crosslisted().

    Required for anything routing a Canadian ticker's data to a US-only
    source (86bbr4azz: CA news merged with the matching Finnhub feed).
    Never derive this by stripping ".TO"/".V" — the map exists precisely
    because that guesses wrong for real tickers, e.g. CNR.TO's US symbol
    is CNI, not "CNR" (Core Natural Resources, an unrelated company —
    see that entry's own manually_confirmed note)."""
    entry = _load_crosslisting_map().get(ca_ticker)
    return entry.get("us_ticker") if entry else None


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


async def _get_native_filing_item(cik: int, form: str, attr: str) -> str | None:
    """`attr` is a property name on edgartools' native parsed-filing object
    — edgar.company_reports.ten_k.TenK for `form="10-K"`
    ("management_discussion"/"business"), or edgar.company_reports.
    twenty_f.TwentyF for `form="20-F"` (same two attribute names, confirmed
    live). Both are parsed natively by edgartools — no exhibit-hunting
    needed, unlike 40-F (see _find_40f_exhibit_text). amendments=False is
    required: an amendment only carries the amended sections (usually just
    Part III exec-comp items for a 10-K), not the full filing — confirmed
    live this drops Item 7 (MD&A) entirely if the amendment is fetched by
    mistake."""
    company = await asyncio.to_thread(Company, cik)
    filings = await asyncio.to_thread(
        lambda: company.get_filings(form=form, amendments=False).head(1)
    )
    if not filings:
        return None
    obj = await asyncio.to_thread(filings[0].obj)
    return getattr(obj, attr, None)


# Returns None/[] for any ticker not in the mapping — the caller's existing
# canadian_data_limited fallback handles that case; nothing here ever
# guesses a cross-listing. Use is_crosslisted() first if you need to tell
# "not mapped" apart from "mapped but this section's extraction failed."


# Form types edgartools parses natively via .obj().management_discussion /
# .business (confirmed live for both) — everything else (40-F) needs the
# exhibit-search path instead. 20-F confirmed live 2026-08-03 against 3 real
# filers (Canada Goose, Celestica, Lithium Americas) found by the ClickUp
# 86bb7h6zt resolver run — turned out simpler than expected: no exhibit
# markers needed at all, same native-parse shape as 10-K.
_NATIVELY_PARSED_FORMS = ("10-K", "20-F")

# Foreign-private-issuer forms — file 6-K for interim filings, unlike a 10-K
# filer (8-K instead, out of scope here). A different partition than
# _NATIVELY_PARSED_FORMS above (which is about extraction *method*, not FPI
# status) — 40-F is FPI but NOT natively parsed; don't conflate the two.
_FOREIGN_PRIVATE_ISSUER_FORMS = ("40-F", "20-F")


async def get_crosslisted_mda(ca_ticker: str) -> str | None:
    """MD&A-equivalent narrative: a 40-F's "Financial Review" exhibit, or
    the native Item 7/Item 5 for a 10-K/20-F filer, depending on the
    entity's actual SEC filer status — see data/ca_us_crosslisting.json's
    `form_type` field. Filer status isn't fixed (CP's 2021 merger changed
    it from 40-F to 10-K), so this always reads the mapping rather than
    assuming 40-F for every entry.

    Returns raw text only, no filing date — the underlying filing object
    does carry `.filing_date` (see get_crosslisted_interim_exhibits, which
    already surfaces it for 6-Ks), so it's cheap to add here once a real
    caller needs it (research_sources.py's `latest_filing_age_days`
    reliability input) rather than guessing the right return shape now."""
    entry = _load_crosslisting_map().get(ca_ticker)
    if entry is None:
        logger.info("crosslisting_not_mapped", ca_ticker=ca_ticker)
        return None
    if entry["form_type"] in _NATIVELY_PARSED_FORMS:
        return await _get_native_filing_item(
            entry["cik"], entry["form_type"], "management_discussion"
        )
    return await _find_40f_exhibit_text(
        entry["cik"], _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS
    )


async def get_crosslisted_business_overview(ca_ticker: str) -> str | None:
    """Business-section equivalent: a 40-F's "Annual Information Form"
    exhibit, or the native Business item for a 10-K/20-F filer. See
    get_crosslisted_mda's docstring re: no filing date in the return value
    yet."""
    entry = _load_crosslisting_map().get(ca_ticker)
    if entry is None:
        logger.info("crosslisting_not_mapped", ca_ticker=ca_ticker)
        return None
    if entry["form_type"] in _NATIVELY_PARSED_FORMS:
        return await _get_native_filing_item(entry["cik"], entry["form_type"], "business")
    return await _find_40f_exhibit_text(
        entry["cik"], _BUSINESS_TITLE_MARKERS, _BUSINESS_TITLE_MARKERS
    )


async def get_crosslisted_interim_exhibits(ca_ticker: str, limit: int = 3) -> list[dict]:
    """Recent 6-K interim exhibits (press releases, some earnings-related)
    — supplementary, near-monthly cadence, not a verbatim transcript. Both
    40-F and 20-F filers are foreign private issuers and use 6-K for
    interim filings; a 10-K filer like CP uses 8-K instead, which this
    function doesn't cover — out of scope for this ticket."""
    entry = _load_crosslisting_map().get(ca_ticker)
    if entry is None or entry["form_type"] not in _FOREIGN_PRIVATE_ISSUER_FORMS:
        return []
    company = await asyncio.to_thread(Company, entry["cik"])
    filings = await asyncio.to_thread(lambda: company.get_filings(form="6-K").head(limit))
    return [
        {"filing_date": str(filing.filing_date), "accession_no": filing.accession_no}
        for filing in filings
    ]
