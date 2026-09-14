"""SEC cross-listing filing extraction for Canadian companies (ClickUp 86bb47560).

176 Canadian tickers are also NYSE/NASDAQ-cross-listed foreign private
issuers (expanded from an initial 13 via 86bb7h6zt), so their MD&A/
business-overview content is reachable through a US SEC filing (40-F,
10-K for entities that have become US domestic filers, or 20-F) instead
of the SEDAR+ scraper originally planned (rejected — no official API,
confirmed too fragile).

Lives in providers/ but deliberately does not implement StockDataProvider —
every other file in this directory does. get_crosslisted_mda/
get_crosslisted_business_overview do the same kind of work as
edgartools.py's other methods (fetch from SEC EDGAR, locate the right
piece of content, return it), just for a different filing shape (a
40-F/10-K's MD&A/business exhibit) and gated on a fixed, CA-only ticker
subset (the crosslisting map) — not a general US-stock capability, so
they were carved out of EdgarToolsDataProvider rather than added to it.
Free functions, not a class: never on any ABC, nothing calls them
through a class today.

get_native_filing_section() (86ban0x1u/2a), the shared retrieval helper
underneath those two, is NOT CA-only at the mechanism level — it takes a
raw CIK or ticker, so EdgarToolsDataProvider.get_filing_section() also
calls it directly, for plain US tickers with no crosslisting-map entry
at all. Only the CIK-resolution path differs: the CA callers below
resolve a CIK from ca_us_crosslisting.json first; the US caller passes
the raw ticker straight through, since edgar.Company() accepts either.

Relationship to sibling files:
- data/tools/resolve_ca_crosslisting.py — the offline resolver that WRITES
  ca_us_crosslisting.json (CA ticker -> CIK + filer form type). Never run
  from here; this module only reads what that tool already verified.
- data/ca_us_crosslisting.json — the mapping this module reads.
- data/ca_us_crosslisting_review.md — tickers the resolver couldn't
  cleanly verify; not in the mapping, so they resolve to None/[] here,
  same as any other unmapped ticker.
- precompute/filing_summarizer.py — the consumer: takes a
  NormalizedFilingSection's `text` and produces a <=250-token FilingDigest.
  This module does retrieval only, never summarization.
- edgartools.py — get_filing_section() reuses get_native_filing_section()
  for plain US tickers outside the crosslisting map (see above).
"""

import asyncio
import json
import os
import re
from datetime import date
from pathlib import Path

from edgar import Company, Filing, set_identity
import structlog

from data.providers.base import NormalizedFilingSection

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
# both patterns exist, against the original 13-ticker map this was built
# and tested against; not re-verified across the full 176-ticker map).
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


def get_expected_cik(ca_ticker: str) -> int | None:
    """The mapped CIK for a cross-listed CA ticker, or None — same
    "mapping-only, no I/O beyond the JSON read" contract as
    is_crosslisted()/get_us_ticker(). Named "expected" (not just
    get_cik()) because this module never fetches a filing to confirm it
    live — it's the static map's own stored value, for a caller (86ban0x1u/
    2b's cik_verified) that wants to cross-check it against a fresh, live
    CIK resolution to catch a stale map entry."""
    entry = _load_crosslisting_map().get(ca_ticker)
    return entry.get("cik") if entry else None


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
) -> NormalizedFilingSection | None:
    company = await asyncio.to_thread(Company, cik)
    filings = await asyncio.to_thread(
        lambda: company.get_filings(form="40-F", amendments=False).head(1)
    )
    if not filings:
        return None
    filing = filings[0]
    attachments = await asyncio.to_thread(lambda: list(filing.attachments))

    def _section(text: str) -> NormalizedFilingSection:
        return {
            "text": text,
            "accession_no": filing.accession_no,
            "filing_date": str(filing.filing_date),
            "source_form": "40-F",
        }

    # Tier 1: description-based match — cheap, no extra HTTP calls.
    for att in attachments:
        desc = _normalize(getattr(att, "description", "") or "")
        if any(marker in desc for marker in description_markers):
            return _section(await asyncio.to_thread(att.text))

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
            return _section(text)
    return None


# --- MD&A source for 40-F filers ---------------------------------------------
#
# The one reliable signal is a document self-identifying in its opening lines:
# a "Management's Discussion and Analysis" / "Financial Review" header, or a
# dated "<Company> reports <period> results" earnings-release headline. So:
# walk the recent 6-Ks newest-first and take the first exhibit that
# self-identifies; fall back to the 40-F's own MD&A exhibit; else None.
#
# Picking by document size instead (an earlier approach) kept grabbing the
# wrong thing - a governance bundle, an NI 43-101 technical report, a
# bid-defence press release, a 100k-char table of contents - each needing its
# own patch. Positive self-identification needs none of them.
# Full findings: docs/technical/ca-crosslisted-filing-digest-findings.md.

_SIX_K_LOOKBACK = 8
_MAX_6K_AGE_DAYS = 300  # past this, the recent 6-Ks don't include a current earnings one
# (the filer went quiet, or files mostly non-earnings 6-Ks) - use the 40-F instead
_PREFER_40F_AGE_DAYS = 120  # a 6-K candidate older than this may predate a just-filed
# 40-F (Q4/annual results land ~90 days after year-end); compare dates and take the newer
_MIN_MDA_CHARS = 8_000  # below this an exhibit is a cover page / stub, not disclosure

_RVIEWER_RE = re.compile(r"r\d+\.html?$", re.I)  # XBRL "R1.htm" viewer files
_SIX_K_COVER_RE = re.compile(r"(^|[_-])6-?k\.html?$", re.I)  # the 6-K cover doc itself
_CERT_NAME_RE = re.compile(r"ex-?(hibit)?[_-]?3[12][_.]|dex3[12]|cert", re.I)

# The MD&A self-title, whitespace-flexible so a heading boxed with pipes/borders
# ("Management's | Discussion | and | Analysis") still matches in un-normalized text.
_MDA_PHRASE = r"management['’]s\s+discussion\s+(?:and|&)\s+analysis"
_MDA_HEADER_RE = re.compile(_MDA_PHRASE + r"|\bfinancial review\b", re.I)
_RESULTS_LANG_RE = re.compile(  # actual financial-results vocabulary, not a headline word
    r"net (?:income|earnings|loss)|\brevenue|diluted (?:eps|earnings per share)"
    r"|adjusted (?:ebitda|earnings|net)|results of operations",
    re.I,
)
# The three ingredients of an earnings-release headline, checked for co-occurrence
# near the top rather than as one brittle ordered pattern.
_ACTION_RE = re.compile(r"\b(report|announce|deliver|post|release|provide)\w*", re.I)
_PERIOD_RE = re.compile(
    r"\b((first|second|third|fourth)[- ]quarter|q[1-4]\b|fiscal (?:year|20\d\d)"
    r"|full[- ]year|half[- ]year|interim|(?:three|six|nine|twelve)[- ]month)\b",
    re.I,
)
_RESULTS_RE = re.compile(r"\b(results|earnings)\b", re.I)


def _strip_borders(text: str) -> str:
    return re.sub(r"[+\-|]{2,}", " ", text)


def _looks_like_mda_or_earnings_release(text: str) -> bool:
    """True only if the exhibit self-identifies in its opening lines - a real
    MD&A / "Financial Review" header, or a dated earnings-release headline
    (an action verb + a period + "results"/"earnings" co-occurring in the
    title zone). This is the single reliable signal; size or a stray "net
    income" mention misfires on governance filings, technical reports and
    bid-defence press releases."""
    stripped = text.lstrip()
    if not stripped or stripped[0].islower():
        return False  # an extraction fragment, not a document start (CAE's bad exhibit)
    head = _strip_borders(_normalize(stripped[:900]))
    if _MDA_HEADER_RE.search(head[:400]):
        return True
    title = head[:300]
    return bool(_ACTION_RE.search(title) and _PERIOD_RE.search(title) and _RESULTS_RE.search(head))


_BODY_HEADER_RE = re.compile(
    r"(?:corporate structure"
    r"|general development of (?:the |our )?business"
    r"|(?:narrative )?description of (?:the |our )?business"
    r"|management['’]s discussion (?:and|&) analysis"
    r"|results of operations|overview(?: of (?:the |our )?(?:business|results))?"
    r"|financial (?:review|highlights|condition))"
    r"(?![\s|+\-.]*\d)",  # a TOC entry is followed by a page number; a real header is not
    re.I,
)
_TOC_LINE_RE = re.compile(r"^\s*\S.{0,70}?\s+\d{1,3}\s*$", re.M)  # "Section .... 12"


def _skip_front_matter(text: str) -> str:
    """A filing that opens with a table of contents / glossary can bury the
    real narrative deep in the document (GFL's "Description of the Business"
    header is at char 150,000, well past the 30k the summarizer reads). Skip
    to the first real section header - one that is not a TOC entry (those
    carry a trailing page number). No-op if the text doesn't open with
    front matter or no such header is found."""
    head = text[:4_000].lower()
    looks_like_toc = "table of contents" in head or len(_TOC_LINE_RE.findall(text[:6_000])) >= 4
    if not looks_like_toc:
        return text
    for m in _BODY_HEADER_RE.finditer(text[:250_000]):
        if m.start() > 1_200:
            return text[m.start() :]
    return text


def _within_days(filing_date: date | str, days: int) -> bool:
    """edgartools gives `.filing_date` as a date; be tolerant of a string too."""
    if isinstance(filing_date, str):
        try:
            filing_date = date.fromisoformat(filing_date[:10])
        except ValueError:
            return False
    return (date.today() - filing_date).days <= days


async def _first_self_identifying_exhibit(filing: Filing) -> str | None:
    """Text of the first `.htm` exhibit in this 6-K that self-identifies as an
    MD&A or a dated earnings release (see _looks_like_mda_or_earnings_release).
    None if the 6-K has none - it is a dividend / transaction / governance
    filing, not an earnings one."""
    attachments = await asyncio.to_thread(lambda: list(filing.attachments))
    for att in attachments:
        doc = (getattr(att, "document", "") or "").lower()
        if not doc.endswith((".htm", ".html")):
            continue
        if _RVIEWER_RE.search(doc) or _SIX_K_COVER_RE.search(doc) or _CERT_NAME_RE.search(doc):
            continue
        try:
            text = (await asyncio.to_thread(att.text)) or ""
        except Exception:  # noqa: BLE001 (one bad exhibit shouldn't sink the filing)
            continue
        if len(text) >= _MIN_MDA_CHARS and _looks_like_mda_or_earnings_release(text):
            return text
    return None


_STALE_40F_DAYS = 550  # an annual filer's current 40-F is at most ~16 months old just
# before the next one; older means the filer stopped (BAM's / SHOP's last 40-F is 2024)


async def _annual_40f_mda(cik: int) -> NormalizedFilingSection | None:
    """The 40-F's own MD&A exhibit, front matter skipped. None if the exhibit
    can't be located, if the most recent 40-F is too old to be current, or if
    after skipping the front matter it still doesn't read as MD&A prose (an
    extraction fragment or a table-of-contents-only match - BN)."""
    section = await _find_40f_exhibit_text(cik, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)
    if section is None or not _within_days(section["filing_date"], _STALE_40F_DAYS):
        return None
    skipped = _skip_front_matter(section["text"]).lstrip()
    if not skipped or skipped[0].islower():
        return None  # started mid-sentence - an extraction fragment, not the MD&A (BN)
    body = _normalize(skipped[:3_000])
    if _MDA_HEADER_RE.search(body) or (_PERIOD_RE.search(body) and _RESULTS_LANG_RE.search(body)):
        return {**section, "text": skipped}
    return None


async def _find_mda_source(cik: int) -> NormalizedFilingSection | None:
    """MD&A text for a 40-F filer: the most recent self-identifying earnings
    6-K, or the 40-F's own MD&A exhibit - whichever is fresher. None if
    neither resolves. See the module comment above."""
    company = await asyncio.to_thread(Company, cik)
    six_ks = await asyncio.to_thread(
        lambda: list(company.get_filings(form="6-K", amendments=False).head(_SIX_K_LOOKBACK))
    )

    candidate: NormalizedFilingSection | None = None
    for filing in six_ks:  # newest first
        if not _within_days(filing.filing_date, _MAX_6K_AGE_DAYS):
            break  # older filings are only older - stop walking back
        text = await _first_self_identifying_exhibit(filing)
        if text:
            candidate = {
                "text": text,
                "accession_no": filing.accession_no,
                "filing_date": str(filing.filing_date),
                "source_form": "6-K",
            }
            break

    # A recent 6-K wins outright. An older one (or none) is weighed against the
    # 40-F's MD&A - just after year-end the annual filing is the fresher one.
    if candidate is not None and _within_days(candidate["filing_date"], _PREFER_40F_AGE_DAYS):
        return candidate
    annual = await _annual_40f_mda(cik)
    if candidate is None:
        return annual
    if annual is None:
        return candidate
    return candidate if candidate["filing_date"] >= annual["filing_date"] else annual


async def get_native_filing_section(
    cik_or_ticker: int | str, form: str, attr: str
) -> NormalizedFilingSection | None:
    """`attr` is a property name on edgartools' native parsed-filing object
    — edgar.company_reports.ten_k.TenK for `form="10-K"`
    ("management_discussion"/"business"), or edgar.company_reports.
    twenty_f.TwentyF for `form="20-F"` (same two attribute names, confirmed
    live). Both are parsed natively by edgartools — no exhibit-hunting
    needed, unlike 40-F (see _find_40f_exhibit_text). amendments=False is
    required: an amendment only carries the amended sections (usually just
    Part III exec-comp items for a 10-K), not the full filing — confirmed
    live this drops Item 7 (MD&A) entirely if the amendment is fetched by
    mistake.

    Public (no leading underscore) and takes `cik_or_ticker`, not just a
    CIK: edgar.Company() accepts either natively, and edgartools.py's
    get_filing_section() (86ban0x1u/2a) calls this directly with a plain
    ticker for non-crosslisted US names — the CA callers below still pass
    a real CIK resolved from ca_us_crosslisting.json, since guessing a
    ticker->CIK mapping live is exactly what this module exists to avoid
    (see the module docstring).

    Real, confirmed gap closed here: the CA callers only ever pass a CIK
    already verified by the offline resolver, so `Company()` raising
    "not found" was never a real path for them — but a raw ticker string
    (the new US caller) can be genuinely invalid, and edgar.Company()
    raises CompanyNotFoundError rather than returning None for one.
    Confirmed live with a real invalid ticker before this guard existed:
    it propagated uncaught, contradicting this function's own "degrades
    to None on any failure" contract. Catches broadly (like this file's
    other external-library boundaries, e.g. the filing.obj() parse in
    edgartools.py's get_insider_trading) rather than importing edgar's
    exact exception type, since the library can raise more than one
    thing for "this isn't a real company."""
    try:
        company = await asyncio.to_thread(Company, cik_or_ticker)
    except Exception:
        logger.info("edgar_company_not_found", cik_or_ticker=cik_or_ticker)
        return None
    filings = await asyncio.to_thread(
        lambda: company.get_filings(form=form, amendments=False).head(1)
    )
    if not filings:
        return None
    filing = filings[0]
    obj = await asyncio.to_thread(filing.obj)
    text = getattr(obj, attr, None)
    if not text:
        return None
    return {
        "text": text,
        "accession_no": filing.accession_no,
        "filing_date": str(filing.filing_date),
        "source_form": form,
    }


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


async def get_crosslisted_mda(ca_ticker: str) -> NormalizedFilingSection | None:
    """MD&A-equivalent narrative plus its filing provenance (see
    NormalizedFilingSection). Source depends on the entity's actual SEC filer
    status (data/ca_us_crosslisting.json's `form_type`), which isn't fixed
    (CP's 2021 merger changed it from 40-F to 10-K):

    - 10-K / 20-F filer: the natively-parsed Item 7 / Item 5 MD&A.
    - 40-F filer: the quarterly earnings 6-K's MD&A or press-release
      exhibit (`_find_mda_source`), falling back to the annual 40-F MD&A
      exhibit. The 6-K disclosures are smaller, quarterly-fresh, and
      self-identify more often than the 1-2M-char annual exhibit, which is
      inconsistently titled and returns nothing for several banks
      (BNS/BMO). See docs/technical/ca-crosslisted-filing-digest-findings.md.
    """
    entry = _load_crosslisting_map().get(ca_ticker)
    if entry is None:
        logger.info("crosslisting_not_mapped", ca_ticker=ca_ticker)
        return None
    if entry["form_type"] in _NATIVELY_PARSED_FORMS:
        return await get_native_filing_section(
            entry["cik"], entry["form_type"], "management_discussion"
        )
    return await _find_mda_source(entry["cik"])


async def get_crosslisted_business_overview(ca_ticker: str) -> NormalizedFilingSection | None:
    """Business-section equivalent plus filing provenance (see
    NormalizedFilingSection): a 40-F's "Annual Information Form" exhibit, or the
    native Business item for a 10-K/20-F filer. No quarterly equivalent
    exists for the AIF, so 40-F filers always use the annual exhibit here.

    Known residual gap: CNR files no separate AIF exhibit in its 40-F
    (financials incorporated by reference), so this returns None for it —
    graceful, same as any unmapped ticker. Its MD&A digest still resolves
    via the 6-K path in get_crosslisted_mda.
    """
    entry = _load_crosslisting_map().get(ca_ticker)
    if entry is None:
        logger.info("crosslisting_not_mapped", ca_ticker=ca_ticker)
        return None
    if entry["form_type"] in _NATIVELY_PARSED_FORMS:
        return await get_native_filing_section(entry["cik"], entry["form_type"], "business")
    section = await _find_40f_exhibit_text(
        entry["cik"], _BUSINESS_TITLE_MARKERS, _BUSINESS_TITLE_MARKERS
    )
    if section is None:
        return None
    return {**section, "text": _skip_front_matter(section["text"])}


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
