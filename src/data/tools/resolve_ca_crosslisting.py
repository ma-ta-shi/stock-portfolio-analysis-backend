"""Resolves Canadian tickers to their US SEC cross-listing (CIK + filer form
type), for ClickUp 86bb47560. Offline maintenance tool — run whenever the
watchlist changes, never from the live agent pipeline. Writes two artifacts:

- data/ca_us_crosslisting.json — auto-verified entries, read at runtime by
  edgartools.py's get_crosslisted_* methods.
- data/ca_us_crosslisting_review.md — anything that didn't cleanly verify,
  in plain language with the specific reason. Not written to the mapping
  file; that ticker keeps the existing canadian_data_limited fallback until
  someone resolves the flagged entry (see the review file for a suggested
  candidate, if one was found).

Usage: python -m data.tools.resolve_ca_crosslisting

Design, confirmed against live SEC data (2026-07-30):
- Never live-guess a US ticker from a CA one inside the agent pipeline —
  edgartools' own company search produced confident false positives (VGRO ->
  "Virtus ETF Trust II", CRE -> "Cre8 Enterprise Ltd", GOP -> "Tidal Trust
  I", all unrelated companies).
- SEC's own officially-maintained company_tickers.json is a much better
  starting point (free, no key, self-updating) — resolved 12 of 13 target
  tickers correctly with zero manual research.
- A ticker match alone isn't enough verification, though: the one miss
  (CNR -> "Core Natural Resources, Inc.", an unrelated US 10-K filer) is
  caught by checking whether the candidate CIK's filing history actually
  contains 40-F/20-F/10-K — a categorical signal, not fuzzy matching, and
  it's what a bare ticker match can't tell you. CNR's real match (Canadian
  National Railway, ticker CNI) only turns up via the name-based fallback
  search, confirmed to file 40-F.
- Filer status isn't a fixed property of "being a cross-listed Canadian
  company" — CP (Canadian Pacific Kansas City) files 10-K, not 40-F, because
  its 2021 merger with Kansas City Southern changed its SEC filer
  classification to a US domestic filer. Check for 40-F, 20-F, AND 10-K,
  and record which one so edgartools.py knows which extraction path to use.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def _user_agent() -> str:
    """SEC's fair-use policy requires a real "Name email@domain.com"-shaped
    contact string in the User-Agent, not just a descriptive label — a
    generic placeholder gets a 403, confirmed live. Reuse edgartools'
    SP_EDGAR_IDENTITY env var rather than inventing a second identity."""
    identity = os.environ.get("SP_EDGAR_IDENTITY")
    if not identity:
        raise RuntimeError(
            "SP_EDGAR_IDENTITY is not set. SEC requires a 'Name email@example.com' "
            "contact string in the User-Agent header for data.sec.gov/www.sec.gov requests."
        )
    return identity


# Forms that indicate "this CIK has SEC-filed MD&A-style content available."
# 40-F/20-F = foreign private issuer; 10-K = a cross-listed company that has
# become (or always was) a US domestic filer — confirmed live via CP.
RELEVANT_FORMS = ("40-F", "20-F", "10-K")

# Between-request pacing for data.sec.gov calls — this tool runs rarely (only
# when the watchlist changes), so courtesy pacing costs nothing and SEC's fair
# use terms don't document a hard rate limit to target instead.
_REQUEST_DELAY_SECONDS = 0.25

# The watchlist's Canadian subset (frontend/src/hooks/use-ticker-search.ts).
# Update this list when the watchlist changes, then re-run this tool.
WATCHLIST_CA_TICKERS: list[tuple[str, str]] = [
    ("RY.TO", "Royal Bank of Canada"),
    ("TD.TO", "Toronto-Dominion Bank"),
    ("BMO.TO", "Bank of Montreal"),
    ("BNS.TO", "Bank of Nova Scotia"),
    ("CM.TO", "CIBC"),
    ("WSP.TO", "WSP Global Inc."),
    ("CNR.TO", "Canadian National Railway"),
    ("CP.TO", "Canadian Pacific Kansas City"),
    ("ATD.TO", "Alimentation Couche-Tard"),
    ("SU.TO", "Suncor Energy"),
    ("ENB.TO", "Enbridge Inc."),
    ("BCE.TO", "BCE Inc."),
    ("SHOP.TO", "Shopify Inc."),
    ("MFC.TO", "Manulife Financial"),
    ("TRI.TO", "Thomson Reuters"),
]

_NAME_NOISE_WORDS = {
    "INC",
    "INC.",
    "CORP",
    "CORP.",
    "CORPORATION",
    "LTD",
    "LTD.",
    "LIMITED",
    "CO",
    "CO.",
    "COMPANY",
    "THE",
    "GROUP",
    "GLOBAL",
    "INTERNATIONAL",
    "HOLDINGS",
    "HOLDING",
    "WORLDWIDE",
    "ENTERPRISES",
    "/CAN/",
    "/CN/",
}


def _name_tokens(name: str) -> set[str]:
    """Uppercase, strip punctuation, drop common corporate-suffix/generic-
    descriptor noise words — good enough to tell "Royal Bank of Canada" apart
    from "Core Natural Resources, Inc." without needing a real entity-
    resolution library. Generic words like GLOBAL/HOLDINGS are stripped
    because they're too weak a signal on their own — confirmed live this
    caused "WSP Global Inc." to false-match "S&P Global Inc." on the single
    shared word GLOBAL."""
    cleaned = re.sub(r"[^\w\s]", " ", name.upper())
    return {tok for tok in cleaned.split() if tok not in _NAME_NOISE_WORDS}


def _names_plausibly_match(expected: str, candidate: str) -> bool:
    """Deliberately strict: ALL of the expected name's meaningful tokens must
    appear in the candidate, not just a majority — this is what auto-accepts
    a name-fallback search result, so a loose threshold is a false-positive
    risk, not just a false-negative one. A 50%-overlap threshold previously
    let "WSP Global" match "S&P Global" on the word GLOBAL alone.

    This is still a cheap heuristic, not real entity resolution — it doesn't
    reliably distinguish an acronym (CIBC vs "Canadian Imperial Bank of
    Commerce", zero token overlap despite being the same company) from an
    actually-wrong match (CNR vs "Core Natural Resources", also zero
    overlap). Both score the same here on purpose: when this returns False,
    the caller must not silently reject either — flag for a human glance."""
    expected_tokens = _name_tokens(expected)
    candidate_tokens = _name_tokens(candidate)
    if not expected_tokens or not candidate_tokens:
        return False
    return expected_tokens <= candidate_tokens


@dataclass
class ResolvedEntry:
    ca_ticker: str
    us_ticker: str
    cik: int
    form_type: str  # "40-F" | "20-F" | "10-K"
    company_name: str
    match_method: str  # "ticker" | "name_fallback"


@dataclass
class FlaggedEntry:
    ca_ticker: str
    expected_name: str
    reason: str
    candidate: dict | None = field(default=None)


async def fetch_sec_ticker_index(session: aiohttp.ClientSession) -> dict[str, dict]:
    """SEC's officially-maintained ticker->CIK registry. Returns {ticker: {cik, title}}."""
    async with session.get(SEC_TICKERS_URL) as response:
        response.raise_for_status()
        data = await response.json()
    return {row["ticker"]: {"cik": row["cik_str"], "title": row["title"]} for row in data.values()}


async def fetch_relevant_form(session: aiohttp.ClientSession, cik: int) -> str | None:
    """Which of RELEVANT_FORMS (if any) this CIK has actually filed. None if
    none of them appear — the categorical signal that catches wrong matches
    a bare ticker/name lookup can't (see CNR in the module docstring)."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    async with session.get(url) as response:
        if response.status == 404:
            return None
        response.raise_for_status()
        data = await response.json()
    forms = set(data.get("filings", {}).get("recent", {}).get("form", []))
    for form in RELEVANT_FORMS:
        if form in forms:
            return form
    return None


def find_name_fallback_candidate(
    index: dict[str, dict], expected_name: str
) -> tuple[str, dict] | None:
    """Scan the full SEC ticker index for a plausible name match, for when
    the direct ticker lookup is missing or wrong (CNR's case: SEC's `CNR`
    belongs to an unrelated company, but "Canadian National Railway Co" is
    in the index under ticker CNI)."""
    for ticker, entry in index.items():
        if _names_plausibly_match(expected_name, entry["title"]):
            return ticker, entry
    return None


async def resolve_ticker(
    session: aiohttp.ClientSession, index: dict[str, dict], ca_ticker: str, expected_name: str
) -> ResolvedEntry | FlaggedEntry:
    us_ticker_guess = ca_ticker.removesuffix(".TO").removesuffix(".V")
    ticker_candidate = index.get(us_ticker_guess)

    if ticker_candidate is not None:
        await asyncio.sleep(_REQUEST_DELAY_SECONDS)
        form_type = await fetch_relevant_form(session, ticker_candidate["cik"])
        name_ok = _names_plausibly_match(expected_name, ticker_candidate["title"])

        if form_type is not None and name_ok:
            return ResolvedEntry(
                ca_ticker=ca_ticker,
                us_ticker=us_ticker_guess,
                cik=ticker_candidate["cik"],
                form_type=form_type,
                company_name=ticker_candidate["title"],
                match_method="ticker",
            )

        if form_type is not None and not name_ok:
            # Ticker match + real SEC filer, but the name comparison didn't
            # pass — often just an abbreviation (CIBC vs "Canadian Imperial
            # Bank of Commerce") rather than a wrong company, and this cheap
            # heuristic can't tell the two apart from text alone. Don't
            # silently discard a plausibly-correct match OR silently accept
            # an unconfirmed one — surface it with full context instead.
            alt = find_name_fallback_candidate(index, expected_name)
            reason = (
                f"Ticker '{us_ticker_guess}' matches SEC filer '{ticker_candidate['title']}' "
                f"(CIK {ticker_candidate['cik']}, files {form_type}) but its name doesn't "
                f"textually match '{expected_name}' — likely just an abbreviation, but needs "
                "a quick human confirmation before adding."
            )
            if alt is not None and alt[0] != us_ticker_guess:
                alt_ticker, alt_entry = alt
                reason += (
                    f" Alternative found by name search: '{alt_entry['title']}' "
                    f"(CIK {alt_entry['cik']}, ticker {alt_ticker})."
                )
            # No `candidate=` here on purpose: the reason text above already
            # states the ticker-matched company and, when found, the correct
            # alternative — a separate "possible candidate" block would
            # redundantly re-surface the UNCONFIRMED (possibly wrong, as in
            # CNR's case) match as if it were the recommended one.
            return FlaggedEntry(ca_ticker=ca_ticker, expected_name=expected_name, reason=reason)
        # form_type is None: ticker exists but this CIK has never filed
        # 40-F/20-F/10-K — almost certainly the wrong company (confirmed
        # live: CNR's SEC ticker belongs to an unrelated 10-K-only US
        # filer). Fall through to the name-based search instead of trusting it.

    fallback = find_name_fallback_candidate(index, expected_name)
    if fallback is None:
        return FlaggedEntry(
            ca_ticker=ca_ticker,
            expected_name=expected_name,
            reason=(
                f"No SEC entity found matching '{expected_name}' by ticker "
                f"('{us_ticker_guess}') or by name search."
            ),
        )
    us_ticker_guess, candidate = fallback

    await asyncio.sleep(_REQUEST_DELAY_SECONDS)
    form_type = await fetch_relevant_form(session, candidate["cik"])
    if form_type is None:
        return FlaggedEntry(
            ca_ticker=ca_ticker,
            expected_name=expected_name,
            reason=(
                f"Name match found ('{candidate['title']}', CIK {candidate['cik']}, "
                f"ticker {us_ticker_guess}) but it has never filed 40-F, 20-F, or 10-K — "
                "no MD&A-style content available via this approach."
            ),
            candidate={
                "us_ticker": us_ticker_guess,
                "cik": candidate["cik"],
                "company_name": candidate["title"],
            },
        )

    return ResolvedEntry(
        ca_ticker=ca_ticker,
        us_ticker=us_ticker_guess,
        cik=candidate["cik"],
        form_type=form_type,
        company_name=candidate["title"],
        match_method="name_fallback",
    )


async def build_mapping(
    session: aiohttp.ClientSession,
    watchlist: list[tuple[str, str]],
    already_confirmed: set[str] = frozenset(),
) -> tuple[list[ResolvedEntry], list[FlaggedEntry]]:
    """`already_confirmed` — ca_tickers with a manually-confirmed mapping
    entry already. Skipped entirely rather than re-resolved: a human already
    settled these after the automated heuristic flagged them, and re-running
    the same heuristic would just flag them again forever, burying genuinely
    new review items under repeat noise."""
    index = await fetch_sec_ticker_index(session)
    resolved: list[ResolvedEntry] = []
    flagged: list[FlaggedEntry] = []
    for ca_ticker, expected_name in watchlist:
        if ca_ticker in already_confirmed:
            logger.info("crosslisting_skip_manually_confirmed", ca_ticker=ca_ticker)
            continue
        result = await resolve_ticker(session, index, ca_ticker, expected_name)
        if isinstance(result, ResolvedEntry):
            logger.info(
                "crosslisting_resolved",
                ca_ticker=ca_ticker,
                us_ticker=result.us_ticker,
                form_type=result.form_type,
                match_method=result.match_method,
            )
            resolved.append(result)
        else:
            logger.warning("crosslisting_flagged", ca_ticker=ca_ticker, reason=result.reason)
            flagged.append(result)
    return resolved, flagged


def write_mapping_file(resolved: list[ResolvedEntry], path: Path) -> None:
    """Merges into the existing file rather than overwriting it — entries
    for tickers this run didn't auto-resolve (most importantly, anything a
    human manually confirmed after being flagged, like CM.TO/CNR.TO) must
    survive a re-run. Auto-resolved entries always overwrite their own
    ca_ticker key, so re-running does refresh data for tickers the resolver
    successfully re-verifies."""
    existing: dict = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

    for entry in resolved:
        existing[entry.ca_ticker] = {
            "us_ticker": entry.us_ticker,
            "cik": entry.cik,
            "form_type": entry.form_type,
            "company_name": entry.company_name,
        }

    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_review_file(flagged: list[FlaggedEntry], path: Path) -> None:
    lines = [
        "# Canadian cross-listing resolver — needs attention",
        "",
        "Generated by `src/data/tools/resolve_ca_crosslisting.py`. These tickers "
        "did NOT get added to `data/ca_us_crosslisting.json` — each one keeps the "
        "existing `canadian_data_limited` fallback until resolved here. Nothing is "
        "broken; this is a coverage improvement opportunity, not an error.",
        "",
    ]
    if not flagged:
        lines.append("Nothing flagged — every watchlist ticker resolved cleanly.")
    for entry in flagged:
        lines.append(f"## {entry.ca_ticker} — {entry.expected_name}")
        lines.append("")
        lines.append(entry.reason)
        if entry.candidate:
            lines.append("")
            lines.append(
                f"Possible candidate found: **{entry.candidate['company_name']}** "
                f"(CIK {entry.candidate['cik']}, ticker `{entry.candidate['us_ticker']}`) — "
                "review before adding manually; not auto-added because it didn't "
                "pass verification."
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_manually_confirmed_tickers(mapping_path: Path) -> set[str]:
    if not mapping_path.exists():
        return set()
    existing = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {ca_ticker for ca_ticker, entry in existing.items() if "manually_confirmed" in entry}


async def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    mapping_path = data_dir / "ca_us_crosslisting.json"
    already_confirmed = _load_manually_confirmed_tickers(mapping_path)

    async with aiohttp.ClientSession(headers={"User-Agent": _user_agent()}) as session:
        resolved, flagged = await build_mapping(session, WATCHLIST_CA_TICKERS, already_confirmed)

    write_mapping_file(resolved, mapping_path)
    write_review_file(flagged, data_dir / "ca_us_crosslisting_review.md")
    logger.info(
        "crosslisting_resolver_done",
        resolved=len(resolved),
        flagged=len(flagged),
        skipped_manually_confirmed=len(already_confirmed),
    )


if __name__ == "__main__":
    asyncio.run(main())
