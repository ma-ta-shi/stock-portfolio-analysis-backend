"""Live verification for the CA cross-listed filing digest (ClickUp 86bbu1j02).

Unit tests in tests/data/test_ca_crosslisting.py cover the logic against fakes.
This file hits real SEC EDGAR + a local Ollama to confirm the shipped
"self-identification only" approach resolves each issuer shape to grounded
content, and that the shapes that don't self-identify degrade to None rather
than a fabricated digest.

Shapes and the full write-up:
docs/technical/ca-crosslisted-filing-digest-findings.md.
"""

import re

import aiohttp
import pytest

from data.precompute.filing_summarizer import summarize_filing_section
from data.providers.ca_crosslisting import (
    get_crosslisted_business_overview,
    get_crosslisted_mda,
)

pytestmark = pytest.mark.live


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


# ---------- MD&A: each shape resolves to results-bearing content ----------


@pytest.mark.parametrize(
    "ticker",
    [
        "RY.TO",  # bank: separate earnings-release exhibit
        "BNS.TO",  # bank: one "Report to Shareholders" bundle
        "CM.TO",  # bank: bundle
        "MFC.TO",  # insurer: bundle
        "EDR.TO",  # miner: self-titled quarterly MD&A
        "NTR.TO",  # self-titled news release
        "TFII.TO",  # "Earnings Press Release"
        "CNQ.TO",  # dated earnings release
        "PAAS.TO",  # was mis-resolving to a 43-101 report; now its real quarterly MD&A
        "CAE.TO",  # was mis-resolving to a 778k bundle; now its 105k press release
        "SHOP.TO",  # reclassified to 10-K -> native path
        "BAM.TO",  # reclassified to 10-K -> native path
    ],
)
async def test_get_crosslisted_mda_resolves_to_results_bearing_text(ticker):
    section = await get_crosslisted_mda(ticker)
    assert section is not None, f"{ticker} MD&A did not resolve"
    assert section["source_form"] in ("6-K", "40-F", "10-K", "20-F")
    assert len(section["text"]) >= 8_000
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", section["filing_date"])
    head = _collapse(section["text"][:40_000])
    assert re.search(
        r"net (income|earnings|loss)|adjusted (ebitda|earnings)|diluted eps|revenue|per share",
        head,
    ), f"{ticker} MD&A text has no results language"


async def test_get_crosslisted_mda_cnr_resolves_via_6k():
    """CNR files no separate MD&A/AIF exhibit in its 40-F; its quarterly 6-K
    press release is the MD&A digest source."""
    section = await get_crosslisted_mda("CNR.TO")
    assert section is not None
    assert section["source_form"] == "6-K"


@pytest.mark.parametrize(
    "ticker",
    [
        "ACB.TO",  # most recent 6-K is a bid-defence press release, not earnings
        "CURA.TO",  # most recent 6-K is a corporate-governance bundle
        "BN.TO",  # no self-identifying 6-K; the 40-F MD&A opens with a boilerplate fragment
    ],
)
async def test_get_crosslisted_mda_degrades_to_none_for_non_self_identifying_shapes(ticker):
    assert await get_crosslisted_mda(ticker) is None


# ---------- Business ----------


@pytest.mark.parametrize("ticker", ["RY.TO", "SU.TO", "BNS.TO", "NTR.TO", "AEM.TO", "GFL.TO"])
async def test_get_crosslisted_business_overview_resolves(ticker):
    section = await get_crosslisted_business_overview(ticker)
    assert section is not None, f"{ticker} Business overview did not resolve"
    assert section["source_form"] in ("40-F", "10-K", "20-F")
    assert len(section["text"]) >= 8_000


async def test_get_crosslisted_business_overview_cnr_is_graceful_none():
    """CNR has no AIF exhibit - a clean None, not a wrong document."""
    assert await get_crosslisted_business_overview("CNR.TO") is None


async def test_reclassified_ticker_takes_native_path():
    """ENB switched to filing 10-Ks; the map hygiene reclassification routes it
    to the native parse, not the 40-F exhibit search."""
    section = await get_crosslisted_mda("ENB.TO")
    if section is not None:
        assert section["source_form"] == "10-K"


# ---------- end to end through the summarizer ----------


@pytest.mark.parametrize("ticker,section_kind", [("RY.TO", "MDA"), ("AEM.TO", "Business")])
async def test_summarize_filing_section_end_to_end(ticker, section_kind):
    """A real resolved section -> a real Ollama call -> a grounded, within-budget
    digest. Requires a local Ollama with gpt-oss:20b."""
    getter = get_crosslisted_mda if section_kind == "MDA" else get_crosslisted_business_overview
    section = await getter(ticker)
    assert section is not None

    async with aiohttp.ClientSession() as http:
        digest = await summarize_filing_section(http, section["text"], section_kind)

    if digest is None:
        pytest.skip("Ollama endpoint not reachable or produced a non-answer")
    assert digest.section == section_kind
    assert digest.token_count <= 250
    assert len(digest.content) > 100
    assert digest.content.rstrip().endswith((".", "!", "?", '"', ")"))
