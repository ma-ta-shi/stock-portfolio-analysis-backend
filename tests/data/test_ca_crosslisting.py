from datetime import date, timedelta

import pytest

from data.providers.ca_crosslisting import (
    _MDA_CONTENT_MARKERS,
    _MDA_DESCRIPTION_MARKERS,
    _annual_40f_mda,
    _find_40f_exhibit_text,
    _find_mda_source,
    _first_self_identifying_exhibit,
    _get_native_filing_item,
    _is_candidate_40f_exhibit,
    _looks_like_mda_or_earnings_release,
    _normalize,
    _skip_front_matter,
    get_crosslisted_business_overview,
    get_crosslisted_interim_exhibits,
    get_crosslisted_mda,
    is_crosslisted,
)


# --- Fakes standing in for the edgar (edgartools) library's object graph ---


class FakeFilingsResult(list):
    def head(self, n: int) -> "FakeFilingsResult":
        return FakeFilingsResult(self[:n])


class FakeCompany:
    def __init__(self, filings_by_form: dict[str, list] | None = None) -> None:
        self._filings_by_form = filings_by_form or {}

    def get_filings(self, form: str | None = None, amendments: bool = True) -> FakeFilingsResult:
        if form is not None and form in self._filings_by_form:
            return FakeFilingsResult(self._filings_by_form[form])
        return FakeFilingsResult([])


class FakeExhibitAttachment:
    def __init__(
        self, document: str, description: str = "", text: str = "", document_type: str = ""
    ) -> None:
        self.document = document
        self.description = description
        self.document_type = document_type
        self._text = text

    def text(self) -> str:
        return self._text


class Fake40FFiling:
    def __init__(
        self,
        attachments: list[FakeExhibitAttachment],
        accession_no: str = "0000000000-00-000000",
        filing_date: str = "2026-02-15",
    ) -> None:
        self.attachments = attachments
        self.accession_no = accession_no
        self.filing_date = filing_date


class FakeTenKObj:
    def __init__(
        self, management_discussion: str | None = None, business: str | None = None
    ) -> None:
        self.management_discussion = management_discussion
        self.business = business


class Fake10KFiling:
    def __init__(
        self,
        obj_result: FakeTenKObj | None,
        accession_no: str = "0000000000-00-000000",
        filing_date: str = "2026-03-01",
    ) -> None:
        self._obj_result = obj_result
        self.accession_no = accession_no
        self.filing_date = filing_date

    def obj(self) -> FakeTenKObj | None:
        return self._obj_result


class Fake6KFiling:
    def __init__(
        self, filing_date: str, accession_no: str, attachments: list | None = None
    ) -> None:
        self.filing_date = filing_date
        self.accession_no = accession_no
        self.attachments = attachments or []


def _patch_crosslisting_map(monkeypatch, mapping: dict) -> None:
    monkeypatch.setattr("data.providers.ca_crosslisting._load_crosslisting_map", lambda: mapping)


def _recent() -> str:
    return (date.today() - timedelta(days=20)).isoformat()


def _stale() -> str:
    return (date.today() - timedelta(days=400)).isoformat()


# --- is_crosslisted ---


def test_is_crosslisted_true_for_mapped_ticker(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    assert is_crosslisted("RY.TO") is True


def test_is_crosslisted_false_for_unmapped_ticker(monkeypatch):
    _patch_crosslisting_map(monkeypatch, {})
    assert is_crosslisted("ZZZZ.TO") is False


# --- _normalize ---


def test_normalize_converts_curly_apostrophe():
    assert _normalize("Management’s Discussion") == "management's discussion"


def test_normalize_strips_border_runs():
    assert _normalize("+---+ Title +---+") == "title"


def test_normalize_strips_table_pipes_of_any_count():
    """Regression test: BNS's real AIF title was rendered as isolated single
    pipes between every word ('annual | | information | form'), which a
    3+-run-only border strip doesn't touch — breaks an exact substring match
    even after whitespace collapsing."""
    assert _normalize("annual | | information | form") == "annual information form"


def test_normalize_collapses_whitespace_and_lowercases():
    assert _normalize("  Financial   Review  \n\n") == "financial review"


# --- _is_candidate_40f_exhibit ---


def test_is_candidate_rejects_non_htm():
    assert _is_candidate_40f_exhibit("chart.jpg", "") is False


def test_is_candidate_rejects_graphic_and_xbrl():
    assert _is_candidate_40f_exhibit("x.htm", "GRAPHIC") is False
    assert _is_candidate_40f_exhibit("x.htm", "IDEA: XBRL Document") is False


def test_is_candidate_rejects_certifications_and_consents():
    assert _is_candidate_40f_exhibit("x.htm", "EX-31.1 Certification") is False
    assert _is_candidate_40f_exhibit("x.htm", "Consent of Independent Auditor") is False


def test_is_candidate_accepts_plain_exhibit():
    assert _is_candidate_40f_exhibit("ex991.htm", "EX-99.1") is True


# --- _find_40f_exhibit_text ---


async def test_find_40f_exhibit_tier1_description_match(monkeypatch):
    attachments = [
        FakeExhibitAttachment("ex1.htm", "EX-1 Annual Information Form", text="AIF body"),
        FakeExhibitAttachment("ex2.htm", "EX-2 Financial Review", text="MDA body"),
    ]
    filing = Fake40FFiling(
        attachments, accession_no="0000950170-26-000123", filing_date="2026-02-20"
    )
    fake = FakeCompany(filings_by_form={"40-F": [filing]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1000275, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result == {
        "text": "MDA body",
        "accession_no": "0000950170-26-000123",
        "filing_date": "2026-02-20",
        "source_form": "40-F",
    }


async def test_find_40f_exhibit_tier2_content_match_within_window(monkeypatch):
    """TD-style filer: no descriptive labels, but the exhibit's own title
    appears near the very start of its text."""
    attachments = [
        FakeExhibitAttachment("ex991.htm", "EX-99.1", text="Some AIF content, not the MDA"),
        FakeExhibitAttachment(
            "ex992.htm",
            "EX-99.2",
            text="TD BANK GROUP - MANAGEMENT'S DISCUSSION AND ANALYSIS\n\nBody...",
        ),
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(947263, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result["text"].startswith("TD BANK GROUP")
    assert result["source_form"] == "40-F"


async def test_find_40f_exhibit_tier2_rejects_match_outside_window(monkeypatch):
    """Regression test for the real TD bug: an exhibit that only mentions the
    MD&A well past the title window (a cross-reference deep in body text,
    not its own title) must NOT be returned — even though the marker phrase
    genuinely appears somewhere in its text."""
    far_mention = ("filler " * 100) + "management's discussion and analysis of the bank"
    attachments = [
        FakeExhibitAttachment("ex991.htm", "EX-99.1", text=far_mention),
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result is None


async def test_find_40f_exhibit_no_filings_returns_none(monkeypatch):
    fake = FakeCompany(filings_by_form={"40-F": []})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result is None


async def test_find_40f_exhibit_no_candidate_matches_returns_none(monkeypatch):
    attachments = [
        FakeExhibitAttachment("ex994.htm", "EX-99.4", text="Return on equity ratios only")
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result is None


async def test_find_40f_exhibit_skips_certification_without_fetching_text(monkeypatch):
    """A certification exhibit should never even have .text() called on it —
    cheap to rule out by description alone."""
    calls = []

    class TrackedAttachment(FakeExhibitAttachment):
        def text(self) -> str:
            calls.append(self.document)
            return super().text()

    attachments = [TrackedAttachment("ex995.htm", "EX-31.1 Certification", text="irrelevant")]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert calls == []


# --- _get_native_filing_item ---


async def test_get_native_filing_item_returns_requested_attribute(monkeypatch):
    obj = FakeTenKObj(management_discussion="MDA text", business="Business text")
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    mda = await _get_native_filing_item(16875, "10-K", "management_discussion")
    biz = await _get_native_filing_item(16875, "10-K", "business")

    assert mda["text"] == "MDA text"
    assert mda["source_form"] == "10-K"
    assert biz["text"] == "Business text"


async def test_get_native_filing_item_uses_the_given_form(monkeypatch):
    """20-F filers parse natively too (confirmed live 2026-08-03 against
    Canada Goose/Celestica/Lithium Americas) — must query the form the
    caller actually asked for, not hardcode 10-K."""
    obj = FakeTenKObj(management_discussion="20-F MDA text")
    fake = FakeCompany(filings_by_form={"20-F": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _get_native_filing_item(1690511, "20-F", "management_discussion")

    assert result["text"] == "20-F MDA text"
    assert result["source_form"] == "20-F"


async def test_get_native_filing_item_no_filings_returns_none(monkeypatch):
    fake = FakeCompany(filings_by_form={"10-K": []})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _get_native_filing_item(16875, "10-K", "management_discussion")

    assert result is None


async def test_get_native_filing_item_missing_attr_returns_none(monkeypatch):
    obj = FakeTenKObj(management_discussion=None)
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _get_native_filing_item(16875, "10-K", "management_discussion")

    assert result is None


# --- get_crosslisted_mda / get_crosslisted_business_overview ---


async def test_get_crosslisted_mda_ticker_not_mapped_returns_none(monkeypatch):
    _patch_crosslisting_map(monkeypatch, {})

    result = await get_crosslisted_mda("ZZZZ.TO")

    assert result is None


async def test_get_crosslisted_mda_10k_filer_uses_10k_path(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"CP.TO": {"us_ticker": "CP", "cik": 16875, "form_type": "10-K", "company_name": "CPKC"}},
    )
    obj = FakeTenKObj(management_discussion="CP's real Item 7 text")
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_mda("CP.TO")

    assert result["text"] == "CP's real Item 7 text"


async def test_get_crosslisted_mda_20f_filer_uses_native_path(monkeypatch):
    """Confirmed live 2026-08-03: 20-F parses natively via edgartools, same
    shape as 10-K — no exhibit-hunting needed, contrary to the original
    assumption that it would need the 40-F exhibit-search path."""
    _patch_crosslisting_map(
        monkeypatch,
        {
            "GOOS.TO": {
                "us_ticker": "GOOS",
                "cik": 1690511,
                "form_type": "20-F",
                "company_name": "Canada Goose",
            }
        },
    )
    obj = FakeTenKObj(management_discussion="Canada Goose's real MD&A text")
    fake = FakeCompany(filings_by_form={"20-F": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_mda("GOOS.TO")

    assert result["text"] == "Canada Goose's real MD&A text"


async def test_get_crosslisted_mda_40f_filer_uses_exhibit_path(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    attachments = [
        FakeExhibitAttachment(
            "ex2.htm",
            "EX-2 Financial Review",
            text="Financial Review\nNet income for the fourth quarter was $4 billion...",
        )
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_mda("RY.TO")

    # No 6-K in the fake, so _find_mda_source falls back to the annual 40-F exhibit.
    assert result["text"].startswith("Financial Review")
    assert result["source_form"] == "40-F"


async def test_get_crosslisted_business_overview_ticker_not_mapped_returns_none(monkeypatch):
    _patch_crosslisting_map(monkeypatch, {})

    result = await get_crosslisted_business_overview("ZZZZ.TO")

    assert result is None


async def test_get_crosslisted_business_overview_10k_filer_uses_business_attr(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"CP.TO": {"us_ticker": "CP", "cik": 16875, "form_type": "10-K", "company_name": "CPKC"}},
    )
    obj = FakeTenKObj(business="CP's real Item 1 text")
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_business_overview("CP.TO")

    assert result["text"] == "CP's real Item 1 text"


async def test_get_crosslisted_business_overview_20f_filer_uses_native_path(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {
            "GOOS.TO": {
                "us_ticker": "GOOS",
                "cik": 1690511,
                "form_type": "20-F",
                "company_name": "Canada Goose",
            }
        },
    )
    obj = FakeTenKObj(business="Canada Goose's real business text")
    fake = FakeCompany(filings_by_form={"20-F": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_business_overview("GOOS.TO")

    assert result["text"] == "Canada Goose's real business text"


async def test_get_crosslisted_business_overview_40f_filer_uses_aif_exhibit(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    attachments = [
        FakeExhibitAttachment("ex1.htm", "EX-1 Annual Information Form", text="RY's AIF text")
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_business_overview("RY.TO")

    assert result["text"] == "RY's AIF text"
    assert result["source_form"] == "40-F"


# --- _looks_like_mda_or_earnings_release ---


@pytest.mark.parametrize(
    "text",
    [
        "Endeavour Silver Corp. Management's Discussion & Analysis For the Three Months Ended...",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "Royal Bank of Canada\nFinancial Review\nFor the quarter ended April 30, 2026",
        "SECOND QUARTER 2026 EARNINGS RELEASE\nRoyal Bank of Canada reports second quarter 2026 results",
        "Quarterly Report to Shareholders\nScotiabank reports third quarter results\nTORONTO...",
        "TFI International Announces 2026 Second Quarter Results\nMONTREAL, July 27, 2026 -- TFI...",
    ],
)
def test_looks_like_mda_or_release_accepts_real_documents(text):
    assert _looks_like_mda_or_earnings_release(text + " " + ("x " * 5000)) is True


@pytest.mark.parametrize(
    "text",
    [
        # an extraction fragment - starts mid-word, lowercase (CAE's bad exhibit)
        "e most directly comparable measure under IFRS. For forward-looking measures, refer to...",
        # a bid-defence press release - no period + results headline
        "AURORA CANNABIS CORRECTS INACCURATE STATEMENTS MADE IN SUPPORT OF A HOSTILE BID",
        # a governance filing
        "Articles of EXAMPLE HOLDINGS, INC.\nTABLE OF CONTENTS\n1. Interpretation\n2. Shares",
        # an NI 43-101 technical report
        "EXAMPLE MINING CORP.\nNI 43-101 Technical Report for the Example Property\nEffective Date...",
        # the bare 6-K cover form
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nWashington, D.C. 20549\nFORM 6-K",
    ],
)
def test_looks_like_mda_or_release_rejects_wrong_documents(text):
    assert _looks_like_mda_or_earnings_release(text + " " + ("x " * 5000)) is False


# --- _skip_front_matter ---


def test_skip_front_matter_noop_without_front_matter():
    text = "Management's Discussion and Analysis\nThis MD&A should be read alongside the..."
    assert _skip_front_matter(text) is text


def test_skip_front_matter_skips_toc_to_first_real_header():
    text = (
        "ANNUAL INFORMATION FORM\nTABLE OF CONTENTS\n"
        "Corporate Structure .......... 3\n"
        "General Development of the Business .......... 5\n"
        "GLOSSARY .......... 40\n" + ("filler glossary term definitions " * 200) + "\n"
        "General Development of the Business\n"
        "Over the past three financial years the Company acquired several regional operators..."
    )
    result = _skip_front_matter(text)
    assert result.startswith("General Development of the Business\nOver the past three")


def test_skip_front_matter_triggers_on_toc_density_without_the_literal_phrase():
    text = (
        "Overview 1\nBasis of Presentation 2\nResults of Operations 3\nLiquidity 8\n"
        "Capital Resources 12\n" + ("front matter prose " * 300) + "\n"
        "Results of Operations\nRevenue for the quarter increased 12% to $410 million..."
    )
    result = _skip_front_matter(text)
    assert result.startswith("Results of Operations\nRevenue for the quarter")


def test_skip_front_matter_noop_when_no_real_header_found():
    text = "TABLE OF CONTENTS\nItem 1 .......... 2\nItem 2 .......... 5\n" + ("prose " * 5000)
    assert _skip_front_matter(text) is text


# --- _first_self_identifying_exhibit ---


async def test_first_self_identifying_exhibit_returns_the_matching_one():
    filing = Fake6KFiling(
        _recent(),
        "acc1",
        attachments=[
            FakeExhibitAttachment(
                "cover.htm", "EX-99.1", text="short cover", document_type="EX-99.1"
            ),
            FakeExhibitAttachment(
                "dex992.htm",
                "EX-99.2",
                text="MANAGEMENT'S DISCUSSION AND ANALYSIS\n" + ("body detail " * 3000),
                document_type="EX-99.2",
            ),
        ],
    )
    result = await _first_self_identifying_exhibit(filing)
    assert result.startswith("MANAGEMENT'S DISCUSSION AND ANALYSIS")


async def test_first_self_identifying_exhibit_skips_xbrl_and_certs():
    filing = Fake6KFiling(
        _recent(),
        "acc1",
        attachments=[
            FakeExhibitAttachment(
                "R1.htm", "IDEA: XBRL", text="xbrl " * 9000, document_type="HTML"
            ),
            FakeExhibitAttachment(
                "dex311.htm", "EX-31.1", text="I certify the report " * 900, document_type="EX-31.1"
            ),
        ],
    )
    assert await _first_self_identifying_exhibit(filing) is None


async def test_first_self_identifying_exhibit_none_for_a_dividend_6k():
    filing = Fake6KFiling(
        _recent(),
        "acc1",
        attachments=[
            FakeExhibitAttachment(
                "ex991.htm",
                "EX-99.1",
                text="Example Corp declares a quarterly dividend of $0.25 per share, payable "
                + ("to holders of record " * 500),
                document_type="EX-99.1",
            ),
        ],
    )
    assert await _first_self_identifying_exhibit(filing) is None


# --- _annual_40f_mda ---


async def test_annual_40f_mda_returns_skipped_body_when_it_reads_as_mda(monkeypatch):
    attachments = [
        FakeExhibitAttachment(
            "ex2.htm",
            "EX-2 Financial Review",
            text="Financial Review\nNet income for the fourth quarter was $2 billion...",
        )
    ]
    fake = FakeCompany(
        filings_by_form={"40-F": [Fake40FFiling(attachments, accession_no="40facc")]}
    )
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _annual_40f_mda(1)

    assert result["accession_no"] == "40facc"
    assert result["text"].startswith("Financial Review")


async def test_annual_40f_mda_none_when_exhibit_is_only_a_table_of_contents(monkeypatch):
    toc = (
        "Corporate Dividends 57\nAccounting Policies 58\nSummary of Results 59\n"
        "Controls 60\nOther Matters 61\n" + ("toc noise " * 200)
    )
    attachments = [
        FakeExhibitAttachment("ex2.htm", "EX-2 Management's Discussion and Analysis", text=toc)
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    assert await _annual_40f_mda(1) is None


async def test_annual_40f_mda_none_when_no_40f(monkeypatch):
    fake = FakeCompany(filings_by_form={})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)
    assert await _annual_40f_mda(1) is None


async def test_annual_40f_mda_none_when_the_most_recent_40f_is_stale(monkeypatch):
    """BAM/SHOP: their last 40-F is from 2024 - the filer stopped, so a
    two-year-old annual MD&A is not a usable digest source."""
    attachments = [
        FakeExhibitAttachment(
            "ex2.htm",
            "EX-2 Financial Review",
            text="Financial Review\nNet income was $2 billion...",
        )
    ]
    old = (date.today() - timedelta(days=700)).isoformat()
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments, filing_date=old)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    assert await _annual_40f_mda(1) is None


async def test_annual_40f_mda_none_when_body_starts_mid_sentence(monkeypatch):
    """BN: the exhibit text begins with a lowercase fragment of a
    forward-looking-statements sentence, not the MD&A."""
    attachments = [
        FakeExhibitAttachment(
            "ex2.htm",
            "EX-2 Management's Discussion and Analysis",
            text="financial condition, expected results, prospects and targets " * 300,
        )
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    assert await _annual_40f_mda(1) is None


# --- _find_mda_source ---


def _patch_company_6ks(monkeypatch, filings):
    fake = FakeCompany(filings_by_form={"6-K": filings})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)


async def test_find_mda_source_uses_the_most_recent_self_identifying_6k(monkeypatch):
    filings = [Fake6KFiling(_recent(), "newest"), Fake6KFiling(_recent(), "older")]
    _patch_company_6ks(monkeypatch, filings)

    async def fake_self_id(filing):
        return "MD&A body " * 2000 if filing.accession_no == "newest" else None

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", fake_self_id
    )

    result = await _find_mda_source(1)

    assert result["source_form"] == "6-K"
    assert result["accession_no"] == "newest"
    assert result["filing_date"] == _recent()


async def test_find_mda_source_walks_back_past_a_non_earnings_6k(monkeypatch):
    filings = [Fake6KFiling(_recent(), "dividend"), Fake6KFiling(_recent(), "earnings")]
    _patch_company_6ks(monkeypatch, filings)

    async def fake_self_id(filing):
        return "MD&A body " * 2000 if filing.accession_no == "earnings" else None

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", fake_self_id
    )

    result = await _find_mda_source(1)
    assert result["accession_no"] == "earnings"


async def test_find_mda_source_stops_at_a_stale_6k_and_uses_the_40f(monkeypatch):
    filings = [Fake6KFiling(_stale(), "stale"), Fake6KFiling(_stale(), "staler")]
    _patch_company_6ks(monkeypatch, filings)

    called = []

    async def fake_self_id(filing):
        called.append(filing.accession_no)
        return "MD&A body " * 2000

    async def fake_annual(cik):
        return {
            "text": "annual mda body",
            "accession_no": "40facc",
            "filing_date": _recent(),
            "source_form": "40-F",
        }

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", fake_self_id
    )
    monkeypatch.setattr("data.providers.ca_crosslisting._annual_40f_mda", fake_annual)

    result = await _find_mda_source(1)

    assert called == []  # never looked inside a stale 6-K
    assert result["source_form"] == "40-F"


async def test_find_mda_source_recent_6k_wins_without_fetching_the_40f(monkeypatch):
    _patch_company_6ks(monkeypatch, [Fake6KFiling(_recent(), "6kacc")])

    async def fake_self_id(filing):
        return "MD&A body " * 2000

    async def fail_annual(cik):
        raise AssertionError("should not fetch the 40-F when the 6-K is recent")

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", fake_self_id
    )
    monkeypatch.setattr("data.providers.ca_crosslisting._annual_40f_mda", fail_annual)

    result = await _find_mda_source(1)
    assert result["accession_no"] == "6kacc"


async def test_find_mda_source_prefers_a_newer_40f_over_an_older_6k(monkeypatch):
    old_6k_date = (date.today() - timedelta(days=200)).isoformat()
    _patch_company_6ks(monkeypatch, [Fake6KFiling(old_6k_date, "old6k")])

    async def fake_self_id(filing):
        return "MD&A body " * 2000

    async def fake_annual(cik):
        return {
            "text": "fresher annual mda",
            "accession_no": "new40f",
            "filing_date": _recent(),
            "source_form": "40-F",
        }

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", fake_self_id
    )
    monkeypatch.setattr("data.providers.ca_crosslisting._annual_40f_mda", fake_annual)

    result = await _find_mda_source(1)
    assert result["accession_no"] == "new40f"


async def test_find_mda_source_keeps_an_older_6k_when_the_40f_is_older_still(monkeypatch):
    old_6k_date = (date.today() - timedelta(days=200)).isoformat()
    _patch_company_6ks(monkeypatch, [Fake6KFiling(old_6k_date, "old6k")])

    async def fake_self_id(filing):
        return "MD&A body " * 2000

    async def fake_annual(cik):
        return {
            "text": "even older annual mda",
            "accession_no": "old40f",
            "filing_date": (date.today() - timedelta(days=300)).isoformat(),
            "source_form": "40-F",
        }

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", fake_self_id
    )
    monkeypatch.setattr("data.providers.ca_crosslisting._annual_40f_mda", fake_annual)

    result = await _find_mda_source(1)
    assert result["accession_no"] == "old6k"


async def test_find_mda_source_none_when_nothing_resolves(monkeypatch):
    _patch_company_6ks(monkeypatch, [Fake6KFiling(_recent(), "6kacc")])

    async def no_self_id(filing):
        return None

    async def no_annual(cik):
        return None

    monkeypatch.setattr(
        "data.providers.ca_crosslisting._first_self_identifying_exhibit", no_self_id
    )
    monkeypatch.setattr("data.providers.ca_crosslisting._annual_40f_mda", no_annual)

    assert await _find_mda_source(1) is None


# --- get_crosslisted_interim_exhibits ---


async def test_get_crosslisted_interim_exhibits_ticker_not_mapped_returns_empty(monkeypatch):
    _patch_crosslisting_map(monkeypatch, {})

    result = await get_crosslisted_interim_exhibits("ZZZZ.TO")

    assert result == []


async def test_get_crosslisted_interim_exhibits_10k_filer_returns_empty(monkeypatch):
    """CP is a 10-K filer, which uses 8-K for interim filings, not 6-K — out
    of scope for this function, must not attempt a 6-K fetch."""
    _patch_crosslisting_map(
        monkeypatch,
        {"CP.TO": {"us_ticker": "CP", "cik": 16875, "form_type": "10-K", "company_name": "CPKC"}},
    )
    fake = FakeCompany(filings_by_form={"6-K": [Fake6KFiling("2026-01-01", "0001")]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_interim_exhibits("CP.TO")

    assert result == []


async def test_get_crosslisted_interim_exhibits_40f_filer_returns_recent_filings(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    fake = FakeCompany(
        filings_by_form={
            "6-K": [
                Fake6KFiling("2026-03-01", "0003"),
                Fake6KFiling("2026-02-01", "0002"),
                Fake6KFiling("2026-01-01", "0001"),
            ]
        }
    )
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_interim_exhibits("RY.TO", limit=2)

    assert result == [
        {"filing_date": "2026-03-01", "accession_no": "0003"},
        {"filing_date": "2026-02-01", "accession_no": "0002"},
    ]


async def test_get_crosslisted_interim_exhibits_20f_filer_returns_recent_filings(monkeypatch):
    """20-F filers are foreign private issuers too and use 6-K for interim
    filings, same as 40-F — must not be excluded like the 10-K/8-K case."""
    _patch_crosslisting_map(
        monkeypatch,
        {
            "GOOS.TO": {
                "us_ticker": "GOOS",
                "cik": 1690511,
                "form_type": "20-F",
                "company_name": "Canada Goose",
            }
        },
    )
    fake = FakeCompany(filings_by_form={"6-K": [Fake6KFiling("2026-03-01", "0001")]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_interim_exhibits("GOOS.TO")

    assert result == [{"filing_date": "2026-03-01", "accession_no": "0001"}]
