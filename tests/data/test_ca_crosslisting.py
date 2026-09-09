from data.providers.ca_crosslisting import (
    _MDA_CONTENT_MARKERS,
    _MDA_DESCRIPTION_MARKERS,
    _find_40f_exhibit_text,
    _get_native_filing_item,
    _is_candidate_40f_exhibit,
    _normalize,
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
    def __init__(self, document: str, description: str = "", text: str = "") -> None:
        self.document = document
        self.description = description
        self._text = text

    def text(self) -> str:
        return self._text


class Fake40FFiling:
    def __init__(self, attachments: list[FakeExhibitAttachment]) -> None:
        self.attachments = attachments


class FakeTenKObj:
    def __init__(
        self, management_discussion: str | None = None, business: str | None = None
    ) -> None:
        self.management_discussion = management_discussion
        self.business = business


class Fake10KFiling:
    def __init__(self, obj_result: FakeTenKObj | None) -> None:
        self._obj_result = obj_result

    def obj(self) -> FakeTenKObj | None:
        return self._obj_result


class Fake6KFiling:
    def __init__(self, filing_date: str, accession_no: str) -> None:
        self.filing_date = filing_date
        self.accession_no = accession_no


def _patch_crosslisting_map(monkeypatch, mapping: dict) -> None:
    monkeypatch.setattr("data.providers.ca_crosslisting._load_crosslisting_map", lambda: mapping)


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
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1000275, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result == "MDA body"


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

    assert result.startswith("TD BANK GROUP")


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

    assert mda == "MDA text"
    assert biz == "Business text"


async def test_get_native_filing_item_uses_the_given_form(monkeypatch):
    """20-F filers parse natively too (confirmed live 2026-08-03 against
    Canada Goose/Celestica/Lithium Americas) — must query the form the
    caller actually asked for, not hardcode 10-K."""
    obj = FakeTenKObj(management_discussion="20-F MDA text")
    fake = FakeCompany(filings_by_form={"20-F": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await _get_native_filing_item(1690511, "20-F", "management_discussion")

    assert result == "20-F MDA text"


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

    assert result == "CP's real Item 7 text"


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

    assert result == "Canada Goose's real MD&A text"


async def test_get_crosslisted_mda_40f_filer_uses_exhibit_path(monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    attachments = [FakeExhibitAttachment("ex2.htm", "EX-2 Financial Review", text="RY's MDA text")]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.ca_crosslisting.Company", lambda cik: fake)

    result = await get_crosslisted_mda("RY.TO")

    assert result == "RY's MDA text"


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

    assert result == "CP's real Item 1 text"


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

    assert result == "Canada Goose's real business text"


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

    assert result == "RY's AIF text"


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
