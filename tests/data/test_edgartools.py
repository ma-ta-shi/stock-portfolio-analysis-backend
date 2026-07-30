import pandas as pd
import pytest
import structlog

from data.providers.base import StockDataProvider
from data.providers.edgartools import (
    _MDA_CONTENT_MARKERS,
    _MDA_DESCRIPTION_MARKERS,
    EdgarToolsDataProvider,
    _find_40f_exhibit_text,
    _get_10k_item,
    _is_candidate_40f_exhibit,
    _normalize,
)
from data.providers.yfinance import YFinanceDataProvider


# --- Fakes standing in for the edgar (edgartools) library's object graph ---


class FakeStatement:
    def __init__(self, df: pd.DataFrame | None) -> None:
        self._df = df

    def to_dataframe(self) -> pd.DataFrame | None:
        return self._df


class FakeFinancials:
    def __init__(self, income_df=None, balance_df=None, cashflow_df=None) -> None:
        self._income_df = income_df
        self._balance_df = balance_df
        self._cashflow_df = cashflow_df

    def income_statement(self) -> FakeStatement:
        return FakeStatement(self._income_df)

    def balance_sheet(self) -> FakeStatement:
        return FakeStatement(self._balance_df)

    def cashflow_statement(self) -> FakeStatement:
        return FakeStatement(self._cashflow_df)


class FakeForm4:
    def __init__(self, df: pd.DataFrame | None, raises: bool = False) -> None:
        self._df = df
        self._raises = raises

    def to_dataframe(self) -> pd.DataFrame | None:
        if self._raises:
            raise ValueError("malformed Form 4 XML")
        return self._df


class FakeFiling:
    def __init__(self, accession_no: str, form4_df=None, raises: bool = False) -> None:
        self.accession_no = accession_no
        self._form4_df = form4_df
        self._raises = raises

    def obj(self) -> FakeForm4:
        return FakeForm4(self._form4_df, raises=self._raises)


class FakeFilingsResult(list):
    def head(self, n: int) -> "FakeFilingsResult":
        return FakeFilingsResult(self[:n])


class FakeCompany:
    def __init__(
        self,
        annual=None,
        quarterly=None,
        filings: list[FakeFiling] | None = None,
        filings_by_form: dict[str, list] | None = None,
    ) -> None:
        self._annual = annual
        self._quarterly = quarterly
        self._filings = filings or []
        self._filings_by_form = filings_by_form or {}

    def get_financials(self) -> FakeFinancials | None:
        return self._annual

    def get_quarterly_financials(self) -> FakeFinancials | None:
        return self._quarterly

    def get_filings(self, form: str | None = None, amendments: bool = True) -> FakeFilingsResult:
        if form is not None and form in self._filings_by_form:
            return FakeFilingsResult(self._filings_by_form[form])
        return FakeFilingsResult(self._filings)


# --- Fixtures ---


def _income_df(concepts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"concept": concepts, "label": concepts})


def _yfinance_style_df() -> pd.DataFrame:
    """Shape yfinance actually returns: columns are recent period-end
    Timestamps, index is line-item labels. correct_alignment() expects this."""
    return pd.DataFrame({pd.Timestamp.now().normalize(): [100.0]}, index=["Total Revenue"])


FULL_INCOME_CONCEPTS = [
    "us-gaap_Revenues",
    "us-gaap_GrossProfit",
    "us-gaap_OperatingIncomeLoss",
    "us-gaap_NetIncomeLoss",
    "us-gaap_EarningsPerShareBasic",
]

# A thin/partial statement — e.g. a bank with no GrossProfit/OperatingIncomeLoss
# line — no longer triggers a fallback (see module docstring in edgartools.py for
# why); it should just pass through as-is.
THIN_INCOME_CONCEPTS = ["us-gaap_Revenues", "us-gaap_NetIncomeLoss"]


@pytest.fixture
def provider(monkeypatch):
    return EdgarToolsDataProvider(fallback=YFinanceDataProvider())


def _patch_company(monkeypatch, company_factory):
    monkeypatch.setattr("data.providers.edgartools.Company", company_factory)


# --- get_financials ---


def test_provider_is_a_stock_data_provider(provider):
    assert isinstance(provider, StockDataProvider)


async def test_get_financials_annual_income_returns_full_dataframe(provider, monkeypatch):
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(FULL_INCOME_CONCEPTS)))
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("AAPL", "income", "annual")

    assert list(df["concept"]) == FULL_INCOME_CONCEPTS


async def test_get_financials_quarterly_uses_quarterly_accessor(provider, monkeypatch):
    fake = FakeCompany(
        quarterly=FakeFinancials(balance_df=pd.DataFrame({"concept": ["us-gaap_Assets"]}))
    )
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("AAPL", "balance", "quarterly")

    assert list(df["concept"]) == ["us-gaap_Assets"]


async def test_get_financials_invalid_statement_raises(provider, monkeypatch):
    _patch_company(monkeypatch, lambda ticker: FakeCompany())

    with pytest.raises(ValueError, match="Invalid statement"):
        await provider.get_financials("AAPL", "notarealstatement", "annual")


async def test_get_financials_thin_statement_passes_through_without_fallback(provider, monkeypatch):
    """A partial income statement (e.g. a bank with no GrossProfit line) is not,
    by itself, a reason to distrust the data — see edgartools.py module docstring."""
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(THIN_INCOME_CONCEPTS)))
    _patch_company(monkeypatch, lambda ticker: fake)

    df = await provider.get_financials("JPM", "income", "annual")

    assert list(df["concept"]) == THIN_INCOME_CONCEPTS


async def test_get_financials_none_financials_falls_back_to_yfinance(provider, monkeypatch):
    """get_financials()/get_quarterly_financials() can return None outright —
    not just a thin statement — and must also trigger the fallback."""
    fake = FakeCompany(annual=None)
    _patch_company(monkeypatch, lambda ticker: fake)

    fallback_df = _yfinance_style_df()

    async def fake_fallback_get_financials(ticker, statement, period):
        return fallback_df

    monkeypatch.setattr(provider._fallback, "get_financials", fake_fallback_get_financials)

    with structlog.testing.capture_logs() as logs:
        df = await provider.get_financials("NODATA", "income", "annual")

    assert df is fallback_df
    assert any(log["event"] == "edgar_financials_missing" for log in logs)


# --- get_insider_trading ---


def _form4_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


async def test_get_insider_trading_filters_by_days_and_maps_fields(provider, monkeypatch):
    now = pd.Timestamp.now()
    recent_row = {
        "Date": now - pd.Timedelta(days=5),
        "Shares": 100,
        "Price": 25.5,
        "Insider": "Jane Doe",
        "Transaction Type": "Purchase",
        "Code": "P",
    }
    old_row = {**recent_row, "Date": now - pd.Timedelta(days=400), "Insider": "Old Insider"}
    filing = FakeFiling("0001-1", form4_df=_form4_df([recent_row, old_row]))
    fake = FakeCompany(filings=[filing])
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.get_insider_trading("AAPL", days=90)

    assert len(result) == 1
    assert result[0]["insider_name"] == "Jane Doe"
    assert result[0]["shares"] == 100.0
    assert result[0]["price"] == 25.5
    assert result[0]["transaction_type"] == "Purchase"


async def test_get_insider_trading_skips_unparseable_filing_and_logs(provider, monkeypatch):
    now = pd.Timestamp.now()
    good_row = {
        "Date": now,
        "Shares": 50,
        "Price": 10.0,
        "Insider": "Good Filing",
        "Transaction Type": "Sale",
        "Code": "S",
    }
    broken_filing = FakeFiling("0001-broken", raises=True)
    good_filing = FakeFiling("0001-good", form4_df=_form4_df([good_row]))
    fake = FakeCompany(filings=[broken_filing, good_filing])
    _patch_company(monkeypatch, lambda ticker: fake)

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_insider_trading("AAPL", days=90)

    assert len(result) == 1
    assert result[0]["insider_name"] == "Good Filing"
    assert any(
        log["event"] == "edgar_form4_parse_failed" and log["accession"] == "0001-broken"
        for log in logs
    )


async def test_get_insider_trading_no_filings_returns_empty_list(provider, monkeypatch):
    fake = FakeCompany(filings=[])
    _patch_company(monkeypatch, lambda ticker: fake)

    result = await provider.get_insider_trading("AAPL", days=90)

    assert result == []


# --- 5 US stocks of varying market cap, per ticket's unit-test requirement ---


@pytest.mark.parametrize(
    "ticker,concepts",
    [
        ("AAPL", FULL_INCOME_CONCEPTS),  # mega-cap
        ("JPM", THIN_INCOME_CONCEPTS),  # mega-cap bank — no COGS model, partial by design
        ("DELL", FULL_INCOME_CONCEPTS),  # mid-cap
        ("PLUG", THIN_INCOME_CONCEPTS),  # small-cap
        ("GRPN", ["us-gaap_Revenues"]),  # micro-cap, sparse statement
    ],
)
async def test_get_financials_across_market_caps(provider, monkeypatch, ticker, concepts):
    fake = FakeCompany(annual=FakeFinancials(income_df=_income_df(concepts)))
    _patch_company(monkeypatch, lambda t, _fake=fake: _fake)

    df = await provider.get_financials(ticker, "income", "annual")

    assert list(df["concept"]) == concepts


# --- Out-of-scope methods ---


@pytest.mark.parametrize(
    "method,args",
    [
        ("get_price_history", ("AAPL", "1y", "1d")),
        ("get_company_info", ("AAPL",)),
        ("get_analyst_estimates", ("AAPL",)),
        ("get_analyst_ratings", ("AAPL",)),
        ("get_peers", ("AAPL",)),
        ("get_earnings_calendar", ("AAPL",)),
        ("get_dividend_history", ("AAPL", "2024-01-01", "2024-12-31")),
    ],
)
async def test_out_of_scope_methods_raise_not_implemented(provider, method, args):
    with pytest.raises(NotImplementedError):
        await getattr(provider, method)(*args)


# --- Canadian cross-listed filing coverage (ClickUp 86bb47560) ---


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
    monkeypatch.setattr("data.providers.edgartools._load_crosslisting_map", lambda: mapping)


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
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

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
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

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
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result is None


async def test_find_40f_exhibit_no_filings_returns_none(monkeypatch):
    fake = FakeCompany(filings_by_form={"40-F": []})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert result is None


async def test_find_40f_exhibit_no_candidate_matches_returns_none(monkeypatch):
    attachments = [
        FakeExhibitAttachment("ex994.htm", "EX-99.4", text="Return on equity ratios only")
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

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
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    await _find_40f_exhibit_text(1, _MDA_DESCRIPTION_MARKERS, _MDA_CONTENT_MARKERS)

    assert calls == []


# --- _get_10k_item ---


async def test_get_10k_item_returns_requested_attribute(monkeypatch):
    obj = FakeTenKObj(management_discussion="MDA text", business="Business text")
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    mda = await _get_10k_item(16875, "management_discussion")
    biz = await _get_10k_item(16875, "business")

    assert mda == "MDA text"
    assert biz == "Business text"


async def test_get_10k_item_no_filings_returns_none(monkeypatch):
    fake = FakeCompany(filings_by_form={"10-K": []})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await _get_10k_item(16875, "management_discussion")

    assert result is None


async def test_get_10k_item_missing_attr_returns_none(monkeypatch):
    obj = FakeTenKObj(management_discussion=None)
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await _get_10k_item(16875, "management_discussion")

    assert result is None


# --- get_crosslisted_mda / get_crosslisted_business_overview ---


async def test_get_crosslisted_mda_ticker_not_mapped_returns_none(provider, monkeypatch):
    _patch_crosslisting_map(monkeypatch, {})

    result = await provider.get_crosslisted_mda("ZZZZ.TO")

    assert result is None


async def test_get_crosslisted_mda_10k_filer_uses_10k_path(provider, monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"CP.TO": {"us_ticker": "CP", "cik": 16875, "form_type": "10-K", "company_name": "CPKC"}},
    )
    obj = FakeTenKObj(management_discussion="CP's real Item 7 text")
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await provider.get_crosslisted_mda("CP.TO")

    assert result == "CP's real Item 7 text"


async def test_get_crosslisted_mda_40f_filer_uses_exhibit_path(provider, monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    attachments = [FakeExhibitAttachment("ex2.htm", "EX-2 Financial Review", text="RY's MDA text")]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await provider.get_crosslisted_mda("RY.TO")

    assert result == "RY's MDA text"


async def test_get_crosslisted_business_overview_ticker_not_mapped_returns_none(
    provider, monkeypatch
):
    _patch_crosslisting_map(monkeypatch, {})

    result = await provider.get_crosslisted_business_overview("ZZZZ.TO")

    assert result is None


async def test_get_crosslisted_business_overview_10k_filer_uses_business_attr(
    provider, monkeypatch
):
    _patch_crosslisting_map(
        monkeypatch,
        {"CP.TO": {"us_ticker": "CP", "cik": 16875, "form_type": "10-K", "company_name": "CPKC"}},
    )
    obj = FakeTenKObj(business="CP's real Item 1 text")
    fake = FakeCompany(filings_by_form={"10-K": [Fake10KFiling(obj)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await provider.get_crosslisted_business_overview("CP.TO")

    assert result == "CP's real Item 1 text"


async def test_get_crosslisted_business_overview_40f_filer_uses_aif_exhibit(provider, monkeypatch):
    _patch_crosslisting_map(
        monkeypatch,
        {"RY.TO": {"us_ticker": "RY", "cik": 1000275, "form_type": "40-F", "company_name": "RBC"}},
    )
    attachments = [
        FakeExhibitAttachment("ex1.htm", "EX-1 Annual Information Form", text="RY's AIF text")
    ]
    fake = FakeCompany(filings_by_form={"40-F": [Fake40FFiling(attachments)]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await provider.get_crosslisted_business_overview("RY.TO")

    assert result == "RY's AIF text"


# --- get_crosslisted_interim_exhibits ---


async def test_get_crosslisted_interim_exhibits_ticker_not_mapped_returns_empty(
    provider, monkeypatch
):
    _patch_crosslisting_map(monkeypatch, {})

    result = await provider.get_crosslisted_interim_exhibits("ZZZZ.TO")

    assert result == []


async def test_get_crosslisted_interim_exhibits_10k_filer_returns_empty(provider, monkeypatch):
    """CP is a 10-K filer, which uses 8-K for interim filings, not 6-K — out
    of scope for this method, must not attempt a 6-K fetch."""
    _patch_crosslisting_map(
        monkeypatch,
        {"CP.TO": {"us_ticker": "CP", "cik": 16875, "form_type": "10-K", "company_name": "CPKC"}},
    )
    fake = FakeCompany(filings_by_form={"6-K": [Fake6KFiling("2026-01-01", "0001")]})
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await provider.get_crosslisted_interim_exhibits("CP.TO")

    assert result == []


async def test_get_crosslisted_interim_exhibits_40f_filer_returns_recent_filings(
    provider, monkeypatch
):
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
    monkeypatch.setattr("data.providers.edgartools.Company", lambda cik: fake)

    result = await provider.get_crosslisted_interim_exhibits("RY.TO", limit=2)

    assert result == [
        {"filing_date": "2026-03-01", "accession_no": "0003"},
        {"filing_date": "2026-02-01", "accession_no": "0002"},
    ]
