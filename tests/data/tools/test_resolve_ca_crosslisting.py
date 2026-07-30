import json

from data.tools.resolve_ca_crosslisting import (
    FlaggedEntry,
    ResolvedEntry,
    _name_tokens,
    _names_plausibly_match,
    build_mapping,
    fetch_sec_ticker_index,
    find_name_fallback_candidate,
    resolve_ticker,
    write_mapping_file,
    write_review_file,
)


# --- Fakes standing in for aiohttp ---


class FakeResponse:
    def __init__(self, status: int, json_data=None) -> None:
        self.status = status
        self._json_data = json_data

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def json(self):
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class FakeSession:
    """Routes by URL: the SEC ticker index (one fixed payload) vs a per-CIK
    submissions lookup (a dict of {cik: [forms]})."""

    def __init__(self, ticker_index_rows: dict, forms_by_cik: dict[int, list[str]]) -> None:
        self._ticker_index_rows = ticker_index_rows
        self._forms_by_cik = forms_by_cik
        self.requested_urls: list[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.requested_urls.append(url)
        if "company_tickers.json" in url:
            return FakeResponse(200, json_data=self._ticker_index_rows)
        # submissions URL: .../CIK{10-digit}.json
        cik = int(url.rsplit("CIK", 1)[1].removesuffix(".json"))
        forms = self._forms_by_cik.get(cik)
        if forms is None:
            return FakeResponse(404)
        return FakeResponse(200, json_data={"filings": {"recent": {"form": forms}}})


def _ticker_rows(*entries: tuple[str, int, str]) -> dict:
    """entries: (ticker, cik, title)"""
    return {
        str(i): {"ticker": t, "cik_str": cik, "title": title}
        for i, (t, cik, title) in enumerate(entries)
    }


# --- _name_tokens / _names_plausibly_match ---


def test_name_tokens_strips_noise_words():
    assert _name_tokens("WSP Global Inc.") == {"WSP"}
    assert _name_tokens("BCE Inc.") == {"BCE"}


def test_names_plausibly_match_requires_full_subset():
    assert _names_plausibly_match("Royal Bank of Canada", "ROYAL BANK OF CANADA") is True
    assert (
        _names_plausibly_match("Canadian National Railway", "CANADIAN NATIONAL RAILWAY CO") is True
    )


def test_names_plausibly_match_rejects_generic_word_only_overlap():
    """Regression test: 'WSP Global Inc.' must NOT match 'S&P Global Inc.' —
    confirmed live this happened when GLOBAL alone counted as a match."""
    assert _names_plausibly_match("WSP Global Inc.", "S&P Global Inc.") is False


def test_names_plausibly_match_rejects_acronym_with_zero_overlap():
    assert _names_plausibly_match("CIBC", "CANADIAN IMPERIAL BANK OF COMMERCE /CAN/") is False


def test_names_plausibly_match_empty_inputs_do_not_match():
    assert _names_plausibly_match("", "ANYTHING") is False
    assert _names_plausibly_match("ANYTHING", "") is False


# --- find_name_fallback_candidate ---


def test_find_name_fallback_candidate_finds_match_by_name():
    index = {
        "CNR": {"cik": 1710366, "title": "Core Natural Resources, Inc."},
        "CNI": {"cik": 16868, "title": "CANADIAN NATIONAL RAILWAY CO"},
    }
    result = find_name_fallback_candidate(index, "Canadian National Railway")
    assert result == ("CNI", {"cik": 16868, "title": "CANADIAN NATIONAL RAILWAY CO"})


def test_find_name_fallback_candidate_no_match_returns_none():
    index = {"XYZ": {"cik": 1, "title": "Totally Unrelated Corp"}}
    assert find_name_fallback_candidate(index, "Canadian National Railway") is None


# --- resolve_ticker ---


async def test_resolve_ticker_direct_match_resolves_cleanly():
    session = FakeSession(
        _ticker_rows(("RY", 1000275, "ROYAL BANK OF CANADA")),
        forms_by_cik={1000275: ["10-K", "40-F", "6-K"]},
    )
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "RY.TO", "Royal Bank of Canada")

    assert isinstance(result, ResolvedEntry)
    assert result.us_ticker == "RY"
    assert result.cik == 1000275
    assert result.form_type == "40-F"  # first match in RELEVANT_FORMS priority order
    assert result.match_method == "ticker"


async def test_resolve_ticker_prefers_10k_when_no_foreign_filer_forms():
    session = FakeSession(
        _ticker_rows(("CP", 16875, "CANADIAN PACIFIC KANSAS CITY LTD/CN")),
        forms_by_cik={16875: ["10-K", "10-Q", "8-K"]},
    )
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "CP.TO", "Canadian Pacific Kansas City")

    assert isinstance(result, ResolvedEntry)
    assert result.form_type == "10-K"


async def test_resolve_ticker_wrong_company_with_valid_form_flags_with_alternative():
    """CNR's actual real-world case: the ticker exists AND happens to file a
    relevant form (10-K) — but it's a completely unrelated company. Must
    flag rather than auto-accept, and must surface the correct alternative
    (found by name search) in the reason text."""
    session = FakeSession(
        _ticker_rows(
            ("CNR", 1710366, "Core Natural Resources, Inc."),
            ("CNI", 16868, "CANADIAN NATIONAL RAILWAY CO"),
        ),
        forms_by_cik={1710366: ["10-K", "8-K"], 16868: ["40-F", "6-K"]},
    )
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "CNR.TO", "Canadian National Railway")

    assert isinstance(result, FlaggedEntry)
    assert "Core Natural Resources" in result.reason
    assert "CANADIAN NATIONAL RAILWAY CO" in result.reason
    assert "CNI" in result.reason


async def test_resolve_ticker_wrong_ticker_no_relevant_form_falls_through_to_name_search():
    """When the direct ticker candidate has NO relevant form at all (not
    even a coincidental one), it's unambiguous enough to fall through
    automatically rather than flag."""
    session = FakeSession(
        _ticker_rows(
            ("XYZ", 1710366, "Totally Unrelated Shell Co"),
            ("CNI", 16868, "CANADIAN NATIONAL RAILWAY CO"),
        ),
        forms_by_cik={1710366: ["S-1"], 16868: ["40-F", "6-K"]},
    )
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "XYZ.TO", "Canadian National Railway")

    assert isinstance(result, ResolvedEntry)
    assert result.us_ticker == "CNI"
    assert result.cik == 16868
    assert result.match_method == "name_fallback"


async def test_resolve_ticker_name_mismatch_with_valid_form_flags_not_rejects():
    """CIBC's real-world case: ticker + filing type both check out, but the
    name doesn't textually match — must flag for review, not silently
    discard the plausibly-correct match."""
    session = FakeSession(
        _ticker_rows(("CM", 1045520, "CANADIAN IMPERIAL BANK OF COMMERCE /CAN/")),
        forms_by_cik={1045520: ["40-F", "6-K"]},
    )
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "CM.TO", "CIBC")

    assert isinstance(result, FlaggedEntry)
    assert "CANADIAN IMPERIAL BANK OF COMMERCE" in result.reason
    assert result.candidate is None  # reason text carries full context instead


async def test_resolve_ticker_no_match_anywhere_flags_with_clear_reason():
    session = FakeSession(_ticker_rows(("ZZZ", 1, "Unrelated Co")), forms_by_cik={})
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "ATD.TO", "Alimentation Couche-Tard")

    assert isinstance(result, FlaggedEntry)
    assert "No SEC entity found" in result.reason


async def test_resolve_ticker_name_fallback_with_no_relevant_form_flags():
    session = FakeSession(
        _ticker_rows(("XYZ", 99, "Totally Different Corp"), ("ABC", 42, "MATCHING NAME CORP")),
        forms_by_cik={42: ["S-1"]},
    )
    index = await fetch_sec_ticker_index(session)

    result = await resolve_ticker(session, index, "MATCHING.TO", "Matching Name")

    assert isinstance(result, FlaggedEntry)
    assert "never filed" in result.reason
    assert result.candidate == {"us_ticker": "ABC", "cik": 42, "company_name": "MATCHING NAME CORP"}


# --- build_mapping (skip-already-confirmed behavior) ---


async def test_build_mapping_skips_already_confirmed_tickers():
    session = FakeSession(
        _ticker_rows(("RY", 1000275, "ROYAL BANK OF CANADA")),
        forms_by_cik={1000275: ["40-F"]},
    )
    resolved, flagged = await build_mapping(
        session,
        [("RY.TO", "Royal Bank of Canada"), ("CM.TO", "CIBC")],
        already_confirmed={"CM.TO"},
    )
    assert [r.ca_ticker for r in resolved] == ["RY.TO"]
    assert flagged == []


# --- write_mapping_file (merge behavior) ---


def test_write_mapping_file_preserves_untouched_existing_entries(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "CM.TO": {
                    "us_ticker": "CM",
                    "cik": 1,
                    "form_type": "40-F",
                    "company_name": "X",
                    "manually_confirmed": "y",
                }
            }
        ),
        encoding="utf-8",
    )

    write_mapping_file(
        [ResolvedEntry("RY.TO", "RY", 1000275, "40-F", "ROYAL BANK OF CANADA", "ticker")],
        path,
    )

    result = json.loads(path.read_text(encoding="utf-8"))
    assert "CM.TO" in result  # preserved, not clobbered
    assert "RY.TO" in result  # newly added


def test_write_mapping_file_overwrites_stale_data_for_tickers_it_resolves(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {"RY.TO": {"us_ticker": "RY", "cik": 999, "form_type": "10-K", "company_name": "STALE"}}
        ),
        encoding="utf-8",
    )

    write_mapping_file(
        [ResolvedEntry("RY.TO", "RY", 1000275, "40-F", "ROYAL BANK OF CANADA", "ticker")],
        path,
    )

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["RY.TO"]["cik"] == 1000275
    assert result["RY.TO"]["form_type"] == "40-F"


def test_write_mapping_file_no_existing_file_creates_fresh(tmp_path):
    path = tmp_path / "mapping.json"
    write_mapping_file(
        [ResolvedEntry("RY.TO", "RY", 1000275, "40-F", "ROYAL BANK OF CANADA", "ticker")], path
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result == {
        "RY.TO": {
            "us_ticker": "RY",
            "cik": 1000275,
            "form_type": "40-F",
            "company_name": "ROYAL BANK OF CANADA",
        }
    }


# --- write_review_file ---


def test_write_review_file_utf8_round_trips_em_dash(tmp_path):
    """Regression test: the review file previously mangled em-dashes on
    Windows because write_text() wasn't given an explicit encoding."""
    path = tmp_path / "review.md"
    write_review_file([FlaggedEntry("X.TO", "X Corp — Y", "some reason")], path)
    text = path.read_text(encoding="utf-8")
    assert "X Corp — Y" in text
    assert "�" not in text  # no replacement-character mangling


def test_write_review_file_empty_flagged_list_says_so(tmp_path):
    path = tmp_path / "review.md"
    write_review_file([], path)
    text = path.read_text(encoding="utf-8")
    assert "resolved cleanly" in text
