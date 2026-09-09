import pytest
import structlog

from data.providers.stats_canada import StatsCanadaProvider, _shift_months, _shift_year


@pytest.fixture
def provider():
    return StatsCanadaProvider()


def _point(
    ref_per: str,
    value: float | None,
    scalar_factor_code: int = 0,
    released: str = "2026-07-10T08:30",
) -> dict:
    return {
        "refPer": ref_per,
        "value": value,
        "scalarFactorCode": scalar_factor_code,
        "releaseTime": released,
    }


def _success_row(*points: dict, vector_id: int | None = None) -> dict:
    return {"status": "SUCCESS", "object": {"vectorId": vector_id, "vectorDataPoint": list(points)}}


def _failed_row(vector_id: int | None = None) -> dict:
    return {"status": "FAILED", "object": {"vectorId": vector_id, "vectorDataPoint": []}}


def _mock_fetch(monkeypatch, provider, rows_by_call: list[list[dict]]):
    """Each call to _fetch_vectors returns the next list in rows_by_call, in order."""
    calls = []

    async def fake_fetch_vectors(vector_ids, latest_n=1):
        calls.append((vector_ids, latest_n))
        return rows_by_call[len(calls) - 1]

    monkeypatch.setattr(provider, "_fetch_vectors", fake_fetch_vectors)
    return calls


# --- _shift_year ---


def test_shift_year_back_one():
    assert _shift_year("2026-04-01", -1) == "2025-04-01"


def test_shift_year_forward_one():
    assert _shift_year("2025-04-01", 1) == "2026-04-01"


# --- _shift_months ---


def test_shift_months_back_three():
    assert _shift_months("2026-06-01", -3) == "2026-03-01"


def test_shift_months_back_across_year_boundary():
    assert _shift_months("2026-01-01", -3) == "2025-10-01"


def test_shift_months_forward_three():
    assert _shift_months("2025-10-01", 3) == "2026-01-01"


# --- _scaled_value ---


def test_scaled_value_applies_scalar_factor(provider):
    point = _point("2026-06-01", 238.971, scalar_factor_code=3)
    assert provider._scaled_value(point) == pytest.approx(238971.0)


def test_scaled_value_zero_factor_is_unscaled(provider):
    point = _point("2026-06-01", 6.5, scalar_factor_code=0)
    assert provider._scaled_value(point) == 6.5


# --- _latest_point ---


def test_latest_point_picks_max_ref_per(provider):
    row = _success_row(
        _point("2026-04-01", 1.0), _point("2026-06-01", 3.0), _point("2026-05-01", 2.0)
    )
    point = provider._latest_point(row)
    assert point["refPer"] == "2026-06-01"


def test_latest_point_failed_status_returns_none(provider):
    assert provider._latest_point(_failed_row()) is None


def test_latest_point_empty_datapoints_returns_none(provider):
    row = _success_row()
    assert provider._latest_point(row) is None


def test_latest_point_ignores_null_values(provider):
    row = _success_row(_point("2026-06-01", None))
    assert provider._latest_point(row) is None


# --- get_unemployment_rate ---


async def test_get_unemployment_rate_returns_value_period_and_6m_delta(provider, monkeypatch):
    calls = _mock_fetch(
        monkeypatch,
        provider,
        [[_success_row(_point("2025-12-01", 6.2), _point("2026-06-01", 6.5))]],
    )

    result = await provider.get_unemployment_rate()

    assert calls[0][1] == 8  # latest_n
    assert result == {
        "value": 6.5,
        "delta_6m_pp": pytest.approx(
            0.3
        ),  # 6.5 now vs 6.2 six months back (raw; rounded in macro_sources)
        "reference_period": "2026-06-01",
        "released": "2026-07-10T08:30",
    }


async def test_get_unemployment_rate_6m_delta_none_when_no_prior_point(provider, monkeypatch):
    _mock_fetch(monkeypatch, provider, [[_success_row(_point("2026-06-01", 6.5))]])

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_unemployment_rate()

    assert result["value"] == 6.5
    assert result["delta_6m_pp"] is None
    assert any(log["event"] == "statcan_unemployment_no_prior_6m" for log in logs)


async def test_get_unemployment_rate_failed_returns_none_and_logs(provider, monkeypatch):
    _mock_fetch(monkeypatch, provider, [[_failed_row()]])

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_unemployment_rate()

    assert result is None
    assert any(log["event"] == "statcan_unemployment_missing" for log in logs)


# --- get_housing_starts ---


async def test_get_housing_starts_applies_scaling(provider, monkeypatch):
    _mock_fetch(
        monkeypatch, provider, [[_success_row(_point("2026-06-01", 238.971, scalar_factor_code=3))]]
    )

    result = await provider.get_housing_starts()

    assert result["value"] == pytest.approx(238971.0)
    assert result["reference_period"] == "2026-06-01"


async def test_get_housing_starts_failed_returns_none(provider, monkeypatch):
    _mock_fetch(monkeypatch, provider, [[_failed_row()]])

    result = await provider.get_housing_starts()

    assert result is None


# --- get_retail_sales_yoy ---


async def test_get_retail_sales_yoy_computes_correct_percentage(provider, monkeypatch):
    row = _success_row(
        _point("2025-05-01", 100_000.0, scalar_factor_code=3),
        _point("2026-05-01", 110_000.0, scalar_factor_code=3),
    )
    _mock_fetch(monkeypatch, provider, [[row]])

    result = await provider.get_retail_sales_yoy()

    assert result["value"] == pytest.approx(10.0)
    assert result["reference_period"] == "2026-05-01"


async def test_get_retail_sales_yoy_requests_14_periods(provider, monkeypatch):
    calls = _mock_fetch(
        monkeypatch,
        provider,
        [[_success_row(_point("2025-05-01", 100.0), _point("2026-05-01", 110.0))]],
    )

    await provider.get_retail_sales_yoy()

    _, latest_n = calls[0]
    assert latest_n == 14


async def test_get_retail_sales_yoy_missing_prior_year_returns_none(provider, monkeypatch):
    row = _success_row(_point("2026-05-01", 110_000.0))
    _mock_fetch(monkeypatch, provider, [[row]])

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_retail_sales_yoy()

    assert result is None
    assert any(log["event"] == "statcan_retail_sales_no_prior_year" for log in logs)


async def test_get_retail_sales_yoy_failed_status_returns_none(provider, monkeypatch):
    _mock_fetch(monkeypatch, provider, [[_failed_row()]])

    result = await provider.get_retail_sales_yoy()

    assert result is None


async def test_get_retail_sales_yoy_all_null_values_returns_none(provider, monkeypatch):
    row = _success_row(_point("2026-05-01", None))
    _mock_fetch(monkeypatch, provider, [[row]])

    result = await provider.get_retail_sales_yoy()

    assert result is None


# --- _yoy_pct_for_period ---


def test_yoy_pct_for_period_computes_correct_percentage(provider):
    points = {
        "2025-06-01": _point("2025-06-01", 100.0),
        "2026-06-01": _point("2026-06-01", 110.0),
    }
    assert provider._yoy_pct_for_period(points, "2026-06-01") == pytest.approx(10.0)


def test_yoy_pct_for_period_missing_current_returns_none(provider):
    points = {"2025-06-01": _point("2025-06-01", 100.0)}
    assert provider._yoy_pct_for_period(points, "2026-06-01") is None


def test_yoy_pct_for_period_missing_prior_year_returns_none(provider):
    points = {"2026-06-01": _point("2026-06-01", 110.0)}
    assert provider._yoy_pct_for_period(points, "2026-06-01") is None


# --- get_cpi_national ---


async def test_get_cpi_national_computes_yoy_and_pp_delta(provider, monkeypatch):
    """delta_3m_pp is a percentage-point change in the YoY rate (YoY now
    minus YoY as of 3 months ago), not a percent change of the index —
    needs 4 points: latest, 3mo-ago, 12mo-ago, and 15mo-ago (the base for
    'YoY as of 3 months ago')."""
    row = _success_row(
        _point("2025-03-01", 100.0),  # 15mo ago
        _point("2025-06-01", 100.0),  # 12mo ago
        _point("2026-03-01", 106.0),  # 3mo ago
        _point("2026-06-01", 110.0),  # latest
    )
    _mock_fetch(monkeypatch, provider, [[row]])

    result = await provider.get_cpi_national()

    assert result["value"] == pytest.approx(110.0)
    assert result["yoy_pct"] == pytest.approx(10.0)  # 110 vs 100
    # YoY now (10.0) minus YoY 3mo ago (106 vs 100 = 6.0) = 4.0pp
    assert result["delta_3m_pp"] == pytest.approx(4.0)
    assert result["reference_period"] == "2026-06-01"


async def test_get_cpi_national_requests_17_periods(provider, monkeypatch):
    calls = _mock_fetch(monkeypatch, provider, [[_success_row(_point("2026-06-01", 110.0))]])

    await provider.get_cpi_national()

    _, latest_n = calls[0]
    assert latest_n == 17


async def test_get_cpi_national_missing_lookback_points_returns_none_deltas(provider, monkeypatch):
    """Only the latest point exists — both yoy_pct and delta_3m_pp fail
    for the same underlying reason (no 12mo-back point at all). Only
    no_prior_year should log; no_prior_3m_yoy would be a misleading
    second warning for what's really one root cause."""
    row = _success_row(_point("2026-06-01", 110.0))
    _mock_fetch(monkeypatch, provider, [[row]])

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_cpi_national()

    assert result["value"] == pytest.approx(110.0)
    assert result["yoy_pct"] is None
    assert result["delta_3m_pp"] is None
    assert any(log["event"] == "statcan_cpi_national_no_prior_year" for log in logs)
    assert not any(log["event"] == "statcan_cpi_national_no_prior_3m_yoy" for log in logs)


async def test_get_cpi_national_missing_only_15mo_point_returns_none_delta(provider, monkeypatch):
    """3mo-ago point exists (so YoY-now resolves fine), but its own
    prior-year point (15mo back) is missing — delta_3m_pp must still
    come back None, not a wrong number computed from a partial YoY."""
    row = _success_row(
        _point("2025-06-01", 100.0),  # 12mo ago
        _point("2026-03-01", 106.0),  # 3mo ago
        _point("2026-06-01", 110.0),  # latest
    )
    _mock_fetch(monkeypatch, provider, [[row]])

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_cpi_national()

    assert result["yoy_pct"] == pytest.approx(10.0)
    assert result["delta_3m_pp"] is None
    assert any(log["event"] == "statcan_cpi_national_no_prior_3m_yoy" for log in logs)


async def test_get_cpi_national_failed_status_returns_none(provider, monkeypatch):
    _mock_fetch(monkeypatch, provider, [[_failed_row()]])

    result = await provider.get_cpi_national()

    assert result is None


# --- get_real_gdp_index ---


async def test_get_real_gdp_index_computes_annualized_qoq_and_yoy(provider, monkeypatch):
    row = _success_row(
        _point("2025-01-01", 100.0),  # 4 quarters ago
        _point("2025-10-01", 100.0),  # prior quarter
        _point("2026-01-01", 110.0),  # latest
    )
    _mock_fetch(monkeypatch, provider, [[row]])

    result = await provider.get_real_gdp_index()

    assert result["value"] == pytest.approx(110.0)
    # ((110/100)**4 - 1) * 100
    assert result["qoq_annualized_pct"] == pytest.approx(46.41, abs=0.01)
    assert result["yoy_pct"] == pytest.approx(10.0)
    assert result["reference_period"] == "2026-01-01"


async def test_get_real_gdp_index_requests_6_periods(provider, monkeypatch):
    calls = _mock_fetch(monkeypatch, provider, [[_success_row(_point("2026-01-01", 110.0))]])

    await provider.get_real_gdp_index()

    _, latest_n = calls[0]
    assert latest_n == 6


async def test_get_real_gdp_index_missing_lookback_points_returns_none_deltas(
    provider, monkeypatch
):
    row = _success_row(_point("2026-01-01", 110.0))
    _mock_fetch(monkeypatch, provider, [[row]])

    with structlog.testing.capture_logs() as logs:
        result = await provider.get_real_gdp_index()

    assert result["value"] == pytest.approx(110.0)
    assert result["qoq_annualized_pct"] is None
    assert result["yoy_pct"] is None
    assert any(log["event"] == "statcan_gdp_index_no_prior_quarter" for log in logs)
    assert any(log["event"] == "statcan_gdp_index_no_prior_year" for log in logs)


async def test_get_real_gdp_index_failed_status_returns_none(provider, monkeypatch):
    _mock_fetch(monkeypatch, provider, [[_failed_row()]])

    result = await provider.get_real_gdp_index()

    assert result is None


# --- Session lifecycle ---


class FakeClientSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_context_manager_opens_and_closes_owned_session(monkeypatch):
    created = []

    def fake_client_session():
        session = FakeClientSession()
        created.append(session)
        return session

    monkeypatch.setattr("data.providers.stats_canada.aiohttp.ClientSession", fake_client_session)

    async with StatsCanadaProvider() as provider:
        assert provider.session is created[0]
        assert provider.session.closed is False

    assert created[0].closed is True


async def test_context_manager_does_not_close_externally_provided_session():
    external = FakeClientSession()
    provider = StatsCanadaProvider(session=external)

    async with provider:
        pass

    assert external.closed is False
