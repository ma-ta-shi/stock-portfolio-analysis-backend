import pytest

from data.degradation import DegradationCollector, reset_collector, set_collector


@pytest.fixture
def collector():
    """A degradation collector installed for the test, as the orchestrator does for a
    real run (86bc997wr). Provider tests use it to assert what a failing path reports
    and that a healthy path reports nothing."""
    c = DegradationCollector()
    token = set_collector(c)
    yield c
    reset_collector(token)


@pytest.fixture
def parity():
    """Reporting must never change what a provider returns. `await parity(make_call)`
    runs the call twice, once with no collector and once with one, and returns
    (result_without, result_with, events). `make_call` is a no-argument function
    returning an awaitable, and its result must be comparable with ==."""

    async def run(make_call):
        plain = await make_call()
        c = DegradationCollector()
        token = set_collector(c)
        try:
            with_collector = await make_call()
        finally:
            reset_collector(token)
        return plain, with_collector, c.drain()

    return run
