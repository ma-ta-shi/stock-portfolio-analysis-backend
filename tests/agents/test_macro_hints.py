"""Tests for agents/macro_hints.py and how the Macro Economist prompt is filled with them."""

from types import SimpleNamespace

import pytest

from agents.macro_hints import _SECTOR_HINTS, currency_exposure_hint, sector_macro_hint
from agents.pass1_macro_economist import MacroEconomistRunner
from data.sector_names import CANONICAL_SECTORS


def test_every_canonical_sector_has_its_own_hint():
    assert set(_SECTOR_HINTS) == set(CANONICAL_SECTORS)


@pytest.mark.parametrize(
    "sector, expected_fragment",
    [
        ("Finance", "net_interest_margin"),  # a Canadian bank, TMX spelling
        ("Financial Services", "net_interest_margin"),
        ("Technology", "duration risk"),
        ("Energy", "oil_price"),
        ("Basic Materials", "china_demand"),
        ("Consumer Defensive", "input_costs"),
    ],
)
def test_the_hint_follows_the_sector_whatever_the_provider_calls_it(sector, expected_fragment):
    assert expected_fragment in sector_macro_hint(sector)


@pytest.mark.parametrize("sector", [None, "", "Cryptocurrency"])
def test_an_unknown_sector_gets_the_default_row(sector):
    assert sector_macro_hint(sector) == "interest_rates, inflation, economic_growth, usd_strength"


def test_currency_hint_differs_for_us_and_canadian_listings():
    assert "primarily_usd" in currency_exposure_hint(False)
    assert "primarily_cad" in currency_exposure_hint(True)


async def test_the_runner_fills_both_hints_into_the_prompt(monkeypatch):
    from tests.agents.test_pass1_macro_economist import _bundle

    captured = {}

    async def fake_call(self, system_prompt, user_msg, validator, **kwargs):
        captured["system"] = system_prompt
        return {}, []

    monkeypatch.setattr(MacroEconomistRunner, "call_with_validation", fake_call)
    bundle = _bundle(is_ca=True)
    bundle.canadian_data_flags = SimpleNamespace()  # a Canadian listing
    await MacroEconomistRunner().run(bundle)

    prompt = captured["system"]
    assert "net_interest_margin" in prompt  # the sector list, after "SECTOR MACRO SENSITIVITIES:"
    assert "Currency exposure: primarily_cad" in prompt
    assert "{sector_macro_hint}" not in prompt and "{currency_exposure_hint}" not in prompt
