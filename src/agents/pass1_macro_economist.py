"""Pass 1 — Macro Economist runner (v1.2).

Production port of `simulation/runners/pass1_macro_economist.py` (86bbuhjup).
NOT a clean port -- `MacroSourcesBundle` (`data/schemas/macro_sources_bundle.py`)
is dramatically richer than the harness fixture's flat 8-field `macro_data`
dict: real trend classifications (`rate_trend`, `cpi_trend`, `cad_trend`,
`vix_regime`, `us_gdp_4q_trend`, `ca_gdp_4q_trend`), deltas
(`policy_rate_90d_delta_bp`, `cpi_3m_delta_pp`, ...), yield curve shapes,
central bank stance notes, and CA-specific StatCan fields the harness fixture
never had. The real prompt's own text confirms these trends are meant to
arrive pre-computed ("Direction is already computed (cite RATE)", "Trend
already computed (cite CPI)") -- this renders the real classified fields, not
raw series the LLM would have to re-derive trends from itself.

Field-by-field notes:
- `sector_tailwinds`/`sector_headwinds` (harness fixture: input data) are
  actually Macro's own OUTPUT fields -- confirmed via `agents/pass2_view.py`'s
  MACRO branch, which reads them from the agent's own `structured_data`
  output, never from bundle input. Omitted from the input payload, not
  fabricated as pre-supplied data (same "don't spoon-feed the verdict"
  pattern as Fundamental's guidance_vs_consensus / Stock Researcher's
  competitive_position).
- CA-specific fields (`boc_rate`, `canada_bond_*`, `ca_cpi_*`, `ca_gdp_*`,
  `statcan_*`) are rendered only when the stock is Canadian
  (`bundle.canadian_data_flags is not None`) -- for a US stock they are
  always None per the schema's own "None if... the stock isn't Canadian"
  documentation, not a real gap.
- `sector_commodity_*` fields are rendered only when
  `sector_commodity_relevant` is True -- the schema's own validator enforces
  they're None otherwise, so gating the render on the same flag avoids a
  wall of "N/A" lines for sectors with no real commodity linkage.
- "BASE RELIABILITY SCORE" block: omitted, same D6-retirement reasoning as
  every other Pass 1 runner in this port.

86bbummwp Tier 1a: `data_coverage_line` is now built from the same
`field_presence` map `build_user_message()` already computes for
`input_field_coverage`, via the shared `render_data_coverage_line()` helper --
was hardcoded to "Data coverage: standard." on every run before this. The
`statcan` key is only ever present in `field_presence` at all for a Canadian
stock (see `build_user_message()`'s own docstring) -- the shared helper treats
a missing key as "not applicable" and a present-but-`False` key as a real gap,
so a US stock never gets a spurious StatCan mention.

Found and fixed during this same round's own critical review, not caught by
the first pass: `field_presence["commodities"]` is `False` unconditionally
whenever `sector_commodity_relevant` is `False` (the schema's own validator
enforces `sector_commodity_age_days=None` for a non-commodity sector -- see
the field-by-field note above) -- that's the correct, intended `False` for
`input_field_coverage`'s own purpose, but it made every non-commodity-sensitive
stock (the large majority) get a false "no sector commodity data available"
line, exactly the active-misinformation problem this ticket exists to remove.
`_coverage_presence()` now drops the `commodities` key entirely when the
sector isn't commodity-relevant, the same "absent means not applicable"
treatment already used for `statcan` -- confirmed live: AAPL/SHOP.TO/WELL.TO/
RDDT/RY.TO (none commodity-sensitive) all reported this false gap before the
fix, and reported "standard."/no commodities mention after it.

86bbummwp Tier 2: `_coverage_presence()` is factored out of `_data_coverage_line()`
(pulled out during this round, not duplicated) so the one piece of real
conditional logic here -- dropping `commodities` when not sector-relevant --
has a single source of truth, shared by both the prose line
(`render_data_coverage_line()`) and the new structured `data_coverage` flag
(`to_data_coverage()`).

86bbummwp Tier 2, `stale_data`: real per-series age fields already exist
(`*_age_days`) but were, until now, only ever read as presence booleans
(`age_days is not None`), never compared to a threshold. A UNIFORM threshold
across all 8 series would be wrong -- checked `_latest_age_days()` directly
(below): it's the age of the most recently *released* data point, and
economic series don't release on the same cadence. Daily/market series
(policy rate, FX, VIX) get a short threshold; monthly series (CPI,
unemployment, StatCan) get a threshold sized to survive a normal release lag,
not a calendar guess; GDP (quarterly) wider still. `age is None` (a fetch
failure) is NOT flagged here -- that's `data_coverage`'s job, and staleness
can't be computed without a value to measure in the first place.

86bbummwp Tier 2, `anomalies`: a threshold layer on top of already-computed
classifiers (`_curve_shape()`, `_vix_regime()`) and delta fields, none of
which were ever flagged as noteworthy before now -- just rendered as plain
values.
"""
from functools import partial

from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.utils import render_data_coverage_line, render_data_warnings, to_data_coverage
from agents.validators.common import validate_confidence_requires_caveat_when_flagged
from agents.validators.pass1 import validate_macro_economist
from data.schemas.data_bundle import DataBundle
from data.schemas.macro_sources_bundle import MacroSourcesBundle

_COVERAGE_GAP_SENTENCES = {
    "rate": "no policy rate data available",
    "yield_curve": "no bond yield data available",
    "cpi": "no inflation data available",
    "gdp": "no GDP data available",
    "employment": "no employment data available",
    "fx": "no FX data available",
    "vix": "no volatility index data available",
    "commodities": "no sector commodity data available",
    "statcan": "no Statistics Canada demand indicator data available",
}

_STALE_DAILY_DAYS = 7        # policy rate, FX, VIX, commodities -- real market/daily series
_STALE_MONTHLY_DAYS = 45     # CPI, unemployment, StatCan -- survives a normal ~30d release lag
_STALE_QUARTERLY_DAYS = 100  # GDP -- survives a normal ~90d release lag

# series -> MacroSourcesBundle age field, for the always-applicable series
# (commodities/statcan are conditional -- handled separately below).
_STALE_THRESHOLDS = {
    "rate": ("policy_rate_age_days", _STALE_DAILY_DAYS),
    "cpi": ("cpi_age_days", _STALE_MONTHLY_DAYS),
    "gdp": ("gdp_age_days", _STALE_QUARTERLY_DAYS),
    "employment": ("unemployment_age_days", _STALE_MONTHLY_DAYS),
    "fx": ("cad_usd_age_days", _STALE_DAILY_DAYS),
    "vix": ("vix_age_days", _STALE_DAILY_DAYS),
}

# field -> "an absolute delta at or beyond this magnitude is an unusually
# large move" -- not exact science, starting points for a real design
# decision (see module docstring).
_LARGE_DELTA_THRESHOLDS = {
    "policy_rate_90d_delta_bp": 100.0,     # a full percentage point in a quarter
    "cpi_3m_delta_pp": 1.0,                # a 1pp inflation shift in 3 months
    "cad_usd_90d_change_pct": 10.0,        # a 10% FX move in a quarter
    "unemployment_6m_delta": 1.0,          # a 1pp unemployment shift in 6 months
}


def _stale_data(bundle: DataBundle) -> list[str]:
    m = bundle.macro_sources
    stale = [
        series for series, (field, threshold) in _STALE_THRESHOLDS.items()
        if (age := getattr(m, field)) is not None and age > threshold
    ]
    if (
        m.sector_commodity_relevant
        and m.sector_commodity_age_days is not None
        and m.sector_commodity_age_days > _STALE_DAILY_DAYS
    ):
        stale.append("commodities")
    if (
        bundle.canadian_data_flags is not None
        and m.statcan_age_days is not None
        and m.statcan_age_days > _STALE_MONTHLY_DAYS
    ):
        stale.append("statcan")
    return stale


def _anomalies(bundle: DataBundle) -> list[str]:
    m = bundle.macro_sources
    flags = []
    if m.us_curve_shape == "inverted":
        flags.append("US yield curve is inverted")
    if bundle.canadian_data_flags is not None and m.ca_curve_shape == "inverted":
        flags.append("Canada yield curve is inverted")
    if m.vix_regime == "high":
        flags.append(f"VIX is in a high-volatility regime ({m.vix})")
    for field, threshold in _LARGE_DELTA_THRESHOLDS.items():
        value = getattr(m, field)
        if value is not None and abs(value) >= threshold:
            flags.append(f"{field}={value} is an unusually large move")
    return flags


def _coverage_presence(field_presence: dict[str, bool], sector_commodity_relevant: bool) -> dict[str, bool]:
    """The one real conditional in this agent's coverage story: `commodities`
    is not applicable at all (not a gap) for a non-commodity-sensitive sector.
    Shared by both `_data_coverage_line()` and the structured `data_coverage`
    flag so this isn't reimplemented twice."""
    if sector_commodity_relevant:
        return field_presence
    return {k: v for k, v in field_presence.items() if k != "commodities"}


def _data_coverage_line(field_presence: dict[str, bool], sector_commodity_relevant: bool) -> str:
    return render_data_coverage_line(
        _coverage_presence(field_presence, sector_commodity_relevant), _COVERAGE_GAP_SENTENCES
    )


def _fmt(v, suffix: str = ""):
    return "N/A" if v is None else f"{v}{suffix}"


def _rate_block(m: MacroSourcesBundle, is_ca: bool) -> str:
    lines = [
        f"  Fed Funds Rate: {_fmt(m.fed_funds_rate, '%')} | Trend: {_fmt(m.rate_trend)} | 90d delta: {_fmt(m.policy_rate_90d_delta_bp, 'bp')}",
    ]
    if is_ca:
        lines.append(
            f"  BoC Rate: {_fmt(m.boc_rate, '%')} | Trend: {_fmt(m.boc_rate_trend)} | 90d delta: {_fmt(m.boc_rate_90d_delta_bp, 'bp')}"
        )
    if m.cb_stance_note:
        lines.append(f"  Central bank stance: {m.cb_stance_note}")
    return "\n".join(lines)


def _yield_block(m: MacroSourcesBundle, is_ca: bool) -> str:
    lines = [
        f"  US Treasury 2y/5y/10y: {_fmt(m.treasury_2y)}/{_fmt(m.treasury_5y)}/{_fmt(m.treasury_10y)} | Curve shape: {_fmt(m.us_curve_shape)}",
    ]
    if is_ca:
        lines.append(
            f"  Canada Bond 2y/5y/10y: {_fmt(m.canada_bond_2y)}/{_fmt(m.canada_bond_5y)}/{_fmt(m.canada_bond_10y)} | Curve shape: {_fmt(m.ca_curve_shape)}"
        )
    if not m.bond_yields_available:
        lines.append("  (bond yield data unavailable this run)")
    return "\n".join(lines)


def _cpi_block(m: MacroSourcesBundle, is_ca: bool) -> str:
    lines = [
        f"  US CPI YoY: {_fmt(m.us_cpi_yoy, '%')} | Core: {_fmt(m.us_core_cpi_yoy, '%')} | Trend: {_fmt(m.cpi_trend)} | 3mo delta: {_fmt(m.cpi_3m_delta_pp, 'pp')}",
    ]
    if is_ca:
        lines.append(
            f"  Canada CPI YoY: {_fmt(m.ca_cpi_yoy, '%')} | Trend: {_fmt(m.ca_cpi_trend)} | 3mo delta: {_fmt(m.ca_cpi_3m_delta, 'pp')}"
        )
    return "\n".join(lines)


def _gdp_block(m: MacroSourcesBundle, is_ca: bool) -> str:
    lines = [f"  US GDP QoQ (annualized): {_fmt(m.us_gdp_qoq, '%')} | 4q trend: {_fmt(m.us_gdp_4q_trend)}"]
    if is_ca:
        lines.append(
            f"  Canada GDP QoQ (annualized): {_fmt(m.ca_gdp_qoq, '%')} | 4q trend: {_fmt(m.ca_gdp_4q_trend)}"
        )
    return "\n".join(lines)


def _employ_block(m: MacroSourcesBundle, is_ca: bool) -> str:
    lines = [
        f"  US Unemployment Rate: {_fmt(m.unemployment, '%')} | 6mo delta: {_fmt(m.unemployment_6m_delta, 'pp')}",
    ]
    if is_ca:
        lines.append(
            f"  Canada Unemployment Rate (LFS): {_fmt(m.statcan_unemployment_ca, '%')} | 6mo delta: {_fmt(m.ca_unemployment_6m_delta, 'pp')}"
        )
    return "\n".join(lines)


def _fx_block(m: MacroSourcesBundle, currency: str, exchange: str) -> str:
    return (
        f"  USD/CAD: {_fmt(m.cad_usd)} | Trend: {_fmt(m.cad_trend)} | 90d change: {_fmt(m.cad_usd_90d_change_pct, '%')}\n"
        f"  Stock currency: {currency} | Listed: {exchange}"
    )


def _vix_block(m: MacroSourcesBundle) -> str:
    return f"  VIX: {_fmt(m.vix)} | Regime: {_fmt(m.vix_regime)} | 30d avg: {_fmt(m.vix_30d_avg)}"


def _commod_block(m: MacroSourcesBundle) -> str:
    if not m.sector_commodity_relevant:
        return "  Not commodity-sensitive for this sector."
    return (
        f"  {_fmt(m.sector_commodity_name)}: {_fmt(m.sector_commodity_level)} "
        f"({_fmt(m.sector_commodity_direction)}, 90d change: {_fmt(m.commodity_90d_change_pct, '%')})\n"
        f"  WTI Crude: {_fmt(m.wti_crude)}"
    )


def _statcan_block(m: MacroSourcesBundle) -> str:
    return (
        f"  Housing starts (SAAR): {_fmt(m.statcan_housing_starts)}\n"
        f"  Retail sales YoY: {_fmt(m.statcan_retail_sales_yoy, '%')}"
    )


def build_user_message(bundle: DataBundle) -> tuple[str, dict[str, bool]]:
    """Returns (rendered user message, field presence map) -- the second
    element is 86bbwachy Phase 4's own new addition. Computed directly from
    MacroSourcesBundle's own per-series `*_age_days`/`bond_yields_available`
    fields (confirmed real, already-exposed signals -- compute_macro_sources()
    sets a series' own age field to None on a fetch failure, e.g.
    _fetch_statcan_fields()'s own try/except), not by wrapping each block
    helper in a RenderedField: these functions build a MULTI-LINE block from
    several related series at once (US + CA figures together), and every
    series within one block fails together (one API call, one age field) --
    tracking at the block level is both accurate and matches the prompt's
    own section headers (RATE/YIELD/CPI/...).
    """
    ctx = bundle.context
    company_info = bundle.company_info
    m = bundle.macro_sources
    is_ca = bundle.canadian_data_flags is not None

    statcan_section = f"\n\nSTATISTICS CANADA (CA demand indicators):\n{_statcan_block(m)}" if is_ca else ""

    text = f"""{bundle.stock.ticker} ({company_info.get('name')}) | {company_info.get('sector')} | {bundle.stock.exchange} | {bundle.stock.currency}
Timeline: {ctx.timeline} | Account: {ctx.account_type} | As of: {bundle.data_vintage.isoformat()}

INTEREST RATES (RATE):
{_rate_block(m, is_ca)}

YIELD CURVE (YIELD):
{_yield_block(m, is_ca)}

INFLATION (CPI):
{_cpi_block(m, is_ca)}

ECONOMIC GROWTH (GDP):
{_gdp_block(m, is_ca)}

EMPLOYMENT (EMPLOY):
{_employ_block(m, is_ca)}

CURRENCY (FX):
{_fx_block(m, bundle.stock.currency, bundle.stock.exchange)}

VOLATILITY (VIX):
{_vix_block(m)}

COMMODITIES (COMMOD):
{_commod_block(m)}{statcan_section}

SECTOR CONTEXT:
  Company sector: {company_info.get('sector')}

REMINDER: Your narrative must be 80-120 words (480-720 chars). This is strictly enforced."""

    field_presence = {
        "rate": m.policy_rate_age_days is not None,
        "yield_curve": m.bond_yields_available,
        "cpi": m.cpi_age_days is not None,
        "gdp": m.gdp_age_days is not None,
        "employment": m.unemployment_age_days is not None,
        "fx": m.cad_usd_age_days is not None,
        "vix": m.vix_age_days is not None,
        "commodities": m.sector_commodity_age_days is not None,
    }
    # Only present in the map at all for a CA stock -- the whole block
    # itself only renders then (statcan_section, above); a US stock never
    # had this data attempted, not "attempted and absent."
    if is_ca:
        field_presence["statcan"] = m.statcan_age_days is not None

    return text, field_presence


def _validate_with_caveats(
    output: dict,
    material_absent: list[str],
    anomalies: list[str],
    stale_data: list[str],
) -> tuple[bool, list[str]]:
    """Composing validator (86bbummwp follow-on) -- this agent had no
    caveat-specific validator to compose with `validate_macro_economist`
    before now (unlike Technical/Sentiment/Stock Researcher), so this
    wrapper is new here, not extended, but follows the identical pattern:
    the base schema check plus the new confidence/data-quality coupling
    rule, the backstop for the real, live case this whole follow-on plan
    started from -- a real Macro Economist run claiming
    `analysis_confidence: "high"` while its own stale_data flag showed 4
    genuinely stale series. See
    `validate_confidence_requires_caveat_when_flagged`'s own docstring."""
    passed, errors = validate_macro_economist(output)
    cq_passed, cq_errors = validate_confidence_requires_caveat_when_flagged(
        output,
        is_high=output.get("analysis_confidence") == "high",
        material_absent=material_absent,
        anomalies=anomalies,
        stale_data=stale_data,
    )
    return passed and cq_passed, errors + cq_errors


class MacroEconomistRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "MACRO"
        ctx = bundle.context
        user_msg, field_presence = build_user_message(bundle)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        is_commodity_relevant = bundle.macro_sources.sector_commodity_relevant
        # 86bbummwp Tier 2 -- D6's structured data_coverage flag, from the same
        # coverage_presence() filtering the prose line below already applies.
        self.last_data_coverage = to_data_coverage(
            _coverage_presence(field_presence, is_commodity_relevant), _COVERAGE_GAP_SENTENCES
        )
        self.last_stale_data = _stale_data(bundle)
        self.last_anomalies = _anomalies(bundle)
        system_prompt = fill(
            load_template("macro_economist"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": _data_coverage_line(field_presence, is_commodity_relevant),
                "data_warnings": render_data_warnings(self.last_anomalies, self.last_stale_data),
                "memory_brief": "",
                "accuracy_brief": "",
                "currency_exposure_hint": "",
                "sector_macro_hint": "",
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            partial(
                _validate_with_caveats,
                material_absent=self.last_data_coverage["absent"],
                anomalies=self.last_anomalies,
                stale_data=self.last_stale_data,
            ),
            max_tokens=3000,
            temperature=0.3,
        )
