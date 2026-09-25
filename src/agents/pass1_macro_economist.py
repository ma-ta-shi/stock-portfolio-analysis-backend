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
"""
from agents.base import BaseRunner
from agents.prompts import fill, load_template
from agents.validators.pass1 import validate_macro_economist
from data.schemas.data_bundle import DataBundle
from data.schemas.macro_sources_bundle import MacroSourcesBundle


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


class MacroEconomistRunner(BaseRunner):
    async def run(self, bundle: DataBundle) -> tuple[dict, list[str]]:
        self.current_agent = "MACRO"
        ctx = bundle.context
        user_msg, field_presence = build_user_message(bundle)
        # 86bbwachy Phase 4 -- set before the LLM call is attempted, so a
        # failed call still records whether its own input was already
        # incomplete.
        self.last_field_coverage = field_presence
        system_prompt = fill(
            load_template("macro_economist"),
            {
                "canonical_ticker": bundle.stock.ticker,
                "company_name": bundle.company_info.get("name"),
                "sector": bundle.company_info.get("sector"),
                "timeline": ctx.timeline,
                "timeline_instruction": f"Timeline: {ctx.timeline}.",
                "data_coverage_line": "Data coverage: standard.",
                "data_warnings": "",
                "memory_brief": "",
                "accuracy_brief": "",
                "currency_exposure_hint": "",
                "sector_macro_hint": "",
            },
        )
        return await self.call_with_validation(
            system_prompt,
            user_msg,
            validate_macro_economist,
            max_tokens=3000,
            temperature=0.3,
        )
