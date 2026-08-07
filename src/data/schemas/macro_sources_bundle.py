"""MacroSourcesBundle contract model (ClickUp 86bawp819).

Full field spec: docs/technical/data-pipeline.md §3 (precompute/macro_sources.py
contract), §4 (MacroSourcesBundle). Removed fields — don't reintroduce:
geopolitical_news_items (G{num} pipeline is dead infrastructure, removed in
Macro v1.2).
"""

from typing import Literal

from pydantic import NonNegativeInt, model_validator

from data.schemas.base import ContractModel
from data.schemas.common import CBCommentaryItem


class MacroSourcesBundle(ContractModel):
    """Full input contract for the Macro Economist agent. Replaces the old
    flat macro_us/macro_canada/commodity_prices/currency_data/sector_macro
    dict fields with one typed object, populated per precompute/
    macro_sources.py (not yet built).

    Type definition only — trend classification happens in the precompute
    module before this is constructed, not here. The two model_validators
    below are basic data-validity invariants (a count that must match its
    own list, conditional-null fields that must actually be null when the
    doc says they should be), not business/scoring logic, so they stay in
    scope per the same reasoning applied in CanadianDataFlags (86bawp80f).

    Design decision (2026-08-05, confirmed with user, deviates from the
    ticket's literal spec — every field below was originally required
    float/Literal): the Macro Economist should degrade gracefully, not
    fail the whole run, when a single FRED/BoC series fails to resolve.
    All independently-fetchable series, the deltas/trends computed from
    them, and their corresponding age fields are Optional so precompute/
    macro_sources.py can construct a partial bundle instead of being
    forced to retry-until-success or fabricate a value. Deliberately NOT
    adding cross-field validators pairing each value with its age/trend
    (e.g. "vix is None implies vix_regime must be None") — the exact
    value-to-age-to-trend mapping isn't fully unambiguous from the doc
    (e.g. does cad_usd_age_days track cad_usd, cad_usd_fred, or both?),
    and inventing that mapping here would be guessing at precompute's own
    construction logic, not enforcing a real type-level invariant.
    """

    # --- FRED series ---
    fed_funds_rate: float | None
    treasury_2y: float | None
    treasury_5y: float | None
    treasury_10y: float | None
    cpi: float | None
    core_cpi: float | None
    gdp: float | None
    unemployment: float | None
    vix: float | None
    cad_usd_fred: float | None
    wti_crude: float | None

    # --- BoC Valet series ---
    boc_rate: float | None
    cad_usd: float | None
    canada_bond_2y: float | None
    canada_bond_5y: float | None
    canada_bond_10y: float | None
    canada_cpi: float | None

    # --- Orchestrator-computed deltas ---
    # Optional because each is computed from one or more of the series
    # above — can't compute a delta/trend from a series that failed to
    # resolve.
    policy_rate_90d_delta_bp: float | None
    cpi_3m_delta_pp: float | None
    cad_usd_90d_change_pct: float | None
    commodity_90d_change_pct: float | None  # None when sector_commodity_relevant is False
    unemployment_6m_delta: float | None

    # --- Trend classification enums (assigned pre-LLM) ---
    rate_trend: Literal["tightening", "pausing", "easing"] | None
    cpi_trend: Literal["rising", "stable", "falling"] | None
    cad_trend: Literal["cad_strengthening", "stable", "cad_weakening"] | None
    vix_regime: Literal["low", "elevated", "high"] | None

    # --- Sector-commodity relevance ---
    sector_commodity_relevant: bool
    sector_commodity_name: str | None
    sector_commodity_level: float | None
    sector_commodity_direction: str | None  # doc gives no closed enum for this one
    sector_commodity_age_days: NonNegativeInt | None

    # --- Availability flags ---
    bond_yields_available: bool
    usd_revenue_exposure_pct: float | None

    # --- Central bank commentary (pre-summarized) ---
    cb_commentary_items: list[CBCommentaryItem]
    cb_commentary_count: NonNegativeInt

    # --- Series age fields for reliability scoring ---
    # Note: sector_commodity_age_days_reliability is a DIFFERENT field from
    # sector_commodity_age_days above, despite the near-identical name — the
    # doc lists both separately (one under "sector-commodity relevance," one
    # under "series age fields for reliability scoring"). Kept distinct per
    # the doc rather than assumed to be a duplicate/typo.
    # Optional alongside their corresponding series above — an age in days
    # only means something for a value that actually resolved.
    policy_rate_age_days: NonNegativeInt | None
    cpi_age_days: NonNegativeInt | None
    gdp_age_days: NonNegativeInt | None
    unemployment_age_days: NonNegativeInt | None
    cad_usd_age_days: NonNegativeInt | None
    vix_age_days: NonNegativeInt | None
    sector_commodity_age_days_reliability: NonNegativeInt | None

    # --- Statistics Canada supplementary fields ---
    # All Optional — None if fetch fails or the stock isn't Canadian.
    statcan_unemployment_ca: float | None  # LFS overall unemployment rate
    statcan_housing_starts: float | None  # Annualized housing starts (SAAR)
    statcan_retail_sales_yoy: float | None  # Retail sales YoY %
    statcan_cpi_by_province: dict[str, float] | None
    statcan_age_days: NonNegativeInt | None

    @model_validator(mode="after")
    def _check_sector_commodity_nulls(self) -> "MacroSourcesBundle":
        """Ticket's own explicit requirement: sector_commodity_relevant=False
        implies name/level/direction/age_days are all None. Extended to also
        cover commodity_90d_change_pct — not explicitly named in the
        ticket's validator instruction, but the doc's own inline comment on
        that field ("None when sector_commodity_relevant is False") ties it
        to the same rule; treating it as an oversight in the ticket's list
        rather than a deliberate exclusion."""
        if not self.sector_commodity_relevant:
            conditional_fields = {
                "sector_commodity_name": self.sector_commodity_name,
                "sector_commodity_level": self.sector_commodity_level,
                "sector_commodity_direction": self.sector_commodity_direction,
                "sector_commodity_age_days": self.sector_commodity_age_days,
                "commodity_90d_change_pct": self.commodity_90d_change_pct,
            }
            non_null = [name for name, value in conditional_fields.items() if value is not None]
            if non_null:
                raise ValueError(
                    f"sector_commodity_relevant=False requires these fields to be "
                    f"None, but got real values: {', '.join(non_null)}"
                )
        return self

    @model_validator(mode="after")
    def _check_cb_commentary_count_matches(self) -> "MacroSourcesBundle":
        if self.cb_commentary_count != len(self.cb_commentary_items):
            raise ValueError(
                f"cb_commentary_count ({self.cb_commentary_count}) must match "
                f"len(cb_commentary_items) ({len(self.cb_commentary_items)})"
            )
        return self
