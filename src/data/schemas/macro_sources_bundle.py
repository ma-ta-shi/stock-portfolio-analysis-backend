"""MacroSourcesBundle contract model (ClickUp 86bawp819).

Full field spec: docs/technical/data-pipeline.md §3 (precompute/macro_sources.py
contract), §4 (MacroSourcesBundle). Removed fields — don't reintroduce:
geopolitical_news_items (G{num} pipeline is dead infrastructure, removed in
Macro v1.2).

Amended 2026-08-07 (contract-vs-consumer audit, see docs/decision-log.md):
data-pipeline.md's own dataclass sketch was cross-checked only against
itself when this model was first built, never against the actual live
Macro Economist Agent Prompt.md that consumes it. Three real gaps found
and closed here:
- `us_curve_shape`/`ca_curve_shape` — the doc never mentioned yield-curve-
  shape classification at all, but the prompt's payload template expects
  it for both jurisdictions (`{us_curve_shape}`/`{ca_curve_shape}`). Pure
  computation from treasury_2y/10y and canada_bond_2y/10y, no new external
  dependency.
- `cb_stance_note` — the doc's `cb_commentary_items`/`cb_commentary_count`
  fields (kept below, still used for raw grounding) reflect an OLDER
  design the live prompt explicitly says it replaced in v1.2: "`CB_COMM`
  as a separate citation token and payload block [removed]. Material
  central bank stance is rolled into the RATE block as an optional one-
  liner" (prompt line 13). The prompt reads `{cb_stance_note}`, a single
  optional string, not the list.
Resolved 2026-08-15 (ClickUp 86bbahum6): `ca_cpi_trend`/`ca_cpi_3m_delta`/
`ca_cpi_yoy`, `ca_gdp_qoq`/`ca_gdp_4q_trend` were deliberately NOT added at
first pass, pending a real data source (confirmed live 2026-08-07: both
FRED's OECD-mirrored Canada CPI [CPALTT01CAM657N, stale since 2024-02] and
Canada GDP [CANRGDPR, stale since 2011] series are dead infrastructure,
not usable). A working source was found — Statistics Canada's own Web
Data Service (table 18100004 for CPI, table 36100105 for GDP), reachable
through the same StatsCanadaProvider integration already used for the
other statcan_* fields below. All 5 fields are now populated from that
source instead; the dead FRED calls are removed, not just unused.
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
    module before this is constructed, not here. The model_validators
    below are basic data-validity invariants (a count that must match its
    own list, conditional-null fields that must actually be null when the
    doc says they should be, a derived field requiring its own source
    data), not business/scoring logic, so they stay in scope per the same
    reasoning applied in CanadianDataFlags (86bawp80f).

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

    # --- Yield curve shape per jurisdiction (added 2026-08-07 — real gap
    # vs. the live Macro Economist prompt, missing from data-pipeline.md
    # entirely) ---
    us_curve_shape: Literal["normal", "flat", "inverted"] | None
    ca_curve_shape: Literal["normal", "flat", "inverted"] | None

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
    # Added 2026-08-07: the actual field the live Macro Economist prompt
    # reads (`{cb_stance_note}`) — a single optional one-liner derived
    # from cb_commentary_items, not the raw list itself. cb_commentary_items
    # is kept for raw grounding/citation, not dead weight.
    cb_stance_note: str | None

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
    # statcan_housing_starts / statcan_retail_sales_yoy are computed and
    # provisioned but have NO payload placeholder yet — kept (cheap, real CA
    # demand indicators) pending a Macro prompt revision that adds a block +
    # CoT guidance. Decision owner: prompt-revision-protocol.md §30.9-30.11.
    # (statcan_cpi_by_province was dropped in 86bbq8rj1 — see stats_canada.py.)
    statcan_housing_starts: float | None  # Annualized housing starts (SAAR)
    statcan_retail_sales_yoy: float | None  # Retail sales YoY %
    statcan_age_days: NonNegativeInt | None

    # --- Statistics Canada CPI/GDP trend fields (86bbahum6) ---
    # Same "independently-optional, no age/value pairing enforced" pattern
    # as the rest of this block — see class docstring. No new cross-field
    # validator: ca_cpi_trend/ca_gdp_4q_trend are single-source pure
    # derivations (one classifier function, one input each), exactly the
    # same shape as cpi_trend/rate_trend above, which are deliberately
    # left unvalidated for the same reason (the classifier's own
    # if-None-return-None branch already guarantees the invariant).
    ca_cpi_yoy: float | None
    ca_cpi_3m_delta: float | None
    ca_cpi_trend: Literal["rising", "stable", "falling"] | None
    ca_gdp_qoq: float | None  # annualized
    ca_gdp_4q_trend: Literal["rising", "stable", "falling"] | None

    # --- US CPI/GDP/VIX + BoC rate delta (86bbq8rj1) ---
    # One block, grouped by ticket rather than by section semantics —
    # same convention as the 86bbahum6 block above. US mirrors of the
    # ca_* CPI/GDP fields, the VIX 30-day average, the BoC rate 90-day
    # delta + direction (paired with the Fed policy_rate_90d_delta_bp/
    # rate_trend), and the CA unemployment 6-month delta. No new
    # validators — single-source pure derivations, like ca_cpi_trend.
    # us_cpi_yoy / us_core_cpi_yoy / us_gdp_qoq are read by the prompt's
    # merge_output_macro; us_gdp_4q_trend / vix_30d_avg / the BoC pair are
    # payload-template placeholders. us_gdp_qoq / us_gdp_4q_trend come
    # from FRED GDPC1 (real GDP), not the nominal `gdp` series, to stay
    # methodologically identical to ca_gdp_qoq.
    us_cpi_yoy: float | None
    us_core_cpi_yoy: float | None
    us_gdp_qoq: float | None  # annualized, from real GDP
    us_gdp_4q_trend: Literal["rising", "stable", "falling"] | None
    vix_30d_avg: float | None
    boc_rate_90d_delta_bp: float | None
    boc_rate_trend: Literal["tightening", "pausing", "easing"] | None
    ca_unemployment_6m_delta: float | None  # pp change vs 6 months ago

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

    @model_validator(mode="after")
    def _check_cb_stance_note_requires_commentary_items(self) -> "MacroSourcesBundle":
        """One-directional: cb_stance_note is derived FROM cb_commentary_items
        (see class docstring), so an empty list can't have produced a real
        stance note. The reverse isn't constrained — having items doesn't
        guarantee one was material enough to produce a note."""
        if not self.cb_commentary_items and self.cb_stance_note is not None:
            raise ValueError("cb_stance_note must be None when cb_commentary_items is empty")
        return self

    @model_validator(mode="after")
    def _check_curve_shape_requires_both_yield_points(self) -> "MacroSourcesBundle":
        """Real gap caught on review: us_curve_shape/ca_curve_shape were
        added without enforcing the invariant their own existence implies
        — a curve shape can only be classified when BOTH yield-curve
        endpoints resolved, and if both did resolve, classification is
        deterministic (no missing-data path), so it must not be left
        unresolved. Bidirectional, unlike this class's other value/age
        pairings, precisely because there's no ambiguity here: the mapping
        is exactly two named fields, not a guess at precompute's internal
        construction logic."""
        if (self.treasury_2y is None or self.treasury_10y is None) != (self.us_curve_shape is None):
            raise ValueError(
                "us_curve_shape must be set if and only if both treasury_2y "
                "and treasury_10y resolved"
            )
        if (self.canada_bond_2y is None or self.canada_bond_10y is None) != (
            self.ca_curve_shape is None
        ):
            raise ValueError(
                "ca_curve_shape must be set if and only if both canada_bond_2y "
                "and canada_bond_10y resolved"
            )
        return self
