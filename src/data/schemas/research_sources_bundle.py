"""ResearchSourcesBundle contract model (ClickUp 86bawp83r).

Full field spec: docs/technical/data-pipeline.md §3 (precompute/
research_sources.py contract), §4 (ResearchSourcesBundle). company_info
stays a top-level DataBundle field (used by multiple agents), not part of
this bundle. analyst_reports_summary intentionally excluded — no FMP
endpoint produces summarized analyst reports at the current provider tier;
re-add only when a provider method for get_analyst_reports() exists.
"""

from typing import Literal

from pydantic import NonNegativeInt, model_validator

from data.schemas.base import ContractModel
from data.schemas.common import FilingDigest, NewsItem, PeerBlock, TranscriptExcerpt


class ManagementSignals(ContractModel):
    """Structured management-activity signals. Single-use by
    ResearchSourcesBundle only (not a cross-bundle shared type like
    data.schemas.common's submodels), so it's colocated here rather than
    in common.py.

    insider_net_direction_90d is Optional, deviating from the ticket's
    literal required-Literal spec: a 3-value enum with no None conflates
    "insiders traded net-neutral" (a real, resolved reading) with "we
    couldn't determine this" (e.g. edgartools fetch failed) under the same
    "neutral" value — the same class of ambiguity fixed in
    CanadianDataFlags/MacroSourcesBundle by keeping "unknown" and "known,
    resolved to a specific value" distinguishable. buyback_activity/
    dividend_activity/changes_detail stay required str — an empty string
    is already a natural, unambiguous "nothing to report" for free text,
    unlike a closed enum."""

    c_suite_changes_12mo: bool
    changes_detail: str
    insider_net_direction_90d: Literal["buying", "neutral", "selling"] | None
    buyback_activity: str
    dividend_activity: str


class ResearchSourcesBundle(ContractModel):
    """Full input contract for the Stock Researcher agent. Replaces the old
    flat filings_summary/earnings_transcript_summary/news_articles/
    analyst_reports_summary/peer_company_data dict fields with one typed
    object, populated per precompute/research_sources.py (not yet built).

    Type definition only — the model_validators below are data-validity
    invariants (a count that must match its own list), not business logic,
    same reasoning as CanadianDataFlags (86bawp80f) and MacroSourcesBundle
    (86bawp819).

    latest_filing_age_days/latest_transcript_age_days/latest_news_age_days
    are Optional, deviating from the ticket's literal required-int spec —
    consistent with the graceful-degradation decision made for
    MacroSourcesBundle (2026-08-05): an age-of-the-latest-X is meaningless
    when zero X exist (e.g. a recent IPO with no filing history, or a
    company with no earnings-call transcripts yet), and forcing a non-null
    age would make it impossible to construct a valid bundle for exactly
    that real, plausible scenario.
    """

    filing_digests: list[FilingDigest]
    transcript_excerpts: list[TranscriptExcerpt]
    news_items: list[NewsItem]
    peer_blocks: list[PeerBlock]
    management_signals: ManagementSignals

    dual_class_flag: bool
    cik_verified: bool
    sedar_filing_available: bool

    # Reliability counters for compute_base_reliability_researcher()
    latest_filing_age_days: NonNegativeInt | None
    latest_transcript_age_days: NonNegativeInt | None
    latest_news_age_days: NonNegativeInt | None
    transcript_count: NonNegativeInt
    news_item_count: NonNegativeInt

    @model_validator(mode="after")
    def _check_counts_match_their_lists(self) -> "ResearchSourcesBundle":
        if self.transcript_count != len(self.transcript_excerpts):
            raise ValueError(
                f"transcript_count ({self.transcript_count}) must match "
                f"len(transcript_excerpts) ({len(self.transcript_excerpts)})"
            )
        if self.news_item_count != len(self.news_items):
            raise ValueError(
                f"news_item_count ({self.news_item_count}) must match "
                f"len(news_items) ({len(self.news_items)})"
            )
        return self

    @model_validator(mode="after")
    def _check_age_nullability_matches_list_emptiness(self) -> "ResearchSourcesBundle":
        """Real gap caught on review: the docstring claimed "age-of-the-
        latest-X is meaningless when zero X exist," but nothing enforced
        it — filing_digests=[] paired with latest_filing_age_days=999 (or
        the reverse: real items with a None age) would have constructed
        silently. Age is None if and only if the corresponding list/count
        is zero, for all three list/age pairs."""
        pairs = (
            (
                "filing_digests",
                len(self.filing_digests),
                "latest_filing_age_days",
                self.latest_filing_age_days,
            ),
            (
                "transcript_excerpts",
                self.transcript_count,
                "latest_transcript_age_days",
                self.latest_transcript_age_days,
            ),
            ("news_items", self.news_item_count, "latest_news_age_days", self.latest_news_age_days),
        )
        for list_name, count, age_name, age in pairs:
            if count == 0 and age is not None:
                raise ValueError(f"{age_name} must be None when {list_name} is empty")
            if count > 0 and age is None:
                raise ValueError(f"{age_name} must not be None when {list_name} is non-empty")
        return self
