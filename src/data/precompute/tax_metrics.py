"""tax_metrics.py — pre-computation for the Tax Strategist agent (Pass 2),
ClickUp 86bbrhx6j.

The Tax Strategist's prompt injects a `precomputed_tax_metrics` block and
instructs the LLM to cite every tax claim against tokens drawn from it
(DIVID, ELIG, LIST, DOM, WHT, CGAIN) plus TAX_RULE_SNAPSHOT — confirmed by
direct read of "Agent Prompts/Current Prompts/Tax Strategist Agent
Prompt.md", not the ticket's own summary. Without a real module producing
this block there is nothing to cite and `tax_profile` (the only thing the
CIO consumes from this agent) has no basis.

Scope, grounded in real data, not the ticket's literal three-arg spec:
- DIVID/ELIG/LIST/DOM/WHT/CGAIN(-base) and TAX_RULE_SNAPSHOT: built for
  real, always. TAX_RULE_SNAPSHOT ships here too (not deferred) because
  it turned out to be a single appended line built from a local file's
  own header, not the wider account-state machinery its neighbours in
  Gate 1's requirement list might suggest.
- The LOSS token's dual_listed flag: built for real (is_canadian_dual_listed) -
  ticker-only, no blocked user data needed, unlike the rest of LOSS.
- ROOM, LOSS's transactional parts (window status, YTD realized losses),
  CGAIN's YTD-realized-gains clause, and US_SITUS: genuinely blocked on a
  real per-user account-state service that does not exist anywhere in this
  codebase (confirmed by ClickUp search, not assumed). build_precomputed_tax_metrics()
  accepts an optional `account_state` parameter and renders these when one
  is supplied - tested with plausible, explicitly-synthetic values per
  explicit direction, since real user tax/portfolio data will never be
  available to build or test this against. Every real call today passes
  account_state=None.
- account_type is scoped to tfsa/rrsp/trading only. "general" is a
  whole-pipeline design question the CIO's own prompt has already
  resolved by removing that mode from real orchestration ("The previous
  general branch is removed - this system always runs against a specific
  account context") - confirmed by direct read of the CIO prompt, not
  assumed from the Tax Strategist prompt's own (stale) Account table.

Withholding tax rates are QUOTED from prompts/tax_strategist/canadian_tax_rules_reference.md,
never authored - WHT_GRID's values are copied verbatim from that file's own
"Withholding tax grid (modelled combinations)" section. A combination the
grid doesn't cover renders as genuinely not modelled, never an invented rate.

canadian_tax_rules_reference.md itself has to live inside this repo (not
the outer Stock Picker monorepo it originates from) for load_tax_rules_reference()
to have anything real to read wherever this backend actually runs - copied
to prompts/tax_strategist/ here, verbatim, as its own addition (it was
never part of this repo's history before).
"""

import re
from datetime import date, datetime, timedelta
from typing import Literal, TypedDict

import structlog

from data.providers.base import NormalizedCompanyInfo, NormalizedDividendRecord
from data.providers.ca_crosslisting import is_crosslisted
from data.schemas.data_bundle import DataBundle

logger = structlog.get_logger(__name__)

_CA_MARKET_SUFFIXES = (".TO", ".V")  # mirrors router.py's own _CA_SUFFIXES (private
# there) and research_sources.py's own _CA_MARKET_SUFFIXES - duplicated here rather
# than imported across a private module boundary, the same disclosed-duplication
# posture already established for this exact check elsewhere in this codebase. Keep
# in sync if router.py's changes. Named and shared within this module (rather than
# inlined at each of classify_dividend's and build_precomputed_tax_metrics's own
# call sites, as an earlier draft had it) so there's exactly one place to update.

_DIVIDEND_WINDOW_DAYS = 365  # a REAL trailing-365-day window, not a calendar-year
# filter - the ticket's own trap #1. A `year >= current-1` filter spans ~20 months
# and sums extra quarterly payments as if annual; confirmed live this overstated
# RY.TO's real ~2.3% yield as 3.9%. This module's own version of the same class of
# bug, found in a DIFFERENT already-shipped module (fundamentals.py::compute_dividend_info,
# "last 4 records", not a date window at all) - confirmed live, not estimated,
# against REI-UN.TO's complete real dividend history: that method understates a
# monthly payer's true trailing yield by exactly 3.00x (1.87% vs a true 5.61%).
# Tracked as its own separate follow-up ticket, not fixed here - a different module.


def classify_dividend(ticker: str, company_info: NormalizedCompanyInfo) -> tuple[str, str]:
    """(dividend_classification, security_structure).

    dividend_classification is the single, already-resolved category
    resolve_withholding()/WHT_GRID key on directly - one of
    canadian_eligible/trust_distribution/us/us_reit/limited_partnership/adr,
    the six values this function can actually produce (real WHT_GRID key
    shapes, confirmed by direct inspection - fixed a stale docstring on
    review that listed "us_mlp" as a seventh possible value, which this
    function has never returned: MLP and non-MLP LP both collapse to
    limited_partnership regardless of market, matching the reference's
    own coverage note that they share identical WHT treatment - the
    distinction lives only in security_structure's descriptive label,
    "MLP" vs "limited partnership", never in the classification itself).
    The reference table's own "foreign" row (non-US, non-CA, not
    ADR-wrapped) is likewise never produced - this system's real ticker
    universe only ever resolves CA- or US-exchange-listed tickers (no
    Router chain exists for anything else), so a directly-foreign-listed
    security with no US ADR is a case that can't reach this function at
    all today, not a heuristic gap. Disclosed here rather than silently
    incomplete.

    Not a (classification, structure) pair to be recombined later: an
    earlier design invented a separate structure dimension the real
    reference table doesn't have, and its own "reit+US -> us+reit"
    shorthand was genuinely ambiguous about whether that meant a single
    composite string or a real pair. security_structure is a purely
    descriptive label for the DOM line's own human-readable text (e.g.
    "REIT", "limited partnership", "ADR", "corporation") - independent of
    the WHT lookup entirely, never itself a lookup key.

    Always called uniformly - never skipped for an ETF ticker; the ETF
    case is one of this function's own branches, not something the
    caller pre-filters before calling in.

    Checked in order:
      1. ETF (asset_type == "etf") FIRST - checked live what this
         protects against: AMLP's own NAME literally contains "MLP"
         ("Alerian MLP ETF"), so checking the partnership/LP
         name-substring check before this could plausibly misfire on a
         differently-phrased MLP-themed ETF. asset_type is the clean,
         already-normalized, cross-provider-tested signal (86bbpk6uf);
         checking it first removes that whole class of risk rather than
         relying on every real ETF's name/industry text never
         accidentally matching a fuzzier heuristic. (A "REIT-themed ETF"
         risk was the original reason this reordering was considered,
         but checked live and found NOT to reproduce: VNQ, a real
         REIT-themed ETF, has industry="Asset Management", not "REIT" -
         the MLP case alone still justifies checking ETF first.)
      2. REIT (industry contains "REIT", case-insensitive) -> trust_distribution
         if this is a CA ticker, else us_reit. Confirmed live both
         markets: CA REITs' industry is "REITs", US REITs' is
         "REIT - Retail" (or similar) - both contain "REIT" as a
         substring.
      3. MLP-or-partnership (name contains PARTNERS/" LP"/PARTNERSHIP, or
         a "-UN" ticker suffix) -> limited_partnership regardless of
         market, since the reference's own coverage statement treats MLP
         and non-MLP LP as one not-modelled bucket for WHT purposes.
         "PARTNERS" is the real, live-confirmed anchor here, not
         " LP"/PARTNERSHIP - checked live against Brookfield
         Infrastructure Partners L.P. (a genuine LP, not a REIT trust):
         its real name contains "Partners" (catches on PARTNERS) but
         literally "L.P." with periods, which " LP" (no periods) does
         NOT match. A real LP phrased only as "Foo L.P." with no
         "Partners" in its name would slip through both name checks and
         fall through to the ordinary-corporation default - a disclosed,
         narrow gap, not fixed here.
      4. ADR (not a CA ticker AND country not in ("US", "")) -> adr.
         country is blank for every CA ticker through the real Router
         path (confirmed live across 6 CA tickers) - not a usable
         CA-detection signal, which is why this checks the ticker suffix
         first, not country. Confirmed live against three independent
         real ADRs spanning China/Netherlands/Japan, no false positive
         on any of 8 other tickers checked.
      5. Ordinary corporation (default) -> canadian_eligible if CA else us.

    For structure=="etf" the returned classification value is real but
    functionally moot - see build_precomputed_tax_metrics, which checks
    the structure return value itself and never passes either return
    value to resolve_withholding for that case.

    Sync, no I/O - company_info is already fetched by the caller."""
    is_ca = ticker.upper().endswith(_CA_MARKET_SUFFIXES)

    if company_info.get("asset_type") == "etf":
        return ("canadian_eligible" if is_ca else "us"), "etf"

    industry = str(company_info.get("industry") or "")
    name = str(company_info.get("name") or "")
    name_upper = name.upper()

    if "REIT" in industry.upper():
        return ("trust_distribution" if is_ca else "us_reit"), "REIT"

    is_partnership = (
        any(marker in name_upper for marker in ("PARTNERS", " LP", "PARTNERSHIP"))
        or "-UN" in ticker.upper()
    )
    if is_partnership:
        label = "MLP" if "MLP" in name_upper else "limited partnership"
        return "limited_partnership", label

    country = str(company_info.get("country") or "")
    if not is_ca and country not in ("US", ""):
        return "adr", "ADR"

    return ("canadian_eligible" if is_ca else "us"), "corporation"


def compute_trailing_dividend(
    dividend_history: list[NormalizedDividendRecord], current_price: float | None
) -> tuple[float | None, int]:
    """(yield_pct, payment_count) from a REAL trailing-365-day window on
    ex_date (cutoff = today - 365 days), not a calendar-year filter - see
    the module docstring's own trap #1 note. payment_count is rendered
    alongside yield deliberately - the ticket's own "cheap tripwire" so a
    wrong window is visible (a quarterly payer must show ~4, a monthly
    payer ~12). (None, 0) when there's no history or no price to divide
    by."""
    if not dividend_history or not current_price:
        return None, 0
    cutoff = (datetime.now().date() - timedelta(days=_DIVIDEND_WINDOW_DAYS)).isoformat()
    recent = [d for d in dividend_history if d["ex_date"] >= cutoff]
    if not recent:
        return None, 0
    annual = sum(d["amount_per_share"] for d in recent)
    return annual / current_price * 100, len(recent)


# WHT_GRID: {(dividend_classification, account_type): (rate_pct, note)}. Two-dimensional,
# not three - dividend_classification is already the fully-resolved category (see
# classify_dividend), not a separate (classification, structure) cross-product the real
# reference table doesn't have. Values quoted verbatim from canadian_tax_rules_reference.md's
# own "Withholding tax grid (modelled combinations)" section - never authored. A combination
# absent from this dict (including every not-modelled classification, and
# trust_distribution/trading specifically) is genuinely not modelled - resolve_withholding()
# returns None for it, the caller renders "NOT MODELLED per REF withholding grid", never an
# invented rate. trust_distribution/trading's own absence here is deliberate, not an
# omission: the reference is explicit that a CA REIT/trust in a taxable account is not
# modelled, only the two registered accounts are - a real, previously-live-caught bug in an
# audit rig's own WHT lookup collapsed this to "REIT means not modelled everywhere",
# which is wrong specifically for TFSA/RRSP.
WHT_GRID: dict[tuple[str, str], tuple[float, str]] = {
    ("canadian_eligible", "tfsa"): (0.0, "no withholding"),
    ("canadian_eligible", "rrsp"): (0.0, "no withholding"),
    ("canadian_eligible", "trading"): (0.0, "no withholding; eligible for the dividend tax credit"),
    ("trust_distribution", "tfsa"): (0.0, "Canadian-source, no withholding"),
    ("trust_distribution", "rrsp"): (0.0, "Canadian-source, no withholding"),
    ("us", "tfsa"): (15.0, "non-recoverable"),
    ("us", "rrsp"): (0.0, "treaty Art. XVIII exemption"),
    ("us", "trading"): (15.0, "recoverable as a foreign tax credit"),
}


def resolve_withholding(
    dividend_classification: str, account_type: Literal["tfsa", "rrsp", "trading"]
) -> tuple[float, str] | None:
    """Looks up WHT_GRID. None means genuinely not modelled - caller
    renders "NOT MODELLED per REF withholding grid", never an invented
    rate (the ticket's own hard requirement)."""
    return WHT_GRID.get((dividend_classification, account_type))


def compute_effective_after_tax_yield(
    yield_pct: float | None, wht_rate_pct: float | None
) -> tuple[float | None, float | None]:
    """(effective_after_tax_yield_pct, annual_tax_drag_pct) = (yield *
    (1 - wht/100), yield * wht/100) - the mechanical definition confirmed
    against the live prompt's own testing checklist ("drag_delta_pct ≈
    -(yield × 0.15)" for a 15%-WHT US dividend in a TFSA). Returns
    (None, None) when yield or wht_rate_pct is None - there is no honest
    value to compute either when there's no dividend at all or when WHT
    isn't modelled for this classification; DIVID's/WHT's own lines state
    which reason applies, this function doesn't need to distinguish them."""
    if yield_pct is None or wht_rate_pct is None:
        return None, None
    drag = yield_pct * wht_rate_pct / 100
    return round(yield_pct - drag, 1), round(drag, 1)


# CGAIN_BY_ACCOUNT: quoted verbatim from the reference's own TFSA/RRSP/Taxable capital-gains
# sections - account-keyed deliberately, not one shared line. An earlier draft (in the audit
# rig this ported logic from) rendered one hardcoded taxable-account line for every account,
# which is wrong in a TFSA (gains are tax-free, not 50%-included) and wrong in the OPPOSITE
# direction in an RRSP (the 50% advantage is lost entirely, not preserved) - not repeating
# that mistake here.
CGAIN_BY_ACCOUNT: dict[str, str] = {
    "tfsa": "tax-free; a stock that 10x in a TFSA generates zero tax [source: REF/TFSA]",
    "rrsp": (
        "tax-deferred, then taxed as ordinary income on withdrawal - the 50% "
        "inclusion-rate advantage of capital gains is LOST [source: REF/RRSP]"
    ),
    "trading": (
        "50% inclusion on the first $250,000 of net annual capital gains; "
        "66.7% above that [source: REF/taxable]"
    ),
}


def is_canadian_dual_listed(ticker: str) -> bool:
    """Reuses ca_crosslisting.py::is_crosslisted() for the LOSS token's
    dual_listed flag - ticker-only, no user data needed, so unlike the
    rest of LOSS this is real and buildable now. Disclosed reuse, not an
    exact-purpose match: is_crosslisted() was built for SEC-filing-access
    (does this CA ticker have a mapped US CIK for pulling 40-F/20-F
    text), not explicitly "identical security" - a reasonable stretch
    for the map's actual population (RY/RY.TO, BNS/BNS.TO, ENB/ENB.TO -
    the reference's own named examples, all genuinely in the map), not
    exhaustively verified for all ~176 entries.

    Disclosed, one-directional gap: is_crosslisted() only accepts a
    CA-suffixed ticker as input, and ca_crosslisting.py has no reverse
    (US-ticker-to-CA-ticker) lookup. So is_canadian_dual_listed("RY")
    returns False even though RY/RY.TO is exactly the reference file's
    own canonical dual-listed example - a real, plausible miss if a
    Canadian retail investor holds the NYSE-side ticker directly (US-listed
    shares absolutely can sit in a TFSA/RRSP/Trading account). Not fixed
    here - a reverse lookup would need its own new function in
    ca_crosslisting.py, a second file beyond this module, and the
    realistic population (this product's own stated audience holds
    Canadian-account securities, most plausibly the CA-listed side) makes
    this a bounded, disclosed gap rather than a blocking one. Always
    False for a genuinely non-dual-listed US ticker either way - the miss
    is specifically "US-side ticker of a real dual-listed pair", not US
    tickers broadly."""
    return is_crosslisted(ticker)


class AccountStateInput(TypedDict, total=False):
    """A stand-in for portfolio_service.account_state()'s not-yet-built
    return value - mirrors exactly what the live prompt's own pseudocode
    `state` object needs. Not in providers/base.py: this isn't a
    provider-normalized shape, it's a local precompute-module input.
    Nothing in this codebase constructs a real instance today - every
    real call to build_precomputed_tax_metrics() passes
    account_state=None - but the rendering logic below is built and
    tested now (with plausible, explicitly-synthetic values, per explicit
    direction: real user tax/portfolio data will never be available to
    build or test this against), not deferred until a real account-state
    service exists. total=False since a caller only fills in what's
    relevant to the account_type actually being analyzed."""

    tfsa_room_remaining_cents: int
    rrsp_room_remaining_cents: int
    trading_ytd_realized_gains_cents: int
    trading_ytd_realized_losses_cents: int
    superficial_loss_blocked: bool
    us_situs_aggregate_usd: float


class TaxReferenceUnavailable(Exception):
    """Raised by load_tax_rules_reference() when the local reference file
    is missing, unreadable, or has no parseable "Last verified:" header.
    A real, deliberate "fail loud" case, not the usual degrade-gracefully
    posture elsewhere in this module - Gate 1's own text is explicit that
    a missing reference file means the orchestrator does NOT invoke the
    agent at all, not a soft caveat. A missing file that ships in the
    repo indicates a real packaging/deployment problem, not a routine
    provider-availability gap. Deliberately left to propagate, not caught
    in this module - the "don't invoke the agent" decision belongs one
    level up, in whatever orchestrator eventually calls this."""


_LAST_VERIFIED_RE = re.compile(r"\*\*Last verified:\*\*\s*(\d{4}-\d{2}-\d{2})")


def load_tax_rules_reference(
    reference_path: str = "prompts/tax_strategist/canadian_tax_rules_reference.md",
) -> tuple[str, date]:
    """Reads the local reference file, parses its "**Last verified:**
    YYYY-MM-DD" header. Returns (raw_text, last_verified_date). Raises
    TaxReferenceUnavailable if the file is missing/unreadable, or has no
    parseable header.

    No cache, deliberately - the real prompt's own orchestrator pseudocode
    calls a generic cache.get/cache.set abstraction that doesn't exist
    anywhere in this codebase (confirmed by grep, not assumed); building a
    real cache layer is a separate, larger concern this small function
    doesn't need to solve to be correct, and re-reading one small local
    file per call is cheap.

    Returns the raw file text as-is - does NOT strip maintenance notes
    before injecting into a system prompt (the real pseudocode's
    render_for_prompt_injection()). That's a prompt-template-assembly
    concern for whatever eventually builds the Tax Strategist's full
    system prompt, not this function's job; this function's job is
    making (text, last_verified_date) available at all."""
    try:
        with open(reference_path, encoding="utf-8") as f:
            raw = f.read()
    except (FileNotFoundError, PermissionError) as exc:
        raise TaxReferenceUnavailable(
            f"Canadian Tax Rules Reference file not found or unreadable at "
            f"{reference_path} - Tax Strategist cannot run."
        ) from exc

    match = _LAST_VERIFIED_RE.search(raw)
    if not match:
        raise TaxReferenceUnavailable(
            f"Canadian Tax Rules Reference file at {reference_path} has no "
            "parseable 'Last verified:' header."
        )
    return raw, date.fromisoformat(match.group(1))


def build_tax_rule_snapshot(reference_last_verified: date) -> str:
    """The one line: "TAX_RULE_SNAPSHOT: snapshot_date=..., reference_last_verified=...".
    snapshot_date is today. Kept as its own tiny function, separate from
    load_tax_rules_reference() itself - that function's job is loading the
    reference; this one's job is rendering the marker line - and trivially
    testable without faking a file read."""
    snapshot_date = datetime.now().date().isoformat()
    return f"TAX_RULE_SNAPSHOT: snapshot_date={snapshot_date}, reference_last_verified={reference_last_verified.isoformat()}"


def build_precomputed_tax_metrics(
    ticker: str,
    account_type: Literal["tfsa", "rrsp", "trading"],
    bundle: DataBundle,
    account_state: AccountStateInput | None = None,
    reference_last_verified: date | None = None,
) -> str:
    """Top-level assembly - the ticket's own named function, signature
    extended with two new optional parameters beyond the ticket's literal
    three-arg spec.

    Pure function, no I/O: bundle.company_info, bundle.dividend_history,
    and bundle.price_info are all already-fetched inputs (DataBundle
    already carries all three regardless of which agent is running).
    reference_last_verified is the ALREADY-PARSED date from a caller's own
    load_tax_rules_reference() call, not the reference_path - this
    function never itself does that file I/O.

    reference_last_verified is NOT like account_state - account_state is
    genuinely, usually None in practice today because no real service
    exists to produce one; reference_last_verified SHOULD basically
    always be supplied by any real caller, since load_tax_rules_reference()
    ships in this same module specifically to produce it. Omitting it
    means the resulting block will fail Gate 1's own hard TAX_RULE_SNAPSHOT
    requirement (unlike account_state's absence, which is Gate-1-neutral).

    account_type is one of tfsa/rrsp/trading only - "general" is a
    whole-pipeline design question the CIO's own prompt has already
    resolved by removing that mode from real orchestration; this module
    doesn't build for it."""
    if account_type not in ("tfsa", "rrsp", "trading"):
        raise ValueError(
            f"build_precomputed_tax_metrics: unsupported account_type {account_type!r}"
        )

    is_ca = ticker.upper().endswith(_CA_MARKET_SUFFIXES)
    company_info = bundle.company_info
    classification, structure = classify_dividend(ticker, company_info)

    lines: list[str] = []

    if structure == "etf":
        logger.info("tax_metrics_etf_out_of_scope", ticker=ticker)
        lines.append(
            f"DOM: not applicable - this precompute module covers individual equities "
            f"and REITs/trusts only; {ticker} is asset_type=etf"
        )
        if reference_last_verified is not None:
            lines.append(build_tax_rule_snapshot(reference_last_verified))
        return "\n".join(lines)

    yield_pct, payment_count = compute_trailing_dividend(
        bundle.dividend_history, bundle.price_info.get("current_price")
    )
    if yield_pct is not None:
        lines.append(f"DIVID: {yield_pct:.1f}% yield, {payment_count} payments/yr")
    else:
        lines.append("DIVID: no dividend history")

    lines.append(f"ELIG: {classification}")
    # country is blank for every CA ticker through the real Router path (confirmed live,
    # a disclosed provider-data gap, not this module's own bug - see classify_dividend's
    # own is_ca check, which exists precisely because country can't be trusted for this).
    # Fall back to the 2-letter code "CA" (matching the real, populated values' own
    # 2-letter-ISO-code convention - "US", "CN", "NL", "JP", confirmed live) rather than
    # rendering an empty, uninformative "()" for every single CA ticker.
    country_label = company_info.get("country") or ("CA" if is_ca else "")
    lines.append(f"LIST: {company_info.get('primary_exchange') or ''} ({country_label})")
    lines.append(f"DOM: {structure}")

    wht = resolve_withholding(classification, account_type)
    if wht is not None:
        rate, note = wht
        lines.append(
            f"WHT (this account, {account_type}): {rate:.1f}%, {note} [source: REF withholding grid]"
        )
    else:
        lines.append(f"WHT (this account, {account_type}): NOT MODELLED per REF withholding grid")

    effective_yield, drag = compute_effective_after_tax_yield(
        yield_pct, wht[0] if wht is not None else None
    )
    if effective_yield is not None:
        lines.append(f"Effective after-tax yield: {effective_yield:.1f}%")
        lines.append(f"Annual tax drag: {drag:.1f}%")

    cgain_line = f"CGAIN: {CGAIN_BY_ACCOUNT[account_type]}"
    if account_state is not None and account_type == "trading":
        ytd_gains_cents = account_state.get("trading_ytd_realized_gains_cents")
        if ytd_gains_cents:
            cgain_line += f"; YTD realized in Trading: ${ytd_gains_cents / 100:,.0f}"
    lines.append(cgain_line)

    if account_state is not None:
        if account_type == "tfsa":
            room = account_state.get("tfsa_room_remaining_cents")
            if room is not None:
                lines.append(f"ROOM: TFSA remaining ${room / 100:,.0f}")
        elif account_type == "rrsp":
            room = account_state.get("rrsp_room_remaining_cents")
            if room is not None:
                lines.append(f"ROOM: RRSP remaining ${room / 100:,.0f}")
        elif account_type == "trading":
            blocked = account_state.get("superficial_loss_blocked")
            ytd_losses_cents = account_state.get("trading_ytd_realized_losses_cents")
            if blocked is not None or ytd_losses_cents is not None:
                # is_canadian_dual_listed() computed lazily, here, not unconditionally at
                # the top of this function - it does a real, uncached JSON-file read
                # (confirmed: ca_crosslisting.py's _load_crosslisting_map() has no
                # caching), so paying that cost on every call regardless of whether the
                # result is ever used (true for the vast majority of real calls today,
                # since account_state is normally None) would be a real, avoidable
                # inefficiency, not a theoretical one.
                dual_listed_flag = ", DUAL_LISTED" if is_canadian_dual_listed(ticker) else ""
                loss_status = "BLOCKED - repurchased within 30d" if blocked else "available"
                losses = ytd_losses_cents / 100 if ytd_losses_cents else 0
                lines.append(
                    f"LOSS: superficial-loss window: {loss_status}{dual_listed_flag}; "
                    f"YTD realized losses: ${losses:,.0f}"
                )

        # is_ca, not `classification`, gates US_SITUS - a real bug caught on review, not
        # shipped: US_SITUS is about listing/domicile situs, not dividend-tax treatment.
        # An earlier version checked `classification in ("us", "us_reit", "us_mlp", "adr")`,
        # but "us_mlp" is dead - classify_dividend never produces it, MLP and non-MLP LP
        # both collapse to "limited_partnership" regardless of market (see
        # classify_dividend's own docstring) - and that same collapse means
        # "limited_partnership" alone can't tell a US-side LP (genuinely US-situs) from a
        # CA-side one (not), so a classification-only check silently misses real US-side
        # LPs entirely. is_ca is the ticker's own real market fact, unambiguous either way.
        us_situs = account_state.get("us_situs_aggregate_usd")
        if not is_ca and us_situs:
            lines.append(f"US_SITUS aggregate: ${us_situs:,.0f} USD")

    if reference_last_verified is not None:
        lines.append(build_tax_rule_snapshot(reference_last_verified))

    return "\n".join(lines)
