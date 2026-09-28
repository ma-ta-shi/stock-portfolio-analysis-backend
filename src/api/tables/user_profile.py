from sqlalchemy import String, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from api.database import Base
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    display_name: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    # Tax context (86bc8efe7) -- province is "ON" only for v1, the one province
    # with real backing bracket/DTC data (see the tax-bracket-sourcing ticket);
    # marginal_tax_rate_override_pct lets a user who knows their real rate skip
    # the province+income derivation entirely, in which case province isn't
    # required.
    province: Mapped[Optional[str]] = mapped_column(String(2))  # "ON" only for v1
    marginal_tax_rate_override_pct: Mapped[Optional[float]]
    income_annual: Mapped[Optional[float]]

    # TFSA/RRSP contribution room (86bc8efe7) -- manually entered/updated, not
    # derived from a transaction ledger (portfolio_holdings/portfolio_transactions
    # aren't live). Deliberately flat on this table rather than a separate
    # account_state table: at this scale (2 users), that's designing for a
    # future system's shape before that system's design exists. Moving these
    # into a real account_state table later, once portfolio tracking is real,
    # is a trivial migration -- not a retrofit worth avoiding pre-emptively.
    tfsa_room_remaining: Mapped[Optional[float]]
    tfsa_room_as_of: Mapped[Optional[datetime]]
    rrsp_room_remaining: Mapped[Optional[float]]
    rrsp_room_as_of: Mapped[Optional[datetime]]

    # Time-sensitive cash needs (86bc8efe7) -- [{description, amount, deadline}].
    # No account field: nothing in this codebase earmarks specific holdings
    # against a specific need, so tagging one would be false precision. No
    # current consumer -- stored for the future Portfolio Optimizer chain.
    # A PATCH replaces the whole list; it does not append.
    time_sensitive_cash_needs: Mapped[Optional[list]] = mapped_column(JSON)

    # Portfolio Optimizer inputs (86bc8efe7) -- stored only. Zero consumer
    # anywhere in the live pipeline today; these exist for the not-yet-built
    # Portfolio Optimizer (max_position_size_pct/max_sector_weight_pct ->
    # Health Assessor's concentration check, rrsp_annual_contribution_cap ->
    # Action Synthesizer's placement rule). Do not read these as live-wired.
    max_position_size_pct: Mapped[Optional[float]]
    max_sector_weight_pct: Mapped[Optional[float]]
    rrsp_annual_contribution_cap: Mapped[Optional[float]]