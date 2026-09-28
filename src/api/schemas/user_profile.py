from datetime import date, datetime
from pydantic import BaseModel, ConfigDict
from typing import Literal
from uuid import UUID


class TimeSensitiveCashNeed(BaseModel):
    description: str
    amount: float
    deadline: date


# Shared personalization fields -- everything optional, since a profile is
# usable without any of them filled in. Base for Create/Update/Response so
# adding a field later touches one place, not three. time_sensitive_cash_needs
# replaces the whole list when supplied on an update; it does not append.
class _UserProfileFields(BaseModel):
    province: Literal["ON"] | None = None
    marginal_tax_rate_override_pct: float | None = None
    income_annual: float | None = None
    tfsa_room_remaining: float | None = None
    tfsa_room_as_of: datetime | None = None
    rrsp_room_remaining: float | None = None
    rrsp_room_as_of: datetime | None = None
    time_sensitive_cash_needs: list[TimeSensitiveCashNeed] | None = None
    max_position_size_pct: float | None = None
    max_sector_weight_pct: float | None = None
    rrsp_annual_contribution_cap: float | None = None


# what client sends to create a profile -- a full profile can be set up in
# one call; every personalization field is optional here too.
class UserProfileCreate(_UserProfileFields):
    display_name: str
    is_active: bool = True


# what client sends to update a profile -- everything optional, PATCH semantics.
class UserProfileUpdate(_UserProfileFields):
    display_name: str | None = None
    is_active: bool | None = None


# what API returns
class UserProfileResponse(_UserProfileFields):
    model_config = ConfigDict(from_attributes=True)

    user_id: UUID
    display_name: str
    is_active: bool
