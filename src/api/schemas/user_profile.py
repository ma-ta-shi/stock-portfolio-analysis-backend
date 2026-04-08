from pydantic import BaseModel, EmailStr
from uuid import UUID

# what client sends
class UserProfileCreate(BaseModel):
    display_name: str
    is_active: bool = True

# what API returns
class UserProfileResponse(BaseModel):
    user_id: UUID
    display_name: str
    is_active: bool

    class Config:
        from_attributes = True